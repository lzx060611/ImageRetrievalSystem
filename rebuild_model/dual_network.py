import torch
import torch.nn as nn
import torch.nn.functional as F

class DualBranchHashNet(nn.Module):
    """
    双分支哈希网络（特征拼接版本）
    全局分支：预训练的 Encoder（冻结）
    局部分支：轻量级 CNN 处理 ROI，使用最大池化聚合
    融合层：拼接后生成哈希码和分类 logits
    """
    def __init__(self, global_encoder, num_classes=13, hash_len=64, roi_feat_dim=128):
        super().__init__()
        # ----- 全局编码器（预训练，冻结）-----
        self.global_encoder = global_encoder
        for param in self.global_encoder.parameters():
            param.requires_grad = False   # 冻结全局分支，防止噪声干扰
        self.global_ln = nn.LayerNorm(1024)

        # ----- 局部编码器（可训练）-----
        # 输入：单通道 ROI，已 resize 到 224x224
        # 输出：roi_feat_dim 维特征向量
        self.local_encoder = nn.Sequential(
            # Conv1: 1->32, stride=2, 112x112
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Conv2: 32->64, stride=2, 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Conv3: 64->128, stride=2, 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 全局平均池化 -> 128维
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            # 全连接降至 roi_feat_dim
            nn.Linear(128, roi_feat_dim),
            nn.LayerNorm(roi_feat_dim), # 强制特征均值为0，方差为1
            nn.ReLU(inplace=True) # 保证输出为正，与全局特征(也是ReLU输出)量级对齐
        )

        # 全局特征维度（Encoder 中 fc3 输出 1024）
        self.global_feat_dim = 1024
        self.roi_feat_dim = roi_feat_dim
        fused_dim = self.global_feat_dim + roi_feat_dim

        # ----- 融合层 -----
        self.dropout_global = nn.Dropout(p=0.5) # 新增：50%的概率丢弃全局特征
        self.fc_fuse = nn.Linear(fused_dim, 512)
        self.fc_hash = nn.Linear(512, hash_len)   # 哈希码输出
        self.fc_class = nn.Linear(512, num_classes)  # 分类 logits
        self.tanh = nn.Tanh()

    def get_global_feature(self, x):
        """
        从全局编码器提取 fc3 层的特征（1024维）
        复用 Encoder 的卷积和全连接层，只计算到 fc3
        """
        # 卷积部分
        x = F.relu(self.global_encoder.conv1(x))
        x, _ = F.max_pool2d(x, (3,3), (2,2), return_indices=True)
        x = F.relu(self.global_encoder.conv2(x))
        x, _ = F.max_pool2d(x, (3,3), (2,2), return_indices=True)
        x = F.relu(self.global_encoder.conv3(x))
        x = F.relu(self.global_encoder.conv4(x))
        x = F.relu(self.global_encoder.conv5(x))
        x, _ = F.max_pool2d(x, (3,3), (2,2), return_indices=True)
        # 展平
        x = x.view(x.size(0), 256 * 6 * 6)
        # 全连接部分
        x = F.relu(self.global_encoder.fc1(x))
        x = F.relu(self.global_encoder.fc2(x))
        x = F.relu(self.global_encoder.fc3(x))   # [B, 1024]
        return x

    def forward(self, x, roi_list):
        """
        输入：
            x: 整图张量 [B, 1, 224, 224]
            roi_list: list of list，每个元素是该样本的 ROI 张量列表，每个 ROI [1,224,224]
                      若某样本无 ROI，对应列表为空
        输出：
            class_out: 分类 logits [B, num_classes]
            hash_code: 哈希码（tanh 输出，范围 -1~1） [B, hash_len]
        """
        B = x.size(0)
        device = x.device

        # 1. 全局特征
        global_feat = self.get_global_feature(x)   # [B, 1024]
        global_feat = self.global_ln(global_feat)
        global_feat = self.dropout_global(global_feat)
        # 2. 局部特征聚合（最大池化）
        local_feats = []
        for i in range(B):
            rois = roi_list[i]
            if not rois:   # 没有检测框，局部特征置为零向量
                feat = torch.zeros(self.roi_feat_dim, device=device)
            else:
                # 将所有 ROI 堆叠成一个 batch
                rois_tensor = torch.stack(rois, dim=0).to(device)   # [n, 1, 224, 224]
                roi_outs = self.local_encoder(rois_tensor)         # [n, roi_feat_dim]
                feat = roi_outs.max(dim=0)[0]                      # 最大池化，取最显著特征
            local_feats.append(feat)
        local_feat_batch = torch.stack(local_feats, dim=0)         # [B, roi_feat_dim].

        # ================= 关键修正区域 =================
        # 必须加上 if self.training，保证测试评估时绝对不丢弃特征！
        if self.training:
            # 使用 B 作为 batch size 生成 mask，0.1 是丢弃概率
            roi_mask = (torch.rand(B, 1, device=device) > 0.1).float()
            # 变量名必须是 local_feat_batch
            local_feat_batch = local_feat_batch * roi_mask
        # ================================================

        # 3. 特征拼接与融合
        combined = torch.cat([global_feat, local_feat_batch], dim=1)  # [B, 1024+roi_feat_dim]
        h = F.relu(self.fc_fuse(combined))
        hash_code = self.tanh(self.fc_hash(h))
        class_out = self.fc_class(h)
        return class_out, hash_code