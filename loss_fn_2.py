import torch
import torch.nn.functional as F
import pickle
import os

class MultiLabelHashLoss:
    """
    多标签哈希检索损失函数，包含：
    - Adaptive Hamming Distance Loss (AHDL)
    - Pairwise Multi-label Classification Loss (PMCL)
    """
    def __init__(self, hash_code, distG_path=None, lambda1=1.5, lambda2=1.5):
        """
        Args:
            hash_code: 哈希码长度 K
            distG_path: 预计算距离表文件路径，如果为 None 则自动根据 hash_code 构建默认路径
            lambda1: AHDL 的权重
            lambda2: PMCL 的权重
        """
        self.hash_code = hash_code
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # 加载预计算的距离表 (ground truth Hamming distance)
        if distG_path is None:
            # 默认路径，与原始 train.py 保持一致
            distG_path = f'Distances/distances_{hash_code}.pkl'
        with open(distG_path, 'rb') as f:
            self.distG = pickle.load(f)

        # 分类损失函数 (BCEWithLogitsLoss, reduction='sum' 与原始代码一致)
        self.classification_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')

    def adaptive_hamming_distance(self, h, h1, labels, labels1):
        """
        计算预测汉明距离和对应的目标汉明距离

        Args:
            h, h1: 连续哈希码 (batch_size, hash_code)，已经过 tanh 激活
            labels, labels1: 多标签 (batch_size, num_classes)

        Returns:
            cos_distH: 预测的汉明距离 (batch_size,)
            g_distH: 目标汉明距离 (batch_size,)
        """
        # 1. 计算预测汉明距离（基于余弦相似度）
        cos = F.cosine_similarity(h, h1, dim=1, eps=1e-6)
        cos_distH = F.relu((1 - cos) * self.hash_code / 2)   # shape: (batch,)

        # 2. 计算标签的并集和交集大小
        sum_label = labels + labels1
        union_label = (sum_label >= 1).sum(dim=1, keepdim=False)      # n^(1)
        intersection_label = (sum_label >= 2).sum(dim=1, keepdim=False) # n^(2)

        # 3. 根据并集和交集从距离表中查找目标汉明距离
        # distG 是一个 list of list: distG[union][intersection]
        g_distH_list = [self.distG[u.item()][i.item()] for u, i in zip(union_label, intersection_label)]
        g_distH = torch.tensor(g_distH_list, device=h.device)

        return cos_distH, g_distH

    def ahdl_loss(self, h, h1, labels, labels1):
        """
        Adaptive Hamming Distance Loss
        L_AHDL = sum( log( cosh( (pred_hd - gt_hd) / K ) ) )
        """
        pred_hd, gt_hd = self.adaptive_hamming_distance(h, h1, labels, labels1)
        # 原始代码: (torch.div(cos_distH - g_distH.float(), hash_code)).cosh().log().sum()
        diff = (pred_hd - gt_hd.float()) / self.hash_code
        loss = diff.cosh().log().sum()
        return loss

    def pmcl_loss(self, outputs, labels, outputs1, labels1):
        """
        Pairwise Multi-label Classification Loss
        L_PMCL = BCEWithLogitsLoss(outputs, labels) + BCEWithLogitsLoss(outputs1, labels1)
        """
        loss1 = self.classification_loss_fn(outputs, labels)
        loss2 = self.classification_loss_fn(outputs1, labels1)
        return loss1 + loss2

    def total_loss(self, outputs, labels, outputs1, labels1, h, h1):
        """
        计算加权总损失
        total = lambda1 * AHDL + lambda2 * PMCL
        """
        loss_ahdl = self.ahdl_loss(h, h1, labels, labels1)
        loss_pmcl = self.pmcl_loss(outputs, labels, outputs1, labels1)
        total = self.lambda1 * loss_ahdl + self.lambda2 * loss_pmcl
        return total, loss_ahdl, loss_pmcl