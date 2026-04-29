import os
os.environ["OMP_NUM_THREADS"] = "16"   # 根据你的 CPU 核心数调整
import torch
import torch.nn as nn
from loss_fn_2 import MultiLabelHashLoss
from tqdm import tqdm  # 批量提取时显示进度条
import warnings
import numpy as np
warnings.filterwarnings("ignore")  # 忽略无关警告
from NIH import NIH_dataset 
from torch.utils.data import DataLoader
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from network import Encoder
from train_val_pro import evaluate_hash_retrieval
import torchvision.transforms as T
warnings.filterwarnings("ignore")  # 忽略无关警告import os
# 这一行有时能触发自定义算子的注册
print(f"Torchvision version: {torchvision.__version__}")
torch.multiprocessing.set_sharing_strategy('file_system')

#一、加载模型
# 1. 模型配置（改算法时优先改这部分）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_ROOT = "./images_001/images"  # 你的图片根目录
train_LIST = "images_001/Hashsystem_dataset/train_ids.txt"  # 你的训练集文件
train_label="images_001/Hashsystem_dataset/train_labels.txt" #训练集标签

query_LIST="images_001/Hashsystem_dataset/query.txt"
query_label="images_001/Hashsystem_dataset/query_labels.txt"
val_LIST="images_001/Hashsystem_dataset/train_val.txt" #交叉验证集 
val_label="images_001/Hashsystem_dataset/train_val_label.txt" #交叉验证集标签

gallery_LIST="images_001/Hashsystem_dataset/test_ids.txt"#训练集
gallery_label="images_001/Hashsystem_dataset/test_labels.txt"#训练集标签
BATCH_SIZE = 64          # 你的批次大小
writer=SummaryWriter("JC_log")#训练记录


    
    
# ===================== 模型核心逻辑（封装成可复用函数）=====================
def load_model():
    """
    封装模型加载逻辑：改算法时只需修改这个函数！
    返回：model, preprocess_val（推理预处理）, tokenizer, hash_adapter
    """
    gray_train_transform = T.Compose([
    T.Resize((224, 224)),                  # 统一尺寸
    T.RandomHorizontalFlip(p=0.5),         # 随机水平翻转（和CLIP一致）
    T.ToTensor(),                          # 转为 [1, H, W] 张量
    T.Normalize(mean=[0.485], std=[0.239])     # 灰度图标准化到 [-1,1]，最稳定
    ])

    gray_val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.239])
    ])
    # 加载模型+预处理工具（改算法时替换这里，比如换成ResNet/自定义模型）
    model, preprocess_train, preprocess_val = Encoder(19,64),gray_train_transform, gray_val_transform
    
    # 模型配置（设备分配）
    model = model.to(DEVICE)
    
    print(f"✅ 模型加载完成 | 设备：{DEVICE} ")
    print(f"✅ HashAdapter加载完成 | 输出维度：64 | 初始化：正交")
    return model, preprocess_train,preprocess_val

# 全局加载模型（只加载一次，避免重复加载浪费内存）
model, preprocess_train,preprocess_val= load_model()
base_lr=1e-4
optimizer = torch.optim.Adam([
    {'params': list(model.parameters()), "lr": base_lr},
], weight_decay=1e-3) # 可以加一点 weight_decay 防止过拟合
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,200,0)
# ===================== 完整断点续训：加载 8001 所有参数 =====================
# checkpoint_path = "checkpoint_iter_24001.pth"  # 改成你的8001断点
# checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# 1. 加载模型 + 哈希适配器
# model.load_state_dict(checkpoint["model"])
# hash_adapter.load_state_dict(checkpoint["hash_adapter"])

# 2. 加载优化器 + 学习率调度器（训练状态完全恢复）
# optimizer.load_state_dict(checkpoint["optimizer"])
# scheduler.load_state_dict(checkpoint["scheduler"])

# 3. 自动恢复迭代次数（不用手动改！）
# current_iter = checkpoint["current_iter"]

# print(f"✅ 完美加载断点：{checkpoint_path}")
# print(f"✅ 恢复迭代次数：{current_iter}")
# print(f"✅ 优化器/学习率状态已同步，训练无缝继续！")
# print(f"📌 主模型 (model) 学习率 = {optimizer.param_groups[0]['lr']:.6f}")
# print(f"📌 HashAdapter 学习率   = {optimizer.param_groups[1]['lr']:.6f}")

# print(f"✅ 成功加载断点：{checkpoint_path}")



def custom_collate_fn_train(batch):
    """
    你的自定义collate_fn
    输入：batch是列表，每个元素是数据集返回的 (image, img_path,tag_listrg)
    输出：整理后的 (图像批次张量, 路径列表)
    """
    images, paths,tags = zip(*batch)  # 分离图像和路径（images是PIL图像元组，paths是路径元组）
    N=len(images)
    # 生成所有无序索引对 (i, j), i < j
    indices_i = []
    indices_j = []
    for i in range(N):
        for j in range(i+1, N):
            indices_i.append(i)
            indices_j.append(j)
    # 1. 用preprocess_val逐个处理PIL图像，转为张量（和你的逻辑一致）
    processed_imgs = [preprocess_train(img) for img in images]
    # 2. 将多个张量堆叠成批次（shape: [batch_size, 3, 224, 224]）
    # 根据索引构建输出批次
    image1_batch = torch.stack([processed_imgs[i] for i in indices_i])
    image2_batch = torch.stack([processed_imgs[j] for j in indices_j])
    tag1_batch = torch.stack([tags[i] for i in indices_i])
    tag2_batch = torch.stack([tags[j] for j in indices_j])
    
    # 可选：路径
    path1_batch = [paths[i] for i in indices_i]
    path2_batch = [paths[j] for j in indices_j]
    return image1_batch, image2_batch, tag1_batch, tag2_batch, path1_batch, path2_batch

def custom_collate_fn(batch):
    """
    你的自定义collate_fn
    输入：batch是列表，每个元素是数据集返回的 (image, img_path,tag_listrg)
    输出：整理后的 (图像批次张量, 路径列表)
    """
    images, paths,tags = zip(*batch)  # 分离图像和路径（images是PIL图像元组，paths是路径元组）
    # 1. 用preprocess_val逐个处理PIL图像，转为张量（和你的逻辑一致）
    processed_images = [preprocess_val(img) for img in images]
    # 2. 将多个张量堆叠成批次（shape: [batch_size, 3, 224, 224]）
    image_batch = torch.stack(processed_images)
    tag_batch = torch.stack(tags)
    # 3. 路径直接保持列表形式
    path_batch = list(paths)   
    return image_batch, path_batch,tag_batch




#=================================加载数据集======================
train_dataset=NIH_dataset(img_root=IMG_ROOT,img_list=train_LIST,tag_path=train_label)
train_loader=DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=custom_collate_fn_train,
    num_workers=8,
    pin_memory=True,  # 👈 加上这个，加速数据传输
    prefetch_factor=2,   # 预加载2个批次，让GPU一直有活干
    persistent_workers = True,
    drop_last=True
)
# 加载验证集（和训练集逻辑完全一样）
val_dataset = NIH_dataset(img_root=IMG_ROOT, img_list=val_LIST, tag_path=val_label)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # 评估必须关闭shuffle
    collate_fn=custom_collate_fn,
    num_workers=8, 
    pin_memory=True,  # 👈 加上这个，加速数据传输
    prefetch_factor=2,   # 预加载2个批次，让GPU一直有活干
    persistent_workers = True
)
# 加载验证集（和训练集逻辑完全一样）
query_dataset = NIH_dataset(img_root=IMG_ROOT, img_list=query_LIST, tag_path=query_label)
query_loader = DataLoader(
    dataset=query_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # 评估必须关闭shuffle
    collate_fn=custom_collate_fn,
    num_workers=8, 
    pin_memory=True,  # 👈 加上这个，加速数据传输
    prefetch_factor=2,   # 预加载2个批次，让GPU一直有活干
    persistent_workers = True
)
# 加载验证集（和训练集逻辑完全一样）
gallery_dataset = NIH_dataset(img_root=IMG_ROOT, img_list=gallery_LIST, tag_path=gallery_label)
gallery_loader = DataLoader(
    dataset=gallery_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # 评估必须关闭shuffle
    collate_fn=custom_collate_fn,
    num_workers=8, 
    pin_memory=True,  # 👈 加上这个，加速数据传输
    prefetch_factor=2,   # 预加载2个批次，让GPU一直有活干
    persistent_workers = True
)

loss_fn=MultiLabelHashLoss(64)



# ----------------------配置训练信息----------------------------
current_iter=0#当前迭代次数
total_loss=0#每100次迭代总损失
total_JC_loss=0#相似损失
total_CA_loss=0#分类损失


total_epoch = 200 # 你的总迭代次数
start_time=time.time()

if __name__ == '__main__':
    print("---------------------------训练开始-----------------------")
    model.train()
    print(f"当前学习率为{optimizer.param_groups[0]["lr"]}")
    for epoch in range(total_epoch):
            for img1, img2, target1, target2, path1, path2 in train_loader:
                img1,img2=img1.to(DEVICE),img2.to(DEVICE)
                target1,target2 = target1.to(DEVICE),target2.to(DEVICE)
                #前向传播
                output, h = model(img1)
                output1, h1 = model(img2)
                weighted_loss, loss_ahdl, loss_pmcl = loss_fn.total_loss(
                outputs=output, labels=target1,
                outputs1=output1, labels1=target2,
                h=h, h1=h1
                )
                total_CA_loss+=loss_pmcl.item()
                total_JC_loss+=loss_ahdl.item()
                total_loss+=weighted_loss.item()
                #反向传播
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()


                current_iter+=1
                if current_iter%100==0:
                    current_lr=optimizer.param_groups[0]["lr"]
                    used_time=time.time()-start_time
                    print(f"当前迭代次数为{current_iter},此100次平均损失为{total_loss/100},距离平均损失为{total_JC_loss/100},分类平均损失为{total_CA_loss/100},花费时间为{used_time},当前学习率为{current_lr}")
                    writer.add_scalar("train_loss",total_loss/100,current_iter)
                    writer.add_scalar("train_JC_loss",total_JC_loss/100,current_iter)
                    writer.add_scalar("train_CA_loss",total_CA_loss/100,current_iter)
                    writer.add_scalar("train_lr",current_lr,current_iter)


                    #重置数据
                    total_loss=0
                    total_JC_loss=0
                    total_CA_loss=0
                    start_time=time.time()
                
                if current_iter%2000==1 or current_iter==113600:#根据计算总的迭代次数为12170
                    mapw,aCG,nDCG,top5=evaluate_hash_retrieval(model,gallery_loader,val_loader,DEVICE,5)
                    print(f"\n✅ 验证集评估结果：mAPw={mapw:.4f} | ACG={aCG:4f}| nDCG@100={nDCG:4f}| Precision@5={top5:.4f}")
                    
                     # 2. 写入TensorBoard
                    writer.add_scalar("val_mAPw", mapw, current_iter)
                    writer.add_scalar("val_aCG", aCG, current_iter)
                    writer.add_scalar("val_nDCG@100", nDCG, current_iter)
                    writer.add_scalar("val_Precision@5", top5, current_iter)

                    if current_iter%10000==1:
                        # 3. 保存模型
                        torch.save({
                            "model": model.state_dict(),
                            "mAPw": mapw,
                            "ACG":aCG,
                            "nDCG":nDCG,
                            "top5":top5,
                            # 2. 优化器 + 学习率调度器（完整续训必备！）
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "current_iter":current_iter
                        }, f"check_point/checkpoint_iter_{current_iter}.pth")

                        print("------------------------模型已保存--------------------")
            scheduler.step()

                

                





