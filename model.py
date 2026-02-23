# model.py
import os
import torch
import torch.nn as nn
import open_clip  # 你用的模型加载库
from PIL import Image
import pydicom  # 处理DICOM医学图像
from tqdm import tqdm  # 批量提取时显示进度条
import warnings
import numpy as np
warnings.filterwarnings("ignore")  # 忽略无关警告
from NIH import NIH_dataset 
from torch.utils.data import DataLoader

#一、加载模型
# 1. 模型配置（改算法时优先改这部分）
MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 二. 批量提取配置（根据你的数据集调整路径）
Is_Test=True#是否使用测试集
if Is_Test:
    IMG_ROOT = "./images_001/test_image"  # 你的测试图片(前500)根目录
    IMG_LIST = "images_001\\test_list.txt"  # 你的测试图片列表文件
else:
    IMG_ROOT = "./images_001/images"  # 你的图片根目录
    IMG_LIST = "images_001\\train_val_list.txt"  # 你的图片列表文件
BATCH_SIZE = 50                       # 你的批次大小
SAVE_DIR = "nih_features_db"          # 特征保存文件夹



#三、加载模型
# ===================== HashAdapter 类 =====================
class HashAdapter(nn.Module):
    """
    哈希适配器：将512维特征映射到64维二值哈希码
    """
    def __init__(self, input_dim=512, output_dim=64):
        super().__init__()
        # 添加线性层，无偏置
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        # 正交初始化权重
        nn.init.orthogonal_(self.projection.weight)
    
    def forward(self, x):
        """
        前向传播：输入512维特征，输出64维二值哈希码
        """
        # 线性投影
        x = self.projection(x)
        # 二值化：大于0置为1，否则置为0
        x = (x > 0).float()
        return x

# ===================== 模型核心逻辑（封装成可复用函数）=====================
def load_model():
    """
    封装模型加载逻辑：改算法时只需修改这个函数！
    返回：model, preprocess_val（推理预处理）, tokenizer, hash_adapter
    """
    # 解决国内HF加载慢的问题
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 加载模型+预处理工具（改算法时替换这里，比如换成ResNet/自定义模型）
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    
    # 创建哈希适配器
    hash_adapter = HashAdapter(input_dim=512, output_dim=64)
    
    # 尝试加载已保存的HashAdapter权重
    SAVE_PATH = os.path.join("nih_features_db", "nih_biomedclip_hash_64bit.pt")
    if os.path.exists(SAVE_PATH):
        try:
            features_db = torch.load(SAVE_PATH)
            if "hash_adapter_weight" in features_db:
                hash_adapter.projection.weight.data = features_db["hash_adapter_weight"].to(DEVICE)
                print(f"✅ HashAdapter权重已加载，使用与批量提取相同的权重")
        except Exception as e:
            print(f"⚠️  加载HashAdapter权重失败，使用新的正交初始化权重: {e}")
    
    # 模型配置（评估模式+设备分配）
    model.eval()
    hash_adapter.eval()
    model = model.to(DEVICE)
    hash_adapter = hash_adapter.to(DEVICE)
    
    print(f"✅ 模型加载完成 | 设备：{DEVICE} | 模型：{MODEL_NAME}")
    print(f"✅ HashAdapter加载完成 | 输出维度：64 | 初始化：正交")
    return model, preprocess_val, tokenizer, hash_adapter

# 全局加载模型（只加载一次，避免重复加载浪费内存）
model, preprocess_val, tokenizer, hash_adapter = load_model()


# ----------------------
# 四. 封装特征提取函数（供后端调用）
# 输入：PIL.Image对象
# 输出：归一化后的特征向量（numpy格式，shape=(256,)）
# ----------------------
def extract_image_feature(image):
    """
    功能：单张图片特征提取，返回64维二值哈希码
    参数：image - PIL.Image对象（RGB格式）
    返回：numpy数组（64维0/1哈希码，类型为np.uint8）
    """
    try:
        # 预处理
        processed_img = preprocess_val(image).unsqueeze(0).to(DEVICE)
        
        # 特征提取
        with torch.no_grad():
            img_feature = model.encode_image(processed_img)
            # L2归一化
            img_feature = torch.nn.functional.normalize(img_feature, p=2, dim=1)
            # 哈希映射
            binary_code = hash_adapter(img_feature)
        
        # 转为numpy并去除batch维度，类型转为np.uint8
        binary_code = binary_code.cpu().numpy()[0].astype(np.uint8)
        return binary_code
    
    except Exception as e:
        raise ValueError(f"单张图片特征提取失败：{str(e)}")
    

# ===================== 批量特征提取（完全复用你的DataLoader逻辑）=====================
def custom_collate_fn(batch):
    """
    你的自定义collate_fn
    输入：batch是列表，每个元素是数据集返回的 (image, img_path)
    输出：整理后的 (图像批次张量, 路径列表)
    """
    images, paths = zip(*batch)  # 分离图像和路径（images是PIL图像元组，paths是路径元组）
    # 1. 用preprocess_val逐个处理PIL图像，转为张量（和你的逻辑一致）
    processed_images = [preprocess_val(img) for img in images]
    # 2. 将多个张量堆叠成批次（shape: [batch_size, 3, 224, 224]）
    image_batch = torch.stack(processed_images)
    # 3. 路径直接保持列表形式
    path_batch = list(paths)
    return image_batch, path_batch

def batch_extract_features():
    """
    批量提取特征，返回64维二值哈希码
    无需传参，直接用全局配置项
    """
    # 1. 加载你的NIH数据集
    print(f"📂 加载NIH数据集 | 图片根路径：{IMG_ROOT} | 列表文件：{IMG_LIST}")
    NiH_data = NIH_dataset(IMG_ROOT, IMG_LIST)
    
    # 2. 创建DataLoader
    NIH_loader = DataLoader(
        NiH_data, 
        batch_size=BATCH_SIZE, 
        collate_fn=custom_collate_fn
    )
    print(f"✅ DataLoader创建完成 | 批次大小：{BATCH_SIZE} | 总批次：{len(NIH_loader)}")

    # 3. 批量提取特征
    all_binary_codes = []
    all_paths = []

    print("\n🚀 开始批量提取哈希特征...")
    with torch.no_grad():  # 禁用梯度，节省内存
        # 加进度条，方便看提取进度
        for batch_images, batch_paths in tqdm(NIH_loader, desc="提取进度"):
            # 图像移到设备
            batch_images = batch_images.to(DEVICE)
            
            # 提取特征
            batch_features = model.encode_image(batch_images)
            # L2归一化
            batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
            # 哈希映射
            batch_binary_codes = hash_adapter(batch_features)
            
            # 保存哈希码和路径
            all_binary_codes.append(batch_binary_codes.cpu())
            all_paths.extend(batch_paths)

    # 4. 合并所有哈希码
    all_binary_codes = torch.cat(all_binary_codes, dim=0)
    print(f"\n📊 提取完成 | 哈希码形状：{all_binary_codes.shape} | 有效图片数：{len(all_paths)}")

    # 5. 保存哈希码、路径和HashAdapter权重
    os.makedirs(SAVE_DIR, exist_ok=True)  # 创建保存文件夹
    # 更新保存路径和键名
    SAVE_PATH = os.path.join(SAVE_DIR, "nih_biomedclip_hash_64bit.pt")
    features_db = {
        "binary_codes": all_binary_codes,  # 二值哈希码 (N, 64)
        "image_paths": all_paths,   # 对应图像路径列表
        "hash_adapter_weight": hash_adapter.projection.weight.cpu()  # 保存HashAdapter权重
    }
    torch.save(features_db, SAVE_PATH)

    print(f"✅ 哈希特征数据库已保存到：{SAVE_PATH}")
    print(f"哈希码形状：{all_binary_codes.shape}，包含 {len(all_paths)} 张图像")
    print(f"✅ HashAdapter权重已保存，将在后续检索中使用相同权重")

# ===================== 一键运行批量提取（直接执行model.py即可）=====================
if __name__ == "__main__":
    # 运行批量提取（和你的逻辑完全一致）
    batch_extract_features()