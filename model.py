# model.py
import os
import torch
import open_clip  # 你用的模型加载库
from PIL import Image
import pydicom  # 处理DICOM医学图像
from tqdm import tqdm  # 批量提取时显示进度条
import warnings
warnings.filterwarnings("ignore")  # 忽略无关警告
from NIH import NIH_dataset 
from torch.utils.data import DataLoader

#一、加载模型
# 1. 模型配置（改算法时优先改这部分）
MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 二. 批量提取配置（根据你的数据集调整路径）
IMG_ROOT = "./images_001/images"  # 你的图片根目录
IMG_LIST = "images_001\\train_val_list.txt"  # 你的图片列表文件
BATCH_SIZE = 50                       # 你的批次大小
SAVE_DIR = "nih_features_db"          # 特征保存文件夹
SAVE_PATH = os.path.join(SAVE_DIR, "nih_biomedclip_features.pt")  # 特征保存路径
#三、加载模型
# ===================== 模型核心逻辑（封装成可复用函数）=====================
def load_model():
    """
    封装模型加载逻辑：改算法时只需修改这个函数！
    返回：model, preprocess_val（推理预处理）, tokenizer
    """
    # 解决国内HF加载慢的问题
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 加载模型+预处理工具（改算法时替换这里，比如换成ResNet/自定义模型）
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    
    # 模型配置（评估模式+设备分配）
    model.eval()
    model = model.to(DEVICE)
    
    print(f"✅ 模型加载完成 | 设备：{DEVICE} | 模型：{MODEL_NAME}")
    return model, preprocess_val, tokenizer

# 全局加载模型（只加载一次，避免重复加载浪费内存）
model, preprocess_val, tokenizer = load_model()


# ----------------------
# 四. 封装特征提取函数（供后端调用）
# 输入：PIL.Image对象
# 输出：归一化后的特征向量（numpy格式，shape=(256,)）
# ----------------------
def extract_image_feature(image):
    """
    功能：单张图片特征提取（和你批量提取的逻辑一致）
    参数：image - PIL.Image对象（RGB格式）
    返回：numpy数组（256维特征向量）
    """
    try:
        # 预处理（和你的collate_fn里的预处理逻辑一致）
        processed_img = preprocess_val(image).unsqueeze(0).to(DEVICE)
        
        # 特征提取（和你的批量提取逻辑一致）
        with torch.no_grad():
            img_feature = model.encode_image(processed_img)
        
        # L2归一化（和你的批量处理逻辑完全一致）
        img_feature = torch.nn.functional.normalize(img_feature, p=2, dim=1)
        
        # 转为numpy并去除batch维度（(1, 512) → (512,)）
        return img_feature.cpu().numpy()[0]
    
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
    批量提取特征（完全复用你的DataLoader逻辑）
    无需传参，直接用全局配置项（和你的路径/参数对齐）
    """
    # 1. 加载你的NIH数据集（和你的代码一致）
    print(f"📂 加载NIH数据集 | 图片根路径：{IMG_ROOT} | 列表文件：{IMG_LIST}")
    NiH_data = NIH_dataset(IMG_ROOT, IMG_LIST)
    
    # 2. 创建DataLoader（复用你的collate_fn和batch_size）
    NIH_loader = DataLoader(
        NiH_data, 
        batch_size=BATCH_SIZE, 
        collate_fn=custom_collate_fn  # 用你的自定义collate_fn
    )
    print(f"✅ DataLoader创建完成 | 批次大小：{BATCH_SIZE} | 总批次：{len(NIH_loader)}")

    # 3. 批量提取特征（和你的逻辑完全一致）
    all_features = []
    all_paths = []

    print("\n🚀 开始批量提取特征...")
    with torch.no_grad():  # 禁用梯度，节省内存
        # 加进度条，方便看提取进度
        for batch_images, batch_paths in tqdm(NIH_loader, desc="提取进度"):
            # 图像移到设备（和你的代码一致）
            batch_images = batch_images.to(DEVICE)
            
            # 提取特征（和你的代码一致）
            batch_features = model.encode_image(batch_images)
            batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
            
            # 保存特征和路径（和你的代码一致）
            all_features.append(batch_features.cpu())
            all_paths.extend(batch_paths)

    # 4. 合并所有特征（和你的代码一致）
    all_features = torch.cat(all_features, dim=0)
    print(f"\n📊 提取完成 | 特征形状：{all_features.shape} | 有效图片数：{len(all_paths)}")

    # 5. 保存特征和路径（和你的格式完全一致）
    os.makedirs(SAVE_DIR, exist_ok=True)  # 创建保存文件夹
    features_db = {
        "features": all_features,  # 特征向量 (N, 512)，和你的维度一致
        "image_paths": all_paths   # 对应图像路径列表
    }
    torch.save(features_db, SAVE_PATH)

    print(f"✅ 特征数据库已保存到：{SAVE_PATH}")
    print(f"特征形状：{all_features.shape}，包含 {len(all_paths)} 张图像")

# ===================== 一键运行批量提取（直接执行model.py即可）=====================
if __name__ == "__main__":
    # 运行批量提取（和你的逻辑完全一致）
    batch_extract_features()
