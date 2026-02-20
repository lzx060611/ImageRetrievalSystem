# load_features_to_faiss.py
import torch
import faiss
import numpy as np
import os

# ----------------------
# 1. 配置路径（替换为你的实际路径）
# ----------------------
FEATURES_PT_PATH = "./nih_features_db/nih_biomedclip_features.pt"  # 你的特征文件路径
SAVE_INDEX_PATH = "features.index"  # 生成的FAISS索引保存路径
SAVE_PATHS_TXT = "image_paths.txt"  # 生成的图片路径清单

# ----------------------
# 2. 加载你已保存的特征和路径
# ----------------------
print("📂 正在加载你已保存的特征文件...")
features_db = torch.load(FEATURES_PT_PATH)
all_features = features_db["features"]  # 你的特征张量
all_paths = features_db["image_paths"]  # 对应的图片路径列表

print(f"✅ 加载完成：特征形状 {all_features.shape}，共 {len(all_paths)} 张图片")

# ----------------------
# 3. 将PyTorch张量转为numpy（FAISS需要numpy格式）
# ----------------------
all_features_np = all_features.cpu().numpy()

# ----------------------
# 4. 构建FAISS索引（自动匹配真实维度）
# ----------------------
# 关键：获取真实的特征维度（512）
actual_dim = all_features_np.shape[1]
print(f"🔍 检测到特征真实维度：{actual_dim}")

# index = faiss.IndexFlatL2(actual_dim)  # 用真实维度创建索引(L2距离)
index = faiss.IndexFlatIP(actual_dim)  # IP=内积，等价于归一化后的余弦相似度
index.add(all_features_np)  # 将所有特征加入索引
print(f"✅ FAISS索引构建完成，包含 {index.ntotal} 个特征")

# ----------------------
# 5. 保存索引和路径（供后端加载）
# ----------------------
# 保存FAISS索引
faiss.write_index(index, SAVE_INDEX_PATH)
print(f"📌 FAISS索引已保存到：{SAVE_INDEX_PATH}")

# 保存图片路径清单
with open(SAVE_PATHS_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(all_paths))
print(f"📌 图片路径清单已保存到：{SAVE_PATHS_TXT}")