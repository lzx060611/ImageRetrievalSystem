# 重写 load_features_to_faiss.py 文件

# load_features_to_faiss.py
import torch
import faiss
import numpy as np
import os

# ----------------------
# 1. 配置路径（修改为哈希特征文件路径）
# ----------------------
FEATURES_PT_PATH = "./nih_features_db/nih_biomedclip_hash_64bit.pt"  # 哈希特征文件路径
SAVE_INDEX_PATH = "nih_hash_64bit.index"  # 生成的FAISS索引保存路径
SAVE_PATHS_TXT = "image_paths.txt"  # 生成的图片路径清单

# ----------------------
# 2. 加载你已保存的哈希特征和路径
# ----------------------
print("📂 正在加载你已保存的哈希特征文件...")
features_db = torch.load(FEATURES_PT_PATH)
all_binary_codes = features_db["binary_codes"]  # 二值哈希码
all_paths = features_db["image_paths"]  # 对应的图片路径列表

print(f"✅ 加载完成：哈希码形状 {all_binary_codes.shape}，共 {len(all_paths)} 张图片")

# ----------------------
# 3. 将PyTorch张量转为numpy并执行位压缩
# ----------------------
all_binary_codes_np = all_binary_codes.cpu().numpy()
# 将浮点数转换为整数类型
all_binary_codes_np = all_binary_codes_np.astype(np.uint8)
# 位压缩：将(64,)的0/1数组压缩为(8,)的uint8数组
all_packed_codes = np.packbits(all_binary_codes_np, axis=1)
print(f"✅ 位压缩完成：压缩前形状 {all_binary_codes_np.shape}，压缩后形状 {all_packed_codes.shape}")

# ----------------------
# 4. 构建FAISS二值索引
# ----------------------
# 使用faiss.IndexBinaryFlat，传入总比特数64
index = faiss.IndexBinaryFlat(64)
index.add(all_packed_codes)  # 将压缩后的哈希码加入索引
print(f"✅ FAISS二值索引构建完成，包含 {index.ntotal} 个哈希码")

# ----------------------
# 5. 保存索引和路径（供后端加载）
# ----------------------
# 保存FAISS二值索引
faiss.write_index_binary(index, SAVE_INDEX_PATH)
print(f"📌 FAISS二值索引已保存到：{SAVE_INDEX_PATH}")

# 保存图片路径清单
with open(SAVE_PATHS_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(all_paths))
print(f"📌 图片路径清单已保存到：{SAVE_PATHS_TXT}")
