# app.py
import os
import numpy as np
import faiss
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # 解决前端跨域访问问题
from PIL import Image
import io
# 导入你修改后的model.py中的模型和特征提取函数
from model import extract_image_feature  

# ----------------------
# 1. 初始化Flask应用
# ----------------------
# static_folder="./"：设置静态文件根目录，让前端能访问本地图片
app = Flask(__name__, static_folder="./")  
CORS(app)  # 允许跨域请求（本地前端访问后端必须加）

# ----------------------
# 2. 加载FAISS二值索引和图片路径
# ----------------------
try:
    # 加载FAISS二值索引文件（用于相似检索）
    index = faiss.read_index_binary("nih_hash_64bit.index")  
    # 加载图片路径清单（索引和路径一一对应）
    with open("image_paths.txt", "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f.readlines()]
    print(f"✅ 二值索引加载成功：共 {len(image_paths)} 张图片，索引中 {index.ntotal} 个哈希码")
except FileNotFoundError:
    # 如果没找到文件，提示先运行load_features_to_faiss.py
    raise ValueError("⚠️ 未找到FAISS二值索引文件！请先运行 load_features_to_faiss.py")

# ----------------------
# 3. 静态文件路由：让前端能访问你的本地图片
# 作用：把本地图片路径转为前端能访问的URL（如 http://localhost:5000/images_001/images/xxx.jpg）
# ----------------------
@app.route('/<path:filename>')
def serve_static(filename):
    # 拼接完整路径，返回文件
    return send_from_directory('./', filename)

# ----------------------
# 4. 核心接口：相似图片检索
# 前端上传图片 → 提取特征 → FAISS检索 → 返回结果
# ----------------------
@app.route('/search', methods=['POST'])
def search_similar():
    try:
        # 步骤1：检查前端是否上传了图片
        if 'image' not in request.files:
            return jsonify({"error": "请上传图片文件！"}), 400  # 400是HTTP错误码（请求错误）
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "图片文件名不能为空！"}), 400

        # 步骤2：加载上传的图片并转为PIL格式
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        print(f"✅ 图片加载成功：{file.filename}")

        # 步骤3：提取上传图片的哈希码（调用model.py中的函数）
        print("🔧 开始提取哈希码...")
        query_code = extract_image_feature(image)  # 得到(64,)的0/1数组
        print(f"✅ 哈希码提取成功，形状：{query_code.shape}，类型：{query_code.dtype}")
        
        # 增加batch维度并执行位压缩
        query_code = np.expand_dims(query_code, axis=0)
        print(f"🔧 增加batch维度后：{query_code.shape}")
        
        # 确保数据类型为uint8
        query_code = query_code.astype(np.uint8)
        print(f"🔧 类型转换后：{query_code.dtype}")
        
        # 位压缩
        packed_query = np.packbits(query_code, axis=1)
        print(f"✅ 位压缩完成，形状：{packed_query.shape}")

        # 步骤4：获取前端指定的返回数量（默认10张，可自定义）
        top_n = int(request.form.get('top_n', 10))  # 从表单中取top_n参数，默认10
        # 限制最大返回数量（避免性能问题）
        top_n = min(top_n, 50)  
        print(f"🔧 检索参数：top_n={top_n}")

        # 步骤5：FAISS相似检索（使用汉明距离）
        print("🔍 开始FAISS检索...")
        # distances：每个相似图片的汉明距离（越小越相似），indices：相似图片的索引
        distances, indices = index.search(packed_query, top_n)  
        print(f"✅ 检索完成，距离形状：{distances.shape}，索引形状：{indices.shape}")

        # 步骤6：整理检索结果（供前端显示）
        results = []
        print("📋 整理检索结果...")
        # 遍历检索结果（indices[0]是检索到的图片索引列表）
        for i, idx in enumerate(indices[0]):
            img_local_path = image_paths[idx]  # 获取本地图片路径
            # 转为前端能访问的URL（拼接本地服务器地址）
            img_url = f"http://localhost:5000/{img_local_path}"
            # 整理单条结果：URL、汉明距离（相似度）、本地路径
            results.append({
                "image_url": img_url,  # 前端显示图片用
                "distance": int(distances[0][i]),  # 汉明距离，越小越相似
                "local_path": img_local_path  # 供调试查看
            })
            if i < 3:  # 只打印前3个结果
                print(f"  结果{i+1}: 距离={int(distances[0][i])}, 路径={img_local_path}")

        # 步骤7：返回JSON格式结果（前端能解析）
        print(f"✅ 结果整理完成，共 {len(results)} 个结果")
        return jsonify({
            "code": 200,  # 200是HTTP成功码
            "msg": "检索成功",
            "data": results  # 核心结果列表
        })

    # 捕获异常，返回详细错误信息（方便调试）
    except Exception as e:
        error_msg = f"检索失败：{str(e)}"
        print(f"❌ 错误：{error_msg}")
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        return jsonify({"error": error_msg}), 500  # 500是服务器错误码

# ----------------------
# 5. 前端页面路由：访问http://localhost:5000直接打开前端页面
# ----------------------
@app.route('/')
def index_page():
    return send_from_directory('./', 'index.html')

# ----------------------
# 6. 启动本地服务器
# ----------------------
if __name__ == '__main__':
    # host='0.0.0.0'：允许本机和局域网访问；port=5000：端口号
    # debug=True：调试模式（代码修改后自动重启服务）
    app.run(host='0.0.0.0', port=5000, debug=True)
