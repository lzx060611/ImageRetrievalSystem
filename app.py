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
# 2. 加载FAISS索引和图片路径（你刚生成的文件）
# ----------------------
try:
    # 加载FAISS索引文件（用于相似检索）
    index = faiss.read_index("features.index")  
    # 加载图片路径清单（索引和路径一一对应）
    with open("image_paths.txt", "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f.readlines()]
    print(f"✅ 索引加载成功：共 {len(image_paths)} 张图片，索引中 {index.ntotal} 个特征")
except FileNotFoundError:
    # 如果没找到文件，提示先运行load_features_to_faiss.py
    raise ValueError("⚠️ 未找到FAISS索引文件！请先运行 load_features_to_faiss.py")

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
        # io.BytesIO(file.read())：把上传的二进制文件转为内存流
        # Image.open(...)：打开图片并转为RGB格式（统一格式，避免模型报错）
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # 步骤3：提取上传图片的特征（调用model.py中的函数）
        query_feat = extract_image_feature(image)  # 得到(256,)的numpy数组
        query_feat = np.expand_dims(query_feat, axis=0)  # 增加batch维度：(256,) → (1,256)（FAISS要求）

        # 步骤4：获取前端指定的返回数量（默认10张，可自定义）
        top_n = int(request.form.get('top_n', 10))  # 从表单中取top_n参数，默认10
        # 限制最大返回数量（避免性能问题）
        top_n = min(top_n, 50)  

        # 步骤5：FAISS相似检索
        # distances：每个相似图片的距离（越小越相似），indices：相似图片的索引
        distances, indices = index.search(query_feat, top_n)  

        # 步骤6：整理检索结果（供前端显示）
        results = []
        # 遍历检索结果（indices[0]是检索到的图片索引列表）
        for i, idx in enumerate(indices[0]):
            img_local_path = image_paths[idx]  # 获取本地图片路径
            # 转为前端能访问的URL（拼接本地服务器地址）
            img_url = f"http://localhost:5000/{img_local_path}"
            # 整理单条结果：URL、距离（相似度）、本地路径
            results.append({
                "image_url": img_url,  # 前端显示图片用
                "distance": float(distances[0][i]),  # 距离越小越相似
                "local_path": img_local_path  # 供调试查看
            })

        # 步骤7：返回JSON格式结果（前端能解析）
        return jsonify({
            "code": 200,  # 200是HTTP成功码
            "msg": "检索成功",
            "data": results  # 核心结果列表
        })

    # 捕获异常，返回错误信息（方便调试）
    except Exception as e:
        return jsonify({"error": f"检索失败：{str(e)}"}), 500  # 500是服务器错误码

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