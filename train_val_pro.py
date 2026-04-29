import torch
import torch.nn.functional as F
from tqdm import tqdm
from metrics import mAPw, aCG, nDCG

def evaluate_hash_retrieval(model, gallery_loader, query_loader, device, K=10):
    """
    对齐源码的哈希检索评估函数
    Args:
        model: 已训练的模型，forward返回 (分类logits, 连续哈希码)
        gallery_loader: gallery集的DataLoader，每个元素为 (image, path, label)
        query_loader: query集的DataLoader，每个元素为 (image, path, label)
        device: 计算设备
        K: Precision@K 中的K值（源码中常用10或100，这里默认为10）
    Returns:
        mapw_avg: 平均加权mAP
        acg_avg: 平均ACG
        ndcg_avg: 平均nDCG
        precision_at_k_avg: 平均Precision@K
    """
    model.eval()
    
    # 1. 提取 gallery 的所有二值哈希码和标签
    gallery_codes = []
    gallery_labels = []
    with torch.no_grad():
        for images, _, labels in tqdm(gallery_loader, desc="Extracting gallery"):
            images = images.to(device)
            _, h = model(images)               # 取第二个返回值：连续哈希码
            binary_codes = torch.sign(h)       # 二值化 {-1, +1}
            gallery_codes.append(binary_codes.cpu())
            gallery_labels.append(labels.cpu())
    gallery_codes = torch.cat(gallery_codes, dim=0)   # [num_gallery, K]
    gallery_labels = torch.cat(gallery_labels, dim=0) # [num_gallery, num_classes]
    
    # 2. 对每个 query 进行检索
    mapw_sum = 0.0
    acg_sum = 0.0
    ndcg_sum = 0.0
    precision_k_sum = 0.0
    num_queries = 0
    
    with torch.no_grad():
        for images, _, q_labels in tqdm(query_loader, desc="Evaluating queries"):
            images = images.to(device)
            _, h_q = model(images)
            q_codes = torch.sign(h_q).cpu()            # [batch_size, K]
            q_labels = q_labels.cpu()                  # [batch_size, num_classes]
            
            # 计算汉明距离：对于 {-1,+1} 码，汉明距离 = (K - dot_product) / 2
            # dot_product 范围 [-K, K]，转换为汉明距离 [0, K]
            dot = torch.matmul(q_codes, gallery_codes.t())   # [batch_size, num_gallery]
            hamming_dist = (q_codes.size(1) - dot) / 2.0    # [batch_size, num_gallery]
            
            # 对每个 query 样本计算指标
            for i in range(images.size(0)):
                # 当前 query 的标签
                q_label = q_labels[i].unsqueeze(0)          # [1, num_classes]
                # 计算与所有 gallery 样本的标签交集大小（即共享标签数）
                # 注意：标签为 0/1 浮点，交集 = 点积
                intersection = torch.matmul(q_label, gallery_labels.t()).squeeze(0)  # [num_gallery]
                relevance_list = intersection.tolist()       # 整数列表，如 [3, 1, 0, ...]
                
                # 按汉明距离升序排序，距离越小越相似
                sorted_indices = torch.argsort(hamming_dist[i])
                sorted_relevance = [relevance_list[idx] for idx in sorted_indices.tolist()]
                
                # 理想相关性列表（降序排列）
                ideal_relevance = sorted(sorted_relevance, reverse=True)
                # 在计算 ndcg 时只取前 100
                topk = 100
                sorted_relevance_100 = sorted_relevance[:topk]
                ideal_relevance_100 = ideal_relevance[:topk]  # 注意理想列表也要截断
                ndcg_sum += nDCG(sorted_relevance_100, ideal_relevance_100)
                # 计算三个指标
                mapw_sum += mAPw(sorted_relevance_100)   # 使用前100个
                acg_sum += aCG(sorted_relevance_100)     # 使用前100个
                
                # Precision@K：前K个中相关性 >0 的比例
                topk_rel = sorted_relevance[:K]
                precision_k = sum(1 for r in topk_rel if r > 0) / K
                precision_k_sum += precision_k
                num_queries += 1
    
    # 平均指标
    mapw_avg = mapw_sum / num_queries
    acg_avg = acg_sum / num_queries
    ndcg_avg = ndcg_sum / num_queries
    precision_k_avg = precision_k_sum / num_queries
    
    model.train()
    return mapw_avg, acg_avg, ndcg_avg, precision_k_avg