import json
import torch
from transformers import CLIPTokenizer, CLIPModel

# ------------------------------
# 1. 配置
# ------------------------------
json_path = "/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/text_cat10.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 下载了 PubMedCLIP，替换这里的路径
local_model_path = "/root/userfolder/Projects/RLCCD/weight/pubmed-clip-vit-base-patch32"

# ------------------------------
# 2. 读取 JSON
# ------------------------------
with open(json_path, "r", encoding="utf-8") as f:
    text_data = json.load(f)

categories = list(text_data.keys())
print(f"检测到 {len(categories)} 个类别: {categories}")

# ------------------------------
# 3. 初始化 CLIP tokenizer 和 text encoder
# ------------------------------
print(f"正在加载模型: {local_model_path} ...")
tokenizer = CLIPTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = CLIPModel.from_pretrained(local_model_path, local_files_only=True).to(device)
model.eval()

# ------------------------------
# 4. 编码 (Prompt Engineering + Encoding)
# ------------------------------
all_feats = []

print("开始编码...")
with torch.no_grad():
    for cat in categories:
        raw_texts = text_data[cat]["text"]
        
        # [修改 1] Prompt Engineering: 结合类别名称和描述
        # 这种写法告诉模型："这是 [类别名] 类型的宫颈细胞，它的特征是 [描述]"
        # 即使描述很像，前面的类别名也能在大语义上拉开距离
        # texts = [f"A micrograph of {cat} cervical cell, characterized by {t}" for t in raw_texts]   
        # texts = [f"A micrograph of {cat} cervical cell" for t in raw_texts]   
        texts = [f"{cat}" for t in raw_texts] 

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # 提取特征
        feats = model.get_text_features(**inputs)
        
        # 先归一化每条描述的特征
        feats = feats / feats.norm(dim=-1, keepdim=True)

        # 取该类别所有描述的平均值，作为一个鲁棒的类别中心
        cat_feat = feats.mean(dim=0, keepdim=True)
        print(cat_feat.shape)
        
        # 此时先不归一化，保留幅值信息用于后续去均值
        all_feats.append(cat_feat)

# 堆叠所有类别特征 [Category_Num, Dim]
text_feats = torch.cat(all_feats, dim=0)

print(f"原始特征方差 (区分度): {torch.var(text_feats, dim=0).mean().item():.6f}")

# ------------------------------
# [修改 2] 特征去中心化 (De-mean / Centering) - 核心步骤
# ------------------------------
# 医学图像描述通常包含大量共有词汇(cell, nucleus, cytoplasm)，导致所有向量挤在一起。
# 减去平均向量，相当于移除了所有细胞的“共性”，只留下了“个性”（差异）。
mean_feat = text_feats.mean(dim=0, keepdim=True)
text_feats = text_feats - mean_feat

print(f"去中心化后方差 (区分度): {torch.var(text_feats, dim=0).mean().item():.6f}")

# ------------------------------
# [修改 3] 重新归一化
# ------------------------------
# 用于后续的余弦相似度计算 (Cosine Similarity)
text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
print(torch.var(text_feats, dim=0).mean())

# ------------------------------
# 5. 相似度分析 (验证效果)
# ------------------------------
sim = torch.matmul(text_feats, text_feats.t())
mask = torch.eye(len(categories), device=sim.device).bool()
others = sim[~mask]

print("-" * 30)
print(f"结果验证 (目标是越低越好):")
print(f"不同类别间的平均相似度: {others.mean().item():.6f}")
print(f"不同类别间的最小相似度: {others.min().item():.6f}")
print(f"不同类别间的最大相似度: {others.max().item():.6f}")
print("-" * 30)

# ------------------------------
# 6. 保存文本向量
# ------------------------------
# save_path = "/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/pubmed_text10_feats.pt"
save_path = "/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/pubmed_text10_name_feats.pt"
torch.save({
    "categories": categories,
    "text_feats": text_feats.cpu()
}, save_path)

print(f"文本编码完成，已保存至: {save_path}")

import torch  # 确保导入 torch，不需要 numpy

print("\n" + "="*50)
print("🔍 正在进行特征质量深度体检...")
print("="*50)

# 确保在 CPU 上计算
feats = text_feats.cpu()
num_cls = len(categories)

# 计算相似度矩阵 [N, N]
sim_matrix = torch.matmul(feats, feats.t())

# 获取上三角索引（排除对角线）
# triu_indices 返回的是 (row_indices, col_indices)
row_idx, col_idx = torch.triu_indices(num_cls, num_cls, offset=1)

# 提取所有非对角线相似度
off_diag_sims = sim_matrix[row_idx, col_idx]

avg_sim = off_diag_sims.mean().item()
max_sim = off_diag_sims.max().item()
min_sim = off_diag_sims.min().item()

print(f"📉 指标统计：")
print(f"   - 平均类间相似度 (Avg): {avg_sim:.4f}  (越低越好，理想 < 0.2)")
print(f"   - 最大类间相似度 (Max): {max_sim:.4f}  (越低越好，警告 > 0.6)")
print(f"   - 最小类间相似度 (Min): {min_sim:.4f}")

# 找出最像的“死对头” (Top-3)
# 排序：从大到小
sorted_indices = torch.argsort(off_diag_sims, descending=True)
top_k = min(3, len(off_diag_sims))

print(f"\n⚠️ 最容易混淆的前 {top_k} 对类别 (Risk Top-{top_k})：")
risk_level = 0
for k in range(top_k):
    idx = sorted_indices[k]
    i = row_idx[idx].item()
    j = col_idx[idx].item()
    sim = off_diag_sims[idx].item()
    c1, c2 = categories[i], categories[j]
    
    tag = "✅安全"
    if sim > 0.6: 
        tag = "🔴危险"; risk_level = max(risk_level, 2)
    elif sim > 0.4: 
        tag = "🟡警告"; risk_level = max(risk_level, 1)
    
    print(f"   {k+1}. [{tag}] {c1} <--> {c2} : {sim:.4f}")

print("\n" + "-"*50)
print("📢 最终结论建议：")

if avg_sim > 0.5:
    print("❌ 【不可用】平均相似度太高！")
    print("   原因：所有类别特征依然挤在一起。")
    print("   对策：请检查是否忘记执行 '特征去均值 (De-mean)' 步骤。")
elif risk_level == 2:
    idx = sorted_indices[0]
    i = row_idx[idx].item()
    j = col_idx[idx].item()
    c1, c2 = categories[i], categories[j]
    print("❌ 【高风险】存在极度相似的类别对 (>0.6)。")
    print(f"   最混淆的是: {c1} 和 {c2}")
    print("   对策：对比学习可能会强行拉近这两类的距离，导致误分类。建议修改这两个类的 Prompt 描述，使其更具差异化。")
elif avg_sim < 0.0:
    print("✅ 【完美】特征分布已正交化甚至互斥。")
    print("   这套特征非常适合用于对比损失 (Contrastive Loss)！")
elif avg_sim < 0.2:
    print("✅ 【优秀】特征区分度良好。")
    print("   可以放心使用。")
else:
    print("⚠️ 【可用但需谨慎】特征稍有重叠。")
    print("   建议降低对比损失的权重 (loss_weight < 0.1)。")

print("="*50 + "\n")

# ===== 打印每个类别之间的相似度 =====
print("\n" + "="*50)
print("📊 类别两两相似度矩阵（cosine similarity）")
print("矩阵形式（行=row类别，列=col类别）：\n")
header = " " * 15 + " ".join([f"{c[:6]:>8}" for c in categories])
print(header)
for i in range(num_cls):
    row_vals = " ".join([f"{sim_matrix[i, j].item():8.3f}" for j in range(num_cls)])
    print(f"{categories[i][:12]:>15} {row_vals}")

# import json
# import torch
# from transformers import CLIPTokenizer, CLIPModel

# # ------------------------------
# # 1. 配置
# # ------------------------------
# json_path = "/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/text_cat10.json"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# local_model_path = "/root/userfolder/Projects/RL-CCD/weight/models--openai--clip-vit-base-patch32"

# # ------------------------------
# # 2. 读取 JSON
# # ------------------------------
# with open(json_path, "r", encoding="utf-8") as f:
#     text_data = json.load(f)

# categories = list(text_data.keys())

# # ------------------------------
# # 3. 初始化 CLIP tokenizer 和 text encoder
# # ------------------------------
# tokenizer = CLIPTokenizer.from_pretrained(local_model_path, local_files_only=True)
# model = CLIPModel.from_pretrained(local_model_path, local_files_only=True).to(device)
# model.eval()

# # ------------------------------
# # 4. 对每个类别的多条文本进行编码（自动适配条数）
# # ------------------------------
# all_feats = []

# with torch.no_grad():
#     for cat in categories:
#         texts = text_data[cat]["text"]  # 取该类别所有描述（不管多少条）

#         inputs = tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             return_tensors="pt"
#         ).to(device)

#         feats = model.get_text_features(**inputs)
#         feats = feats / feats.norm(dim=-1, keepdim=True)

#         # 取平均作为该类别向量
#         cat_feat = feats.mean(dim=0, keepdim=True)
#         cat_feat = cat_feat / cat_feat.norm(dim=-1, keepdim=True)
#         print(cat_feat.shape)

#         all_feats.append(cat_feat)

# text_feats = torch.cat(all_feats, dim=0)  # [类别数, dim]

# print(torch.var(text_feats, dim=0).mean())

# # ------------------------------
# # 5. 相似度分析
# # ------------------------------
# sim = torch.matmul(text_feats, text_feats.t())
# mask = torch.eye(len(categories), device=sim.device).bool()
# others = sim[~mask]

# print(f"不同类别间的平均相似度: {others.mean().item():.6f}")
# print(f"不同类别间的最小相似度: {others.min().item():.6f}")

# # ------------------------------
# # 6. 保存文本向量
# # ------------------------------
# torch.save({
#     "categories": categories,
#     "text_feats": text_feats.cpu()
# }, "/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/text10_feats.pt")

# print("文本编码完成，保存为 text10_feats.pt")

