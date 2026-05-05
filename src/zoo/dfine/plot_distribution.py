import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import os

# 假设你的10个类别名称是这些（请按你的实际TCT类别修改）
CLASS_NAMES = ['Normal0', 'ASC-US1', 'ASC-H2', 'LSIL3', 'HSIL/SCC4', 'AGC5', 'VAG6', 'MON7', 'DYS8', 'EC9']

class FeatureAccumulator:
    """全局单例的特征收集器"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.all_feats = []
        self.all_labels = []
        self.all_preds = []
        
    def update(self, feats, true_labels, pred_labels=None):
        # 立即转移到 CPU 测试，防止撑爆显卡
        self.all_feats.append(feats.detach().cpu())
        self.all_labels.append(true_labels.detach().cpu())
        if pred_labels is not None:
            self.all_preds.append(pred_labels.detach().cpu())
            
    def plot_and_clear(self, save_dir, epoch=0):
        if len(self.all_feats) == 0:
            print("没有收集到任何特征，跳过画图。")
            return
            
        print(f"正在准备绘制 Epoch {epoch} 的汇总分布图...")
        os.makedirs(save_dir, exist_ok=True)
        
        # 将整个 Epoch 断断续续的 batch 拼接成一个大 tensor
        feats_cat = torch.cat(self.all_feats, dim=0).numpy()
        labels_cat = torch.cat(self.all_labels, dim=0).numpy()
        
        # 降采样限制：如果框特别多，t-SNE 会极其缓慢，超过5000随机抽样
        n_samples = feats_cat.shape[0]
        if n_samples > 5000:
            indices = np.random.choice(n_samples, 5000, replace=False)
            plot_feats = feats_cat[indices]
            plot_labels = labels_cat[indices]
            print(f"样本过多 ({n_samples})，随机抽样 5000 个进行 t-SNE...")
        else:
            plot_feats = feats_cat
            plot_labels = labels_cat
            print(f"总计收集到 {n_samples} 个样本，开始绘制 t-SNE...")

        tsne_path = os.path.join(save_dir, f"epoch_{epoch}_tsne_dist.png")
        plot_tsne_feature_distribution(plot_feats, plot_labels, tsne_path)
        
        if len(self.all_preds) > 0:
            preds_cat = torch.cat(self.all_preds, dim=0).numpy()
            cm_path = os.path.join(save_dir, f"epoch_{epoch}_confusion_matrix.png")
            plot_confusion_matrix_heatmap(labels_cat, preds_cat, cm_path)
            
        self.reset() # 画完图后清空，为下一个 Epoch 腾出空间

# 实例化一个全局对象供 VPE 随时导入调用
epoch_visualizer = FeatureAccumulator()

def plot_tsne_feature_distribution(features, labels, save_path="/root/userfolder/Projects/RLCCD/output/dfine_hgnetv2_m_ccd/6_1/tsne_distribution.png"):
    """
    绘制 t-SNE 特征降维分布图
    features: numpy array [N, C] (比如 [N, 256])
    labels: numpy array [N] (取值 0~9的真实标签)
    """
    print("正在计算 t-SNE，这可能需要一点时间...")
    # 使用 t-SNE 将高维特征降到 2 维
    n_samples = features.shape[0]
    perplexity = min(30, max(2, n_samples // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    reduced_feats = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    # 设置调色板，确保有10种颜色
    palette = sns.color_palette("tab10", len(np.unique(labels)))
    
    # 绘制散点图
    ax = sns.scatterplot(
        x=reduced_feats[:, 0], y=reduced_feats[:, 1],
        hue=[CLASS_NAMES[int(lbl)] for lbl in labels],
        palette=palette,
        legend="full",
        alpha=0.7,
        s=30, # 点的大小
        edgecolor=None
    )
    
    plt.title("TCT Cells Feature Distribution (t-SNE)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    
    # 调整图例位置
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE 特征分布图已保存至: {save_path}")

def plot_confusion_matrix_heatmap(y_true, y_pred, save_path="/root/userfolder/Projects/RLCCD/output/dfine_hgnetv2_m_ccd/6_1/confusion_matrix.png"):
    """
    绘制混淆矩阵热力图
    y_true: numpy array [N] 真实标签
    y_pred: numpy array [N] 预测标签
    """
    labels_range = np.arange(len(CLASS_NAMES)) 
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    
    # 归一化混淆矩阵，显示百分比
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized, 
        annot=True,     # 在格子里显示数字
        fmt=".2f",      # 两位小数
        cmap="Blues",   # 蓝色主题
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES,
        vmin=0, vmax=1  # 色条范围 0~1
    )
    
    plt.title("Confusion Matrix (Normalized)", fontsize=16)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"混淆矩阵图已保存至: {save_path}")


if __name__ == "__main__":
    # --- 这里是模拟测试代码 ---
    # 实际应用中，你需要将验证集(Validation)推断得到的 tensor 传进去
    num_samples = 1000
    num_classes = 10
    
    # 模拟收集到的高维特征 (例如 pred_vpe_feats)
    mock_features = np.random.randn(num_samples, 256) 
    
    # 模拟收集到的真实标签和预测标签
    mock_true_labels = np.random.randint(0, num_classes, num_samples)
    mock_pred_labels = mock_true_labels.copy()
    
    # 人为制造一点 2, 6, 7, 9 的预测误差
    error_mask = np.random.rand(num_samples) < 0.2
    mock_pred_labels[error_mask] = np.random.randint(0, num_classes, error_mask.sum())
    
    # 1. 画 t-SNE
    plot_tsne_feature_distribution(mock_features, mock_true_labels)
    # 2. 画 混淆矩阵
    plot_confusion_matrix_heatmap(mock_true_labels, mock_pred_labels)