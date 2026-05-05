import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def compute_iou(box1, box2):
    """
    计算两个框的 IoU
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def iou_matching_max_iou(json_file_path, iou_threshold=0.5):
    """
    每个 GT 与 IoU 最大的预测框匹配（只保留 IoU > threshold 的匹配）
    
    Args:
        json_file_path: JSON 文件路径
        iou_threshold: IoU 阈值，默认 0.5，只有 IoU > 该值才算匹配
    
    Returns:
        matched_scores_per_class: dict, key=类别, value=匹配上的分数列表
        matched_details: dict, 包含详细的匹配信息
    """
    # 加载 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 存储每个类别匹配上的分数
    matched_scores_per_class = defaultdict(list)
    
    # 存储详细信息
    matched_details = {
        'total_gt': 0,
        'total_matched': 0,
        'matches': []
    }
    
    # 遍历每张图片
    for img_data in data['detections']:
        image_id = img_data['image_id']
        ground_truths = img_data['ground_truths']
        predictions = img_data['predictions']
        
        # 如果没有 GT 或预测，跳过
        if len(ground_truths) == 0 or len(predictions) == 0:
            continue
        
        # 按类别分组处理
        gt_by_class = defaultdict(list)
        for gt in ground_truths:
            label = gt['label']
            gt_by_class[label].append(gt)
        
        pred_by_class = defaultdict(list)
        for pred in predictions:
            label = pred['label']
            pred_by_class[label].append(pred)
        
        # 对每个类别分别进行匹配
        for class_id in gt_by_class.keys():
            gt_list = gt_by_class[class_id]
            pred_list = pred_by_class.get(class_id, [])
            
            if len(pred_list) == 0:
                continue
            
            matched_details['total_gt'] += len(gt_list)
            
            # 构建 IoU 矩阵 [num_pred, num_gt]
            iou_matrix = np.zeros((len(pred_list), len(gt_list)))
            for i, pred in enumerate(pred_list):
                for j, gt in enumerate(gt_list):
                    iou_matrix[i, j] = compute_iou(pred['bbox'], gt['bbox'])
            
            # 每个 GT 找 IoU 最大的预测框（一对一匹配）
            matched_pairs = []
            used_preds = set()
            
            # 为每个 GT 找到 IoU 最大的预测框
            for gt_idx in range(len(gt_list)):
                # 找到当前 GT 的所有 IoU
                ious_for_gt = iou_matrix[:, gt_idx]
                
                # 找到 IoU 最大的预测框
                best_pred_idx = np.argmax(ious_for_gt)
                best_iou = ious_for_gt[best_pred_idx]
                
                # 只有 IoU > 阈值才算匹配
                if best_iou > iou_threshold:
                    # 检查这个预测框是否已经被其他 GT 匹配了
                    if best_pred_idx not in used_preds:
                        # 预测框未被匹配，直接匹配
                        matched_pairs.append((best_pred_idx, gt_idx, best_iou))
                        used_preds.add(best_pred_idx)
                    else:
                        # 预测框已被匹配，比较 IoU，保留更大的那个
                        for idx, (p_idx, g_idx, iou_val) in enumerate(matched_pairs):
                            if p_idx == best_pred_idx:
                                if best_iou > iou_val:
                                    # 替换匹配，旧的 GT 失去匹配
                                    matched_pairs[idx] = (best_pred_idx, gt_idx, best_iou)
                                break
            
            # 记录匹配上的分数（只记录最终的匹配）
            for pred_idx, gt_idx, iou_value in matched_pairs:
                score = pred_list[pred_idx]['score']
                matched_scores_per_class[class_id].append(score)
                matched_details['total_matched'] += 1
                matched_details['matches'].append({
                    'image_id': image_id,
                    'class_id': class_id,
                    'score': score,
                    'iou': iou_value,
                    'pred_bbox': pred_list[pred_idx]['bbox'],
                    'gt_bbox': gt_list[gt_idx]['bbox']
                })
    
    # 打印统计信息
    print(f"📊 IoU Matching Statistics (Max IoU matching, threshold > {iou_threshold}):")
    print(f"  - Total GT boxes: {matched_details['total_gt']}")
    print(f"  - Total matched: {matched_details['total_matched']}")
    if matched_details['total_gt'] > 0:
        print(f"  - Match rate: {matched_details['total_matched']/matched_details['total_gt']*100:.2f}%")
    print(f"\n📈 Matched scores per class:")
    for class_id, scores in sorted(matched_scores_per_class.items()):
        print(f"  - Class {class_id}: {len(scores)} matches, "
              f"score range: [{min(scores):.3f}, {max(scores):.3f}], "
              f"mean: {np.mean(scores):.3f}, std: {np.std(scores):.3f}")
    
    return matched_scores_per_class, matched_details

def iou_matching_one_to_many(json_file_path, iou_threshold=0.5):
    """
    每个 GT 与 IoU 最大的预测框匹配（一对多模式）
    修改点：移除了 used_preds 检查，允许一个预测框被多个 GT 匹配
    """
    # 加载 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 存储每个类别匹配上的分数
    matched_scores_per_class = defaultdict(list)
    # 存储详细信息
    matched_details = {
        'total_gt': 0,
        'total_matched': 0,
        'matches': []
    }

    # 遍历每张图片
    for img_data in data['detections']:
        image_id = img_data['image_id']
        ground_truths = img_data['ground_truths']
        predictions = img_data['predictions']

        # 如果没有 GT 或预测，跳过
        if len(ground_truths) == 0 or len(predictions) == 0:
            continue

        # 按类别分组处理
        gt_by_class = defaultdict(list)
        for gt in ground_truths:
            label = gt['label']
            gt_by_class[label].append(gt)

        pred_by_class = defaultdict(list)
        for pred in predictions:
            label = pred['label']
            pred_by_class[label].append(pred)

        # 对每个类别分别进行匹配
        for class_id in gt_by_class.keys():
            gt_list = gt_by_class[class_id]
            pred_list = pred_by_class.get(class_id, [])
            
            if len(pred_list) == 0:
                continue
                
            matched_details['total_gt'] += len(gt_list)

            # 构建 IoU 矩阵 [num_pred, num_gt]
            iou_matrix = np.zeros((len(pred_list), len(gt_list)))
            for i, pred in enumerate(pred_list):
                for j, gt in enumerate(gt_list):
                    iou_matrix[i, j] = compute_iou(pred['bbox'], gt['bbox'])

            # --- 核心修改区域 ---
            # 1. 移除了 used_preds = set()
            # 2. 移除了复杂的 matched_pairs 列表和替换逻辑
            
            # 直接遍历每个 GT，找到 IoU 最大的预测框并记录
            for gt_idx in range(len(gt_list)):
                # 找到当前 GT 的所有 IoU
                ious_for_gt = iou_matrix[:, gt_idx]
                # 找到 IoU 最大的预测框
                best_pred_idx = np.argmax(ious_for_gt)
                best_iou = ious_for_gt[best_pred_idx]

                # 只有 IoU > 阈值才算匹配
                if best_iou > iou_threshold:
                    # 直接记录，不再检查 best_pred_idx 是否在 used_preds 中
                    score = pred_list[best_pred_idx]['score']
                    
                    matched_scores_per_class[class_id].append(score)
                    matched_details['total_matched'] += 1
                    matched_details['matches'].append({
                        'image_id': image_id,
                        'class_id': class_id,
                        'score': score,
                        'iou': best_iou,
                        'pred_bbox': pred_list[best_pred_idx]['bbox'],
                        'gt_bbox': gt_list[gt_idx]['bbox']
                    })
            # --- 修改结束 ---

    # 打印统计信息
    print(f"📊 IoU Matching Statistics (One-to-Many matching, threshold > {iou_threshold}):")
    print(f" - Total GT boxes: {matched_details['total_gt']}")
    print(f" - Total matched pairs: {matched_details['total_matched']}")
    
    if matched_details['total_gt'] > 0:
        print(f" - Avg matches per GT: {matched_details['total_matched']/matched_details['total_gt']:.2f}")
    
    print(f"\n📈 Matched scores per class:")
    for class_id, scores in sorted(matched_scores_per_class.items()):
        if len(scores) > 0:
            print(f" - Class {class_id}: {len(scores)} matches, " 
                  f"score range: [{min(scores):.3f}, {max(scores):.3f}], " 
                  f"mean: {np.mean(scores):.3f}, std: {np.std(scores):.3f}")
        else:
            print(f" - Class {class_id}: 0 matches")

    return matched_scores_per_class, matched_details


def plot_score_distribution(matched_scores_per_class, save_path=None, figsize=(12, 8), iou_threshold=0.5):
    """
    绘制每个类别匹配上的框的分数分布图（子图形式）
    
    Args:
        matched_scores_per_class: dict, key=类别, value=分数列表
        save_path: 保存图片的路径
        figsize: 图片大小
        iou_threshold: IoU 阈值，用于标题显示
    """
    if not matched_scores_per_class:
        print("⚠️ No matched scores to plot!")
        return
    
    num_classes = len(matched_scores_per_class)
    
    # 计算子图布局
    n_cols = min(3, num_classes)
    n_rows = (num_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if num_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for idx, (class_id, scores) in enumerate(sorted(matched_scores_per_class.items())):
        ax = axes[idx]
        
        # 绘制直方图
        ax.hist(scores, bins=20, alpha=0.7, color=colors[idx], 
                edgecolor='black', linewidth=1.5, density=True)
        
        # 添加 KDE 曲线
        from scipy import stats
        kde = stats.gaussian_kde(scores)
        x_range = np.linspace(0, 1, 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # 统计信息
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_score:.3f}')
        ax.axvline(median_score, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_score:.3f}')
        
        ax.set_xlabel('Confidence Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Class {class_id} (n={len(scores)})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
    
    # 隐藏多余的子图
    for idx in range(num_classes, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Distribution of Matched Box Confidence Scores by Class\n(IoU > {iou_threshold} Matching)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.show()

def plot_score_histogram_combined(matched_scores_per_class, save_path=None, figsize=(10, 6), iou_threshold=0.5):
    """
    绘制叠加的直方图，对比各类别的分数分布
    
    Args:
        matched_scores_per_class: dict, key=类别, value=分数列表
        save_path: 保存路径
        figsize: 图片大小
        iou_threshold: IoU 阈值，用于标题显示
    """
    if not matched_scores_per_class:
        print("⚠️ No matched scores to plot!")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(matched_scores_per_class)))
    
    for idx, (class_id, scores) in enumerate(sorted(matched_scores_per_class.items())):
        ax.hist(scores, bins=30, alpha=0.5, color=colors[idx], 
                label=f'Class {class_id} (n={len(scores)})', density=True)
    
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Matched Box Confidence Score Distribution by Class\n(IoU > {iou_threshold} Matching)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.show()

def plot_score_boxplot(matched_scores_per_class, save_path=None, figsize=(10, 6), iou_threshold=0.5):
    """
    绘制箱线图对比各类别的分数分布
    
    Args:
        matched_scores_per_class: dict, key=类别, value=分数列表
        save_path: 保存路径
        figsize: 图片大小
        iou_threshold: IoU 阈值，用于标题显示
    """
    if not matched_scores_per_class:
        print("⚠️ No matched scores to plot!")
        return
    
    # 准备数据
    class_ids = []
    scores_list = []
    for class_id, scores in sorted(matched_scores_per_class.items()):
        class_ids.append(str(class_id))
        scores_list.append(scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制箱线图
    bp = ax.boxplot(scores_list, labels=class_ids, patch_artist=True,
                   showmeans=True, meanline=True, showfliers=True)
    
    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_ids)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Class ID', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title(f'Matched Box Confidence Score Distribution by Class\n(IoU > {iou_threshold} Matching)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    # 添加统计信息
    total_matches = sum(len(s) for s in scores_list)
    stats_text = f"Total matches: {total_matches}\nTotal classes: {len(class_ids)}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.show()


def iou_matching_with_fp(json_file_path, iou_threshold=0.5):
    """
    严谨的一对一匹配：同时收集 TP 和 FP 的分数
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    tp_scores_per_class = defaultdict(list)
    fp_scores_per_class = defaultdict(list)
    
    for img_data in data['detections']:
        ground_truths = img_data['ground_truths']
        predictions = img_data['predictions']
        
        if len(predictions) == 0: continue
        
        # 按类别分组
        gt_by_class = defaultdict(list)
        for gt in ground_truths: gt_by_class[gt['label']].append(gt)
        
        pred_by_class = defaultdict(list)
        for pred in predictions: pred_by_class[pred['label']].append(pred)
        
        # 获取所有涉及的类别
        all_classes = set(list(gt_by_class.keys()) + list(pred_by_class.keys()))
        
        for class_id in all_classes:
            gts = gt_by_class.get(class_id, [])
            preds = sorted(pred_by_class.get(class_id, []), key=lambda x: x['score'], reverse=True)
            
            if not preds: continue
            if not gts:
                # 该图此类别无GT，所有预测均为FP
                for p in preds: fp_scores_per_class[class_id].append(p['score'])
                continue

            # 构建 IoU 矩阵并进行一对一匹配
            iou_matrix = np.zeros((len(preds), len(gts)))
            for i, p in enumerate(preds):
                for j, g in enumerate(gts):
                    iou_matrix[i, j] = compute_iou(p['bbox'], g['bbox'])
            
            matched_gts = set()
            for i in range(len(preds)):
                best_iou = 0
                best_gt_idx = -1
                for j in range(len(gts)):
                    if j not in matched_gts and iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_gt_idx = j
                
                if best_iou >= iou_threshold:
                    tp_scores_per_class[class_id].append(preds[i]['score'])
                    matched_gts.add(best_gt_idx)
                else:
                    # 未匹配成功（IoU低、或该GT已被抢），判定为 FP
                    fp_scores_per_class[class_id].append(preds[i]['score'])
                    
    return tp_scores_per_class, fp_scores_per_class

def plot_tp_fp_distribution(tp_dict, fp_dict, save_path=None):
    """
    绘制 TP vs FP 置信度对比直方图
    """
    classes = sorted(list(set(list(tp_dict.keys()) + list(fp_dict.keys()))))
    n_cols = 3
    n_rows = (len(classes) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, cid in enumerate(classes):
        ax = axes[i]
        tp_s = tp_dict.get(cid, [])
        fp_s = fp_dict.get(cid, [])
        
        # 绘制 TP 为蓝色，FP 为红色，重叠部分透明
        if tp_s:
            ax.hist(tp_s, bins=25, alpha=0.5, color='blue', label=f'TP (n={len(tp_s)})', density=True)
        if fp_s:
            ax.hist(fp_s, bins=25, alpha=0.5, color='red', label=f'FP (n={len(fp_s)})', density=True)
            
        ax.set_title(f'Class {cid}: TP vs FP Score Dist')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Density')
        ax.legend(fontsize='small')
        ax.set_xlim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ TP/FP Distribution saved to: {save_path}")
    plt.show()


# 使用示例
if __name__ == "__main__":
    json_file = "/root/userfolder/Projects/RLCCD/output/dfine_hgnetv2_m_ccd/6_1/26/detection_results/detection_results_epoch-1_20260423_115938.json"
    
    # 设置 IoU 阈值
    IOU_THRESHOLD = 0.5
    
    # 1. 进行最大 IoU 匹配（IoU > 0.5 才算匹配）
    # matched_scores, matched_details = iou_matching_max_iou(json_file, iou_threshold=IOU_THRESHOLD)
    # matched_scores, matched_details = iou_matching_one_to_many(json_file, iou_threshold=IOU_THRESHOLD)
    
    # # 2. 绘制子图分布图
    # plot_score_distribution(matched_scores, 
    #                        save_path="/root/userfolder/Projects/RLCCD/output/score_distribution_by_class.png",
    #                        iou_threshold=IOU_THRESHOLD)
    
    # 3. 绘制叠加直方图
    # plot_score_histogram_combined(matched_scores, 
    #                               save_path="/root/userfolder/Projects/R-CCD/output/score_histogram_combined.png",
    #                               iou_threshold=IOU_THRESHOLD)
    
    # # 4. 绘制箱线图
    # plot_score_boxplot(matched_scores, 
    #                   save_path="/root/userfolder/Projects/RLCCD/output/score_boxplot.png",
    #                   iou_threshold=IOU_THRESHOLD)
    
    # # 5. 打印详细匹配信息
    # print(f"\n📋 Detailed matches (first 10):")
    # for match in matched_details['matches'][:10]:
    #     print(f"  Image {match['image_id']}, Class {match['class_id']}, "
    #           f"Score: {match['score']:.4f}, IoU: {match['iou']:.4f}")

    # 1. 同时获取 TP 和 FP 分数
    tp_scores, fp_scores = iou_matching_with_fp(json_file, iou_threshold=IOU_THRESHOLD)
    
    # 2. 绘制对比图
    plot_tp_fp_distribution(tp_scores, fp_scores, 
                            save_path="/root/userfolder/Projects/RLCCD/output/tp_fp_comparison.png")