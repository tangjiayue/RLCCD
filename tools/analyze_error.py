import json
import sys
from collections import defaultdict

def box_xyxy_to_cxcywh(box):
    """将 xyxy 格式的框转换为 cxcywh"""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def box_cxcywh_to_xyxy(box):
    """将 cxcywh 格式的框转换为 xyxy"""
    cx, cy, w, h = box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def compute_iou(box1, box2):
    """计算两个框的 IoU（输入 xyxy 格式）"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou

def iou_one_to_one_match_with_score_sorting(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """
    先按置信度排序，再 IoU 一对一匹配（COCO标准方式）
    """
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    
    if num_gt == 0 or num_pred == 0:
        return [], list(range(num_gt)), list(range(num_pred))
    
    # 按置信度降序排序
    sorted_indices = sorted(range(num_pred), key=lambda i: pred_scores[i], reverse=True)
    sorted_pred_boxes = [pred_boxes[i] for i in sorted_indices]
    sorted_pred_labels = [pred_labels[i] for i in sorted_indices]
    sorted_pred_scores = [pred_scores[i] for i in sorted_indices]
    original_indices = sorted_indices
    
    matched_gt = set()
    matched_pred = set()
    matches = []
    
    for pred_pos, (pred_box, pred_label, pred_score) in enumerate(zip(sorted_pred_boxes, sorted_pred_labels, sorted_pred_scores)):
        if len(matched_gt) == num_gt:
            break
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            matches.append((best_gt_idx, original_indices[pred_pos], best_iou, pred_score))
            matched_gt.add(best_gt_idx)
            matched_pred.add(original_indices[pred_pos])
    
    unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
    unmatched_pred = [i for i in range(num_pred) if i not in matched_pred]
    
    return matches, unmatched_gt, unmatched_pred

def iou_one_to_many_match(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """
    一对多匹配：每个GT可以匹配多个预测框
    同一个预测框只能匹配一个GT（取IoU最高的）
    """
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    
    if num_gt == 0 or num_pred == 0:
        return [], list(range(num_gt)), list(range(num_pred))
    
    # 按置信度降序排序
    sorted_indices = sorted(range(num_pred), key=lambda i: pred_scores[i], reverse=True)
    sorted_pred_boxes = [pred_boxes[i] for i in sorted_indices]
    sorted_pred_labels = [pred_labels[i] for i in sorted_indices]
    sorted_pred_scores = [pred_scores[i] for i in sorted_indices]
    original_indices = sorted_indices
    
    # 记录每个预测框匹配到的GT（用于去重）
    pred_to_gt = {}  # pred_idx -> gt_idx
    gt_matched_count = {i: 0 for i in range(num_gt)}  # 记录每个GT匹配了多少个预测框
    matches = []
    
    # 计算所有 IoU 矩阵
    iou_matrix = [[compute_iou(gt_box, pred_box) for pred_box in sorted_pred_boxes] for gt_box in gt_boxes]
    
    for pred_pos, (pred_box, pred_label, pred_score) in enumerate(zip(sorted_pred_boxes, sorted_pred_labels, sorted_pred_scores)):
        if pred_pos in pred_to_gt:
            continue
        
        best_iou = 0
        best_gt_idx = -1
        
        # 找与当前预测框 IoU 最高的 GT（允许已匹配的GT重复匹配）
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            iou = iou_matrix[gt_idx][pred_pos]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            pred_to_gt[pred_pos] = best_gt_idx
            gt_matched_count[best_gt_idx] += 1
            matches.append((best_gt_idx, original_indices[pred_pos], best_iou, pred_score))
    
    # 找出未匹配的GT和预测框
    matched_gt = set(pred_to_gt.values())
    unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
    unmatched_pred = [i for i in range(num_pred) if original_indices[i] not in [m[1] for m in matches]]
    
    return matches, unmatched_gt, unmatched_pred

def iou_one_to_many_match_all(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """
    一对多匹配的宽松版本：每个GT可以匹配多个预测框，每个预测框也可以匹配多个GT
    （不要求预测框唯一，真正的"一对多"）
    """
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    
    if num_gt == 0 or num_pred == 0:
        return [], list(range(num_gt)), list(range(num_pred))
    
    # 按置信度降序排序
    sorted_indices = sorted(range(num_pred), key=lambda i: pred_scores[i], reverse=True)
    sorted_pred_boxes = [pred_boxes[i] for i in sorted_indices]
    sorted_pred_labels = [pred_labels[i] for i in sorted_indices]
    sorted_pred_scores = [pred_scores[i] for i in sorted_indices]
    original_indices = sorted_indices
    
    matches = []
    gt_matched_count = {i: 0 for i in range(num_gt)}
    
    # 计算所有 IoU 矩阵
    iou_matrix = [[compute_iou(gt_box, pred_box) for pred_box in sorted_pred_boxes] for gt_box in gt_boxes]
    
    # 对每个预测框，找所有IoU超过阈值的GT
    for pred_pos, (pred_box, pred_label, pred_score) in enumerate(zip(sorted_pred_boxes, sorted_pred_labels, sorted_pred_scores)):
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            iou = iou_matrix[gt_idx][pred_pos]
            if iou >= iou_threshold:
                matches.append((gt_idx, original_indices[pred_pos], iou, pred_score))
                gt_matched_count[gt_idx] += 1
    
    # 找出未匹配的GT
    matched_gt = set(m[0] for m in matches)
    unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
    
    # 找出未匹配的预测框
    matched_pred = set(m[1] for m in matches)
    unmatched_pred = [i for i in range(num_pred) if i not in matched_pred]
    
    return matches, unmatched_gt, unmatched_pred

def iou_one_to_one_match_with_iou_sorting(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """
    先按 IoU 降序排序，再 IoU 一对一匹配
    每个 GT 只能匹配一个预测框，每个预测框只能匹配一个 GT
    """
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    
    if num_gt == 0 or num_pred == 0:
        return [], list(range(num_gt)), list(range(num_pred))
    
    # 计算所有 IoU 矩阵
    iou_matrix = []
    for gt_box in gt_boxes:
        row = []
        for pred_box in pred_boxes:
            row.append(compute_iou(gt_box, pred_box))
        iou_matrix.append(row)
    
    # 收集所有 IoU >= 阈值的配对
    candidates = []
    for gt_idx in range(num_gt):
        for pred_idx in range(num_pred):
            iou = iou_matrix[gt_idx][pred_idx]
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))
    
    # 按 IoU 降序排序
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    matched_gt = set()
    matched_pred = set()
    matches = []
    
    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matches.append((gt_idx, pred_idx, iou, pred_scores[pred_idx]))
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
    
    unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
    unmatched_pred = [i for i in range(num_pred) if i not in matched_pred]
    
    return matches, unmatched_gt, unmatched_pred


def analyze_classification_errors(json_path, iou_threshold=0.5, match_mode="one_to_one"):
    """
    分析分类错误：匹配后，GT类别与预测类别不一致的框
    
    match_mode:
        - "one_to_one": 一对一匹配（标准评估，按置信度排序）
        - "one_to_one_iou": 一对一匹配（按 IoU 排序）
        - "one_to_many": 一对多匹配（预测框唯一，GT可多）
        - "one_to_many_all": 一对多匹配（完全松散，用于统计所有可能混淆）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    CLASS_NAMES = {
        0: "normal",
        1: "ascus",
        2: "asch",
        3: "lsil",
        4: "hsil_scc_omn",
        5: "agc_adenocarcinoma_em",
        6: "vaginalis",
        7: "monilia",
        8: "dysbacteriosis_herpes_act",
        9: "ec"
    }
    
    all_errors = []
    
    for img_data in data['detections']:
        image_id = img_data['image_id']
        gt_list = img_data['ground_truths']
        pred_list = img_data['predictions']
        
        gt_boxes = []
        gt_labels = []
        for gt in gt_list:
            gt_boxes.append(gt['bbox'])
            gt_labels.append(gt['label'])
        
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        for pred in pred_list:
            pred_boxes.append(pred['bbox'])
            pred_labels.append(pred['label'])
            pred_scores.append(pred['score'])
        
        # 根据模式选择匹配方式
        if match_mode == "one_to_one":
            matches, unmatched_gt, unmatched_pred = iou_one_to_one_match_with_score_sorting(
                gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold
            )
        elif match_mode == "one_to_one_iou":
            matches, unmatched_gt, unmatched_pred = iou_one_to_one_match_with_iou_sorting(
                gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold
            )
        elif match_mode == "one_to_many":
            matches, unmatched_gt, unmatched_pred = iou_one_to_many_match(
                gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold
            )
        elif match_mode == "one_to_many_all":
            matches, unmatched_gt, unmatched_pred = iou_one_to_many_match_all(
                gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold
            )
        else:
            raise ValueError(f"Unknown match_mode: {match_mode}")
        
        for gt_idx, pred_idx, iou, score in matches:
            gt_label = gt_labels[gt_idx]
            pred_label = pred_labels[pred_idx]
            
            if gt_label != pred_label:
                error_info = {
                    "image_id": image_id,
                    "gt_label": gt_label,
                    "gt_class": CLASS_NAMES.get(gt_label, str(gt_label)),
                    "pred_label": pred_label,
                    "pred_class": CLASS_NAMES.get(pred_label, str(pred_label)),
                    "pred_score": score,
                    "iou": iou,
                    "gt_bbox": gt_boxes[gt_idx],
                    "pred_bbox": pred_boxes[pred_idx]
                }
                all_errors.append(error_info)
    
    return all_errors


def save_errors_to_txt(errors, output_path="classification_errors.txt"):
    """将错误保存为文本文件，格式与命令行输出一致"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("分类错误分析报告\n")
        f.write("=" * 80 + "\n")
        
        f.write(f"\n总分类错误数: {len(errors)}\n")
        
        # 按真实类别统计错误
        error_by_gt = defaultdict(list)
        for e in errors:
            error_by_gt[e['gt_label']].append(e)
        
        f.write(f"\n按真实类别统计错误:\n")
        for label in sorted(error_by_gt.keys()):
            class_name = error_by_gt[label][0]['gt_class']
            f.write(f"  {class_name}({label}): {len(error_by_gt[label])} 个错误\n")
        
        # 打印详细错误列表
        f.write(f"\n详细错误列表:\n")
        f.write("-" * 80 + "\n")
        for i, e in enumerate(errors):
            f.write(f"\n{i+1}. 图像 {e['image_id']}\n")
            f.write(f"   真实: {e['gt_class']}({e['gt_label']})\n")
            f.write(f"   预测: {e['pred_class']}({e['pred_label']})\n")
            f.write(f"   预测置信度: {e['pred_score']:.4f}\n")
            f.write(f"   IoU: {e['iou']:.4f}\n")
            f.write(f"   真实框: {e['gt_bbox']}\n")
            f.write(f"   预测框: {e['pred_bbox']}\n")
    
    print(f"错误报告已保存至: {output_path}")

if __name__ == "__main__":
    json_path = "/root/userfolder/Projects/R-CCD/output/dfine_hgnetv2_m_ccd/dfine-ccd/detection_results/detection_results_epoch-1_20260508_141545.json"
    
    # 一对一匹配（按置信度排序）
    errors_one_to_one = analyze_classification_errors(json_path, iou_threshold=0.5, match_mode="one_to_one")
    save_errors_to_txt(errors_one_to_one, "classification_errors_one_to_one.txt")
    
    # 一对一匹配（按 IoU 排序）
    errors_one_to_one_iou = analyze_classification_errors(json_path, iou_threshold=0.5, match_mode="one_to_one_iou")
    save_errors_to_txt(errors_one_to_one_iou, "classification_errors_one_to_one_iou.txt")
    
    # # 一对多匹配（预测框唯一）
    # errors_one_to_many = analyze_classification_errors(json_path, iou_threshold=0.5, match_mode="one_to_many")
    # save_errors_to_txt(errors_one_to_many, "classification_errors_one_to_many.txt")
    
    print(f"一对一匹配错误数(按置信度排序): {len(errors_one_to_one)}")
    print(f"一对一匹配错误数(按 IoU 排序): {len(errors_one_to_one_iou)}")
    # print(f"一对多匹配错误数: {len(errors_one_to_many)}")