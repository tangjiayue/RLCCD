import copy
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torchvision.ops import box_iou

import torch
import torch.distributed as dist
import numpy as np
import pickle
from collections import defaultdict

class AdvancedMetricsCalculator:
    def __init__(self, device='cuda', iou_thresholds=None):
        self.device = device
        # 如果没有指定 IoU 阈值，默认使用 COCO 标准的 0.50:0.95
        if iou_thresholds is None:
            self.iou_thresholds = np.linspace(.5, 0.95, int(round((0.95 - .5) / .05)) + 1, endpoint=True)
        else:
            self.iou_thresholds = iou_thresholds

    def _gather_data(self, gt_list, preds_list):
        """
        [显存优化版] 
        1. 先将数据移至 CPU，避免 GPU 显存占用
        2. 在 CPU 上进行序列化和 all_gather
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return gt_list, preds_list

        def _serialize_and_gather(data):
            # --- Step 0: 强制移至 CPU (关键修改) ---
            # 如果 data 是列表/字典结构，需要递归或遍历移动 Tensor
            # 这里假设 data 是 list[dict] 或 list[list] 结构
            cpu_data = []
            for item in data:
                if isinstance(item, dict):
                    # 将字典中的 Tensor 移至 CPU
                    cpu_item = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in item.items()}
                elif isinstance(item, torch.Tensor):
                    cpu_item = item.cpu()
                else:
                    cpu_item = item
                cpu_data.append(cpu_item)
            
            # --- Step 1: 序列化 (在 CPU 上进行) ---
            data_bytes = pickle.dumps(cpu_data)
            byte_list = list(data_bytes)
            local_size = len(byte_list)
            
            # 在 CPU 上创建 Tensor
            local_size_tensor = torch.tensor([local_size], dtype=torch.long)

            # --- Step 2: 收集所有进程的长度信息 ---
            world_size = dist.get_world_size()
            sizes_list = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
            dist.all_gather(sizes_list, local_size_tensor)
            
            # 找出最大的长度
            max_size = max(s.item() for s in sizes_list)
            
            # --- Step 3: 创建统一大小的缓冲区并传输 ---
            # 在 CPU 上创建缓冲区
            buffer = torch.zeros(max_size, dtype=torch.uint8)
            buffer[:local_size] = torch.tensor(byte_list, dtype=torch.uint8)
            
            gathered_buffers = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(world_size)]
            dist.all_gather(gathered_buffers, buffer)
            
            # --- Step 4: 反序列化 ---
            full_data = []
            for i, buf in enumerate(gathered_buffers):
                size = sizes_list[i].item()
                valid_bytes = bytes(buf[:size].tolist())
                full_data.extend(pickle.loads(valid_bytes))
                
            return full_data

        dist.barrier()
        return _serialize_and_gather(gt_list), _serialize_and_gather(preds_list)

    def _compute_iou_matrix(self, boxes1, boxes2):
        """
        计算两个框集合之间的 IoU 矩阵 (NxM)
        向量化操作，速度快，符合论文标准
        """
        # boxes: [N, 4] (x1, y1, x2, y2)
        N, M = len(boxes1), len(boxes2)
        if N == 0 or M == 0:
            return np.zeros((N, M))

        # 扩展维度以进行广播计算
        # boxes1 -> [N, 1, 4], boxes2 -> [1, M, 4]
        b1 = boxes1[:, np.newaxis, :]
        b2 = boxes2[np.newaxis, :, :]

        # 计算交集坐标
        xx1 = np.maximum(b1[..., 0], b2[..., 0]) # x1
        yy1 = np.maximum(b1[..., 1], b2[..., 1]) # y1
        xx2 = np.minimum(b1[..., 2], b2[..., 2]) # x2
        yy2 = np.minimum(b1[..., 3], b2[..., 3]) # y2

        # 计算交集面积
        inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        
        # 计算并集面积
        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-6)

    def _calculate_ap(self, scores, tp, fp, n_gt):
        """
        计算单类别的 AP (Average Precision)
        使用 VOC/COCO 标准的全点插值法
        """
        if n_gt == 0:
            return 0.0
        
        # 按置信度降序排列
        sorted_indices = np.argsort(-scores)
        tp = tp[sorted_indices]
        fp = fp[sorted_indices]

        # 计算累积和
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)

        # 计算 Recall 和 Precision 曲线
        # Recall = TP / Total_GT
        rec = cumsum_tp / n_gt
        # Precision = TP / (TP + FP)
        pre = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-6)

        # --- 论文级插值逻辑 (All-Point Interpolation) ---
        # 这一步是为了消除 PR 曲线的抖动，保证单调性
        # 从后向前遍历，取当前位置之后的最大 Precision
        for i in range(len(pre) - 1, 0, -1):
            pre[i - 1] = np.maximum(pre[i - 1], pre[i])

        # 找出 Recall 发生变化的索引点
        # 我们在 Recall 序列前后补 0 和 1，以便处理边界情况
        m_rec = np.concatenate(([0.0], rec, [1.0]))
        m_pre = np.concatenate(([0.0], pre, [0.0]))

        # 计算 Recall 变化的差值
        i = np.where(m_rec[1:] != m_rec[:-1])[0]

        # 计算曲线下面积 (AUC)
        # Sum of (delta_recall * max_precision_at_that_recall)
        ap = np.sum((m_rec[i + 1] - m_rec[i]) * m_pre[i + 1])
        
        return ap

    def _calculate_f1_curves(self, scores, tp, fp, n_gt, cid, save_path="f1_curves.png"):
        """
        计算单类别的 F1 曲线数据
        """
        if n_gt == 0 or len(scores) == 0:
            return 0.0, 0.0, None, None

        indices = np.argsort(-scores)
        scores = scores[indices]
        tp = tp[indices]
        fp = fp[indices]

        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)

        recalls = cumsum_tp / n_gt
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-6)
        
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        
        max_f1_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[max_f1_idx]
        best_thr = scores[max_f1_idx]
        
        return max_f1, best_thr, scores, f1_scores

    def compute(self, gt_list, preds_list):
        """
        主计算函数：最小化改动，增加 F1 曲线绘图
        """
        # 1. 汇聚所有 GPU 数据 (保持不变)
        all_gt, all_preds = self._gather_data(gt_list, preds_list)
        
        # 2. 统计全局信息 (保持不变)
        cat_ids = set()
        for g in all_gt: cat_ids.update(g['labels'].unique().cpu().numpy().tolist())
        for p in all_preds: cat_ids.update(p['labels'].unique().cpu().numpy().tolist())
        cat_ids = sorted(list(cat_ids)) # 排序使图例有序
        
        n_gt_per_class = {cid: 0 for cid in cat_ids}
        for g in all_gt:
            labels = g['labels'].cpu().numpy()
            for l in labels: n_gt_per_class[int(l)] += 1

        # 3 & 4. 初始化与严谨匹配逻辑 (保持不变)
        eval_data = {i: {cid: {'scores': [], 'tp': [], 'fp': []} for cid in cat_ids} 
                     for i in range(len(self.iou_thresholds))}

        for img_gt, img_pred in zip(all_gt, all_preds):
            gt_boxes, gt_labels = img_gt['boxes'].cpu().numpy(), img_gt['labels'].cpu().numpy()
            pred_boxes, pred_scores, pred_labels = img_pred['boxes'].cpu().numpy(), img_pred['scores'].cpu().numpy(), img_pred['labels'].cpu().numpy()
            if len(pred_boxes) == 0: continue
            ious = self._compute_iou_matrix(pred_boxes, gt_boxes)
            pred_order = np.argsort(-pred_scores)
            
            for i, iou_thr in enumerate(self.iou_thresholds):
                matched_gt = set()
                for p_idx in pred_order:
                    p_score, p_label = pred_scores[p_idx], int(pred_labels[p_idx])
                    best_iou, best_gt_idx = 0, -1
                    for g_idx in range(len(gt_boxes)):
                        if g_idx in matched_gt or gt_labels[g_idx] != p_label: continue
                        if ious[p_idx, g_idx] > best_iou:
                            best_iou, best_gt_idx = ious[p_idx, g_idx], g_idx
                    
                    if best_iou >= iou_thr:
                        eval_data[i][p_label]['scores'].append(p_score); eval_data[i][p_label]['tp'].append(1); eval_data[i][p_label]['fp'].append(0)
                        matched_gt.add(best_gt_idx)
                    else:
                        eval_data[i][p_label]['scores'].append(p_score); eval_data[i][p_label]['tp'].append(0); eval_data[i][p_label]['fp'].append(1)

        # 5. 计算最终指标并绘图
        aps = []
        max_f1_list = [] # 用于存储每个类别的最大 F1
        is_main = (not dist.is_initialized()) or dist.get_rank() == 0
        
        # --- 绘图初始化 ---
        if is_main: plt.figure(figsize=(10, 7))

        for i, iou_thr in enumerate(self.iou_thresholds):
            class_aps = []
            for cid in cat_ids:
                data = eval_data[i][cid]
                if len(data['scores']) > 0:
                    scores = np.array(data['scores']); tp = np.array(data['tp']); fp = np.array(data['fp'])
                    ap = self._calculate_ap(scores, tp, fp, n_gt_per_class[cid])
                    
                    # --- 核心改动：提取每个类别的 Max F1 ---
                    if i == 0: # 仅在 IoU=0.5 时计算
                        max_f1, best_thr, s_vec, f1_vec = self._calculate_f1_curves(scores, tp, fp, n_gt_per_class[cid], cid)
                        max_f1_list.append(max_f1) # 收集该类别的巅峰 F1
                        if is_main and s_vec is not None:
                            plt.plot(s_vec, f1_vec, label=f'Class {cid} (MaxF1:{max_f1:.4f})')
                            plt.scatter(best_thr, max_f1, s=30)
                else:
                    ap = 0.0
                    if i == 0: max_f1_list.append(0.0)
                
                class_aps.append(ap)
            aps.append(np.mean(class_aps))

        # --- 保存 F1 曲线图 ---
        if is_main:
            plt.title('F1-Score Curves per Class (IoU=0.50)')
            plt.xlabel('Confidence Threshold'); plt.ylabel('F1-Score')
            plt.ylim([0, 1.05]); plt.xlim([0, 1.0]); plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize='small')
            plt.tight_layout()
            plt.savefig('/root/userfolder/Projects/RLCCD/output/f1_curves_iou05.png', dpi=300)
            plt.close()
            print(f"\n[Visual] F1 curves saved to f1_curves_iou05.png")

        # 6. 计算汇总指标 (完全维持你原本的逻辑)
        map_50_95 = np.mean(aps)
        map_50 = aps[0]
        map_75 = aps[5] if len(aps) > 5 else 0.0
        
        total_tp = sum(sum(eval_data[0][cid]['tp']) for cid in cat_ids)
        total_fp = sum(sum(eval_data[0][cid]['fp']) for cid in cat_ids)
        total_gt_all = sum(n_gt_per_class.values())
        total_fn = total_gt_all - total_tp
        
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        avg_max_f1 = np.mean(max_f1_list) if max_f1_list else 0.0

        return {
            "mAP_50_95": map_50_95,
            "mAP_50": map_50,
            "mAP_75": map_75,
            "Max-F1 (Avg)": avg_max_f1,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn
        }



class Validator:
    def __init__(
        self,
        gt: List[Dict[str, torch.Tensor]],
        preds: List[Dict[str, torch.Tensor]],
        conf_thresh=0.5,
        iou_thresh=0.5,
    ) -> None:
        """
        Format example:
        gt = [{'labels': tensor([0]), 'boxes': tensor([[561.0, 297.0, 661.0, 359.0]])}, ...]
        len(gt) is the number of images
        bboxes are in format [x1, y1, x2, y2], absolute values
        """
        self.gt = gt
        self.preds = preds
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.thresholds = np.arange(0.2, 1.0, 0.05)
        self.conf_matrix = None

    def compute_metrics(self, extended=False) -> Dict[str, float]:
        filtered_preds = filter_preds(copy.deepcopy(self.preds), self.conf_thresh)
        metrics = self._compute_main_metrics(filtered_preds)
        if not extended:
            metrics.pop("extended_metrics", None)
        return metrics

    def _compute_main_metrics(self, preds):
        (
            self.metrics_per_class,
            self.conf_matrix,
            self.class_to_idx,
        ) = self._compute_metrics_and_confusion_matrix(preds)
        tps, fps, fns = 0, 0, 0
        ious = []
        extended_metrics = {}
        for key, value in self.metrics_per_class.items():
            tps += value["TPs"]
            fps += value["FPs"]
            fns += value["FNs"]
            ious.extend(value["IoUs"])

            extended_metrics[f"precision_{key}"] = (
                value["TPs"] / (value["TPs"] + value["FPs"])
                if value["TPs"] + value["FPs"] > 0
                else 0
            )
            extended_metrics[f"recall_{key}"] = (
                value["TPs"] / (value["TPs"] + value["FNs"])
                if value["TPs"] + value["FNs"] > 0
                else 0
            )

            extended_metrics[f"iou_{key}"] = np.mean(value["IoUs"])

        precision = tps / (tps + fps) if (tps + fps) > 0 else 0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = np.mean(ious).item() if ious else 0
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "TPs": tps,
            "FPs": fps,
            "FNs": fns,
            "extended_metrics": extended_metrics,
        }

    def _compute_metrics_and_confusion_matrix(self, preds):
        # Initialize per-class metrics
        metrics_per_class = defaultdict(lambda: {"TPs": 0, "FPs": 0, "FNs": 0, "IoUs": []})

        # --- 新增：深度分析统计变量 ---
        self.analysis_total_gt = 0
        self.analysis_matched_loc = 0
        self.analysis_label_errors = 0
        self.analysis_mis_to_other = 0    # 0-4 -> 5-9
        self.analysis_mis_to_ordinal = 0  # 5-9 -> 0-4
        # 统计 Over/Under: {class_id: [matched, correct, over, under]}
        self.analysis_class_details = defaultdict(lambda: {"matched": 0, "correct": 0, "over": 0, "under": 0})

        # Collect all class IDs
        all_classes = set()
        for pred in preds:
            all_classes.update(pred["labels"].tolist())
        for gt in self.gt:
            all_classes.update(gt["labels"].tolist())
        all_classes = sorted(list(all_classes))
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(all_classes)}
        n_classes = len(all_classes)
        conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)  # +1 for background class

        for pred, gt in zip(preds, self.gt):
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            n_preds = len(pred_boxes)
            n_gts = len(gt_boxes)

            self.analysis_total_gt += n_gts # 累加总 GT 数

            if n_preds == 0 and n_gts == 0:
                continue

            ious = box_iou(pred_boxes, gt_boxes) if n_preds > 0 and n_gts > 0 else torch.tensor([])
            # Assign matches between preds and gts
            matched_pred_indices = set()
            matched_gt_indices = set()

            if ious.numel() > 0:
                # For each pred box, find the gt box with highest IoU
                ious_mask = ious >= self.iou_thresh
                pred_indices, gt_indices = torch.nonzero(ious_mask, as_tuple=True)
                iou_values = ious[pred_indices, gt_indices]

                # Sorting by IoU to match highest scores first
                sorted_indices = torch.argsort(iou_values, descending=True, stable=True)
                pred_indices = pred_indices[sorted_indices]
                gt_indices = gt_indices[sorted_indices]
                iou_values = iou_values[sorted_indices]

                for pred_idx, gt_idx, iou in zip(pred_indices, gt_indices, iou_values):
                    if (
                        pred_idx.item() in matched_pred_indices
                        or gt_idx.item() in matched_gt_indices
                    ):
                        continue
                    matched_pred_indices.add(pred_idx.item())
                    matched_gt_indices.add(gt_idx.item())

                    self.analysis_matched_loc += 1

                    pred_label = pred_labels[pred_idx].item()
                    gt_label = gt_labels[gt_idx].item()

                    self.analysis_class_details[gt_label]["matched"] += 1
                    if pred_label != gt_label:
                        self.analysis_label_errors += 1
                        # 方向统计
                        if pred_label < gt_label: self.analysis_class_details[gt_label]["over"] += 1
                        else: self.analysis_class_details[gt_label]["under"] += 1
                        
                        # 跨组统计 (0-4 vs 5-9)
                        if gt_label < 5 and pred_label >= 5: self.analysis_mis_to_other += 1
                        elif gt_label >= 5 and pred_label < 5: self.analysis_mis_to_ordinal += 1
                    else:
                        self.analysis_class_details[gt_label]["correct"] += 1


                    pred_cls_idx = class_to_idx[pred_label]
                    gt_cls_idx = class_to_idx[gt_label]

                    # Update confusion matrix
                    conf_matrix[gt_cls_idx, pred_cls_idx] += 1

                    # Update per-class metrics
                    if pred_label == gt_label:
                        metrics_per_class[gt_label]["TPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(iou.item())
                    else:
                        # Misclassification
                        metrics_per_class[gt_label]["FNs"] += 1
                        metrics_per_class[pred_label]["FPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(0)
                        metrics_per_class[pred_label]["IoUs"].append(0)

            # Unmatched predictions (False Positives)
            unmatched_pred_indices = set(range(n_preds)) - matched_pred_indices
            for pred_idx in unmatched_pred_indices:
                pred_label = pred_labels[pred_idx].item()
                pred_cls_idx = class_to_idx[pred_label]
                # Update confusion matrix: background row
                conf_matrix[n_classes, pred_cls_idx] += 1
                # Update per-class metrics
                metrics_per_class[pred_label]["FPs"] += 1
                metrics_per_class[pred_label]["IoUs"].append(0)

            # Unmatched ground truths (False Negatives)
            unmatched_gt_indices = set(range(n_gts)) - matched_gt_indices
            for gt_idx in unmatched_gt_indices:
                gt_label = gt_labels[gt_idx].item()
                gt_cls_idx = class_to_idx[gt_label]
                # Update confusion matrix: background column
                conf_matrix[gt_cls_idx, n_classes] += 1
                # Update per-class metrics
                metrics_per_class[gt_label]["FNs"] += 1
                metrics_per_class[gt_label]["IoUs"].append(0)

                # self.analysis_label_errors += 1

        # 打印分析结果
        self._print_analysis_report()

        return metrics_per_class, conf_matrix, class_to_idx

    def _print_analysis_report(self):
        """打印深度分析报告的方法 """
        pure_loc_recall = self.analysis_matched_loc / self.analysis_total_gt if self.analysis_total_gt > 0 else 0
        label_error_rate = self.analysis_label_errors / self.analysis_matched_loc if self.analysis_matched_loc > 0 else 0
        
        print("\n" + "=" * 25)
        print("深度匹配分析 (基于 IoU 优先的一对一匹配)")
        print(f"1. 仅位置召回率 (Pure Loc Recall): {pure_loc_recall:.2%} ({self.analysis_matched_loc}/{self.analysis_total_gt})")
        print(f"2. 已匹配分类错误率(Label Error Rate): {label_error_rate:.2%} ({self.analysis_label_errors}个)")
        # print(f"3. 分支判定错误: Ordinal->Other: {self.analysis_mis_to_other} | Other->Ordinal: {self.analysis_mis_to_ordinal}")
        print("-" * 50)
        print(f"{'Class':>5} | {'Matched':>7} | {'Correct':>7} | {'Over':>5} | {'Under':>5}")
        for c in sorted(self.analysis_class_details.keys()):
            s = self.analysis_class_details[c]
            print(f"{c:>5} | {s['matched']:>7} | {s['correct']:>7} | {s['over']:>5} | {s['under']:>5}")
        print("=" * 25 + "\n")

    def save_plots(self, path_to_save) -> None:
        path_to_save = Path(path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)

        if self.conf_matrix is not None:
            class_labels = [str(cls_id) for cls_id in self.class_to_idx.keys()] + ["background"]

            plt.figure(figsize=(10, 8))
            plt.imshow(self.conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(class_labels))
            plt.xticks(tick_marks, class_labels, rotation=45)
            plt.yticks(tick_marks, class_labels)

            # Add labels to each cell
            thresh = self.conf_matrix.max() / 2.0
            for i in range(self.conf_matrix.shape[0]):
                for j in range(self.conf_matrix.shape[1]):
                    plt.text(
                        j,
                        i,
                        format(self.conf_matrix[i, j], "d"),
                        horizontalalignment="center",
                        color="white" if self.conf_matrix[i, j] > thresh else "black",
                    )

            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.savefig(path_to_save / "confusion_matrix.png")
            plt.close()

        thresholds = self.thresholds
        precisions, recalls, f1_scores = [], [], []

        # Store the original predictions to reset after each threshold
        original_preds = copy.deepcopy(self.preds)

        for threshold in thresholds:
            # Filter predictions based on the current threshold
            filtered_preds = filter_preds(copy.deepcopy(original_preds), threshold)
            # Compute metrics with the filtered predictions
            metrics = self._compute_main_metrics(filtered_preds)
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1_scores.append(metrics["f1"])

        # Plot Precision and Recall vs Threshold
        plt.figure()
        plt.plot(thresholds, precisions, label="Precision", marker="o")
        plt.plot(thresholds, recalls, label="Recall", marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("Value")
        plt.title("Precision and Recall vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.savefig(path_to_save / "precision_recall_vs_threshold.png")
        plt.close()

        # Plot F1 Score vs Threshold
        plt.figure()
        plt.plot(thresholds, f1_scores, label="F1 Score", marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.title("F1 Score vs Threshold")
        plt.grid(True)
        plt.savefig(path_to_save / "f1_score_vs_threshold.png")
        plt.close()

        # Find the best threshold based on F1 Score (last occurence)
        best_idx = len(f1_scores) - np.argmax(f1_scores[::-1]) - 1
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        logger.info(
            f"Best Threshold: {round(best_threshold, 2)} with F1 Score: {round(best_f1, 3)}"
        )


def filter_preds(preds, conf_thresh):
    # print(f"当前 Batch 图片张数：{len(preds)}")
    for i, pred in enumerate(preds):
        # before_count = len(pred["scores"])
        # score = pred["scores"]
        # print(f"pred_scores：{score}")

        keep_idxs = pred["scores"] >= conf_thresh
        pred["scores"] = pred["scores"][keep_idxs]
        pred["boxes"] = pred["boxes"][keep_idxs]
        pred["labels"] = pred["labels"][keep_idxs]

        # after_count = len(pred["scores"])
        # print(f"图片 {i}: 过滤前有 {before_count} 个框 -> 过滤后剩 {after_count} 个框")
    return preds


def scale_boxes(boxes, orig_shape, resized_shape):
    """
    boxes in format: [x1, y1, x2, y2], absolute values
    orig_shape: [height, width]
    resized_shape: [height, width]
    """
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes