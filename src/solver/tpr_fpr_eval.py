from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: [N,4] xyxy
    boxes2: [M,4] xyxy
    return: [N,M] IoU
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-9)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add_(self, tp: int = 0, fp: int = 0, fn: int = 0):
        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)

    def merge_(self, other: "Counts"):
        self.add_(other.tp, other.fp, other.fn)

    def metrics(self) -> Dict[str, float]:
        tp, fp, fn = float(self.tp), float(self.fp), float(self.fn)
        tpr = tp / (tp + fn + 1e-9)  # recall
        precision = tp / (tp + fp + 1e-9)
        f1 = 2.0 * precision * tpr / (precision + tpr + 1e-9)

        # detection 常用“误报占比”（严格 FPR 需要 TN，但检测里 TN 无定义）
        fpr_pred = fp / (tp + fp + 1e-9)  # 1-precision

        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TPR": tpr,
            "Precision": precision,
            "F1": f1,
            "FPR_pred": fpr_pred,
        }


@torch.no_grad()
def match_tp_fp_fn_single_image(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_thr: float = 0.5,
    score_thr: float = 0.0,
    per_class: bool = True,
    ignore_label: Optional[int] = None,
) -> Tuple[Counts, Dict[int, Counts]]:
    """
    使用 score 降序的一对一贪心匹配：
    - TP: 预测匹配到一个未匹配GT (IoU>=thr 且类别一致)
    - FP: 预测未匹配到任何GT
    - FN: GT 未被任何预测匹配

    返回：
      overall_counts, per_class_counts
    """
    if ignore_label is not None and gt_labels.numel() > 0:
        keep = gt_labels != int(ignore_label)
        gt_boxes = gt_boxes[keep]
        gt_labels = gt_labels[keep]

    if pred_scores.numel() > 0 and score_thr > 0:
        keep = pred_scores >= float(score_thr)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

    overall = Counts()
    by_cls: Dict[int, Counts] = {}

    # trivial cases
    if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
        return overall, by_cls
    if gt_boxes.numel() == 0 and pred_boxes.numel() > 0:
        overall.add_(fp=int(pred_boxes.shape[0]))
        if per_class:
            for c in pred_labels.unique().tolist():
                c = int(c)
                by_cls.setdefault(c, Counts()).add_(fp=int((pred_labels == c).sum().item()))
        return overall, by_cls
    if gt_boxes.numel() > 0 and pred_boxes.numel() == 0:
        overall.add_(fn=int(gt_boxes.shape[0]))
        if per_class:
            for c in gt_labels.unique().tolist():
                c = int(c)
                by_cls.setdefault(c, Counts()).add_(fn=int((gt_labels == c).sum().item()))
        return overall, by_cls

    # choose classes to evaluate
    if per_class:
        classes = sorted(set(gt_labels.unique().tolist()) | set(pred_labels.unique().tolist()))
    else:
        classes = [-1]  # sentinel meaning "all together"

    for c in classes:
        if per_class:
            gt_mask = gt_labels == int(c)
            pr_mask = pred_labels == int(c)
            gtb = gt_boxes[gt_mask]
            prb = pred_boxes[pr_mask]
            prs = pred_scores[pr_mask]
        else:
            gtb = gt_boxes
            prb = pred_boxes
            prs = pred_scores

        if gtb.numel() == 0 and prb.numel() == 0:
            continue
        if gtb.numel() == 0 and prb.numel() > 0:
            fp = int(prb.shape[0])
            overall.add_(fp=fp)
            if per_class:
                by_cls.setdefault(int(c), Counts()).add_(fp=fp)
            continue
        if gtb.numel() > 0 and prb.numel() == 0:
            fn = int(gtb.shape[0])
            overall.add_(fn=fn)
            if per_class:
                by_cls.setdefault(int(c), Counts()).add_(fn=fn)
            continue

        order = torch.argsort(prs, descending=True)
        prb = prb[order]

        ious = box_iou_xyxy(prb, gtb)  # [P,G]
        gt_used = torch.zeros((gtb.shape[0],), dtype=torch.bool, device=gtb.device)

        tp = 0
        fp = 0
        for pi in range(prb.shape[0]):
            best_iou, best_gi = ious[pi].max(dim=0)
            if float(best_iou.item()) >= float(iou_thr) and (not bool(gt_used[best_gi].item())):
                tp += 1
                gt_used[best_gi] = True
            else:
                fp += 1

        fn = int((~gt_used).sum().item())

        overall.add_(tp=tp, fp=fp, fn=fn)
        if per_class:
            by_cls.setdefault(int(c), Counts()).add_(tp=tp, fp=fp, fn=fn)

    return overall, by_cls


@torch.no_grad()
def evaluate_tpr_fpr_from_gt_preds(
    gt: List[Dict[str, torch.Tensor]],
    preds: List[Dict[str, torch.Tensor]],
    iou_thr: float = 0.5,
    score_thr: float = 0.0,
    per_class: bool = True,
    ignore_label: Optional[int] = -1,
) -> Dict[str, Dict]:
    """
    直接复用 det_engine.evaluate() 已经构造好的 gt/preds 列表。
    每个元素格式：
      gt[i]    = {"boxes": [G,4] xyxy, "labels": [G]}
      preds[i] = {"boxes": [P,4] xyxy, "labels": [P], "scores":[P]}
    """
    assert len(gt) == len(preds), "gt/preds length mismatch"

    total = Counts()
    by_cls_total: Dict[int, Counts] = {}

    for g, p in zip(gt, preds):
        overall_i, by_cls_i = match_tp_fp_fn_single_image(
            gt_boxes=g["boxes"],
            gt_labels=g["labels"],
            pred_boxes=p["boxes"],
            pred_labels=p["labels"],
            pred_scores=p["scores"],
            iou_thr=iou_thr,
            score_thr=score_thr,
            per_class=per_class,
            ignore_label=ignore_label,
        )
        total.merge_(overall_i)
        if per_class:
            for c, cnt in by_cls_i.items():
                by_cls_total.setdefault(int(c), Counts()).merge_(cnt)

    out = {"overall": total.metrics(), "settings": {"iou_thr": iou_thr, "score_thr": score_thr}}
    if per_class:
        out["per_class"] = {int(c): cnt.metrics() for c, cnt in sorted(by_cls_total.items(), key=lambda x: x[0])}
    return out



import os
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def generate_and_plot_roc(
    gt: List[Dict[str, torch.Tensor]],
    preds: List[Dict[str, torch.Tensor]],
    iou_thr: float = 0.5,
    ignore_label: Optional[int] = -1,
    save_path: str = "roc_curve_iou0.5.png"
):
    """
    收集所有图像在指定 IoU 下的匹配结果，按置信度降序计算 TPR 和 FPR，并作图。
    FPR 采用归一化形式：当前累计 FP / 模型产生的总 FP
    """
    total_gt = 0
    all_preds_info = []  # 存储 (score, is_tp_flag)

    for g, p in zip(gt, preds):
        gt_boxes = g["boxes"]
        gt_labels = g["labels"]
        pred_boxes = p["boxes"]
        pred_labels = p["labels"]
        pred_scores = p["scores"]

        # 过滤忽略的类别
        if ignore_label is not None and gt_labels.numel() > 0:
            keep = gt_labels != int(ignore_label)
            gt_boxes = gt_boxes[keep]
            gt_labels = gt_labels[keep]

        total_gt += gt_boxes.shape[0]

        if pred_boxes.numel() == 0:
            continue
            
        if gt_boxes.numel() == 0:
            # 图像无GT，所有预测都是 FP
            for score in pred_scores:
                all_preds_info.append((score.item(), 0))
            continue

        # 按 score 降序贪心匹配
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]

        ious = box_iou_xyxy(pred_boxes, gt_boxes)  # [P, G]
        gt_used = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=gt_boxes.device)

        for pi in range(pred_boxes.shape[0]):
            is_tp = 0
            # 必须类别一致的预测才能匹配 (你可以根据需求决定是否要在算整体ROC时强制类别一致)
            valid_mask = (gt_labels == pred_labels[pi])
            if valid_mask.any():
                valid_ious = ious[pi] * valid_mask.float()
                best_iou, best_gi = valid_ious.max(dim=0)
                if best_iou.item() >= iou_thr and not gt_used[best_gi].item():
                    is_tp = 1
                    gt_used[best_gi] = True
            
            all_preds_info.append((pred_scores[pi].item(), is_tp))

    if total_gt == 0 or len(all_preds_info) == 0:
        print("未检测到有效样本或预测，终止画图。")
        return

    # 按置信度排序
    all_preds_info.sort(key=lambda x: x[0], reverse=True)
    tps = np.array([x[1] for x in all_preds_info])
    fps = 1 - tps

    # 累加前缀和
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)

    # 计算 TPR 和 FPR 数组
    tpr_array = cum_tp / total_gt
    total_fp = cum_fp[-1]
    
    if total_fp > 0:
        fpr_array = cum_fp / total_fp
    else:
        fpr_array = np.zeros_like(cum_fp)

    # 插值寻找 FPR=0.3, FPR=0.5 和 FPR=0.8 时的 TPR 值
    # fpr_array 随阈值降低必定单调递增，可以直接插值
    tpr_at_03 = np.interp(0.3, fpr_array, tpr_array)
    tpr_at_05 = np.interp(0.5, fpr_array, tpr_array)
    tpr_at_08 = np.interp(0.8, fpr_array, tpr_array)

    # 开始绘图
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_array, tpr_array, label=f'ROC Curve (IoU={iou_thr})', color='blue', linewidth=2)
    
    # 标出 0.3, 0.5 和 0.8 的点
    plt.scatter([0.3, 0.5, 0.8], [tpr_at_03, tpr_at_05, tpr_at_08], color='red', zorder=5)
    
    # 画辅助线和文本 (FPR=0.3)
    plt.axvline(x=0.3, color='gray', linestyle='--', alpha=0.6)
    plt.axhline(y=tpr_at_03, color='gray', linestyle='--', alpha=0.6)
    plt.text(0.32, tpr_at_03 - 0.05, f'TPR={tpr_at_03:.4f}', color='red', fontsize=10)

    # 画辅助线和文本 (FPR=0.5)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.6)
    plt.axhline(y=tpr_at_05, color='gray', linestyle='--', alpha=0.6)
    plt.text(0.52, tpr_at_05 - 0.05, f'TPR={tpr_at_05:.4f}', color='red', fontsize=10)

    # 画辅助线和文本 (FPR=0.8)
    plt.axvline(x=0.8, color='gray', linestyle='--', alpha=0.6)
    plt.axhline(y=tpr_at_08, color='gray', linestyle='--', alpha=0.6)
    plt.text(0.82, tpr_at_08 - 0.05, f'TPR={tpr_at_08:.4f}', color='red', fontsize=10)

    plt.title(f'Detection ROC Curve (IoU={iou_thr})')
    plt.xlabel('False Positive Rate (Normalized)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC曲线已保存至 {os.path.abspath(save_path)}")
    print(f" - 当 FPR=0.3 时, TPR={tpr_at_03:.4f}")
    print(f" - 当 FPR=0.5 时, TPR={tpr_at_05:.4f}")
    print(f" - 当 FPR=0.8 时, TPR={tpr_at_08:.4f}")

    return {"FPR_0.3_TPR": tpr_at_03, "FPR_0.5_TPR": tpr_at_05, "FPR_0.8_TPR": tpr_at_08}