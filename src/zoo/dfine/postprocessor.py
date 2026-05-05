# """
# Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
# Copyright(c) 2023 lyuwenyu. All Rights Reserved.
# """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register

__all__ = ["DFINEPostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class DFINEPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self, num_classes=10, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        quality_scores_raw = outputs["quality_score"]

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = torch.sigmoid(logits)
        else:
            scores = F.softmax(logits, dim=-1)

        # --- 在 TopK 之前进行 NMS ---
        processed_logits = logits.clone()
        for i in range(logits.shape[0]):
            # 拿到当前图的预测
            cur_scores = scores[i]      # [300, 10]
            cur_boxes = bbox_pred[i]    # [300, 4] 已经是像素尺度坐标
            
            # 获取每个 Query 的最大得分和对应类别
            max_vals, labels = cur_scores.max(dim=-1)

            # conf_mask = max_vals >= 0.5
            # if not conf_mask.any():
            #     # 如果没有框超过阈值，保留最高分的一个
            #     conf_mask[max_vals.argmax()] = True
            
            # cur_boxes = cur_boxes[conf_mask]
            # max_vals = max_vals[conf_mask]
            # labels = labels[conf_mask]
            # orig_idx = torch.where(conf_mask)[0]  # 记录原始索引
            
            # if len(cur_boxes) == 0:
            #     continue
            
            # 执行 Batched NMS (同类抑制)
            # 因为已经是像素坐标，这里的 0.5 阈值会非常准确
            keep = torchvision.ops.batched_nms(cur_boxes, max_vals, labels, iou_threshold=0.5)
            
            # 创建掩码，将被抑制的 Query 的 Logits 设为极小值
            mask = torch.ones(logits.shape[1], device=logits.device, dtype=torch.bool)
            mask[keep] = False
            processed_logits[i, mask] = -1e6
        
        # 用处理后的 logits 重新计算后续排序用的 scores
        logits = processed_logits
        if self.use_focal_loss:
            scores = torch.sigmoid(logits)
        else:
            scores = F.softmax(logits, dim=-1)
        # --- NMS 结束 ---

        if self.use_focal_loss:
            # scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            labels = mod(index, self.num_classes)
            query_index = index // self.num_classes
            boxes_scaled = bbox_pred.gather(
                dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )
            boxes_norm = boxes.gather(
                dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1])
            )

            quality_scores = torch.gather(quality_scores_raw, dim=1, index=query_index.unsqueeze(-1))

        else:
            # scores = F.softmax(logits, dim=-1)
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, query_index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=query_index)  
                boxes_scaled = torch.gather(
                    bbox_pred, dim=1, index=query_index.unsqueeze(-1).tile(1, 1, bbox_pred.shape[-1])
                )
                boxes_norm = torch.gather(
                    boxes, dim=1, index=query_index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )
                quality_scores = torch.gather(quality_scores_raw, dim=1, index=query_index.unsqueeze(-1))
            else:
                boxes_scaled = bbox_pred
                boxes_norm = boxes
                query_index = torch.arange(boxes.shape[1], device=boxes.device).unsqueeze(0).repeat(boxes.shape[0], 1)
                quality_scores = quality_scores_raw

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes_scaled, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            labels = (
                torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for lab, box, sco, box_n, q_idx, q_sco in zip(labels, boxes_scaled, scores, boxes_norm, query_index, quality_scores):
            result = dict(labels=lab, boxes=box, scores=sco, boxes_norm=box_n, query_index=q_idx, quality_score=q_sco)
            results.append(result)
        
        # for lab, box, sco in zip(labels, boxes, quality):
        #     result = dict(labels=lab, boxes=box, scores=sco)
        #     results.append(result)

        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self

def expert_calibration(scores, class_configs):
    """
    scores: [N, num_classes] 原始得分
    class_configs: 每一类的映射字典 {class_id: (x_nodes, y_nodes)}
    """
    calibrated_scores = scores.clone()
    
    for cid, (x_nodes, y_nodes) in class_configs.items():
        # 获取该列分数
        s = scores[:, cid]
        
        # 构造分段线性映射
        # 原理：利用 torch.bucketize 找到区间，再进行线性内插
        # 这里为了演示清晰使用逻辑掩码，实际大规模部署建议用 interp 函数
        new_s = torch.zeros_like(s)
        for i in range(len(x_nodes) - 1):
            mask = (s >= x_nodes[i]) & (s < x_nodes[i+1])
            if mask.any():
                # 线性插值公式: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                slope = (y_nodes[i+1] - y_nodes[i]) / (x_nodes[i+1] - x_nodes[i])
                new_s[mask] = y_nodes[i] + slope * (s[mask] - x_nodes[i])
        
        # 处理边界 1.0
        new_s[s >= x_nodes[-1]] = y_nodes[-1]
        calibrated_scores[:, cid] = new_s
        
    return calibrated_scores


