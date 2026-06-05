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
        # logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        # vpe_logits = outputs["vpe_logits"]
        logits, boxes = outputs["vpe_logits"], outputs["pred_boxes"]

        # original_scores, _ = torch.sigmoid(logits).max(dim=-1) # [B, 300]
        # replacement_mask = original_scores > 0.65 # [B, 300]
        # logits = torch.where(replacement_mask.unsqueeze(-1), vpe_logits, logits)
        
        quality_scores_raw = outputs["quality_score"]

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        # fusion_weights = torch.tensor([
        #     0.2,  # 0: normal (原始更好)
        #     0.3,  # 1: ascus
        #     0.7,  # 2: asch (弱类别，提高)
        #     0.7,  # 3: lsil (弱类别，提高)
        #     0.3,  # 4: hsil
        #     0.3,  # 5: agc
        #     0.7,  # 6: vaginalis (弱类别)
        #     0.8,  # 7: monilia (样本少，最高)
        #     0.3,  # 8: dys
        #     0.3,  # 9: ec
        # ]).to(boxes.device)
        if self.use_focal_loss:
            scores = torch.sigmoid(logits)

            # scores = (1-fusion_weights) * scores + fusion_weights*torch.sigmoid(outputs["pred_logits"])
            # scores = 0.25 * torch.sigmoid(vpe_logits) + 0.75*torch.sigmoid(outputs["pred_logits"])
        else:
            scores = F.softmax(logits, dim=-1)

        # ========== 在 TopK 之前进行 NMS =============
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
        # ============ NMS 结束 =========================

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

