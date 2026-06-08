"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou, get_pairwise_iou
from .dfine_utils import bbox2distance


@register()
class DFINECriterion(nn.Module):
    """This class computes the loss for D-FINE."""

    __share__ = [
        "num_classes",
    ]
    __inject__ = [
        "matcher",
    ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        match_number=1,
        tau =0.33,
        aux_alpha = 0.25,
        label_gamma = 2,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        
        #中间层参数
        self.match_number = match_number
        self.tau = tau
        self.aux_alpha = aux_alpha
        self.label_gamma = label_gamma
        self.weight_table = torch.exp(-torch.arange(match_number, dtype=torch.float32) / tau)


    def loss_labels_focal(self, outputs, targets, indices, num_boxes, **kwargs):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, loc_weight=1,**kwargs):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]  #[B,N,C]
        prob = src_logits.sigmoid()
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # #修改类别损失，用软标签作为类别损失而不是IOU
        # pos_weights = torch.zeros_like(src_logits)
        # neg_weights =  prob ** self.label_gamma
        # pos_idx_c = idx + (target_classes_o.cpu(), )
        
        # with torch.no_grad():
        #     t = prob[pos_idx_c].detach()**self.aux_alpha * ious ** (1-self.aux_alpha)
        #     t = torch.clamp(t, 0.01).detach()
        #     t = t * loc_weight
        # pos_weights[pos_idx_c] = t.to(pos_weights.dtype) 
        # neg_weights[pos_idx_c] = (1 -t.to(pos_weights.dtype))
        # loss = -pos_weights * prob.log() - neg_weights * (1-prob).log()
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        # return {"loss_vfl": loss}

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype) * loc_weight   #新加loc_weight
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        # # loss_label_mal
        # target_score = target_score.pow(self.gamma)
        # if self.mal_alpha != None:
        #     weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        # else:
        #     weight = pred_score.pow(self.gamma) * (1 - target) + target
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None, loc_weight=1, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]   #[M,4]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        # losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        losses['loss_bbox'] = (loc_weight* loss_bbox.sum(dim=-1)).sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        # losses["loss_giou"] = loss_giou.sum() / num_boxes
        losses['loss_giou'] = (loc_weight * loss_giou).sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5, **kwargs):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        ref_outputs = kwargs.get("ref_outputs", None)
        cfg = kwargs.get("cfg", None)

        losses = {}
        
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))  #[N, 4 × (reg_max + 1)] -> [N × 4, reg_max + 1]
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and "is_dn" in outputs:
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )

            ious = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]), box_cxcywh_to_xyxy(target_boxes)
                )[0]
            )

            # if cfg.grpo_finetune is False:
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )
            # else:
            #     losses["loss_grpo"] = self.grpo_loss(outputs, targets, indices, ref_outputs, cfg.grpo_advantage_weight, cfg.grpo_beta, cfg.grpo_num_mute, cfg.epsilon, cfg.iou_pro_reward)

            if "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                if torch.equal(pred_corners, target_corners):
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype
                    )
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    if "is_dn" not in outputs:
                        batch_scale = (
                            8 / outputs["pred_boxes"].shape[0]
                        )  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses

    def grpo_loss(self, outputs, targets, indices,  ref_outputs, grpo_advantage_weight=10e-3,  grpo_beta=0.04,  grpo_num_mute=300,  epsilon=1e-9, iou_pro_reward=False):
        device = outputs["pred_logits"].device
        losses = []

        # -------- collect layers --------
        cur_layers = [(outputs["pred_logits"], outputs["pred_boxes"])]
        ref_layers = [(ref_outputs["pred_logits"], ref_outputs["pred_boxes"])]

        # --------  decoder layers --------
        for (pred_logits, pred_boxes), (ref_logits, _) in zip(cur_layers, ref_layers):

            layer_losses = []

            # -------- per image (episode) --------
            for b, (src_idx, tgt_idx) in enumerate(indices):
                if src_idx.numel() == 0:
                    continue

                logits_b = pred_logits[b, src_idx]
                boxes_b  = pred_boxes[b, src_idx]
                ref_logits_b = ref_logits[b, src_idx]

                tgt_boxes  = targets[b]["boxes"][tgt_idx]
                tgt_labels = targets[b]["labels"][tgt_idx]

                # -------- advantage = IoU --------
                with torch.no_grad():
                    if iou_pro_reward is False:
                        advantage = torch.diag(
                            box_iou(
                                box_cxcywh_to_xyxy(boxes_b),
                                box_cxcywh_to_xyxy(tgt_boxes)
                            )[0]
                        )
                    else:
                        advantage = self.compute_riou_reward_matched(
                            boxes_b,
                            logits_b,
                            tgt_boxes,
                            tgt_labels,
                            src_idx,
                            tgt_idx,
                            logits_b.shape[0],
                        )

                # -------- per-image normalization --------
                if advantage.numel() > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + epsilon)
                else:
                    advantage = torch.zeros_like(advantage)

                if grpo_num_mute > 0 and advantage.numel() > grpo_num_mute:
                    advantage[-grpo_num_mute:] = 0.0

                # -------- log π --------
                logp = F.log_softmax(logits_b, dim=-1)
                logp = logp.gather(1, tgt_labels[:, None]).squeeze(1)

                ref_logp = F.log_softmax(ref_logits_b, dim=-1)
                ref_logp = ref_logp.gather(1, tgt_labels[:, None]).squeeze(1)

                # -------- KL --------
                p = logp.exp()
                pref = ref_logp.exp()
                kl = (pref + epsilon) / (p + epsilon) \
                    - torch.log(pref + epsilon) \
                    + torch.log(p + epsilon) - 1

                # -------- GRPO loss (per image) --------
                loss_b = (-advantage * grpo_advantage_weight + grpo_beta * kl).mean()
                layer_losses.append(loss_b)

                if not torch.isfinite(loss_b):
                    print("❌ NaN in loss_b")
                    print("advantage:", advantage)
                    print("logp:", logp)
                    print("ref_logp:", ref_logp)
                    print("kl:", kl)
                    print("logits_b min/max:", logits_b.min(), logits_b.max())
                    print("ref_logits_b min/max:", ref_logits_b.min(), ref_logits_b.max())
                    raise RuntimeError("NaN detected in GRPO loss")

            if len(layer_losses) == 0:
                losses.append(torch.tensor(0.0, device=device))
            else:
                losses.append(torch.stack(layer_losses).mean())

        return sum(losses) / len(losses)

    
    def grpo_loss_v1(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        Reward = -Classification_Loss
        作用：一组内，预测类别更好的的采样框类别分数会输出更高，然后让最后的置信度更高，最后选择预测框的时候先根据置信度排序，就会优先选择类被预测更好的框
        """
        final_losses = {}
        if "pred_grpo_logits" in outputs and outputs["pred_grpo_logits"] is not None and outputs["pred_grpo_logits"].numel() != 0:
            ref_cls_outputs = kwargs.get("ref_cls_outputs", None)
            grpo_advantage_weight=0.2
            grpo_beta=0.03
            epsilon=1e-9

            logits_g = outputs["pred_grpo_logits"]  # [B, N, G, C]
            B, N, G, C = logits_g.shape
            device = logits_g.device

            # 提取所有正样本索引 [Total_M]  ,Total_M 是整个 Batch 中所有图片正样本的总和
            idx = self._get_src_permutation_idx(indices)   
            
            src_boxes = outputs["dec_out_grpo_bboxes"][idx]    # [B, N, G, 4] (cxcywh) ->[Total_M, G, 4]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            src_xyxy = box_cxcywh_to_xyxy(src_boxes)            # [Total_M, G, 4]
            tgt_xyxy = box_cxcywh_to_xyxy(target_boxes).unsqueeze(1) # [Total_M, 1, 4]
            curr_ious = get_pairwise_iou(src_xyxy.detach(), tgt_xyxy.detach()) # 直接得到 [Total_M, G]
                  
            # 准备对应的标签 [Total_M] -> 扩展为 [Total_M, G]
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            tgt_labels = target_classes_o.view(-1, 1).expand(-1, G)

            curr_logits = logits_g[idx].float()   # 提取匹配到的 Logits [Total_M, G, C]

            # 计算 Reward 
            # 计算全 Batch 展平后的 CrossEntropy
            individual_loss = F.cross_entropy(
                curr_logits.reshape(-1, C), 
                tgt_labels.reshape(-1), 
                reduction='none'
            ).view(-1, G)
            
            rewards = -individual_loss + 2.0 * curr_ious  # 加上 IoU 项作为拉动力
            
            # 组内标准化 (Advantage)
            mean_r = rewards.mean(dim=1, keepdim=True)
            std_r = rewards.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
            advantages = (rewards - mean_r) / (std_r + epsilon) 

            logp = F.log_softmax(curr_logits, dim=-1)
            logp = logp.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
            if ref_cls_outputs is not None:
                ref_logits_g = ref_cls_outputs.view(B, -1, G, ref_cls_outputs.shape[-1])
                ref_logits = ref_logits_g[idx]
                ref_logp = F.log_softmax(ref_logits, dim=-1)
                ref_logp = ref_logp.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]

                p = logp.exp()
                pref = ref_logp.exp()
                kl = (pref + epsilon) / (p + epsilon) - torch.log(pref + epsilon) + torch.log(p + epsilon) - 1
            else:
                kl = torch.zeros_like(logp)

            # [Total_M, G] 每个点的 policy loss
            element_loss = -grpo_advantage_weight * advantages + grpo_beta * kl 

            # 将 element_loss 填回一个空白的 [B, N, G] 张量, 执行 .mean(1) (即在 Query 维度平均)
            full_loss_tensor = torch.zeros((B, N, G),dtype=curr_logits.dtype,  device=device)
            full_loss_tensor[idx] = element_loss
            
            # 聚合计算：
            # full_loss_tensor.mean(2) -> 先对组内 G 取平均 [B, N]
            # .mean(1)                -> 再对 Query 维度取平均 [B]
            # .sum()                  -> 对 Batch 求和
            # * N                     -> 恢复 Query 量级
            # / num_boxes             -> 最终按目标数归一化
            loss_rl = (full_loss_tensor.mean(dim=2).mean(dim=1).sum() * N) / num_boxes 
            final_losses["loss_rl"] = loss_rl
            del curr_logits, logp, advantages, rewards
            if 'ref_logp' in locals(): del ref_logp

        return final_losses


    def compute_riou_reward_matched(
        self,
        pred_boxes,    # (Nq, 4) cxcywh
        pred_logits,   # (Nq, C)
        gt_boxes,      # (K, 4)
        gt_labels,     # (K,)
        src_idx,       # (K,)
        tgt_idx,       # (K,)
        num_queries,   # Nq
        eps=1e-6
    ):
        device = pred_boxes.device
        K = src_idx.numel()
        Ng = gt_labels.numel()

        if K == 0 or Ng == 0:
            return torch.zeros((), device=device)

        # ---- IoU per matched pair ----
        iou = torch.diag(
            box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(gt_boxes)
            )[0]
        )  # (K,)

        # ---- classification correctness ----
        pred_labels = pred_logits.argmax(dim=-1)
        label_match = (pred_labels == gt_labels).float()

        r_k = iou * label_match  # (K,)

        # ---- precision / recall ----
        recall = r_k.sum() / (Ng + eps)
        precision = r_k.sum() / (num_queries + eps)

        rIoU = 2 * precision * recall / (precision + recall + eps)

        return rIoU
        

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "local": self.loss_local,
            # "rl": self.grpo_loss_v1,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)["indices"]
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for i, aux_outputs in enumerate(outputs["aux_outputs"] + [outputs["pre_outputs"]]):
                indices_aux = self.matcher(aux_outputs, targets)["indices"]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                indices_enc = self.matcher(aux_outputs, targets)["indices"]
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert "aux_outputs" in outputs, ""

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        ref_outputs = kwargs.get("ref_outputs", None)
        cfg = kwargs.get("cfg", None)
        ref_cls_outputs = kwargs.get("ref_cls_outputs", None)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            indices_in = indices_go if loss in ["boxes", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, ref_outputs=ref_outputs, cfg=cfg, ref_cls_outputs=ref_cls_outputs, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                if self.match_number > 1:
                    src_logits = outputs['pred_logits']   #置信度
                    pred_boxes = outputs['pred_boxes']
                    m_aux_indices = self.matcher(aux_outputs, targets, return_topk = self.match_number)["indices_o2m"]
                    target_boxes = torch.cat([t["boxes"][v[1]] for t,v in zip(targets, m_aux_indices)], dim=0)
                    target_classes_o = torch.cat([t["labels"][v[1]] for t,v in zip(targets, m_aux_indices)])
                    
                    pos_idx = self._get_src_permutation_idx(m_aux_indices)
                    pos_idx_c = pos_idx + (target_classes_o.cpu(), )
                    src_boxes = pred_boxes[pos_idx]
                    
                    prob = src_logits.sigmoid()
                    iou_scores = torch.diag(box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))[0])
                    t = prob[pos_idx_c]**self.aux_alpha * iou_scores ** (1-self.aux_alpha)
                    t = torch.clamp(t, 0.01).detach()
                    rank = get_local_rank(t, m_aux_indices)
                    weight_table = self.weight_table.to(rank.device)
                    weight = weight_table[rank]
                    
                for loss in self.losses:
                    if self.match_number == 1:
                        indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[i]
                        num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                        meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    else:
                        indices_in = (
                            m_aux_indices if loss in ["boxes", "vfl"] else
                            cached_indices[i] if loss == "lr" else
                            indices_go
                        )

                        num_boxes_in = (
                            num_boxes * self.match_number if loss in ["boxes", "vfl"] else
                            num_boxes if loss == "lr" else
                            num_boxes_go
                        )
                        weight = weight if loss in ["boxes", "vfl"] else 1
                        meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in, weight)

                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_in, num_boxes_in, ref_outputs=ref_outputs, cfg=cfg, **meta
                    )

                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer.
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in,  ref_outputs=ref_outputs, cfg=cfg, **meta)

                l_dict = {
                    k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                }
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                enc_losses = self.losses.copy()
                enc_losses.append('vfl')
                for loss in enc_losses:
                    indices_in = indices_go if loss == "boxes" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "boxes" else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices_in, num_boxes_in, ref_outputs=ref_outputs, cfg=cfg, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }

                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses. For dfine
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, ref_outputs=ref_outputs, cfg=cfg, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer.
            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, ref_outputs=ref_outputs, cfg=cfg, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices, weight=1):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == "iou":
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == "giou":
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
                )
            )
        else:
            raise AttributeError()

        if loss in ("boxes",):
            meta = {"boxes_weight": iou, "loc_weight": weight}
        elif loss in ("vfl",):
            meta = {"values": iou, "loc_weight": weight}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1)
        opt_list = [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list


def get_local_rank( quality, indices):
    #quality: one-dimension tensor 
    #indices: matching result
    bs = len(indices)
    device = quality.device
    tgt_size = [len(tgt_ind) for _,tgt_ind in indices]
    ind_start = 0
    rank_list = []
    for i in range(bs):
        if  tgt_size[i] == 0:
            rank_list.append(torch.zeros(0,dtype=torch.long,device=device))
            continue     
        num_tgt = max(indices[i][1]) + 1
        # split quality of one item
        quality_per_img = quality[ind_start:ind_start+tgt_size[i]]
        ind_start += tgt_size[i]
        #suppose candidate bag sizes are equal        
        k = torch.div(tgt_size[i], num_tgt,rounding_mode='floor')
        #sort quality in each candidate bag
        quality_per_img = quality_per_img.reshape(num_tgt, k)
        ind = quality_per_img.sort(dim=-1,descending=True)[1]
        #scatter ranks, eg:[0.3,0.6,0.5] -> [2,0,1]
        rank_per_img = torch.zeros_like(quality_per_img, dtype=torch.long, device = device)
        rank_per_img.scatter_(-1, ind, torch.arange(k,device=device, dtype=torch.long).repeat(num_tgt,1))
        rank_list.append(rank_per_img.flatten())

    return torch.cat(rank_list, 0)