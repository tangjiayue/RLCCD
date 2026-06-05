import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, ModuleList
import torchvision.ops as ops
from torchvision.ops import roi_align
from torchvision.ops import batched_nms
import math
import torch.fft

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .dfine_utils import distance2bbox, weighting_function
from .utils import get_activation
from .vpe import MMTREX

__all__ = [
    "VisualClassifier",
    ]

#做原本dfine置信度过滤
CLASS_THRESHOLDS = {
    0: 0.1,  
    1: 0.1,  
    2: 0.1,  
    3: 0.1,  
    4: 0.1,  
    5: 0.1,  
    6: 0.1,  
    7: 0.1,  
    8: 0., 
    9: 0.1,  
}


@register()
class VisualClassifier(nn.Module):
    __inject__ = [
        "matcher",
    ]

    def __init__(self, matcher, weight_dict, num_classes=10, hidden_dim=256, alpha=0.2, gamma=2.0, mal_alpha=None, reg_max=32, reg_scale=4,
            # class_freq=[38103, 26593, 14678, 8899, 11341, 11568, 10995, 2782, 10968, 10111]
            ):
        super().__init__()
        self.weight_dict = weight_dict
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.mal_alpha = mal_alpha
        self.reg_max = reg_max
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([float(reg_scale)]), requires_grad=False)
        self.matcher = matcher

        self.vpe = VisualPromptEncoder(hidden_dim=hidden_dim, depth=1, return_intermediate=True) 
        self.cls_head = ClassificationHead(hidden_dim, num_classes)
        self.cls_head_aux = ClassificationHead(hidden_dim, num_classes)
        
        text_feats_path="/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/pubmed_text10_name_feats.pt"
        text_feats = torch.load(text_feats_path, map_location='cpu')["text_feats"]
        text_dim = text_feats.shape[1]

        self.text_adapter = TextAdapter(text_dim=text_dim, img_dim=hidden_dim, num_layers=1)
        self.register_buffer("class_text_feats", text_feats)
        # 可学习的 Logit Scale (初始化为 1/0.1 ≈ 10，即 ln(10) ≈ 2.3)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.3026) 

    def forward(self, feats, outputs, targets=None):
        #========开始
        pred_boxes = outputs['pred_boxes'] # [B, 300, 4] (cxcywh)
        quality_scores = outputs['quality_score'] # [B, 300, 1]
        device = pred_boxes.device
        B, N, _ = pred_boxes.shape
        
        # 提取特征并分类
        vpe_output = self.vpe(reference_boxes=pred_boxes, multi_scale_feats=feats)  # [B,N,C] 
        if not isinstance(vpe_output, list):
            vpe_output = [vpe_output]
        
        all_layer_vpe_feats = vpe_output  # List of [B, N, hidden_dim]
        num_layers = len(all_layer_vpe_feats)
        
        # 计算每一层的 logits（使用不同的分类头）
        all_layer_logits = []
        for layer_idx, layer_feats in enumerate(all_layer_vpe_feats):
            is_last_layer = (layer_idx == num_layers - 1)
            
            if is_last_layer:
                # 最后一层使用主分类头
                layer_logits = self.cls_head(layer_feats)
                pred_vpe_feats = layer_feats
                pred_vpe_logits = layer_logits
            else:
                # 中间层使用辅助分类头
                layer_logits = self.cls_head_aux(layer_feats)
            
            all_layer_logits.append(layer_logits)
       
        # # 融合质量分数作为最终匹配/预测的依据
        refined_logits = pred_vpe_logits + quality_scores

        if self.training:
            # #===========================匈牙利匹配===========================
            # outputs_for_matcher = {
            #     'pred_logits': refined_logits, # 使用 VPE 修正后的分数
            #     'pred_boxes': pred_boxes
            # }
            # indices = self.matcher(outputs_for_matcher, targets)["indices"]
            
            #=====================iou匹配===========================
            # iou一对多匹配版本
            indices, neg_indices = rcnn_iou_match(
                pred_boxes=pred_boxes,
                targets=targets,
                pos_threshold=0.6,   
            )
            # #iou一对一匹配版本
            # indices, neg_indices = rcnn_iou_match_one_to_one(
            #     pred_boxes=pred_boxes,
            #     targets=targets,
            #     pos_threshold=0.6,   
            # )

            #=======================构造正样本的 Labels 和 IoUs===================
            flat_boxes_cxcywh = pred_boxes.reshape(-1, 4)
            flat_pred_boxes_xyxy = box_cxcywh_to_xyxy(flat_boxes_cxcywh)
            
            flatten_labels = torch.full((B, N), -1, dtype=torch.long, device=device)
            flatten_ious = torch.zeros((B, N), device=device)
            for i in range(B):
                src_idx, tgt_idx = indices[i]
                if len(src_idx) > 0:
                    flatten_labels[i, src_idx] = targets[i]["labels"][tgt_idx]
                    p_boxes = flat_pred_boxes_xyxy.view(B, N, 4)[i, src_idx]
                    t_boxes = box_cxcywh_to_xyxy(targets[i]["boxes"][tgt_idx])
                    ious = torch.diag(box_iou(p_boxes, t_boxes)[0])
                    flatten_ious[i, src_idx] = ious
            flatten_labels = flatten_labels.reshape(-1)
            flatten_ious = flatten_ious.reshape(-1)
            flat_pred_quality = quality_scores.reshape(-1, 1)
            flatten_vpe_logits = pred_vpe_logits.reshape(-1, pred_vpe_logits.shape[-1])

            # 存储中间层的 matched 数据
            layer_matched_aux = []
            for layer_idx, layer_logits in enumerate(all_layer_logits):
                is_last_layer = (layer_idx == num_layers - 1)
                if is_last_layer:
                    continue  # 最后一层的 loss 在主分类头里计算，不参与辅助分类头的损失计算

                indices_aux, neg_indices_aux = rcnn_iou_match(
                    pred_boxes=pred_boxes,
                    targets=targets,
                    pos_threshold=0.6,   
                )
                flatten_labels_aux = torch.full((B, N), -1, dtype=torch.long, device=device)
                flatten_ious_aux = torch.zeros((B, N), device=device)
                for i in range(B):
                    src_idx_aux, tgt_idx_aux = indices_aux[i]
                    if len(src_idx_aux) > 0:
                        flatten_labels_aux[i, src_idx_aux] = targets[i]["labels"][tgt_idx_aux]
                        p_boxes = box_cxcywh_to_xyxy(pred_boxes[i, src_idx_aux])
                        t_boxes = box_cxcywh_to_xyxy(targets[i]["boxes"][tgt_idx_aux])
                        ious = torch.diag(box_iou(p_boxes, t_boxes)[0])
                        flatten_ious_aux[i, src_idx_aux] = ious      
                flatten_labels_aux = flatten_labels_aux.reshape(-1)
                flatten_ious_aux = flatten_ious_aux.reshape(-1)
                flat_pred_quality_aux = quality_scores.reshape(-1, 1)
                flatten_vpe_logits_aux = layer_logits.reshape(-1, layer_logits.shape[-1])

                layer_result = {
                    "matched_vpe_logits_aux": flatten_vpe_logits_aux,
                    "matched_labels_aux": flatten_labels_aux,
                    "matched_ious_aux": flatten_ious_aux,
                    "matched_quality_aux": flat_pred_quality_aux,
                    "layer_idx": layer_idx,
                }
                layer_matched_aux.append(layer_result)

            # --- 获取匹配上的正样本特征 ---
            flatten_feats = pred_vpe_feats.reshape(B*N, -1)  # [BN,C] 
            pos_mask = (flatten_labels != -1).reshape(-1)

            pred_pos_feats = flatten_feats[pos_mask]
            pred_pos_labels = flatten_labels[pos_mask]       # [Num_Pred_Pos]

            # --------------------------------------------------
            # 在构造 pred_pos_feats 时顺便保存对应 GT index
            pred_pos_gt_indices = []
            offset = 0

            for i in range(B):
                src_idx, tgt_idx = indices[i]
                if len(src_idx) > 0:
                    pred_pos_gt_indices.append(tgt_idx + offset)
                offset += len(targets[i]["boxes"])

            if len(pred_pos_gt_indices) > 0:
                pred_pos_gt_indices = torch.cat(pred_pos_gt_indices)
            else:
                pred_pos_gt_indices = torch.empty(0, dtype=torch.long, device=device)

            # ================= [添加到全局收集器] =================
            # 根据特征获取当前的预测类别 (用于画混淆矩阵)
            pred_pos_preds = (flatten_vpe_logits+flat_pred_quality)[pos_mask].detach().argmax(dim=-1)
            from .plot_distribution import epoch_visualizer
            if pred_pos_feats.shape[0] > 0:
                epoch_visualizer.update(
                    feats=pred_pos_feats.detach(), 
                    true_labels=pred_pos_labels.detach(), 
                    pred_labels=pred_pos_preds,
                    all_logits=(flatten_vpe_logits+flat_pred_quality)[pos_mask].detach()
                )
            # ==============================================================
            
            
            # --- 获取 GT 框的特征 ---
            gt_boxes_list = [t["boxes"] for t in targets] # list of [Ni, 4]
            
            # 构造 Batch 化 GT 
            max_gts = max([len(b) for b in gt_boxes_list])
            if max_gts > 0:
                gt_boxes_padded = torch.zeros((B, max_gts, 4), device=device)
                gt_labels_padded = torch.full((B, max_gts), -1, dtype=torch.long, device=device)
                gt_mask = torch.zeros((B, max_gts), dtype=torch.bool, device=device)

                for i, boxes in enumerate(gt_boxes_list):
                    n_gt = len(boxes)
                    if n_gt > 0:
                        gt_boxes_padded[i, :n_gt] = boxes
                        gt_labels_padded[i, :n_gt] = targets[i]["labels"]
                        gt_mask[i, :n_gt] = True
                
                # 提取 GT 特征
                gt_feats= self.vpe(reference_boxes=gt_boxes_padded, multi_scale_feats=feats) # [B, Max_GT, C]
                
                if isinstance(gt_feats, list):
                    gt_feats = gt_feats[-1]  # 只取最后一层

                #只取有效的 GT 特征
                valid_gt_feats = gt_feats[gt_mask]  # [Total_GT, C]
                valid_gt_labels = gt_labels_padded[gt_mask] # [Total_GT]

                # --- 合并两者 ---
                # all_contrast_feats = torch.cat([pred_pos_feats, valid_gt_feats], dim=0) # [Total_Samples, C]
                # all_contrast_labels = torch.cat([pred_pos_labels, valid_gt_labels], dim=0) # [Total_Samples]
                all_contrast_feats = valid_gt_feats # [Total_GT, C]
                all_contrast_labels = valid_gt_labels # [Total_GT]
            else:
                # 如果没有 GT (极端情况)，只用 pred
                all_contrast_feats = pred_pos_feats
                all_contrast_labels = pred_pos_labels

        
            # --- 计算对比损失所需的 Logits ---
            matched_sim_logits = None
            if all_contrast_feats.shape[0] > 0:
                v_proj = F.normalize(all_contrast_feats, dim=-1)
                adapted_text = self.text_adapter(self.class_text_feats)
                t_norm = F.normalize(adapted_text, dim=-1)

                # 使用可学习温度，并进行钳位防止溢出
                logit_scale = self.logit_scale.exp().clamp(max=100.0)
                # 计算相似度矩阵 [Total_Samples, Num_Classes]
                matched_sim_logits = logit_scale * torch.matmul(v_proj, t_norm.t())
            
            #把对比学习的数据单独存一个字段
            adapted_text = self.text_adapter(self.class_text_feats)  # [num_cls, C]
            t_norm = F.normalize(adapted_text, dim=-1)
            contrast_data = {
                "contrast_feats": all_contrast_feats,     # [Total_Samples, C]
                "contrast_labels": all_contrast_labels,   # [Total_Samples]
                "class_prototypes": self.cls_head.head.weight,
                

                "pred_pos_feats": pred_pos_feats,
                "gt_feats": valid_gt_feats,
                "gt_labels": valid_gt_labels,
                "pred_pos_gt_indices": pred_pos_gt_indices,
                "text_feats": t_norm,
                "contrast_logits": matched_sim_logits,  
            }

           
            # grpo_data = self.sample_grpo_features1(feats, outputs, indices, targets, G=14)
            
            outputs['pred_logits'] = refined_logits
            return {
                "matched_vpe_logits": flatten_vpe_logits,
                "matched_labels": flatten_labels,
                "matched_ious": flatten_ious,  #[BN]
                "matched_quality": flat_pred_quality,

                "layer_matched_aux": layer_matched_aux,

                **contrast_data,
                "num_boxes": self._get_num_boxes(targets, device),

                # **grpo_data,
                "outputs":outputs,  #联调用
            }
        else:
            # 推理模式：直接使用前面算好的 refined_logits
            # outputs['pred_logits'] = refined_logits
            outputs['vpe_logits'] = refined_logits        

            # ================= [推理阶段特征与混淆矩阵收集] =================
            if targets is not None:
                try:
                    fake_targets = []
                    for t in targets:
                        boxes_xyxy_norm = t["boxes"] / 640.                     
                        boxes_cxcywh_norm = box_xyxy_to_cxcywh(boxes_xyxy_norm)
                        fake_targets.append({"labels": t["labels"], "boxes": boxes_cxcywh_norm})
       
                    # outputs_for_matcher = {
                    #     'pred_logits': refined_logits, # 使用 VPE 修正后的分数
                    #     'pred_boxes': pred_boxes
                    # }
                    # indices = self.matcher(outputs_for_matcher, targets)["indices"]             
                    indices = rcnn_iou_match(
                        pred_boxes=pred_boxes,
                        targets=fake_targets,
                        pos_threshold=0.6,   
                    )[0]
                    
                    flatten_labels = torch.full((B, N), -1, dtype=torch.long, device=device)
                    for i in range(B):
                        src_idx, tgt_idx = indices[i]
                        if len(src_idx) > 0:
                            flatten_labels[i, src_idx] = targets[i]["labels"][tgt_idx]
                    
                    flatten_labels = flatten_labels.reshape(-1)
                    flatten_vpe_logits = refined_logits.reshape(-1, self.num_classes)
                    flatten_feats = pred_vpe_feats.reshape(B*N, -1)
                    
                    pos_mask = (flatten_labels != -1).reshape(-1)
                    if pos_mask.any():
                        pred_pos_feats = flatten_feats[pos_mask]
                        pred_pos_labels = flatten_labels[pos_mask]
                        pred_pos_preds = flatten_vpe_logits[pos_mask].detach().argmax(dim=-1)
                        
                        from .plot_distribution import epoch_visualizer
                        epoch_visualizer.update(
                            feats=pred_pos_feats, 
                            true_labels=pred_pos_labels, 
                            pred_labels=pred_pos_preds,
                            all_logits=flatten_vpe_logits[pos_mask]
                        )
                except Exception as e:
                    pass
                
            return outputs
        

    def _get_num_boxes(self, targets, device):
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        return torch.clamp(num_boxes / get_world_size(), min=1).item()

    def get_losses(self, outputs, ref_cls_outputs=None, old_cls_outputs=None):
        num_boxes = outputs["num_boxes"]
        losses = {}
        #================vfl损失========================
        # 主分支
        losses_main = self.loss_labels_matched_branch(outputs, num_boxes, branch="main")
        losses.update(losses_main)
        
        if len(outputs["layer_matched_aux"]) > 0:
            losses_aux = self.loss_labels_matched_branch(outputs, num_boxes, branch="aux")
            losses.update(losses_aux)
            
        
        #=================Contrastive Loss====================
        losses.update(self.loss_contrast_matched(outputs, num_boxes))     #all_contrast_feats只有gt

        #==================GRPO Loss======================
        if "grpo_logits" in outputs:
            grpo_loss_dict = self.grpo_loss_v4(outputs, num_boxes, ref_grpo_logits=ref_cls_outputs, old_cls_outputs=old_cls_outputs)
            losses.update(grpo_loss_dict)

        losses = {
            k: losses[k] * self.weight_dict[k] for k in losses if k in self.weight_dict
        }
        losses = {k + "_vpe_cls": v for k, v in losses.items()}

        # # ================== 探测梯度贡献大小 ==================
        # if self.training:
        #     try:
        #         # 这里用分类头的主权重作为观察对象
        #         p = self.cls_head.head.weight

        #         loss_cls_raw = losses.get("loss_cls_vpe_cls")
        #         loss_rl_raw = losses.get("loss_rl_vpe_cls")
        #         loss_contrast_raw = losses.get("loss_contrast_vpe_cls")

        #         def _grad_norm(loss_tensor):
        #             if loss_tensor is None or (not torch.is_tensor(loss_tensor)) or (not loss_tensor.requires_grad):
        #                 return 0.0
        #             g = torch.autograd.grad(
        #                 loss_tensor,
        #                 p,
        #                 retain_graph=True,
        #                 allow_unused=True,
        #                 create_graph=False,
        #             )[0]
        #             if g is None:
        #                 return 0.0
        #             # 用 float 计算更稳，避免 AMP 下溢/上溢影响统计
        #             return float(g.detach().float().norm(p=2).cpu())

        #         g_cls = _grad_norm(loss_cls_raw)
        #         g_rl = _grad_norm(loss_rl_raw)
        #         g_ct = _grad_norm(loss_contrast_raw)

        #         denom = g_cls + g_rl + g_ct + 1e-12
        #         r_cls = g_cls / denom
        #         r_rl = g_rl / denom
        #         r_ct = g_ct / denom

        #         # 限频打印：每 200 step 打一次（用 buffer 计数，不影响 DDP）
        #         if not hasattr(self, "_grad_dbg_step"):
        #             self._grad_dbg_step = 0
        #         self._grad_dbg_step += 1

        #         if self._grad_dbg_step % 1 == 0:
        #             # 只在 rank0 打印，避免多卡刷屏
        #             if not is_dist_available_and_initialized() or torch.distributed.get_rank() == 0:
        #                 print(
        #                     f"[GRAD RATIO][cls_head.weight] "
        #                     f"||g_cls||={g_cls:.4e} ({r_cls:.2%}) | "
        #                     f"||g_rl||={g_rl:.4e} ({r_rl:.2%}) | "
        #                     f"||g_ct||={g_ct:.4e} ({r_ct:.2%})"
        #                 )
        #     except Exception as e:
        #         if not is_dist_available_and_initialized() or torch.distributed.get_rank() == 0:
        #             print(f"[GRAD RATIO] skipped due to error: {type(e).__name__}: {str(e)}")

        return losses


    def loss_labels_matched_branch(self, outputs, num_boxes, branch="main"):
        if branch == "main":
            src_logits = outputs["matched_vpe_logits"]
            target_labels = outputs["matched_labels"]
            ious = outputs["matched_ious"].detach()
            quality = outputs["matched_quality"]
            loss_name = "loss_cls"

            if quality.dim() == 1:
                quality = quality.unsqueeze(-1)
            fused_logits = src_logits + quality

            target_classes = target_labels.clone()
            target_classes[target_classes == -1] = self.num_classes 
            target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()
            target_score = target * ious.view(-1, 1).pow(0.5)

            with torch.no_grad():
                pred_score = fused_logits.sigmoid().detach()
                weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

            # #DEIM权重
            # target_score = target_score.pow(1.5)
            # weight = pred_score.pow(1.5) * (1 - target) + target

            loss = F.binary_cross_entropy_with_logits(
                fused_logits, target_score, weight=weight, reduction="none"
            )   # [B*N,10]

            return {loss_name: loss.sum() / num_boxes}
        else:
            # 辅助分支的输入
            aux_data = outputs["layer_matched_aux"]  # 根据层索引获取对应数据
            total_aux_loss = 0.0
            loss_name = "loss_cls_aux"
            for layer_data in aux_data:
                src_logits = layer_data["matched_vpe_logits_aux"]
                target_labels = layer_data["matched_labels_aux"]
                ious = layer_data["matched_ious_aux"].detach()
                quality = layer_data["matched_quality_aux"]
                layer_idx = layer_data["layer_idx"]
                
                if quality.dim() == 1:
                    quality = quality.unsqueeze(-1)
                fused_logits = src_logits + quality

                target_classes = target_labels.clone()
                target_classes[target_classes == -1] = self.num_classes 
                target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()
                target_score = target * ious.view(-1, 1).pow(0.5)

                with torch.no_grad():
                    pred_score = fused_logits.sigmoid().detach()
                    weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score


                loss = F.binary_cross_entropy_with_logits(
                    fused_logits, target_score, weight=weight, reduction="none"
                )   # [B*N,10]
                total_aux_loss += loss
                total_aux_loss = total_aux_loss.sum() / num_boxes

            return {loss_name: total_aux_loss}


    def loss_contrast_matched2(self, outputs, num_boxes):
        """
        计算图像特征与文本特征的对比损失 (InfoNCE / CrossEntropy)
        """
        sim_logits = outputs.get("contrast_logits") # [Total_Samples, 10]
        targets = outputs.get("contrast_labels")    # [Total_Samples]

        if sim_logits is None or targets is None or sim_logits.shape[0] == 0:
            return {"loss_contrast": torch.tensor(0.0, device=self.cls_head.head.weight.device)}
       
        if torch.isnan(sim_logits).any():
            print("Warning: NaN detected in contrast logits!")
            return {"loss_contrast": torch.tensor(0.0, device=sim_logits.device)}
        # 这里已经是纯正样本了 (Pred_Pos + GT)，直接计算 CE Loss
        loss = F.cross_entropy(sim_logits, targets, reduction="sum")

        # 使用 num_boxes 归一化
        return {"loss_contrast": loss / num_boxes} 
    
    def loss_contrast_matched1(self, outputs, num_boxes):
        """
        基于原型的 CSPCL 机制
        """
        query_feats = outputs.get("contrast_feats")   # [N, C]
        query_labels = outputs.get("contrast_labels") # [N]
        prototypes = outputs.get("class_prototypes")  # [num_classes, C]

        # 修复了原来没改干净的 device 报错隐患
        if query_feats is None or query_labels is None or query_feats.shape[0] == 0:
            return {"loss_contrast": torch.tensor(0.0, device=prototypes.device)}
       
        if torch.isnan(query_feats).any():
            return {"loss_contrast": torch.tensor(0.0, device=prototypes.device)}

        # ========== 确保精度一致 ==========
        # 获取当前精度
        target_dtype = query_feats.dtype  # FP16 或 FP32
        
        # 将 prototypes 转换为相同精度
        if prototypes.dtype != target_dtype:
            prototypes = prototypes.to(target_dtype)
   
        gamma = 5e-3   
        tau = 0.3     
        alpha = 1.0   
        beta = 0.5    

        # 增加 eps 防止除 0 导致归一化 NaN
        q_norm = F.normalize(query_feats, p=2, dim=-1, eps=1e-6)  
        p_norm = F.normalize(prototypes, p=2, dim=-1, eps=1e-6)   
        
        K = prototypes.shape[0]
        N = query_feats.shape[0]

        # ====================================================================
        # 1. ITA Loss (类内截断吸引)
        # ====================================================================
        pos_prototypes = p_norm[query_labels]           
        sim_pos = (q_norm * pos_prototypes).sum(dim=-1) 
        sim_pos = torch.clamp(sim_pos, min=0.0005, max=0.9995)
        
        threshold = 1.0 - gamma
        T_x = torch.where(sim_pos > threshold, 
                          torch.full_like(sim_pos, threshold), 
                          sim_pos.clamp(min=1e-4)) 
        
        loss_ita = -torch.log(T_x).mean()

        # ====================================================================
        # 2. IAR Loss (类间自适应排斥) 绝对安全计算法
        # ====================================================================
        # 把被你删掉的映射公式加回来，统一到 [0, 1] 域
        sim_all = torch.matmul(q_norm, p_norm.t())      
        sim_all = torch.clamp(sim_all, min=0.0005, max=0.9995)
        
        sim_proto = torch.matmul(p_norm, p_norm.t())    
        sim_proto = torch.clamp(sim_proto, min=0.0005, max=0.9995)
        
        mask = ~F.one_hot(query_labels, num_classes=K).bool() # [N, K]
        
        # 提取各个 Query 对应的 GT类别 原型相似度
        proto_sim_to_gt = sim_proto[query_labels] # [N, K]
        
        # 先过滤！只把负类的元素抽出来做计算，彻底屏蔽 1.0 的情况
        valid_proto_sim = proto_sim_to_gt[mask]   # [N * (K-1)] 一维张量
        valid_sim_neg = sim_all[mask]             # [N * (K-1)] 一维张量
        
        # 计算排斥力系数，clamp max 进一步限制在 0.9，防止原型确实高度重合导致指数起飞
        # exp((1-0.3)/0.1) = exp(7) = 1096, 处于极其安全的 float32 范围内
        R_k1_k2 = torch.exp((1.0 - tau) / (1.0 - valid_proto_sim.clamp(max=0.9))) 
        
        # 直接计算一维向量的 loss
        loss_iar = -(R_k1_k2 * torch.log(1.0 - valid_sim_neg.clamp(max=0.999))).mean()

        # ====================================================================
        # 3. 合并返回
        # ====================================================================
        loss_csp = (alpha * loss_ita + beta * loss_iar)

        return {"loss_contrast": loss_csp}

    def loss_contrast_matched(self, outputs, num_boxes):
        pred_feats = outputs["pred_pos_feats"]
        gt_feats = outputs["gt_feats"]
        gt_labels = outputs["gt_labels"]
        match_idx = outputs["pred_pos_gt_indices"]
        t_norm = outputs["text_feats"]
        ious = outputs["matched_ious"]  #[BN]

        gt_text_sim_logits = outputs.get("contrast_logits") # [Total_Samples, 10]

        device = gt_feats.device
        dtype = gt_feats.dtype 

        if gt_feats.shape[0] == 0:
            return {"loss_contrast": torch.tensor(0.0, device=device, dtype=dtype)}

        # 统一 normalize
        v_gt = F.normalize(gt_feats, dim=-1)
        v_pred = F.normalize(pred_feats, dim=-1) if pred_feats is not None else None

        loss = 0.0

        # 权重
        w_inst = 1.0
        w_cls = 0.3
        w_struct = 3

        # ================= ① Pred ↔ GT =================
        if v_pred is not None and match_idx.shape[0] > 0:
            # matched_gt = v_gt[match_idx]
            # sim = (v_pred * matched_gt).sum(-1)
            # loss += w_inst * (1.0 - sim).mean()

            matched_gt = v_gt[match_idx]           # [num_pos, C]
            pos_ious = ious[match_idx]             # [num_pos]

            sim = F.cosine_similarity(v_pred, matched_gt, dim=-1)

            target = (pos_ious * 2 - 1).to(dtype)               # 映射到 [-1,1]
            weight = pos_ious.detach().to(dtype)             # IoU加权
            sim = sim.to(dtype)
            loss_inst = (weight * (sim - target.detach()) ** 2).mean()

            loss += w_inst * loss_inst
            # print(f"Instance-level Contrast Loss: {w_inst * (1.0 - sim).mean():.4f}")

        # # ================= ② GT ↔ Text =================
        loss += w_cls * F.cross_entropy(gt_text_sim_logits, gt_labels).to(dtype)
        # print(f"GT-Text Contrast Loss: {w_cls * F.cross_entropy(gt_text_sim_logits, gt_labels):.4f}")

        # ================= ③ 结构对齐 =================
        if v_gt.shape[0] > 1:
            sim_gt = torch.matmul(v_gt, v_gt.t()).float()

            text_selected = t_norm[gt_labels]
            sim_text = torch.matmul(text_selected, text_selected.t()).float()

            loss += w_struct * F.mse_loss(sim_gt, sim_text.detach())
            # print(f"Structural Alignment Loss: {w_struct * F.mse_loss(sim_gt, sim_text.detach()):.4f}")

        return {"loss_contrast": loss}


    def grpo_loss_v1(self, grpo_data, num_boxes, ref_grpo_logits=None):

        """
        Args:
            grpo_data: sample_grpo_features 返回的字典
                - grpo_logits: [Total_M, G, C]
                - grpo_ious:   [Total_M, G]
                - grpo_labels: [Total_M]
            num_boxes: 归一化因子
            ref_grpo_logits: 参考模型的 Logits, [Total_M * G, C]
        """
        final_losses = {}
        logits_g = grpo_data["grpo_logits"]
    
        if logits_g is None or logits_g.numel() == 0:
            return {"loss_rl": logits_g.sum() * 0.0} # 保持梯度流的 0 损失

        Total_M, G, C = logits_g.shape
        device = logits_g.device

        grpo_advantage_weight = 0.1
        grpo_beta = 0.03
        epsilon = 1e-9

        #准备标签 [Total_M, G]
        tgt_labels = grpo_data["grpo_labels"].view(-1, 1).expand(-1, G)
        curr_ious = grpo_data["grpo_ious"] # [Total_M, G]
        grpo_cost_bbox = grpo_data["grpo_cost_bbox"]

        # #负（类别cost+IOU）作为奖励
        # prob = torch.sigmoid(logits_g)
        # tgt_prob = prob.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
        # neg_cost_class = (
        #     (1 - 0.25) * (tgt_prob**2.0) * (-(1 - tgt_prob + 1e-8).log())
        # )
        # pos_cost_class = (
        #     0.25 * ((1 - tgt_prob) ** 2.0) * (-(tgt_prob + 1e-8).log())
        # )
        # # cost = 2*(pos_cost_class - neg_cost_class) - 2*curr_ious + 5*grpo_cost_bbox
        # cost = 2*(pos_cost_class - neg_cost_class) - 2*curr_ious
        # cost = torch.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=-1e6)
        # rewards = -cost


        # tgt_logits = logits_g.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
        # tgt_iou = curr_ious.clamp(0.0, 1.0)
        # # GT 类上的 soft BCE：IoU 越大 -> 目标越接近 1 -> 奖励越高
        # individual_loss = F.binary_cross_entropy_with_logits(
        #     tgt_logits,
        #     tgt_iou,
        #     reduction="none",
        # )  # [Total_M, G]

        # rewards = -individual_loss

        tgt_iou = curr_ious.clamp(0.0, 1.0)
        rewards = tgt_iou  # [Total_M, G]

        # 组内标准化 (Advantage) - 核心 GRPO 逻辑
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = (rewards - mean_r) / (std_r + epsilon) # [Total_M, G]
        advantages = torch.clamp(advantages, -2, 2).detach()

        #logits_g 维度应该是 [Total_M, G, C]
        tgt_logits = logits_g.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
        logp = F.logsigmoid(tgt_logits)  # [Total_M, G]

        if ref_grpo_logits is not None:
            ref_grpo_logits = ref_grpo_logits.view(Total_M, G, C)
            ref_tgt_logits = ref_grpo_logits.detach().gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)

            p = torch.sigmoid(tgt_logits)         # current prob, [M,G]
            pref = torch.sigmoid(ref_tgt_logits)  # ref prob, [M,G]

            kl = pref * (torch.log(pref + epsilon) - torch.log(p + epsilon)) + \
                 (1.0 - pref) * (torch.log(1.0 - pref + epsilon) - torch.log(1.0 - p + epsilon))
        else:
            kl = torch.zeros_like(logp)

        # 计算 Policy Loss
        # GRPO 的核心：优势函数越大，logp 应该越大；KL 散度作为惩罚
        element_loss = -(logp * advantages.detach()) * grpo_advantage_weight + grpo_beta * kl

        # print("1",advantages)
        # print("2",kl)
        # print("3",element_loss)

        #  归一化与返回
        loss_rl = element_loss.mean()
        final_losses["loss_rl"] = loss_rl

        return final_losses 
   
    def grpo_loss_v2(self, grpo_data, num_boxes, ref_grpo_logits=None):

        """
        Args:
            grpo_data: sample_grpo_features 返回的字典
                - grpo_logits: [Total_M, G, C]
                - grpo_ious:   [Total_M, G]
                - grpo_labels: [Total_M]
            num_boxes: 归一化因子
            ref_grpo_logits: 参考模型的 Logits, [Total_M * G, C]
        """
        final_losses = {}
        logits_g = grpo_data["grpo_logits"]
    
        if logits_g is None or logits_g.numel() == 0:
            return {"loss_rl": logits_g.sum() * 0.0} # 保持梯度流的 0 损失

        Total_M, G, C = logits_g.shape
        device = logits_g.device

        grpo_advantage_weight = 0.1
        grpo_beta = 0.03
        epsilon = 1e-9

        #准备标签 [Total_M, G]
        tgt_labels = grpo_data["grpo_labels"].view(-1, 1).expand(-1, G)
        curr_ious = grpo_data["grpo_ious"] # [Total_M, G]
        grpo_cost_bbox = grpo_data["grpo_cost_bbox"]

        # #负（类别cost+IOU）作为奖励
        # prob = torch.sigmoid(logits_g)
        # tgt_prob = prob.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
        # neg_cost_class = (
        #     (1 - 0.25) * (tgt_prob**2.0) * (-(1 - tgt_prob + 1e-8).log())
        # )
        # pos_cost_class = (
        #     0.25 * ((1 - tgt_prob) ** 2.0) * (-(tgt_prob + 1e-8).log())
        # )
        # # cost = 2*(pos_cost_class - neg_cost_class) - 2*curr_ious + 5*grpo_cost_bbox
        # cost = 2*(pos_cost_class - neg_cost_class) - 2*curr_ious
        # cost = torch.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=-1e6)
        # rewards = -cost


        # tgt_logits = logits_g.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
        # tgt_iou = curr_ious.clamp(0.0, 1.0)
        # # GT 类上的 soft BCE：IoU 越大 -> 目标越接近 1 -> 奖励越高
        # individual_loss = F.binary_cross_entropy_with_logits(
        #     tgt_logits,
        #     tgt_iou,
        #     reduction="none",
        # )  # [Total_M, G]

        # rewards = -individual_loss

        tgt_iou = curr_ious.clamp(0.0, 1.0)
        rewards = tgt_iou  # [Total_M, G]

        # # 组内标准化 (Advantage) - 核心 GRPO 逻辑
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = (rewards - mean_r) / (std_r + epsilon) # [Total_M, G]
        advantages = torch.clamp(advantages, -2, 2).detach()

        bad_mask = tgt_iou < 0.4
        advantages = torch.where(bad_mask, torch.full_like(advantages, -1.0), advantages)
        advantages = torch.clamp(advantages, -2, 2).detach()

        #logits_g 维度应该是 [Total_M, G, C]
        tgt_logits = logits_g.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
        logp = F.logsigmoid(tgt_logits)  # [Total_M, G]

        if ref_grpo_logits is not None:
            ref_grpo_logits = ref_grpo_logits.view(Total_M, G, C)
            ref_tgt_logits = ref_grpo_logits.detach().gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)

            p = torch.sigmoid(tgt_logits)         # current prob, [M,G]
            pref = torch.sigmoid(ref_tgt_logits)  # ref prob, [M,G]

            kl = pref * (torch.log(pref + epsilon) - torch.log(p + epsilon)) + \
                 (1.0 - pref) * (torch.log(1.0 - pref + epsilon) - torch.log(1.0 - p + epsilon))
        else:
            kl = torch.zeros_like(logp)

        # 计算 Policy Loss
        # GRPO 的核心：优势函数越大，logp 应该越大；KL 散度作为惩罚
        element_loss = -(logp * advantages.detach()) * grpo_advantage_weight + grpo_beta * kl

        # #  归一化与返回
        # # 额外的分类一致性奖励（可选）
        # pred_prob = torch.sigmoid(logits_g).gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)
        # consistency_bonus = (pred_prob - curr_ious).abs()  # 偏差越大惩罚越大
        # consistency = consistency_bonus.mean() * 0.04
        # loss_rl = element_loss.mean() +consistency
        loss_rl = element_loss.mean()
        final_losses["loss_rl"] = loss_rl

        return final_losses 
    
    def grpo_loss_v3(self, grpo_data, num_boxes, ref_grpo_logits=None, old_cls_outputs=None):

        """
        Args:
            grpo_data: sample_grpo_features 返回的字典
                - grpo_logits: [Total_M, G, C]
                - grpo_ious:   [Total_M, G]
                - grpo_labels: [Total_M]
            num_boxes: 归一化因子
            ref_grpo_logits: 参考模型的 Logits, [Total_M * G, C]
        """
        final_losses = {}
        logits_g = grpo_data["grpo_logits"]
    
        if logits_g is None or logits_g.numel() == 0:
            return {"loss_rl": logits_g.sum() * 0.0} # 保持梯度流的 0 损失

        Total_M, G, C = logits_g.shape
        device = logits_g.device

        grpo_advantage_weight = 0.1
        grpo_beta = 0.03
        epsilon = 1e-9

        #准备标签 [Total_M, G]
        tgt_labels = grpo_data["grpo_labels"].view(-1, 1).expand(-1, G)
        curr_ious = grpo_data["grpo_ious"] # [Total_M, G]
        grpo_cost_bbox = grpo_data["grpo_cost_bbox"]

        #负（类别cost+IOU）作为奖励
        prob = torch.sigmoid(logits_g)
        tgt_prob = prob.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
        neg_cost_class = (
            (1 - 0.25) * (tgt_prob**2.0) * (-(1 - tgt_prob + 1e-8).log())
        )
        pos_cost_class = (
            0.25 * ((1 - tgt_prob) ** 2.0) * (-(tgt_prob + 1e-8).log())
        )
        # cost = 2*(pos_cost_class - neg_cost_class) - 2*curr_ious + 5*grpo_cost_bbox
        cost = 2*(pos_cost_class - neg_cost_class) - 2*curr_ious
        cost = torch.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=-1e6)
        rewards = -cost

        # tgt_iou = curr_ious.clamp(0.0, 1.0)
        # rewards = tgt_iou  # [Total_M, G]

        # 组内标准化 (Advantage) - 核心 GRPO 逻辑
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = (rewards - mean_r) / (std_r + epsilon) # [Total_M, G]
        advantages = torch.clamp(advantages, -2, 2).detach()

        #logits_g 维度应该是 [Total_M, G, C]
        tgt_logits = logits_g.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
        logp = F.logsigmoid(tgt_logits)  # [Total_M, G]
        
        # 对所有类别的概率做kl散度
        if ref_grpo_logits is not None:
            ref_grpo_logits = ref_grpo_logits.view(Total_M, G, C)
            student_prob = torch.sigmoid(logits_g)

            with torch.no_grad():
                teacher_prob = torch.sigmoid(ref_grpo_logits)

            kl = (
                teacher_prob * (torch.log(teacher_prob + epsilon)- torch.log(student_prob + epsilon))
                +
                (1.0 - teacher_prob) * (torch.log(1.0 - teacher_prob + epsilon) 
                - torch.log(1.0 - student_prob + epsilon))
            )

            # [M,G,C] -> [M,G]
            kl = kl.mean(dim=-1)
        else:
            kl = torch.zeros((Total_M, G), device=device)

        # --------------------------------------------------------
        # 只强化 positive advantage
        # 防止 GT -> -∞
        # --------------------------------------------------------
        positive_advantages = advantages.clamp(min=0.0)
        policy_loss = -(logp * positive_advantages)* grpo_advantage_weight

        confidence_penalty_weight = 0.05
        low_iou_mask = (curr_ious < 0.4).float()
        confidence_penalty = (torch.sigmoid(tgt_logits)* low_iou_mask* confidence_penalty_weight)

        # --------------------------------------------------------
        # Hard competitor suppression
        # --------------------------------------------------------
        non_tgt_mask = torch.ones_like(logits_g,dtype=torch.bool)
        non_tgt_mask.scatter_(2,tgt_labels.unsqueeze(-1),False)
        non_tgt_logits = logits_g.masked_fill(~non_tgt_mask,float("-inf")) # [Total_M, G, C]
        max_comp_logits, _ = non_tgt_logits.max(dim=2)

        # margin ranking
        margin = tgt_logits - max_comp_logits
        dynamic_margin = 0.2 + 0.5 * curr_ious

        ranking_loss = F.relu(dynamic_margin.detach() - margin) * 0.2
        
        
        # 计算 Policy Loss
        total_loss = policy_loss + kl + confidence_penalty + ranking_loss
        loss_rl = total_loss.mean()
        
        # element_loss = -(logp * advantages.detach()) * grpo_advantage_weight + grpo_beta * kl
        # loss_rl = element_loss.mean()

        final_losses["loss_rl"] = loss_rl
        return final_losses 

    def grpo_loss_v4(self, grpo_data, num_boxes, ref_grpo_logits=None, old_cls_outputs=None):
        """
        Args:
            grpo_data: sample_grpo_features 返回的字典
                - grpo_logits: [Total_M, G, C]
                - grpo_ious:   [Total_M, G]
                - grpo_labels: [Total_M] (正确类别的索引)
            num_boxes: 归一化因子
            ref_grpo_logits: 参考模型的 Logits
            old_cls_outputs: 旧策略的分类输出 [Total_M, G, C] 或 [Total_M, G]
        """
        final_losses = {}
        logits_g = grpo_data["grpo_logits"]

        if logits_g is None or logits_g.numel() == 0:
            return {"loss_rl": logits_g.sum() * 0.0}

        Total_M, G, C = logits_g.shape
        device = logits_g.device
        epsilon = 1e-9

        # 超参数配置
        grpo_advantage_weight = 0.1
        grpo_beta = 0.03
        grpo_clip_epsilon = 0.2
        
        # 排名奖励配置
        reward_top1 = 1.0   # 正确类别排第1的奖励
        reward_top2 = 0.5   # 正确类别排第2的奖励
        reward_other = -0.5 # 正确类别排其他的负奖励

        # 准备标签
        tgt_labels = grpo_data["grpo_labels"].view(-1) # [Total_M]
        curr_ious = grpo_data["grpo_ious"]  # [Total_M, G]

        # ========== 1. 重新设计 Rewards 和 Advantages ==========
        
        # 1.1 计算当前模型对各个类别的置信度 (使用 sigmoid)
        probs = torch.sigmoid(logits_g) # [Total_M, G, C]
        
        # 1.2 提取正确类别的置信度，并计算其在组内的排名
        # 获取正确类别的预测概率 [Total_M, G]
        tgt_prob = probs.gather(2, tgt_labels.view(-1, 1, 1).expand(-1, G, 1)).squeeze(-1)
        
        # 计算排名：将组内(G)所有类别的概率与正确类别的概率比较
        # 如果其他类别的概率 >= 正确类别的概率，则排名落后
        # (probs >= tgt_prob.unsqueeze(-1)) 会得到 [Total_M, G, C] 的布尔矩阵
        is_greater_or_equal = (probs >= tgt_prob.unsqueeze(-1)).float()
        # 对每个样本的每个组(G)，统计有多少个类别的概率 >= 正确类别的概率，即为排名
        ranks = is_greater_or_equal.sum(dim=2) # [Total_M, G]
        
        # 1.3 根据排名分配基础奖励
        ranking_reward = torch.where(
            ranks == 1, 
            torch.tensor(reward_top1, device=device),
            torch.where(
                ranks == 2,
                torch.tensor(reward_top2, device=device),
                torch.tensor(reward_other, device=device)
            )
        )
        
        # 1.4 结合 IoU 进行加权 (IoU作为质量系数，范围 [0.5, 1.0])
        # 这样即使排名对了，如果框不准(IoU低)，奖励也会降低
        iou_factor = 0.5 + 0.5 * curr_ious 
        rewards = ranking_reward * iou_factor

        # 1.5 组内标准化 (GRPO 核心)
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = (rewards - mean_r) / (std_r + epsilon)
        advantages = torch.clamp(advantages, -2.0, 2.0).detach()

        # ========== 2. 当前策略的 logp ==========
        # 提取正确类别的 logits
        tgt_logits = logits_g.gather(2, tgt_labels.view(-1, 1, 1).expand(-1, G, 1)).squeeze(-1)
        tgt_logits = torch.clamp(tgt_logits, -15, 15) 
        current_logp = F.logsigmoid(tgt_logits)  # [Total_M, G]

        # ========== 3. 旧策略的 logp ==========
        if old_cls_outputs is not None:
            old_cls_outputs = old_cls_outputs.view(Total_M, G, C) 
            old_tgt_logits = old_cls_outputs.detach().gather(2, tgt_labels.view(-1, 1, 1).expand(-1, G, 1)).squeeze(-1)
            old_tgt_logits = torch.clamp(old_tgt_logits, -15, 15)
            old_logp = F.logsigmoid(old_tgt_logits)
        else:
            old_logp = current_logp.detach()

        # ========== 4. GRPO Policy Loss ==========
        ratio = torch.exp(current_logp - old_logp)  # [Total_M, G]
        ratio = torch.clamp(ratio, 0.0, 3.0) 
        
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - grpo_clip_epsilon, 1 + grpo_clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2) 

        policy_loss = policy_loss.mean() * grpo_advantage_weight

        # ========== 5. KL 散度 ==========
        if old_cls_outputs is not None:
            old_tgt_logits = old_cls_outputs = old_cls_outputs.view(Total_M, G, C) 
            student_prob = torch.sigmoid(logits_g)
            with torch.no_grad():
                teacher_prob = torch.sigmoid(old_tgt_logits)
            
            # 使用二元 KL 散度
            kl = (teacher_prob * (torch.log(teacher_prob + epsilon) - torch.log(student_prob + epsilon)) +
                (1.0 - teacher_prob) * (torch.log(1.0 - teacher_prob + epsilon) - torch.log(1.0 - student_prob + epsilon)))
            kl = kl.mean(dim=-1)  # [Total_M, G]
            kl_loss = (kl * grpo_beta).mean()
        else:
            kl_loss = torch.tensor(0.0, device=device)

        # ========== 6. 辅助损失 (可选，建议保留 Ranking Loss 作为辅助监督) ==========
        # 这里的 Ranking Loss 依然保留，它和上面的 Reward 逻辑是互补的
        non_tgt_mask = torch.ones_like(logits_g, dtype=torch.bool)
        non_tgt_mask.scatter_(2, tgt_labels.view(-1, 1, 1).expand(-1, G, 1), False)
        non_tgt_logits = logits_g.masked_fill(~non_tgt_mask, float("-inf"))
        max_comp_logits, _ = non_tgt_logits.max(dim=2)
        
        margin = tgt_logits - max_comp_logits
        # 动态 margin：IoU 越高，我们希望 margin 越大（区分度越高）
        dynamic_margin = 0.2 + 0.5 * curr_ious
        ranking_loss = F.relu(dynamic_margin.detach() - margin).mean() * 0.1 # 稍微降低权重，让 RL 主导

        # ========== 7. 总损失 ==========
        total_loss = policy_loss + kl_loss + ranking_loss     
        final_losses["loss_rl"] = total_loss

        return final_losses


    def sample_grpo_features(
        self,
        multi_scale_feats,
        outputs,
        indices,
        targets,
        G=16,
        low_iou_frac: float = 0.5,
        low_iou_thr: float = 0.3,
        extra_factor: int = 4,
        temperature: float = 2.0,
    ):
        """
        GRPO 采样：四条边 (l,t,r,b) 独立随机采样，只保证最终框合法
        对所有 GT 框进行采样，不依赖匹配结果
        """
        device = outputs["pred_boxes"].device
        B, N, _ = outputs["pred_boxes"].shape

        # ===== 收集所有 GT 框（不依赖匹配） =====
        all_tgt_boxes_list = []
        all_tgt_labels_list = []
        gt_per_image = []
        
        for i in range(B):
            gt_boxes = targets[i]["boxes"]   # [num_gts, 4] cxcywh
            gt_labels = targets[i]["labels"] # [num_gts]
            num_gts = gt_boxes.shape[0]
            
            if num_gts > 0:
                all_tgt_boxes_list.append(gt_boxes)
                all_tgt_labels_list.append(gt_labels)
                gt_per_image.append(num_gts)
            else:
                gt_per_image.append(0)
        
        if len(all_tgt_boxes_list) == 0:
            return {"grpo_logits": None, "grpo_ious": None, "grpo_labels": None, "grpo_cost_bbox": None}
        
        all_tgt_boxes = torch.cat(all_tgt_boxes_list, dim=0)   # [Total_GT, 4]
        all_tgt_labels = torch.cat(all_tgt_labels_list, dim=0) # [Total_GT]
        Total_M = len(all_tgt_labels)
        
        # 记录每个 GT 属于哪个 batch（用于回填）
        gt_batch_idx = []
        for i, num_gts in enumerate(gt_per_image):
            for _ in range(num_gts):
                gt_batch_idx.append(i)
        gt_batch_idx = torch.tensor(gt_batch_idx, device=device)

        # 调整采样数量（GT 框占用 1 个）
        G = max(1, G - 1)  # 实际采样数
        G_gt = 1

        # ------------------------------------------------------------------
        # 1) 四边独立随机采样：混合【好框(围绕GT)】+【探索框(完全随机)】
        # ------------------------------------------------------------------
        min_wh = 1e-4

        good_frac = 0.55
        G_good = max(1, int(round(G * good_frac)))
        G_rand = G - G_good

        # ========== A) 好框：围绕 GT 框做四边小扰动 ==========
        gt_cx = all_tgt_boxes[:, 0].unsqueeze(1)  # [Total_M, 1]
        gt_cy = all_tgt_boxes[:, 1].unsqueeze(1)
        gt_w = all_tgt_boxes[:, 2].unsqueeze(1)
        gt_h = all_tgt_boxes[:, 3].unsqueeze(1)

        # 将 GT 转为 xyxy
        gt_xyxy = box_cxcywh_to_xyxy(all_tgt_boxes)
        bx1, by1, bx2, by2 = gt_xyxy.unbind(-1)

        # 计算 GT 的 ltrb
        cx0 = (bx1 + bx2) * 0.5
        cy0 = (by1 + by2) * 0.5
        l0 = (cx0 - bx1).clamp(min=min_wh)
        t0 = (cy0 - by1).clamp(min=min_wh)
        r0 = (bx2 - cx0).clamp(min=min_wh)
        b0 = (by2 - cy0).clamp(min=min_wh)

        # 乘性扰动
        sigma = 0.18
        noise = torch.randn((Total_M, G_good, 4), device=device) * sigma
        noise = noise.clamp(-0.7, 0.7)
        mult = noise.exp()

        l_good = l0.unsqueeze(1) * mult[..., 0]
        t_good = t0.unsqueeze(1) * mult[..., 1]
        r_good = r0.unsqueeze(1) * mult[..., 2]
        b_good = b0.unsqueeze(1) * mult[..., 3]

        cx_good = cx0.unsqueeze(1).expand(-1, G_good)
        cy_good = cy0.unsqueeze(1).expand(-1, G_good)

        x1_good = (cx_good - l_good).clamp(0.0, 1.0)
        y1_good = (cy_good - t_good).clamp(0.0, 1.0)
        x2_good = (cx_good + r_good).clamp(0.0, 1.0)
        y2_good = (cy_good + b_good).clamp(0.0, 1.0)

        x2_good = torch.maximum(x2_good, x1_good + min_wh)
        y2_good = torch.maximum(y2_good, y1_good + min_wh)
        bboxes_good_xyxy = torch.stack([x1_good, y1_good, x2_good, y2_good], dim=-1)

        # ========== B) 坏框：随机扰动，产生低 IoU ==========
        if G_rand > 0:
            target_iou = 0.10
            pool_factor = 6
            P = max(G_rand, G_rand * pool_factor)

            sigma_bad = 0.35
            noise_bad = torch.randn((Total_M, P, 4), device=device) * sigma_bad
            noise_bad = noise_bad.clamp(-1.2, 1.2)
            mult_bad = noise_bad.exp()

            l_bad = l0.unsqueeze(1) * mult_bad[..., 0]
            t_bad = t0.unsqueeze(1) * mult_bad[..., 1]
            r_bad = r0.unsqueeze(1) * mult_bad[..., 2]
            b_bad = b0.unsqueeze(1) * mult_bad[..., 3]

            shift_scale = 0.60
            dx = torch.randn((Total_M, P), device=device).clamp(-2.0, 2.0) * (l0 + r0).unsqueeze(1) * 0.5 * shift_scale
            dy = torch.randn((Total_M, P), device=device).clamp(-2.0, 2.0) * (t0 + b0).unsqueeze(1) * 0.5 * shift_scale

            cx_bad = cx0.unsqueeze(1).expand(-1, P) + dx
            cy_bad = cy0.unsqueeze(1).expand(-1, P) + dy

            x1 = (cx_bad - l_bad).clamp(0.0, 1.0)
            y1 = (cy_bad - t_bad).clamp(0.0, 1.0)
            x2 = (cx_bad + r_bad).clamp(0.0, 1.0)
            y2 = (cy_bad + b_bad).clamp(0.0, 1.0)

            x2 = torch.maximum(x2, x1 + min_wh)
            y2 = torch.maximum(y2, y1 + min_wh)

            bboxes_pool_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            bboxes_pool_cxcywh = box_xyxy_to_cxcywh(bboxes_pool_xyxy)

            pool_flat = bboxes_pool_cxcywh.reshape(-1, 4)
            tgt_flat = all_tgt_boxes.unsqueeze(1).expand(-1, P, -1).reshape(-1, 4)

            pool_iou = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(pool_flat),
                    box_cxcywh_to_xyxy(tgt_flat),
                )[0]
            ).view(Total_M, P)

            pool_iou = pool_iou.clamp(0.0, 1.0)
            bad_mask = (pool_iou < 0.30)

            score = (pool_iou - target_iou).abs()
            score = score + (~bad_mask) * 10.0

            _, idx = torch.topk(score, k=G_rand, dim=1, largest=False)

            idx4 = idx.unsqueeze(-1).expand(-1, -1, 4)
            bboxes_rand_xyxy = torch.gather(bboxes_pool_xyxy, dim=1, index=idx4)

            del bboxes_pool_xyxy, bboxes_pool_cxcywh, pool_flat, tgt_flat, pool_iou, score
            torch.cuda.empty_cache()

            bboxes_g_xyxy = torch.cat([bboxes_good_xyxy, bboxes_rand_xyxy], dim=1)
        else:
            bboxes_g_xyxy = bboxes_good_xyxy

        # 将 GT 框本身加入
        gt_boxes_xyxy = box_cxcywh_to_xyxy(all_tgt_boxes).clamp(0.0, 1.0)
        gt_boxes_xyxy = gt_boxes_xyxy.unsqueeze(1)
        bboxes_g_xyxy = torch.cat([bboxes_g_xyxy, gt_boxes_xyxy], dim=1)

        bboxes_g_cxcywh = box_xyxy_to_cxcywh(bboxes_g_xyxy)
        bboxes_g_cxcywh = torch.nan_to_num(bboxes_g_cxcywh, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        G = bboxes_g_cxcywh.shape[1]

        # ------------------------------------------------------------------
        # 2) 计算 IoU / cost
        # ------------------------------------------------------------------
        tgt_expand = all_tgt_boxes.unsqueeze(1).expand(-1, G, -1).reshape(-1, 4)
        samp_flat = bboxes_g_cxcywh.reshape(-1, 4)

        grpo_ious = torch.diag(
            box_iou(
                box_cxcywh_to_xyxy(samp_flat),
                box_cxcywh_to_xyxy(tgt_expand),
            )[0]
        ).view(Total_M, G)

        cost_bbox = F.l1_loss(samp_flat, tgt_expand, reduction="none").sum(dim=-1).view(Total_M, G)

        # ------------------------------------------------------------------
        # 3) padded_boxes -> VPE -> logits
        # ------------------------------------------------------------------
        # 按 batch 组织
        padded_boxes = torch.zeros((B, Total_M * G, 4), device=device)
        mask = torch.zeros((B, Total_M * G), dtype=torch.bool, device=device)
        
        bboxes_g_cxcywh_flat = bboxes_g_cxcywh.reshape(-1, 4)
        
        curr_idx = 0
        for i in range(B):
            n_gts = gt_per_image[i]
            if n_gts > 0:
                start = curr_idx * G
                end = (curr_idx + n_gts) * G
                padded_boxes[i, :n_gts * G] = bboxes_g_cxcywh_flat[start:end]
                mask[i, :n_gts * G] = True
                curr_idx += n_gts

        if not isinstance(multi_scale_feats, (list, tuple)):
            multi_scale_feats = [multi_scale_feats]

        vpe_feats_padded = self.vpe(reference_boxes=padded_boxes, multi_scale_feats=multi_scale_feats)
        if isinstance(vpe_feats_padded, list):
            vpe_feats_padded = vpe_feats_padded[-1]
        box_features = vpe_feats_padded[mask]

        grpo_logits = self.cls_head(box_features).view(Total_M, G, -1)

        # batch 索引
        grpo_batch_idx = []
        for i, num_gts in enumerate(gt_per_image):
            for _ in range(num_gts * G):
                grpo_batch_idx.append(i)
        grpo_batch_idx = torch.tensor(grpo_batch_idx, device=device)

        return {
            "grpo_logits": grpo_logits,
            "grpo_ious": grpo_ious,
            "grpo_cost_bbox": cost_bbox,
            "grpo_labels": all_tgt_labels,
            "grpo_feats": box_features,
            "grpo_boxes": bboxes_g_cxcywh_flat,
            "grpo_batch_idx": grpo_batch_idx,
            "dec_out_grpo_bboxes": None,
            "vpe_multi_scale_feats": [f.detach().float() for f in multi_scale_feats],
        }

    def sample_grpo_features1(
        self,
        multi_scale_feats,
        outputs,
        indices,
        targets,
        G=16,
        low_iou_frac: float = 0.5,
        low_iou_thr: float = 0.3,
        extra_factor: int = 4,
        temperature: float = 2.0,
    ):
        """
        GRPO 采样：沿GT对角线平移采样 (框尺寸不变，IoU 0.35 ~ 1.0, 恒定间隔)
        对所有 GT 框进行采样，不依赖匹配结果
        """
        device = outputs["pred_boxes"].device
        B, N, _ = outputs["pred_boxes"].shape

        # ===== 收集所有 GT 框（不依赖匹配） =====
        all_tgt_boxes_list = []
        all_tgt_labels_list = []
        gt_per_image = []
        
        for i in range(B):
            gt_boxes = targets[i]["boxes"]   # [num_gts, 4] cxcywh
            gt_labels = targets[i]["labels"] # [num_gts]
            num_gts = gt_boxes.shape[0]
            
            if num_gts > 0:
                all_tgt_boxes_list.append(gt_boxes)
                all_tgt_labels_list.append(gt_labels)
                gt_per_image.append(num_gts)
            else:
                gt_per_image.append(0)
        
        if len(all_tgt_boxes_list) == 0:
            return {"grpo_logits": None, "grpo_ious": None, "grpo_labels": None, "grpo_cost_bbox": None}
        
        all_tgt_boxes = torch.cat(all_tgt_boxes_list, dim=0)   # [Total_GT, 4]
        all_tgt_labels = torch.cat(all_tgt_labels_list, dim=0) # [Total_GT]
        Total_M = len(all_tgt_labels)
        
        # 记录每个 GT 属于哪个 batch 和原始索引（用于回填）
        gt_batch_idx = []
        gt_original_idx = []
        for i, num_gts in enumerate(gt_per_image):
            for j in range(num_gts):
                gt_batch_idx.append(i)
                gt_original_idx.append(j)
        gt_batch_idx = torch.tensor(gt_batch_idx, device=device)
        gt_original_idx = torch.tensor(gt_original_idx, device=device)

        # ------------------------------------------------------------------
        # 1) 沿GT对角线平移采样 (框尺寸不变，IoU 0.35 ~ 1.0, 间隔 0.05)
        # ------------------------------------------------------------------
        iou_levels = torch.arange(0.35, 1.01, 0.05, device=device)
        num_levels = len(iou_levels)
        
        if G <= len(iou_levels):
            idx = torch.randperm(len(iou_levels), device=device)[:G]
            target_ious = iou_levels[idx]
        else:
            idx = torch.randint(0, len(iou_levels), (G,), device=device)
            target_ious = iou_levels[idx]
        target_ious = target_ious.unsqueeze(0).expand(Total_M, G)  # [Total_M, G]

        # 计算对角线平移量
        K_shift = 1.0 - torch.sqrt(2.0 * target_ious / (1.0 + target_ious))
        
        # 随机方向
        sign_x = torch.randint(0, 2, (Total_M, G), device=device) * 2 - 1
        sign_y = torch.randint(0, 2, (Total_M, G), device=device) * 2 - 1

        # 提取 GT 信息
        gt_cx = all_tgt_boxes[:, 0].unsqueeze(1)
        gt_cy = all_tgt_boxes[:, 1].unsqueeze(1)
        gt_w = all_tgt_boxes[:, 2].unsqueeze(1)
        gt_h = all_tgt_boxes[:, 3].unsqueeze(1)

        # 沿对角线平移
        new_cx = gt_cx + sign_x * K_shift * gt_w
        new_cy = gt_cy + sign_y * K_shift * gt_h
        new_w = gt_w.expand(-1, G)
        new_h = gt_h.expand(-1, G)

        bboxes_g_cxcywh = torch.stack([new_cx, new_cy, new_w, new_h], dim=-1)
        
        # 钳制到合法范围
        bboxes_g_xyxy = box_cxcywh_to_xyxy(bboxes_g_cxcywh.reshape(-1, 4)).clamp(0.0, 1.0)
        bboxes_g_cxcywh = box_xyxy_to_cxcywh(bboxes_g_xyxy).reshape(Total_M, G, 4)
        bboxes_g_cxcywh = torch.nan_to_num(bboxes_g_cxcywh, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        G = bboxes_g_cxcywh.shape[1]
        
        # ------------------------------------------------------------------
        # 2) 计算 IoU
        # ------------------------------------------------------------------
        tgt_expand = all_tgt_boxes.unsqueeze(1).expand(-1, G, -1).reshape(-1, 4)
        samp_flat = bboxes_g_cxcywh.reshape(-1, 4)

        grpo_ious = torch.diag(
            box_iou(
                box_cxcywh_to_xyxy(samp_flat),
                box_cxcywh_to_xyxy(tgt_expand),
            )[0]
        ).view(Total_M, G)

        cost_bbox = F.l1_loss(samp_flat, tgt_expand, reduction="none").sum(dim=-1).view(Total_M, G)

        # ------------------------------------------------------------------
        # 3) 通过 VPE 提取特征
        # ------------------------------------------------------------------
        # 将采样框按 batch 组织
        padded_boxes = torch.zeros((B, Total_M * G, 4), device=device)
        mask = torch.zeros((B, Total_M * G), dtype=torch.bool, device=device)
        
        bboxes_g_cxcywh_flat = bboxes_g_cxcywh.reshape(-1, 4)
        
        curr_idx = 0
        for i in range(B):
            n_gts = gt_per_image[i]
            if n_gts > 0:
                start = curr_idx * G
                end = (curr_idx + n_gts) * G
                padded_boxes[i, :n_gts * G] = bboxes_g_cxcywh_flat[start:end]
                mask[i, :n_gts * G] = True
                curr_idx += n_gts

        if not isinstance(multi_scale_feats, (list, tuple)):
            multi_scale_feats = [multi_scale_feats]

        vpe_feats_padded = self.vpe(reference_boxes=padded_boxes, multi_scale_feats=multi_scale_feats)
        if isinstance(vpe_feats_padded, list):
            vpe_feats_padded = vpe_feats_padded[-1]
        box_features = vpe_feats_padded[mask]

        grpo_logits = self.cls_head(box_features).view(Total_M, G, -1)

        # 生成 batch 索引
        grpo_batch_idx = []
        for i, num_gts in enumerate(gt_per_image):
            for _ in range(num_gts * G):
                grpo_batch_idx.append(i)
        grpo_batch_idx = torch.tensor(grpo_batch_idx, device=device)

        return {
            "grpo_logits": grpo_logits,
            "grpo_ious": grpo_ious,
            "grpo_cost_bbox": cost_bbox,
            "grpo_labels": all_tgt_labels,
            "grpo_feats": box_features,
            "grpo_boxes": bboxes_g_cxcywh_flat,
            "grpo_batch_idx": grpo_batch_idx,
            "dec_out_grpo_bboxes": None,
            "vpe_multi_scale_feats": [f.detach().float() for f in multi_scale_feats],
        }


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def rcnn_iou_match(pred_boxes, targets, pos_threshold=0.6, neg_threshold=0.3):
    """
    Mask R-CNN / Faster R-CNN 风格匹配
    
    Args:
        pred_boxes: [B, N, 4] (cxcywh)
        targets: list of dicts, 每个包含 'boxes' (cxcywh)
        pos_threshold: 正样本 IoU 阈值 (默认 0.7)
        neg_threshold: 负样本 IoU 阈值 (默认 0.3)
    
    Returns:
        pos_indices: list of (src_idx, tgt_idx) - 正样本匹配
        neg_indices: list of src_idx - 负样本索引（无 GT 匹配）
    """
    B, N, _ = pred_boxes.shape
    device = pred_boxes.device

    # 转 xyxy
    boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)

    pos_indices = []
    neg_indices = []

    for i in range(B):
        gt_boxes = box_cxcywh_to_xyxy(targets[i]["boxes"])

        # 没有 GT 的情况：所有 query 都是负样本
        if len(gt_boxes) == 0:
            pos_indices.append((
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            ))
            neg_indices.append(torch.arange(N, device=device))
            continue

        # IoU: [N, M]
        ious = box_iou(boxes_xyxy[i], gt_boxes)[0]

        # 每个 query 找最大 IoU 的 GT
        max_ious, matched_gt_idx = ious.max(dim=1)

        # 正样本：IoU >= pos_threshold
        pos_mask = max_ious >= pos_threshold
        pos_src_idx = torch.where(pos_mask)[0]
        pos_tgt_idx = matched_gt_idx[pos_mask]
        pos_indices.append((pos_src_idx, pos_tgt_idx))

        # 负样本：IoU < neg_threshold
        neg_mask = max_ious < neg_threshold
        neg_src_idx = torch.where(neg_mask)[0]
        neg_indices.append(neg_src_idx)

    return pos_indices, neg_indices

def rcnn_iou_match_one_to_one(pred_boxes, targets, pos_threshold=0.6, neg_threshold=0.3):
    """
    一对一匹配：每个 GT 最多匹配一个预测框，每个预测框最多匹配一个 GT
    
    Args:
        pred_boxes: [B, N, 4] (cxcywh)
        targets: list of dicts, 每个包含 'boxes' (cxcywh)
        pos_threshold: 正样本 IoU 阈值 (默认 0.7)
        neg_threshold: 负样本 IoU 阈值 (默认 0.3)
    
    Returns:
        pos_indices: list of (src_idx, tgt_idx) - 正样本匹配（一对一）
        neg_indices: list of src_idx - 负样本索引
    """
    B, N, _ = pred_boxes.shape
    device = pred_boxes.device

    # 转 xyxy
    boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)

    pos_indices = []
    neg_indices = []

    for i in range(B):
        gt_boxes = box_cxcywh_to_xyxy(targets[i]["boxes"])
        num_gts = len(gt_boxes)

        # 没有 GT 的情况：所有 query 都是负样本
        if num_gts == 0:
            pos_indices.append((
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            ))
            neg_indices.append(torch.arange(N, device=device))
            continue

        # IoU: [N, M]
        ious = box_iou(boxes_xyxy[i], gt_boxes)[0]  # [N, num_gts]

        # ========== 一对一匹配：贪心匹配 ==========
        # 复制 ious 避免修改原矩阵
        ious_copy = ious.clone()
        
        matched_src = []
        matched_tgt = []
        
        # 重复匹配直到所有高IoU的都被匹配
        while True:
            # 找到最大IoU的位置
            max_iou = ious_copy.max()
            if max_iou < pos_threshold:
                break
            
            # 找到最大IoU对应的 src 和 tgt 索引
            flat_idx = torch.argmax(ious_copy).item()
            src_idx = flat_idx // num_gts
            tgt_idx = flat_idx % num_gts
            
            matched_src.append(src_idx)
            matched_tgt.append(tgt_idx)
            
            # 删除已匹配的行和列
            ious_copy[src_idx, :] = -1  # 该预测框不再参与匹配
            ious_copy[:, tgt_idx] = -1  # 该GT不再参与匹配
        
        # 修复：明确指定 dtype=torch.long
        if len(matched_src) > 0:
            matched_src = torch.tensor(matched_src, device=device, dtype=torch.long)
            matched_tgt = torch.tensor(matched_tgt, device=device, dtype=torch.long)
        else:
            matched_src = torch.empty(0, device=device, dtype=torch.long)
            matched_tgt = torch.empty(0, device=device, dtype=torch.long)

        # 正样本：一对一匹配的结果
        pos_indices.append((matched_src, matched_tgt))

        # 负样本：未被匹配且 IoU < neg_threshold 的预测框
        is_matched = torch.zeros(N, dtype=torch.bool, device=device)
        if len(matched_src) > 0:
            is_matched[matched_src] = True
        
        max_ious, _ = ious.max(dim=1)
        neg_mask = (~is_matched) & (max_ious < neg_threshold)
        neg_src_idx = torch.where(neg_mask)[0]
        neg_indices.append(neg_src_idx)

    return pos_indices, neg_indices

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SimpleMSDeformableAttention(nn.Module):
    """
    简化版 Multi-Scale Deformable Attention
    适合 ROI-like feature extraction
    """

    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.head_dim = embed_dim // num_heads

        # offset prediction
        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )

        # attention weights
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )

        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, reference_points, multi_scale_feats):
        B, N, C = query.shape
        num_levels = len(multi_scale_feats)
        
        # 计算 offsets 和 attention weights
        offsets = self.sampling_offsets(query)  # [B, N, num_heads * num_levels * num_points * 2]
        offsets = offsets.view(B, N, self.num_heads, num_levels, self.num_points, 2)

        attn = self.attention_weights(query)  # [B, N, num_heads * num_levels * num_points]
        attn = attn.view(B, N, self.num_heads, num_levels * self.num_points)
        attn = attn.softmax(-1).view(B, N, self.num_heads, num_levels, self.num_points)

        # 统一对所有尺度的特征进行投影
        value_proj_list = []
        for feat in multi_scale_feats:
            Bf, Cf, H, W = feat.shape
            # 投影并保持空间结构
            feat_flat = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            feat_proj = self.value_proj(feat_flat)  # [B, H*W, C]
            feat_proj = feat_proj.view(B, -1, self.num_heads, self.head_dim)   # [B, H*W, num_heads, head_dim]
            value_proj_list.append(feat_proj)
        
        # 初始化输出 [B, N, num_heads, head_dim]
        output = torch.zeros(B, N, self.num_heads, self.head_dim, device=query.device)

        # 分离中心点和宽高
        ref_cxcy = reference_points[..., :2]  # [B, N, num_levels, 2]
        ref_wh = reference_points[..., 2:4]   # [B, N, num_levels, 2]
        
        # 对每个尺度进行采样
        for lvl in range(num_levels):
            value = value_proj_list[lvl]   # 当前尺度的value: [B, H*W, num_heads, head_dim]
            H, W = multi_scale_feats[lvl].shape[-2:]
            num_tokens = H * W
            
            ref_lvl = ref_cxcy[:, :, lvl]  # [B, N, 2]
            wh_lvl = ref_wh[:, :, lvl]     # [B, N, 2]
            
            # 将参考点从[0,1]映射到[-1,1]用于grid_sample
            ref_lvl = ref_lvl * 2 - 1  # [B, N, 2]

            # 将box_wh也映射到[-1,1]空间（实际上范围是[0,2]）
            # 因为wh在[0,1]，乘以2后范围[0,2]
            box_wh = wh_lvl * 2 # [B, N, 2]
            
            offset_lvl = offsets[:, :, :, lvl]  # 当前尺度的offsets: [B, N, num_heads, num_points, 2]
            
            # 对每个头独立处理
            for h in range(self.num_heads):
                # 当前头的value: [B, H*W, head_dim]
                value_h = value[..., h, :]  # [B, num_tokens, head_dim]
                
                # 当前头的offsets: [B, N, num_points, 2]
                offset_h = offset_lvl[:, :, h]  # [B, N, num_points, 2]
                                           
                # 计算采样点位置
                sample_points = ref_lvl[:, :, None, :] + offset_h * box_wh[:, :, None, :]
                sample_points = sample_points.clamp(-1, 1)  # [B, N, num_points, 2]
                
                # 重塑为grid_sample格式: [B, N*num_points, 1, 2]
                grid = sample_points.view(B, N * self.num_points, 1, 2)
                
                # 准备value用于采样: [B, head_dim, H, W]
                value_h_spatial = value_h.transpose(1, 2).reshape(B, self.head_dim, H, W)
                
                # 采样: [B, head_dim, N*num_points, 1]
                sampled = F.grid_sample(
                    value_h_spatial,
                    grid,
                    align_corners=False,
                    mode='bilinear',
                    padding_mode='zeros'
                )
                
                sampled = sampled.view(B, self.head_dim, N, self.num_points)  #[B, head_dim, N, num_points]               
                sampled = sampled.permute(0, 2, 3, 1)  # [B, N, num_points, head_dim]
                
                attn_h = attn[:, :, h, lvl]  # 当前头的attention weights: [B, N, num_points]
                # 加权求和: [B, N, head_dim]
                weighted_sampled = (sampled * attn_h.unsqueeze(-1)).sum(dim=2)
                
                # 累加到对应头的输出
                output[:, :, h, :] += weighted_sampled
        
        # 合并多头并输出
        output = output.view(B, N, C)  # [B, N, num_heads, head_dim] -> [B, N, C]
        
        return self.output_proj(output)

class VisualPromptEncoderBlock(nn.Module):
    """
    单个 Visual Prompt Encoder Block
    包含：Cross-Attention + Self-Attention + FFN
    """
    def __init__(self, hidden_dim=256, num_heads=8, num_levels=3, num_points=6, ffn_dim=1024):
        super().__init__()
        
        # Cross-Attention
        self.cross_attn = SimpleMSDeformableAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )
        
        # Self-Attention
        self.self_attn = MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # FFN
        self.ffn = SwiGLUFFN(
            in_features=hidden_dim, 
            hidden_features=ffn_dim, 
            out_features=hidden_dim
        )
        
        # Layer Norms (Pre-Norm 风格)
        self.norm_cross = RMSNorm(hidden_dim)
        self.norm_self = RMSNorm(hidden_dim)
        self.norm_ffn = RMSNorm(hidden_dim)
        
        # Dropout
        self.dropout_cross = nn.Dropout(0.1)
        self.dropout_self = nn.Dropout(0.1)
        self.dropout_ffn = nn.Dropout(0.1)
    
    def forward(self, query, multi_scale_feats, reference_points):
        """
        Args:
            query: [B, N, C]
            multi_scale_feats: List of multi-scale features
            reference_points: [B, N, num_levels, 2] or [B, N, 2]
        """
        # Pre-Norm Cross-Attention
        residual = query
        query = self.norm_cross(query)
        attn_out = self.cross_attn(
            query=query,
            reference_points=reference_points,
            multi_scale_feats=multi_scale_feats
        )
        query = residual + self.dropout_cross(attn_out)
        
        # Pre-Norm Self-Attention
        residual = query
        query = self.norm_self(query)
        self_attn_out, _ = self.self_attn(query=query, key=query, value=query)
        query = residual + self.dropout_self(self_attn_out)
        
        # Pre-Norm FFN
        residual = query
        query = self.norm_ffn(query)
        ffn_out = self.ffn(query)
        query = residual + self.dropout_ffn(ffn_out)
        
        return query

class VisualPromptEncoder(nn.Module):
    """
    基于 multi-scale 注意力提取预测框特征，支持多层深度
    """
    def __init__(self, hidden_dim=256, num_heads=8, num_levels=3, num_points=6, 
                 ffn_dim=1024, depth=1, return_intermediate=False):  # 新增 depth 参数
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.depth = depth
        
        # 框坐标编码
        self.visual_bbox_encoding = MLP(hidden_dim * 2 + 8, hidden_dim, hidden_dim, num_layers=2)
        
        # 频域增强
        self.freq_enhancer = MultiScaleFrequencyEnhance(
            channels=hidden_dim,
            num_levels=num_levels
        )
        
        # 纹理特征辅助提取分支
        self.texture_extract = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            # CBAM(hidden_dim),
            nn.AdaptiveAvgPool2d(1)
        )
        self.texture_fusion = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.texture_fusion.weight)
        nn.init.zeros_(self.texture_fusion.bias)
        
        # 堆叠多个 Block
        self.blocks = nn.ModuleList([
            VisualPromptEncoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                ffn_dim=ffn_dim
            )
            for _ in range(depth)
        ])

        self.return_intermediate = return_intermediate  # 是否返回中间层特征
        
        # 最终的 LayerNorm（可选）
        self.final_norm = RMSNorm(hidden_dim) if depth > 0 else nn.Identity()
    
    def forward(self, reference_boxes: torch.Tensor, multi_scale_feats: list):
        """
        reference_boxes: [B, N, 4], cxcywh, 归一化到[0,1]
        multi_scale_feats: List[B, C, H, W], 多尺度特征
        """
        B, N, _ = reference_boxes.shape
        device = reference_boxes.device

        # 频域增强
        multi_scale_feats = self.freq_enhancer(multi_scale_feats)

        # --- 坐标编码 ---
        query_sine_embed = self.coordinate_to_encoding(reference_boxes)
        query = self.visual_bbox_encoding(query_sine_embed)

        # --- 纹理特征辅助 ---
        finest_feat = multi_scale_feats[0]
        _, _, H_feat, W_feat = finest_feat.shape
        
        # 构建 RoIs
        rois_list = []
        for i in range(B):
            boxes_pixels = box_cxcywh_to_xyxy(reference_boxes[i])
            boxes_pixels[:, 0::2] *= W_feat
            boxes_pixels[:, 1::2] *= H_feat
            idx_col = torch.full((N, 1), i, device=device)
            rois_list.append(torch.cat([idx_col, boxes_pixels], dim=1))
        
        all_rois = torch.cat(rois_list, dim=0)
        
        texture_feats = roi_align(
            finest_feat, all_rois, output_size=(7, 7), 
            spatial_scale=1.0, sampling_ratio=-1, aligned=True
        )
        texture_vector = self.texture_extract(texture_feats)
        texture_vector = texture_vector.flatten(1).view(B, N, -1)
        content_embed = self.texture_fusion(texture_vector)
        query = query + content_embed

        # 准备参考点（多尺度）
        reference_points = reference_boxes[:, :, None, :].repeat(1, 1, self.num_levels, 1)
        
        intermediate_outputs = []

        # 通过多个 Block 处理
        for block in self.blocks:
            query = block(query, multi_scale_feats, reference_points)
            # 收集每一层的输出（包括最后一层）
            if self.return_intermediate:
                intermediate_outputs.append(query)
        
        # # 最终归一化
        # if self.final_norm is not None:
        #     query = self.final_norm(query)
        if self.return_intermediate:
            intermediate_outputs[-1] = query  # 替换最后一层为 final_norm 后的版本
        
        if self.return_intermediate:
            return intermediate_outputs  # List of [B, N, hidden_dim], 长度 = depth
        else:
            return query  # [B, N, hidden_dim]
    
    @staticmethod
    def coordinate_to_encoding(boxes: torch.Tensor):
        """坐标编码"""
        boxes = boxes.clamp(min=1e-6, max=1.0)
        B, N, _ = boxes.shape
        device = boxes.device
        
        scale = 1000.0
        boxes_scaled = boxes * scale
        
        dim_t_orig = torch.arange(0, 64, device=device)
        dim_t_orig = 10000 ** (2 * (dim_t_orig // 2) / 128)
        boxes_scaled_orig = boxes_scaled[:, :, :, None] / dim_t_orig
        pos_sine = torch.stack([
            boxes_scaled_orig[:, :, 0].sin(), boxes_scaled_orig[:, :, 0].cos(),
            boxes_scaled_orig[:, :, 1].sin(), boxes_scaled_orig[:, :, 1].cos(),
            boxes_scaled_orig[:, :, 2].sin(), boxes_scaled_orig[:, :, 2].cos(),
            boxes_scaled_orig[:, :, 3].sin(), boxes_scaled_orig[:, :, 3].cos()
        ], dim=-1).flatten(2)
        
        w, h = boxes[..., 2], boxes[..., 3]
        aspect_ratio = w / (h + 1e-6)
        area = w * h
        aspect_ratio = aspect_ratio.clamp(min=1e-5, max=1e5)
        area = area.clamp(min=1e-6)
        geo_feats = torch.stack([
            w, h, 
            aspect_ratio.log(), 
            area.sqrt()
        ], dim=-1)
        geo_embed = torch.cat([geo_feats.sin(), geo_feats.cos()], dim=-1)
        
        pos = torch.cat([pos_sine, geo_embed], dim=-1)
        return pos


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(hidden_dim, num_classes)
        
        prior_prob = 0.01
        # 根据 sigmoid 反函数计算 bias: bias = -log((1 - p) / p)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        # 初始化权重为很小的随机值
        nn.init.normal_(self.head.weight, std=0.01)
        # 初始化偏置，使模型初始输出的概率接近 0.01
        nn.init.constant_(self.head.bias, bias_value)

    def forward(self, x):
        return self.head(x)


class MultiScaleFrequencyEnhance(nn.Module):
    """
    对编码器的多尺度特征图进行频域增强
    针对宫颈细胞的多尺度纹理(核膜/染色质)与形态(N/C比)进行自适应增强
    """
    def __init__(self, channels, num_levels=3):
        super().__init__()
        
        # 每个尺度独立的增强参数
        # 初始值设定：
        # high_boost: 稍微增强高频，用于突出细胞核纹理
        # cutoff: 不同尺度截断频率不同
        self.level_params = nn.ParameterList([
            nn.ParameterDict({
                'high_boost': nn.Parameter(torch.tensor(0.2)), # 初始值0.2，让网络自己学
                'low_atten': nn.Parameter(torch.tensor(1.0)),
                'cutoff': nn.Parameter(torch.tensor(0.25 + i * 0.05)) # 越深层(i越大)，保留的低频越多
            }) for i in range(num_levels)
        ])
        
        # 通道注意力：抑制频域增强带来的噪声
        self.channel_attn = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 16, 1),
                nn.ReLU(),
                nn.Conv2d(channels // 16, channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_levels)
        ])
        
    def forward(self, multi_scale_feats):
        """
        multi_scale_feats: List of [B, C, H, W]
        """
        enhanced_feats = []
        
        for lvl, feat in enumerate(multi_scale_feats):
            B, C, H, W = feat.shape
            # 防止特征层数超过初始化层数
            # if lvl >= len(self.level_params): 
            if lvl >= 1: 
                enhanced_feats.append(feat)
                continue
            
            dtype = feat.dtype
            params = self.level_params[lvl]
            
            # 1. 2D FFT
            fft_x = torch.fft.rfft2(feat.float(), norm='ortho')
            
            # 2. 构建频域掩码 (Soft Mask)
            mask_low, mask_high = self.build_frequency_masks(
                H, W, 
                cutoff=params['cutoff'], # 保持梯度
                device=feat.device
            )
            
            # 3. 分离与增强
            # 允许 high_boost 为负(平滑)或正(锐化)，网络自适应
            fft_enhanced = fft_x * mask_low * params['low_atten'] + \
                           fft_x * mask_high * (1.0 + params['high_boost'])
            
            
            # 4. 逆FFT
            feat_refined = torch.fft.irfft2(fft_enhanced, s=(H, W), norm='ortho')
            
            # 5.由于FFT操作可能改变分布，使用通道注意力重校准
            feat_refined = feat_refined.to(dtype)
            channel_weight = self.channel_attn[lvl](feat_refined)
            
            # 6. 残差连接 (原特征 + 加权的频域修正特征)
            enhanced_feats.append(feat + feat_refined * channel_weight)
        
        return enhanced_feats
    
    def build_frequency_masks(self, H, W, cutoff, device):
        """构建软掩码"""
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W // 2 + 1, device=device)
        y, x = torch.meshgrid(ys, xs, indexing='ij')
        
        # 计算归一化频率半径
        radius = torch.sqrt(y**2 + x**2)
        radius = radius.unsqueeze(0).unsqueeze(0) # [1, 1, H, W/2+1]
        
        # Sigmoid 软截断
        # transition 控制过渡带宽度，越小越像硬截断
        transition = 0.15 
        # 当 radius < cutoff 时，sigmoid 输入为正，mask_low 接近 1
        mask_low = torch.sigmoid((cutoff - radius) / transition)
        mask_high = 1.0 - mask_low
        
        return mask_low, mask_high


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.constant_(self.w12.bias, 0)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.constant_(self.w3.bias, 0)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        output = output * self.scale
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}'


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        B, C, _, _ = x.shape
        avg = self.avg_pool(x).view(B, C)
        max_ = self.max_pool(x).view(B, C)

        attn = self.mlp(avg) + self.mlp(max_)
        attn = torch.sigmoid(attn).view(B, C, 1, 1)
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        concat = torch.cat([avg, max_], dim=1)
        attn = torch.sigmoid(self.conv(concat))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class GatedFFNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * dim * 2) 
        self.w3 = nn.Linear(dim * 2, dim)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x1 = self.norm(x)
        x12 = self.w12(x1)
        x1, x2 = x12.chunk(2, dim=-1)
        gated = F.silu(x1) * x2
        out = x + self.w3(gated)
        return out

class TextAdapter(nn.Module):
    def __init__(self, text_dim: int, img_dim: int, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedFFNBlock(text_dim) for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(text_dim, img_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.w12.weight)
            nn.init.constant_(layer.w12.bias, 0)
            nn.init.xavier_uniform_(layer.w3.weight)
            nn.init.constant_(layer.w3.bias, 0)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, text_feats):
        x = text_feats
        for layer in self.layers:
            x = layer(x)
        return self.proj_out(x)


