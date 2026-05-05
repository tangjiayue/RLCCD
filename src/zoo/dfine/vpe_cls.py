import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, ModuleList
import torchvision.ops as ops
from torchvision.ops import roi_align
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


#用ClassEmbed,注意没有class_prototypes": self.cls_head.head.weight
@register()
class VisualClassifier(nn.Module):
    __inject__ = [
        "matcher",
    ]

    def __init__(self, matcher, weight_dict, num_classes=10, hidden_dim=256, alpha=0.2, gamma=2.0, mal_alpha=None, reg_max=32, reg_scale=4,
            # 新增两个参数
            weak_classes=[2, 6, 7, 9], 
            confidence_threshold=0.55
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

        # self.vpe = VisualPromptEncoder(hidden_dim=hidden_dim, depth=2, return_intermediate=True)
        self.vpe = VisualPromptEncoder(hidden_dim=hidden_dim, depth=1, return_intermediate=True)
        # self.vpe = VisualPromptEncoder(hidden_dim)
        # self.vpe = MMTREX(embed_dim = hidden_dim)  
        self.cls_head = ClassificationHead(hidden_dim, num_classes)
        self.cls_head_aux = ClassificationHead(hidden_dim, num_classes)
        
        # self.cls_head = ClassEmbed(init_bias=100.0, init_scale=15.0)
        
        text_feats_path="/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/pubmed_text10_name_feats.pt"
        text_feats = torch.load(text_feats_path, map_location='cpu')["text_feats"]
        text_dim = text_feats.shape[1]
        # self.contrast_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim), # 先在视觉空间整理一下
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, text_dim)    # 再映射到文本空间
        # )
        self.text_adapter = TextAdapter(text_dim=text_dim, img_dim=hidden_dim, num_layers=1)
        self.register_buffer("class_text_feats", text_feats)
        # 可学习的 Logit Scale (初始化为 1/0.1 ≈ 10，即 ln(10) ≈ 2.3)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.3026) 
        
        # 弱类别列表和阈值
        self.weak_classes = weak_classes if weak_classes is not None else []
        self.confidence_threshold = confidence_threshold

    def forward(self, feats, outputs, targets=None):
        # K = 200 # 目标保留数量
        # #过滤一半query
        # pred_boxes = outputs['pred_boxes']    # [B, 300, 4]
        # pred_logits = outputs['pred_logits']  # [B, 300, num_classes]
        # quality_scores = outputs['quality_score'] # [B, 300, 1]
        # B = pred_boxes.shape[0]
        # device = pred_boxes.device
        # orig_scores = pred_logits.detach().sigmoid()
        # max_scores, _ = orig_scores.max(dim=-1) # [B, 300]
        # _, topk_indices = torch.topk(max_scores, K, dim=1)

        # # 构造 batch 维度的索引以适配多维提取
        # batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, K)

        # # 裁剪后的 boxes: [B, 150, 4]
        # filtered_boxes = pred_boxes[batch_idx, topk_indices]
        
        # # 裁剪后的 logits: [B, 150, num_classes]
        # filtered_logits = pred_logits[batch_idx, topk_indices]
        
        # # 裁剪后的 quality_scores: [B, 150, 1]
        # filtered_quality = quality_scores[batch_idx, topk_indices]

        # # 更新 outputs 字典（如果后续逻辑依赖该字典）
        # # 这一步很关键，确保后续的 VPE 或 Loss 拿到的是裁剪后的 150 个
        # outputs['pred_boxes'] = filtered_boxes
        # outputs['pred_logits'] = filtered_logits
        # outputs['quality_score'] = filtered_quality

        #========开始
        pred_boxes = outputs['pred_boxes'] # [B, 300, 4] (cxcywh)
        quality_scores = outputs['quality_score'] # [B, 300, 1]
        device = pred_boxes.device
        B, N, _ = pred_boxes.shape
        
        #=======掩码===============
        orig_logits = outputs['pred_logits'].detach()
        orig_scores = orig_logits.sigmoid()  # [B, N, num_classes]
        orig_max_scores, orig_labels = orig_scores.max(dim=-1)  # [B, N]
          
        # 计算保留掩码
        keep_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            for i in range(N):
                label = orig_labels[b, i].item()
                thr = CLASS_THRESHOLDS.get(label, 0.3)
                keep_mask[b, i] = orig_max_scores[b, i] >= thr

        #=====================弱类别重分类=====================
        orig_logits = outputs['pred_logits'].detach() # [B, N, num_classes]
        orig_scores = orig_logits.sigmoid()
        orig_classes = orig_scores.argmax(dim=-1) # [B, N]
        
        weak_tensor = torch.tensor(self.weak_classes, device=device)
        # mask: 判断各个 query 是否被 D-FINE 分类为 2, 6, 7, 9
        query_mask = torch.isin(orig_classes, weak_tensor) # [B, N] 布尔值

        # 创建类别掩码，确保 VPE 只修正指定类别的分数
        class_mask = torch.zeros(self.num_classes, device=device)
        class_mask[self.weak_classes] = 1.0 # [10]


        # 提取特征并分类
        vpe_output = self.vpe(reference_boxes=pred_boxes, multi_scale_feats=feats)  # [B,N,C] 
        # pred_vpe_feats = self.vpe(tboxes=pred_boxes, ibfeatures=feats)
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
        # pred_vpe_logits = self.cls_head(pred_vpe_feats)  # [B,N,num_classes]
        ##用ClassEmbed得到logits
        # adapted_text = self.text_adapter(self.class_text_feats) # [num_classes, C]
        # # ClassEmbed 对 lang_embeds 使用了 transpose(2,1)，所以需要加上 Batch 维度 [1, num_classes, C]
        # pred_vpe_logits = self.cls_head(pred_vpe_feats, adapted_text.unsqueeze(0)) # [B, N, num_classes]
 
        # # 融合质量分数作为最终匹配/预测的依据
        refined_logits = pred_vpe_logits + quality_scores
        # refined_logits[~keep_mask] = float('-inf')
        # refined_logits = pred_vpe_logits

        # combined_mask = query_mask.unsqueeze(-1) * class_mask.view(1, 1, -1)
        # vpe_part = gate * pred_vpe_logits
        # refined_logits = (orig_logits-quality_scores) * (1.0 - combined_mask) + vpe_part * combined_mask

        if self.training:
            # ==============过滤 GT，只保留弱类别的 GT 进行匹配===========
            # filtered_targets = []
            # num_weak_boxes = 0
            # for t in targets:
            #     # 找出符合弱类别的 GT 索引
            #     valid_gt_mask = torch.isin(t["labels"], weak_tensor)
            #     filtered_targets.append({
            #         "labels": t["labels"][valid_gt_mask],
            #         "boxes": t["boxes"][valid_gt_mask]
            #     })
            #     num_weak_boxes += valid_gt_mask.sum().item()

            # targets = filtered_targets
            # num_weak_boxes = torch.as_tensor([num_weak_boxes], dtype=torch.float, device=device)
            # if is_dist_available_and_initialized():
            #     torch.distributed.all_reduce(num_weak_boxes)
            # # 防止除以 0
            # num_weak_boxes = torch.clamp(num_weak_boxes / get_world_size(), min=1).item()

            # # 限制匹配器 (Matcher) 只能选用 query_mask == True 的框
            # match_logits = refined_logits.clone() + quality_scores
            # # 强行把 D-FINE 判定为其他类的 query 分数置为 -inf，彻底断绝被匹配成正样本的可能
            # match_logits[~query_mask] = -1e6 
                
            # outputs_for_matcher = {
            #     'pred_logits': match_logits, 
            #     'pred_boxes': pred_boxes
            # }

            # indices = self.matcher(outputs_for_matcher, targets)["indices"]

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
            # flatten_vpe_logits = refined_logits.reshape(-1, self.num_classes)

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
                    pred_labels=pred_pos_preds
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
                # gt_feats= self.vpe(tboxes=gt_boxes_padded, ibfeatures=feats)  #用MMTREX提特征
                
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
                # img_embeds = self.contrast_head(all_contrast_feats)
                # v_proj = F.normalize(img_embeds, dim=-1)
                # t_norm = F.normalize(self.class_text_feats, dim=-1)

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

           
            grpo_data = self.sample_grpo_features(feats, outputs, indices, targets, G=12)
            
            outputs['pred_logits'] = refined_logits
            return {
                "matched_vpe_logits": flatten_vpe_logits,
                "matched_labels": flatten_labels,
                "matched_ious": flatten_ious,  #[BN]
                "matched_quality": flat_pred_quality,
                "query_mask": query_mask.reshape(-1),
                # "neg_indices": neg_indices,

                "layer_matched_aux": layer_matched_aux,


                **contrast_data,
                "num_boxes": self._get_num_boxes(targets, device),
                # "num_boxes": num_weak_boxes,
                **grpo_data,
                "outputs":outputs,  #联调用
            }
        else:
            # 推理模式：直接使用前面算好的 refined_logits
            outputs['pred_logits'] = refined_logits
            # outputs['pred_logits'] = refined_logits + quality_scores          

            # ================= [推理阶段特征与混淆矩阵收集] =================
            if targets is not None:
                try:
                    fake_targets = []
                    for t in targets:
                        # 1. 拿绝对坐标 xyxy 除以 [W, H, W, H] 得到 0~1 的 xyxy
                        boxes_xyxy_norm = t["boxes"] / 640.
                        
                        # 2. 将归一化的 xyxy 转成归一化的 cxcywh (匹配器Matcher需要的唯一格式)
                        boxes_cxcywh_norm = box_xyxy_to_cxcywh(boxes_xyxy_norm)
                        
                        fake_targets.append({"labels": t["labels"], "boxes": boxes_cxcywh_norm})
       
                    outputs_for_matcher = {
                        'pred_logits': refined_logits, # 使用 VPE 修正后的分数
                        'pred_boxes': pred_boxes
                    }
                    indices = self.matcher(outputs_for_matcher, targets)["indices"]             
                    # indices = rcnn_iou_match(
                    #     pred_boxes=pred_boxes,
                    #     targets=fake_targets,
                    #     pos_threshold=0.6,   
                    # )[0]
                    
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
                            pred_labels=pred_pos_preds
                        )
                except Exception as e:
                    # 出现类型异常直接放过，不影响 mAP 评测
                    pass

            return outputs

    def _get_num_boxes(self, targets, device):
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        return torch.clamp(num_boxes / get_world_size(), min=1).item()

    def get_losses(self, outputs, ref_cls_outputs=None):
        num_boxes = outputs["num_boxes"]
        losses = {}
        # 主分支
        losses_main = self.loss_labels_matched_branch(outputs, num_boxes, branch="main")
        losses.update(losses_main)
        
        if len(outputs["layer_matched_aux"]) > 0:
            losses_aux = self.loss_labels_matched_branch(outputs, num_boxes, branch="aux")
            losses.update(losses_aux)

        # losses_main = self.loss_labels_matched(outputs, num_boxes)
            
        
        # Contrastive Loss
        losses.update(self.loss_contrast_matched(outputs, num_boxes))     #all_contrast_feats只有gt
        # losses.update(self.loss_contrast_matched1(outputs, num_boxes))

        if "grpo_logits" in outputs:
            grpo_loss_dict = self.grpo_loss_v1(outputs, num_boxes, ref_grpo_logits=ref_cls_outputs)
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
            ious = outputs["matched_ious"]
            quality = outputs["matched_quality"]
            # query_mask = outputs["query_mask"].float() # [B*N]
            loss_name = "loss_cls"

            if quality.dim() == 1:
                quality = quality.unsqueeze(-1)

            fused_logits = src_logits + quality
            # fused_logits = src_logits

            target_classes = target_labels.clone()
            target_classes[target_classes == -1] = self.num_classes 
            target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()
            target_score = target * ious.view(-1, 1).pow(0.5)

            with torch.no_grad():
                pred_score = fused_logits.sigmoid().detach()
                weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
            # weight = (target_score - pred_score).abs().pow(self.gamma)

            # #DEIM权重
            # target_score = target_score.pow(1.5)
            # weight = pred_score.pow(1.5) * (1 - target) + target

            loss = F.binary_cross_entropy_with_logits(
                fused_logits, target_score, weight=weight, reduction="none"
            )   # [B*N,10]

            # # 只对 D-FINE 判定出的“特定 query”，且只对“特定的列”算 loss
            # class_mask = torch.zeros(self.num_classes, device=src_logits.device)
            # class_mask[self.weak_classes] = 1.0 # 选定那几个特定的维度的权重为1
            # class_mask = class_mask.view(1, self.num_classes)
            # loss = loss * query_mask.unsqueeze(-1) * class_mask

            return {loss_name: loss.sum() / num_boxes}
        else:
            # 辅助分支的输入
            aux_data = outputs["layer_matched_aux"]  # 根据层索引获取对应数据
            total_aux_loss = 0.0
            loss_name = "loss_cls_aux"
            for layer_data in aux_data:
                src_logits = layer_data["matched_vpe_logits_aux"]
                target_labels = layer_data["matched_labels_aux"]
                ious = layer_data["matched_ious_aux"]
                quality = layer_data["matched_quality_aux"]
                layer_idx = layer_data["layer_idx"]
                
                if quality.dim() == 1:
                    quality = quality.unsqueeze(-1)

                fused_logits = src_logits + quality
                # fused_logits = src_logits

                target_classes = target_labels.clone()
                target_classes[target_classes == -1] = self.num_classes 
                target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()
                target_score = target * ious.view(-1, 1)

                with torch.no_grad():
                    pred_score = fused_logits.sigmoid().detach()
                    weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score


                loss = F.binary_cross_entropy_with_logits(
                    fused_logits, target_score, weight=weight, reduction="none"
                )   # [B*N,10]
                total_aux_loss += loss
                total_aux_loss = total_aux_loss.sum() / num_boxes

            return {loss_name: total_aux_loss}

    # def loss_labels_matched_branch(self, outputs, num_boxes, branch="main"):
    #     matched_vpe_logits = outputs["matched_vpe_logits"]  # [BN, num_classes+1]
    #     matched_labels = outputs["matched_labels"]          # [BN]
        
    #     device = matched_vpe_logits.device
    #     num_classes = self.num_classes  # 物体类别数
        
    #     # 过滤忽略样本
    #     valid_mask = matched_labels != -1
    #     valid_logits = matched_vpe_logits[valid_mask]
    #     valid_labels = matched_labels[valid_mask]
        
    #     if len(valid_logits) == 0:
    #         return {f"loss_cls_{branch}": torch.tensor(0.0, device=device, requires_grad=True)}
        
    #     loss_cls = F.cross_entropy(valid_logits, valid_labels, reduction='mean')
    #     return {f"loss_cls": loss_cls}


    # def loss_contrast_matched(self, outputs, num_boxes):
    #     """
    #     计算图像特征与文本特征的对比损失 (InfoNCE / CrossEntropy)
    #     """
    #     sim_logits = outputs.get("contrast_logits") # [Total_Samples, 10]
    #     targets = outputs.get("contrast_labels")    # [Total_Samples]

    #     if sim_logits is None or targets is None or sim_logits.shape[0] == 0:
    #         return {"loss_contrast": torch.tensor(0.0, device=self.cls_head.head.weight.device)}
       
    #     if torch.isnan(sim_logits).any():
    #         print("Warning: NaN detected in contrast logits!")
    #         return {"loss_contrast": torch.tensor(0.0, device=sim_logits.device)}
    #     # 这里已经是纯正样本了 (Pred_Pos + GT)，直接计算 CE Loss
    #     loss = F.cross_entropy(sim_logits, targets, reduction="sum")

    #     # 使用 num_boxes 归一化
    #     return {"loss_contrast": loss / num_boxes} 
    
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

        # rewards = compute_robust_grpo_reward(logits_g, grpo_data["grpo_labels"], grpo_data["grpo_ious"])

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
    

    # def grpo_loss_v1(self, grpo_data, num_boxes, ref_grpo_logits=None):
    #     """
    #     改进的GRPO损失函数，专门用于提升IoU高框的置信度排序能力
    #     修复了混合精度训练中的数据类型错误，特别是quantile函数的问题
    #     Args:
    #         grpo_data: sample_grpo_features 返回的字典
    #             - grpo_logits: [Total_M, G, C]
    #             - grpo_ious:   [Total_M, G]
    #             - grpo_labels: [Total_M]
    #         num_boxes: 归一化因子
    #         ref_grpo_logits: 参考模型的 Logits, [Total_M * G, C]
    #     """
    #     final_losses = {}
    #     logits_g = grpo_data["grpo_logits"]

    #     if logits_g is None or logits_g.numel() == 0:
    #         return {"loss_rl": logits_g.sum() * 0.0}

    #     Total_M, G, C = logits_g.shape
    #     device = logits_g.device
    #     dtype = logits_g.dtype  # 获取正确的数据类型

    #     # 超参数
    #     grpo_advantage_weight = 0.1
    #     grpo_beta = 0.03
    #     iou_regression_weight = 0.5
    #     ranking_weight = 0.3
    #     epsilon = torch.tensor(1e-9, dtype=torch.float32, device=device)  # 使用float32保证数值稳定性

    #     # 准备标签和IoU数据
    #     tgt_labels = grpo_data["grpo_labels"].view(-1, 1).expand(-1, G)
    #     curr_ious = grpo_data["grpo_ious"].to(dtype)  # 转换为正确的数据类型 [Total_M, G]
        
    #     # 提取目标类别的logits
    #     tgt_logits = logits_g.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
        
    #     # 1. IoU回归损失：直接让logits预测IoU值
    #     iou_regression_loss = F.mse_loss(torch.sigmoid(tgt_logits), curr_ious, reduction='none')
        
    #     # 2. 排序损失：鼓励高IoU框的置信度高于低IoU框
    #     ranking_loss = self._compute_pairwise_ranking_loss(tgt_logits, curr_ious, dtype, device)
        
    #     # 3. 改进的优势函数计算
    #     # 使用IoU作为基础奖励信号，并增强对比度
    #     base_rewards = curr_ious.clamp(0.0, 1.0)
        
    #     # 增强高IoU样本的奖励权重
    #     iou_weights = torch.pow(base_rewards, 2.0)  # 平方操作增强高IoU的重要性
        
    #     # 计算相对优势（相对于组内平均）
    #     mean_iou = base_rewards.mean(dim=1, keepdim=True)
    #     relative_advantages = (base_rewards - mean_iou) / (mean_iou + torch.tensor(0.1, dtype=dtype, device=device))  # 避免除零
        
    #     # 添加对比学习成分：高IoU vs 低IoU的对比
    #     contrastive_advantages = self._compute_contrastive_advantages(base_rewards, tgt_logits, dtype, device)
        
    #     # 组合优势函数
    #     combined_advantages = relative_advantages + 0.5 * contrastive_advantages
    #     combined_advantages = torch.clamp(combined_advantages, -3, 3).detach()
        
    #     # 4. 策略梯度损失
    #     logp = F.logsigmoid(tgt_logits)  # [Total_M, G]
        
    #     # KL散度正则化
    #     if ref_grpo_logits is not None:
    #         ref_grpo_logits = ref_grpo_logits.view(Total_M, G, C).to(dtype)  # 确保类型一致
    #         ref_tgt_logits = ref_grpo_logits.detach().gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)
            
    #         p = torch.sigmoid(tgt_logits)
    #         pref = torch.sigmoid(ref_tgt_logits)
            
    #         # 使用数值稳定的KL散度计算
    #         log_p = torch.log(p + epsilon.to(dtype))
    #         log_1_minus_p = torch.log((1 - p).clamp(min=epsilon.item()))
    #         log_pref = torch.log(pref + epsilon.to(dtype))
    #         log_1_minus_pref = torch.log((1 - pref).clamp(min=epsilon.item()))
            
    #         kl_div = pref * (log_pref - log_p) + (1.0 - pref) * (log_1_minus_pref - log_1_minus_p)
    #     else:
    #         kl_div = torch.zeros_like(logp, dtype=dtype)
        
    #     # 总体策略损失
    #     policy_loss = -(logp * combined_advantages) * grpo_advantage_weight + grpo_beta * kl_div
        
    #     # 5. 组合所有损失
    #     total_policy_loss = policy_loss.mean()
    #     total_iou_regression_loss = iou_regression_loss.mean() * iou_regression_weight
    #     total_ranking_loss = ranking_loss * ranking_weight
        
    #     # final_losses["loss_rl"] = total_policy_loss
    #     # final_losses["loss_iou_regression"] = total_iou_regression_loss
    #     # final_losses["loss_ranking"] = total_ranking_loss
    #     final_losses["loss_rl"] = total_policy_loss + total_iou_regression_loss + total_ranking_loss
        
    #     return final_losses

    # def _compute_pairwise_ranking_loss(self, logits, ious, dtype, device):
    #     """
    #     计算成对排序损失：高IoU框的logits应该大于低IoU框的logits
    #     """
    #     batch_size, num_boxes = logits.shape
        
    #     # 构建成对比较矩阵
    #     iou_diff = ious.unsqueeze(2) - ious.unsqueeze(1)  # [B, G, G]
    #     logit_diff = logits.unsqueeze(2) - logits.unsqueeze(1)  # [B, G, G]
        
    #     # 只考虑IoU差异较大的对（避免噪声干扰）
    #     threshold = torch.tensor(0.1, dtype=dtype, device=device)
    #     significant_diff_mask = torch.abs(iou_diff) > threshold
    #     logit_diff_masked = logit_diff * significant_diff_mask.float()
        
    #     # 排序损失：当iou_diff > 0时，期望logit_diff > 0
    #     ranking_loss = F.relu(-logit_diff_masked * torch.sign(iou_diff))
        
    #     # 只计算有意义的排序对
    #     valid_pairs = significant_diff_mask.sum() + torch.tensor(1e-9, dtype=dtype, device=device)
    #     ranking_loss = ranking_loss.sum() / valid_pairs
        
    #     return ranking_loss

    # def _compute_contrastive_advantages(self, base_rewards, logits, dtype, device):
    #     """
    #     计算对比学习优势：通过对比学习增强排序信号
    #     使用更兼容的数据类型处理quantile操作
    #     """
    #     batch_size, num_boxes = base_rewards.shape
        
    #     # 临时转换到float32进行quantile计算，然后转回原dtype
    #     base_rewards_float = base_rewards.to(torch.float32)
    #     threshold_float = torch.quantile(base_rewards_float, 0.5, dim=1, keepdim=True)  # 中位数阈值
    #     threshold = threshold_float.to(dtype)
        
    #     high_iou_mask = (base_rewards >= threshold).float()
    #     low_iou_mask = (base_rewards < threshold).float()
        
    #     # 计算组内平均logits
    #     high_iou_mean_logit = (logits * high_iou_mask).sum(dim=1, keepdim=True) / (high_iou_mask.sum(dim=1, keepdim=True) + torch.tensor(1e-9, dtype=dtype, device=device))
    #     low_iou_mean_logit = (logits * low_iou_mask).sum(dim=1, keepdim=True) / (low_iou_mask.sum(dim=1, keepdim=True) + torch.tensor(1e-9, dtype=dtype, device=device))
        
    #     # 对比优势：每个框相对于组间差异的贡献
    #     group_diff = high_iou_mean_logit - low_iou_mean_logit
    #     contrastive_adv = (logits - low_iou_mean_logit) * high_iou_mask + (high_iou_mean_logit - logits) * low_iou_mask
        
    #     return contrastive_adv.squeeze(-1)




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
        """
        device = outputs["pred_boxes"].device
        B, N, _ = outputs["pred_boxes"].shape

        batch_idx_map = torch.cat([torch.full((len(indices[i][0]),), i, device=device) for i in range(B)])
        src_idx_map = torch.cat([indices[i][0].to(device) for i in range(B)])
        tgt_idx_map = torch.cat([indices[i][1].to(device) for i in range(B)])

        if len(src_idx_map) == 0:
            return {"grpo_logits": None, "grpo_ious": None, "grpo_labels": None, "grpo_cost_bbox": None}

        Total_M = len(src_idx_map)

        # ===== GT 信息 =====
        all_tgt_labels = torch.cat([targets[i]["labels"][indices[i][1]] for i in range(B)])
        all_tgt_boxes = torch.cat([targets[i]["boxes"][indices[i][1]] for i in range(B)])  # [Total_M, 4] (cxcywh)

        # ===== 参考点（中心）=====
        ref_points = outputs["ref_points"][batch_idx_map, src_idx_map].float()  # [Total_M, 2] in [0,1]

        # 调整采样数量（如果使用 GT 框，需要预留 1 个位置）
        G = max(1, G - 1)  # 实际采样数（GT 框占用 1 个）
        G_gt = 1                  # GT 框占 1 个

        # ------------------------------------------------------------------
        # 1) 四边独立随机采样：混合【好框(围绕pred)】+【探索框(完全随机)】
        # ------------------------------------------------------------------
        min_wh = 1e-4

        # 组内比例：多少个“好框”
        good_frac = 0.55  # 0.2~0.5 都可试
        G_good = max(1, int(round(G * good_frac)))
        G_rand = G - G_good

        # ========== A) 好框：围绕 pred_boxes 做四边小扰动 ==========
        # 用 matched query 对应的当前预测框作为 base（比 ref_points 更接近“好框”）
        base_pred_cxcywh = outputs["pred_boxes"][batch_idx_map, src_idx_map].float()  # [Total_M, 4] in [0,1]
        base_xyxy = box_cxcywh_to_xyxy(base_pred_cxcywh).clamp(0.0, 1.0)
        bx1, by1, bx2, by2 = base_xyxy.unbind(-1)

        # 计算 base 的 ltrb（相对中心）
        cx0 = (bx1 + bx2) * 0.5
        cy0 = (by1 + by2) * 0.5
        l0 = (cx0 - bx1).clamp(min=min_wh)
        t0 = (cy0 - by1).clamp(min=min_wh)
        r0 = (bx2 - cx0).clamp(min=min_wh)
        b0 = (by2 - cy0).clamp(min=min_wh)

        # 对 ltrb 乘性扰动：更稳（不会出现负数），同时允许一定扩张/收缩
        # sigma 越大，“好框”也会更散；建议 0.10~0.25
        sigma = 0.18
        noise = torch.randn((Total_M, G_good, 4), device=device) * sigma
        # 限制极端值，避免仍然太离谱
        noise = noise.clamp(-0.7, 0.7)
        mult = noise.exp()  # lognormal multiplier in (0, +inf)

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
        bboxes_good_xyxy = torch.stack([x1_good, y1_good, x2_good, y2_good], dim=-1)  # [Total_M, G_good, 4]

        # ========== B) 坏框：接近好框，但 IoU 尽量接近一个小目标值 ==========
        if G_rand > 0:
            # 目标 IoU（你希望大约 0.1）
            target_iou = 0.10
            # 候选池倍率：越大越容易挑到接近 target_iou 的框，但计算更贵
            pool_factor = 6
            P = max(G_rand, G_rand * pool_factor)

            # 以 pred_box 为中心，做“中等幅度”的乘性扰动，保证不离谱但能产生低 IoU
            # sigma_bad 比 sigma 大一些，且允许一定平移（shift）
            sigma_bad = 0.35  # 0.25~0.45 建议
            noise_bad = torch.randn((Total_M, P, 4), device=device) * sigma_bad
            noise_bad = noise_bad.clamp(-1.2, 1.2)
            mult_bad = noise_bad.exp()

            # 尺度扰动：从 pred 的 ltrb 扩张/收缩
            l_bad = l0.unsqueeze(1) * mult_bad[..., 0]
            t_bad = t0.unsqueeze(1) * mult_bad[..., 1]
            r_bad = r0.unsqueeze(1) * mult_bad[..., 2]
            b_bad = b0.unsqueeze(1) * mult_bad[..., 3]

            # 再加一点“中心平移”，让它更容易从 GT 上挪开（但仍在附近）
            # shift 量与 box 尺度成比例，避免小框平移过大
            shift_scale = 0.60  # 越大越容易低 IoU；0.4~0.8
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

            bboxes_pool_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # [Total_M, P, 4]
            bboxes_pool_cxcywh = box_xyxy_to_cxcywh(bboxes_pool_xyxy)  # [Total_M, P, 4]

            # 计算 pool 中每个候选与对应 GT 的 IoU
            pool_flat = bboxes_pool_cxcywh.reshape(-1, 4)
            tgt_flat = all_tgt_boxes.unsqueeze(1).expand(-1, P, -1).reshape(-1, 4)

            # IoU: [Total_M*P]
            pool_iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(pool_flat),
                    box_cxcywh_to_xyxy(tgt_flat),
                )
            ).view(Total_M, P)

            # 只允许合法 iou 范围；并过滤掉明显“好框”（比如 >0.3）
            pool_iou = pool_iou.clamp(0.0, 1.0)
            bad_mask = (pool_iou < 0.30)

            # 打分：越接近 target_iou 越好；不满足 bad_mask 的给极大惩罚
            score = (pool_iou - target_iou).abs()
            score = score + (~bad_mask) * 10.0

            # 选出每个样本最接近 target_iou 的 G_rand 个
            # idx: [Total_M, G_rand]
            _, idx = torch.topk(score, k=G_rand, dim=1, largest=False)

            # gather 出最终坏框
            idx4 = idx.unsqueeze(-1).expand(-1, -1, 4)  # [Total_M, G_rand, 4]
            bboxes_rand_xyxy = torch.gather(bboxes_pool_xyxy, dim=1, index=idx4)  # [Total_M, G_rand, 4]

            del bboxes_pool_xyxy, bboxes_pool_cxcywh, pool_flat, tgt_flat, pool_iou, score
            torch.cuda.empty_cache() # 配合使用

            bboxes_g_xyxy = torch.cat([bboxes_good_xyxy, bboxes_rand_xyxy], dim=1)
        else:
            bboxes_g_xyxy = bboxes_good_xyxy

        # 将 GT 框转换为 xyxy 格式
        gt_boxes_xyxy = box_cxcywh_to_xyxy(all_tgt_boxes).clamp(0.0, 1.0)  # [Total_M, 4]
        gt_boxes_xyxy = gt_boxes_xyxy.unsqueeze(1)  # [Total_M, 1, 4]
        # 合并：好框 + 坏框 + GT 框
        bboxes_g_xyxy = torch.cat([bboxes_g_xyxy, gt_boxes_xyxy], dim=1)

        bboxes_g_cxcywh = box_xyxy_to_cxcywh(bboxes_g_xyxy)     # [Total_M, G, 4]
        bboxes_g_cxcywh = torch.nan_to_num(bboxes_g_cxcywh, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        G = bboxes_g_cxcywh.shape[1]
        # ------------------------------------------------------------------
        # 2) 计算 IoU / cost
        # ------------------------------------------------------------------
        tgt_expand = all_tgt_boxes.unsqueeze(1).expand(-1, G, -1).reshape(-1, 4)
        samp_flat = bboxes_g_cxcywh.reshape(-1, 4)

        grpo_ious = torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(samp_flat),
                box_cxcywh_to_xyxy(tgt_expand),
            )
        ).view(Total_M, G)

        cost_bbox = F.l1_loss(samp_flat, tgt_expand, reduction="none").sum(dim=-1).view(Total_M, G)

        # 回填供可视化/调试
        dec_out_grpo_bboxes = torch.zeros((B, N, G, 4), device=device)
        dec_out_grpo_bboxes[batch_idx_map, src_idx_map] = bboxes_g_cxcywh

        # ------------------------------------------------------------------
        # 3) padded_boxes -> VPE -> logits
        # ------------------------------------------------------------------
        max_m = max([len(indices[i][0]) for i in range(B)])
        padded_boxes = torch.zeros((B, max_m * G, 4), device=device)
        mask = torch.zeros((B, max_m * G), dtype=torch.bool, device=device)

        bboxes_g_cxcywh_flat = bboxes_g_cxcywh.reshape(-1, 4)

        curr_idx = 0
        for i in range(B):
            n_m = len(indices[i][0])
            if n_m > 0:
                padded_boxes[i, : n_m * G] = bboxes_g_cxcywh_flat[curr_idx * G : (curr_idx + n_m) * G]
                mask[i, : n_m * G] = True
                curr_idx += n_m

        if not isinstance(multi_scale_feats, (list, tuple)):
            multi_scale_feats = [multi_scale_feats]

        vpe_feats_padded = self.vpe(reference_boxes=padded_boxes, multi_scale_feats=multi_scale_feats)
        if isinstance(vpe_feats_padded, list):
            vpe_feats_padded = vpe_feats_padded[-1]
        box_features = vpe_feats_padded[mask]  # [Total_M*G, C]

        # matched_queries = outputs["output"][batch_idx_map, src_idx_map]  # [Total_M, C]
        # matched_queries_expand = matched_queries.unsqueeze(1).expand(-1, G, -1).reshape(-1, matched_queries.shape[-1])
        # fused_feats = box_features + matched_queries_expand

        fused_feats = box_features
        grpo_logits = self.cls_head(fused_feats).view(Total_M, G, -1)

        padded_batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, padded_boxes.shape[1])
        batch_id_for_each_item = padded_batch_idx[mask]
        grpo_boxes = padded_boxes[mask]  # [Total_M*G, 4]  (格式与 self.vpe(reference_boxes=...)一致)
        grpo_batch_idx = batch_id_for_each_item
        return {
            "grpo_logits": grpo_logits,
            "grpo_ious": grpo_ious,
            "grpo_cost_bbox": cost_bbox,
            "grpo_labels": all_tgt_labels,
            "grpo_feats": fused_feats,
            "grpo_boxes": grpo_boxes,              
            "grpo_batch_idx": grpo_batch_idx, 
            "dec_out_grpo_bboxes": dec_out_grpo_bboxes,
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

        # # 统一对所有尺度的特征进行投影
        # value_proj_list = []
        # for feat in multi_scale_feats:
        #     Bf, Cf, H, W = feat.shape
        #     # 投影并保持空间结构
        #     feat_flat = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        #     feat_proj = self.value_proj(feat_flat)  # [B, H*W, C]
        #     feat_proj = feat_proj.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        #     value_proj_list.append(feat_proj)
        
        # ref_cxcy = reference_points[..., :2]   # [B, N, L, 2]
        # ref_wh   = reference_points[..., 2:4]  # [B, N, L, 2]

        # # 初始化输出
        # output = torch.zeros(B, N, C, device=query.device)
        
        # # 对每个尺度采样
        # for lvl in range(num_levels):
        #     feat = value_proj_list[lvl]  # [B, C, H, W]
        #     H, W = feat.shape[-2:]
            
        #     # 当前尺度的参考点
        #     #ref_lvl = reference_points[:, :, lvl]  # [B, N, 2]
        #     ref_lvl = ref_cxcy[:, :, lvl]      # [B, N, 2], in [0,1]
        #     wh_lvl  = ref_wh[:, :, lvl]        # [B, N, 2], in [0,1]
            
        #     # 计算采样位置
        #     offset_lvl = offsets[:, :, :, lvl]  # [B, N, num_heads, num_points, 2]
            
        #     # 参考点 + 偏移 = 采样点
        #     # 参考点在 [0,1] 范围，需要映射到 [-1,1] 给 grid_sample
        #     ref_lvl = ref_lvl * 2 - 1  # [B, N, 2]
        #     box_wh = wh_lvl * 2.0              # [B, N, 2]
            
        #     # 对每个头的每个采样点
        #     sampled_values = []
        #     for h in range(self.num_heads):
        #         head_offset = offset_lvl[:, :, h]  # [B, N, num_points, 2]

        #         # 采样点位置
        #         sample_points = ref_lvl[:, :, None, :] + head_offset * box_wh[:, :, None, :] 
        #         #sample_points = ref_lvl[:, :, None, :] + head_offset * 2.0 / max(H, W)
        #         sample_points = sample_points.clamp(-1, 1)
                
        #         # 重塑为 grid_sample 格式
        #         grid = sample_points.view(B, N * self.num_points, 1, 2)
                
        #         # 采样
        #         sampled = F.grid_sample(
        #             feat[:, None, :, :].repeat(1, self.num_heads, 1, 1, 1)[:, h],
        #             grid,
        #             align_corners=False,
        #             mode='bilinear',
        #             padding_mode='zeros'
        #         )  # [B, C, N*num_points, 1]
                
        #         sampled = sampled.view(B, C, N, self.num_points).permute(0, 2, 1, 3)  # [B, N, C, num_points]
                
        #         # 加权求和
        #         weight = attn[:, :, h, lvl]  # [B, N, num_points]
        #         sampled = (sampled * weight.unsqueeze(2)).sum(-1)  # [B, N, C]
        #         sampled_values.append(sampled)
            
        #     # 合并所有头
        #     lvl_output = torch.stack(sampled_values, dim=2).mean(dim=2)  # [B, N, C]
        #     output += lvl_output

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
            
            # ref_lvl = reference_points[:, :, lvl]  #当前尺度参考点[B, N, 2]
            ref_lvl = ref_cxcy[:, :, lvl]  # [B, N, 2]
            wh_lvl = ref_wh[:, :, lvl]     # [B, N, 2]
            
            # 将参考点从[0,1]映射到[-1,1]用于grid_sample
            ref_lvl = ref_lvl * 2 - 1  # [B, N, 2]

            # 将box_wh也映射到[-1,1]空间（实际上范围是[0,2]）
            # 因为wh在[0,1]，乘以2后范围[0,2]
            box_wh = wh_lvl * 2 # [B, N, 2]
            
            offset_lvl = offsets[:, :, :, lvl]  # 当前尺度的offsets: [B, N, num_heads, num_points, 2]
            
            # # 根据特征图大小计算每个方向的缩放
            # scale_x = 2.0 / W
            # scale_y = 2.0 / H
            # offset_scale = torch.tensor([scale_x, scale_y], device=query.device)
            # 对每个头独立处理
            for h in range(self.num_heads):
                # 当前头的value: [B, H*W, head_dim]
                value_h = value[..., h, :]  # [B, num_tokens, head_dim]
                
                # 当前头的offsets: [B, N, num_points, 2]
                offset_h = offset_lvl[:, :, h]  # [B, N, num_points, 2]
                                           
                # 计算采样点位置
                # 参考点 + 偏移量 (偏移量需要根据特征图大小归一化)
                # sample_points = ref_lvl[:, :, None, :] + offset_h * offset_scale
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

class MSDeformableCrossAttentionCorrect(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, n_points=8, n_levels=3, 
                 base_scale=0.6, offset_scale=0.5):  
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_levels = n_levels
        self.head_dim = embed_dim // n_heads
        self.base_scale = base_scale
        self.offset_scale = offset_scale  

        # linear layers
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, n_heads * n_points * 2 * n_levels)
        self.attn_weights = nn.Linear(embed_dim, n_heads * n_points * n_levels)
        self.value_projs = nn.ModuleList([nn.Conv2d(embed_dim, embed_dim, kernel_size=1) for _ in range(n_levels)])
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_bias_radial()

    def _init_bias_radial(self):
        """放射状初始化"""
        n_h, n_p, n_l = self.n_heads, self.n_points, self.n_levels
        
        thetas = torch.arange(n_h, dtype=torch.float32) * (2.0 * math.pi / n_h)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / (grid_init.abs().max(-1, keepdim=True).values + 1e-6)
        grid_init = grid_init.reshape(n_h, 1, 2).repeat(1, n_p, 1)
        
        scaling = torch.arange(1, n_p + 1, dtype=torch.float32).reshape(1, -1, 1)
        grid_init = grid_init * scaling
        
        bias = torch.zeros((n_h, n_p, 2, n_l), dtype=torch.float32)
        for lvl in range(n_l):
            scale = self.base_scale * (1.0 + lvl / max(1, n_l - 1))
            bias[:, :, :, lvl] = grid_init * scale
        
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(bias.reshape(-1))
            nn.init.constant_(self.sampling_offsets.weight, 0.0)
            nn.init.constant_(self.attn_weights.weight, 0.0)
            nn.init.constant_(self.attn_weights.bias, 0.0)

    def forward(self, query, feat_maps, boxes, padding_mask=None, box_mask=None):
        B, Nq, C = query.shape
        device = query.device

        if box_mask is None:
            box_mask = torch.ones((B, Nq), dtype=torch.float32, device=device)

        q = self.query_proj(query.view(B * Nq, C))
        sampling_offsets = self.sampling_offsets(q).view(B, Nq, self.n_heads, self.n_points, 2, self.n_levels)

        attn_raw = self.attn_weights(q).view(B, Nq, self.n_heads, self.n_points * self.n_levels)
        attn_flat = F.softmax(attn_raw, dim=-1)

        # 解析 boxes
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        bw = (x_max - x_min).clamp(min=1e-6)
        bh = (y_max - y_min).clamp(min=1e-6)

        if padding_mask is not None:
            pad_mask_in = padding_mask.unsqueeze(1).to(dtype=feat_maps[0].dtype)
        else:
            pad_mask_in = None

        sampled_per_level = []

        for lvl_idx, fm in enumerate(feat_maps):
            Bf, Cf, H_l, W_l = fm.shape
            assert Bf == B and Cf == C

            value = self.value_projs[lvl_idx](fm)
            value = value.view(B, self.n_heads, self.head_dim, H_l, W_l)
            value_for_grid = value.view(B * self.n_heads, self.head_dim, H_l, W_l)

            offs = sampling_offsets[..., :, :, :, lvl_idx]

            # 使用 offset_scale
            off_x_norm = offs[..., 0] * bw.view(B, Nq, 1, 1) * self.offset_scale
            off_y_norm = offs[..., 1] * bh.view(B, Nq, 1, 1) * self.offset_scale

            x_sample_norm = (cx.view(B, Nq, 1, 1) + off_x_norm).clamp(0.0, 1.0)
            y_sample_norm = (cy.view(B, Nq, 1, 1) + off_y_norm).clamp(0.0, 1.0)

            x_grid = x_sample_norm * 2.0 - 1.0
            y_grid = y_sample_norm * 2.0 - 1.0

            grid = torch.stack((x_grid, y_grid), dim=-1)
            grid = grid.permute(0, 2, 1, 3, 4).contiguous()
            n_pts = Nq * self.n_points
            grid = grid.view(B * self.n_heads, n_pts, 2).unsqueeze(2)

            sampled = F.grid_sample(value_for_grid, grid, mode='bilinear', align_corners=False)
            sampled = sampled.view(B, self.n_heads, self.head_dim, n_pts).permute(0, 3, 1, 2).contiguous()
            sampled = sampled.view(B, Nq, self.n_points, self.n_heads, self.head_dim).permute(0,1,3,2,4).contiguous()
            sampled_per_level.append(sampled)

        # 直接使用 attn_flat，跳过复杂的 mask 逻辑
        attn_normalized = attn_flat.view(B, Nq, self.n_heads, self.n_points, self.n_levels)

        weighted_levels = []
        for lvl_idx in range(self.n_levels):
            w_lvl = attn_normalized[..., :, lvl_idx].unsqueeze(-1)
            sampled_lvl = sampled_per_level[lvl_idx]
            weighted_lvl = (sampled_lvl * w_lvl).sum(dim=3)
            weighted_levels.append(weighted_lvl)

        weighted = sum(weighted_levels)
        merged = weighted.view(B, Nq, self.n_heads * self.head_dim)

        out = self.output_proj(merged.view(B * Nq, C)).view(B, Nq, C)
        return out

# class VisualPromptEncoder(nn.Module):
#     """
#     基于 multi-scale 注意力提取预测框特征，摒弃 ROIAlign
#     """
#     def __init__(self, hidden_dim=256, num_heads=8, num_levels=3, num_points=6,ffn_dim=1024):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_levels = num_levels
#         # cross-attention: query=框坐标编码, key/value=multi-scale特征
#         self.cross_attn = SimpleMSDeformableAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             num_levels=num_levels,
#             num_points=num_points
#         )
#         # self.cross_attn = MSDeformableCrossAttentionCorrect(
#         #     embed_dim=hidden_dim,
#         #     n_heads=num_heads,
#         #     n_levels=num_levels,
#         #     n_points=num_points,
#         #     base_scale=0.6,
#         #     offset_scale=0.5
#         # )
#         self.self_attn = MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
#         # FFN
#         # self.ffn = nn.Sequential(
#         #     nn.Linear(hidden_dim, ffn_dim),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(0.1),
#         #     nn.Linear(ffn_dim, hidden_dim),
#         # )
#         self.ffn = SwiGLUFFN(
#             in_features=hidden_dim, 
#             hidden_features=ffn_dim, 
#             out_features=hidden_dim
#         )
#         # self.norms = ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
#         self.norms = ModuleList([RMSNorm(hidden_dim) for _ in range(3)])

#         # 框坐标编码
#         self.visual_bbox_encoding = MLP(hidden_dim * 2 + 8, hidden_dim, hidden_dim, num_layers=2)


#         self.freq_enhancer = MultiScaleFrequencyEnhance(
#             channels=hidden_dim,
#             num_levels=num_levels
#         )

#         #纹理特征辅助提取分支
#         self.texture_extract = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
#             nn.GroupNorm(32, hidden_dim),
#             nn.ReLU(inplace=True),
#             # CBAM(hidden_dim),
#             nn.AdaptiveAvgPool2d(1) # 压缩成 vector
#         )
#         self.texture_fusion = nn.Linear(hidden_dim, hidden_dim)
#         nn.init.zeros_(self.texture_fusion.weight) # 初始化为0，避免初始干扰
#         nn.init.zeros_(self.texture_fusion.bias)

#     def forward(self, reference_boxes: torch.Tensor, multi_scale_feats: list):
#         """
#         reference_boxes: [B, N, 4], cxcywh, 归一化到[0,1]
#         multi_scale_feats: List[B, C, H, W], 多尺度特征
#         """
#         B, N, _ = reference_boxes.shape
#         device = reference_boxes.device

#         multi_scale_feats = self.freq_enhancer(multi_scale_feats)

#         # --- 坐标编码 ---
#         query_sine_embed = self.coordinate_to_encoding(reference_boxes)  # [B, N, hidden_dim*2]
#         query_pos = self.visual_bbox_encoding(query_sine_embed)              # [B, N, hidden_dim]
#         query = query_pos

#         #纹理特征辅助提取分支
#         finest_feat = multi_scale_feats[0]
#         B_feat, C_feat, H_feat, W_feat = finest_feat.shape
#         rois_list = []
#         for i in range(B):
#             # 将 [0,1] 坐标转为像素坐标
#             boxes_pixels = box_cxcywh_to_xyxy(reference_boxes[i])
#             boxes_pixels[:, 0::2] *= W_feat
#             boxes_pixels[:, 1::2] *= H_feat

#             idx_col = torch.full((N, 1), i, device=device)
#             rois_list.append(torch.cat([idx_col, boxes_pixels], dim=1))
        
#         all_rois = torch.cat(rois_list, dim=0) # [B*N, 5]
        
#         # 高效采样
#         texture_feats = roi_align(
#             finest_feat, 
#             all_rois, 
#             output_size=(7, 7), 
#             spatial_scale=1.0, 
#             sampling_ratio=-1, 
#             aligned=True
#         ) # [B*N, C, 7, 7]

#         texture_vector = self.texture_extract(texture_feats)
 
#         # [B*N, C, 1, 1] -> [B*N, C] -> [B, N, C]
#         texture_vector = texture_vector.flatten(1).view(B, N, -1)   
#         content_embed = self.texture_fusion(texture_vector) 
#         query = query + content_embed

#         reference_points = reference_boxes[:, :, None, :].repeat(1, 1, self.num_levels, 1)
#         # cross-attention
#         attn_out = self.cross_attn(
#             query=query,
#             reference_points=reference_points,  
#             multi_scale_feats=multi_scale_feats
#         ) # [B, N, C]
#         # reference_boxes = box_cxcywh_to_xyxy(reference_boxes)
#         # attn_out = self.cross_attn(
#         #     query=query,
#         #     boxes=reference_boxes,  
#         #     feat_maps=multi_scale_feats,
#         # ) # [B, N, C]

#         query = self.norms[0](attn_out + query)

#         self_attn_out, _ = self.self_attn(query=query, key=query, value=query)
#         query = self.norms[1](self_attn_out + query)

#         # FFN
#         ffn_out = self.ffn(query)
#         # query = self.norms[2](query)
#         query = self.norms[2](query + ffn_out)
             
#         return query  # [B, N, hidden_dim]

    
#     @staticmethod
#     def coordinate_to_encoding(boxes: torch.Tensor):
#         """
#         boxes: [B, N, 4] cxcywh
#         """
#         boxes = boxes.clamp(min=1e-6, max=1.0) 
#         B, N, _ = boxes.shape
#         device = boxes.device
        
#         # 1. 基础 Sine 编码 (保持不变)
#         scale = 1000.0
#         boxes_scaled = boxes * scale
        
#         # Sine 编码: [B, N, 256]
#         dim_t_orig = torch.arange(0, 64, device=device)
#         dim_t_orig = 10000 ** (2 * (dim_t_orig // 2) / 128)
#         boxes_scaled_orig = boxes_scaled[:, :, :, None] / dim_t_orig
#         pos_sine = torch.stack([
#             boxes_scaled_orig[:, :, 0].sin(), boxes_scaled_orig[:, :, 0].cos(),
#             boxes_scaled_orig[:, :, 1].sin(), boxes_scaled_orig[:, :, 1].cos(),
#             boxes_scaled_orig[:, :, 2].sin(), boxes_scaled_orig[:, :, 2].cos(),
#             boxes_scaled_orig[:, :, 3].sin(), boxes_scaled_orig[:, :, 3].cos()
#         ], dim=-1).flatten(2) # [B, N, 512]

#         # 2. [新增] 显式几何特征 (Explicit Geometric Features)
#         # w, h, aspect_ratio, area
#         w, h = boxes[..., 2], boxes[..., 3]
#         aspect_ratio = w / (h + 1e-6)
#         area = w * h
#         aspect_ratio = aspect_ratio.clamp(min=1e-5, max=1e5)
#         area = area.clamp(min=1e-6)
#         # Log 变换使其分布更均匀
#         geo_feats = torch.stack([
#             w, h, 
#             aspect_ratio.log(), 
#             area.sqrt() # 线性化面积
#         ], dim=-1) # [B, N, 4]
        
#         # 扩展到 sine/cosine 映射
#         geo_embed = torch.cat([geo_feats.sin(), geo_feats.cos()], dim=-1) # [B, N, 8]

#         # 拼接: [B, N, 512 + 8]
#         pos = torch.cat([pos_sine, geo_embed], dim=-1)

#         return pos


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
        """坐标编码（保持不变）"""
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

class ClassEmbed(nn.Module):
    def __init__(
        self,
        init_bias: float = 100.0,
        init_scale: float = 15.0,
    ):
        super().__init__()
        bias_value = -math.log(init_bias)
        self.lang_bias = nn.Parameter(torch.full((), bias_value))
        self.lang_scale = nn.Parameter(torch.tensor(init_scale).log())
        
    def forward(self, image_embeds, lang_embeds, mask=None):
        image_norm = F.normalize(image_embeds, p=2, dim=-1)
        lang_norm = F.normalize(lang_embeds, p=2, dim=-1)
        
        logits = image_norm @ lang_norm.transpose(2, 1)  # [batch_size, num_queries, num_classes]
                
        logits = logits * torch.exp(self.lang_scale) + self.lang_bias

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float('-inf'))
        return logits


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



# #主实验是5开头的
# @register()
# class VisualClassifier(nn.Module):
#     __inject__ = [
#         "matcher",
#     ]

#     def __init__(self, matcher, weight_dict, num_classes=10, hidden_dim=256, alpha=0.2, gamma=2.0, mal_alpha=None, reg_max=32, reg_scale=4):
#         super().__init__()
#         self.weight_dict = weight_dict
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.gamma = gamma
#         self.mal_alpha = mal_alpha
#         self.reg_max = reg_max
#         self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
#         self.reg_scale = nn.Parameter(torch.tensor([float(reg_scale)]), requires_grad=False)
#         self.matcher = matcher

#         self.vpe = SimpleVisualPromptEncoder(hidden_dim)  
#         self.cls_head = ClassificationHead(hidden_dim, num_classes)
#         # self.fusion = ROIQueryFusion(hidden_dim)

#         text_feats_path="/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/text10_feats.pt"
#         text_feats = torch.load(text_feats_path, map_location='cpu')["text_feats"]
#         text_dim = text_feats.shape[1]
#         self.v_to_t_projection = nn.Linear(hidden_dim, text_dim)
#         self.register_buffer("target_text_feats", text_feats)

#         project_sequence = weighting_function(reg_max, self.up, self.reg_scale, deploy=True)
#         self.register_buffer("project", project_sequence)
    
#     # def forward(self, feats, outputs, targets=None):
#     #     pred_boxes = outputs['pred_boxes'] # 归一化 cxcywh [B, N, 4]
#     #     quality_scores = outputs.get('quality_score', None)
#     #     device = pred_boxes.device
#     #     B, N, _ = pred_boxes.shape
#     #     memory_map = feats[-2]
#     #     _, _, m_h, m_w = memory_map.shape

#     #     # ---------------------------------------------------------
#     #     # 训练模式：在原有正样本基础上，加入负样本采样
#     #     # ---------------------------------------------------------
#     #     if self.training:
#     #         outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
#     #         indices = self.matcher(outputs_without_aux, targets)["indices"]
#     #         num_boxes = sum(len(t["labels"]) for t in targets)
#     #         num_boxes = torch.as_tensor(
#     #             [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
#     #         )

#     #         if is_dist_available_and_initialized():
#     #             torch.distributed.all_reduce(num_boxes)
#     #         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#     #         all_matched_vpe_logits = []
#     #         all_matched_sim_logits = []
#     #         all_matched_labels = [] # 这里会包含 -1 代表背景
#     #         all_matched_ious = []
#     #         all_matched_quality = []

#     #         for i in range(B):
#     #             src_idx = indices[i][0].to(device)
#     #             tgt_idx = indices[i][1].to(device)
            
#     #             # --- 负样本采样逻辑：是把背景框也加进来训练 ---
#     #             all_q_idx = torch.arange(N, device=device)
#     #             mask = torch.ones(N, device=device, dtype=torch.bool)
#     #             mask[src_idx] = False
#     #             neg_idx_pool = all_q_idx[mask]          

#     #             # 采样负样本（数量设为正样本的3倍左右，有助于分类头抑制背景）
#     #             num_neg = min(len(neg_idx_pool), len(src_idx) * 3 + 5)
#     #             perm = torch.randperm(len(neg_idx_pool), device=device)
#     #             sampled_neg_idx = neg_idx_pool[perm[:num_neg]]        

#     #             # 合并索引进行处理
#     #             train_idx = torch.cat([src_idx, sampled_neg_idx])
#     #             if len(train_idx) == 0: continue                          

#     #             # --- 准备输入 ---
#     #             cur_boxes = pred_boxes[i:i+1, train_idx]
#     #             Ni_total = cur_boxes.shape[1]
#     #             cur_boxes_xyxy = box_cxcywh_to_xyxy(cur_boxes.squeeze(0))
#     #             cur_boxes_scaled = cur_boxes_xyxy.clone()
#     #             cur_boxes_scaled[:, [0, 2]] *= m_w
#     #             cur_boxes_scaled[:, [1, 3]] *= m_h

#     #             batch_idx_vec = torch.zeros((Ni_total, 1), device=device)
#     #             cur_rois = torch.cat([batch_idx_vec, cur_boxes_scaled], dim=1)
#     #             cur_feat_batch = self.vpe(memory_map[i:i+1], cur_rois, cur_boxes)
#     #             cur_feat = cur_feat_batch.squeeze(0)

#     #             # --- 分类分支 ---
#     #             cur_logits = self.cls_head(cur_feat) # 包含正负样本

#     #             # --- 对比学习分支 (仅对前 len(src_idx) 个正样本计算) ---
#     #             pos_feat = cur_feat[:len(src_idx)]
#     #             v_proj = F.normalize(self.v_to_t_projection(pos_feat), dim=-1)
#     #             t_norm = F.normalize(self.target_text_feats, dim=-1)
#     #             sim_logits = torch.matmul(v_proj, t_norm.t()) / 0.07

#     #             # --- 标签构造 ---
#     #             # 正样本用原标签，负样本设为 -1
#     #             cur_labels = torch.full((Ni_total,), -1, dtype=torch.long, device=device)
#     #             cur_labels[:len(src_idx)] = targets[i]["labels"][tgt_idx]

#     #             # IoU 软标签 (仅正样本)
#     #             target_boxes = targets[i]["boxes"][tgt_idx]
#     #             pos_boxes_xyxy = cur_boxes_xyxy[:len(src_idx)]
#     #             ious_mat, _ = box_iou(pos_boxes_xyxy, box_cxcywh_to_xyxy(target_boxes))
#     #             cur_ious = torch.diag(ious_mat).detach()

#     #             # 收集结果
#     #             if quality_scores is not None:
#     #                 # quality_scores 维度是 [B, N, 1]
#     #                 cur_quality = quality_scores[i, train_idx]
#     #                 all_matched_quality.append(cur_quality) 
#     #             all_matched_vpe_logits.append(cur_logits)
#     #             all_matched_sim_logits.append(sim_logits)
#     #             all_matched_labels.append(cur_labels)
#     #             all_matched_ious.append(cur_ious)
#     #         # grpo_data = self.sample_grpo_features(memory_map, outputs, indices, targets, G=32)
            
#     #         # 封装返回
#     #         res = {
#     #             "pred_logits": outputs['pred_logits'].clone(),
#     #             "matched_vpe_logits": torch.cat(all_matched_vpe_logits, dim=0) if all_matched_vpe_logits else None,
#     #             "matched_sim_logits": torch.cat(all_matched_sim_logits, dim=0) if all_matched_sim_logits else None,
#     #             "matched_labels": torch.cat(all_matched_labels, dim=0) if all_matched_labels else None,
#     #             "matched_ious": torch.cat(all_matched_ious, dim=0) if all_matched_ious else None,
#     #             "matched_quality": torch.cat(all_matched_quality, dim=0) if all_matched_quality else None,
#     #             "num_boxes": num_boxes,
#     #             # **grpo_data
#     #         }

#     #         return res

#     #     # ---------------------------------------------------------
#     #     # 推理模式：加入“背景锁”逻辑
#     #     # ---------------------------------------------------------
#     #     else:
#     #         full_batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1, 1)
#     #         flat_boxes = pred_boxes.reshape(-1, 4)
            
#     #         boxes_xyxy = box_cxcywh_to_xyxy(flat_boxes)
#     #         boxes_xyxy[:, [0, 2]] *= m_w
#     #         boxes_xyxy[:, [1, 3]] *= m_h
#     #         infer_rois = torch.cat([full_batch_idx.float(), boxes_xyxy], dim=1)
            
#     #         # 得到 VPE 分支的预测分数
#     #         box_features = self.vpe(memory_map, infer_rois, pred_boxes)
#     #         new_logits = self.cls_head(box_features).view(B, N, -1)

#     #         #  获取 D-FINE 原生 Logits (已包含类别分数 + 质量分数)
#     #         orig_logits = outputs['pred_logits']

#     #         # 让 VPE 作为一个“视觉修正项”存在
#     #         w = 0.5 
#     #         combined_logits = (1 - w) * orig_logits + w * new_logits
            
#     #         # 5. 更新并返回
#     #         outputs['pred_logits'] = combined_logits
#     #         return outputs

#     #D-FINE的输出得到匹配后再用重分类，但是按前面的匹配结果计算损失
#     # def forward(self, feats, outputs, targets=None):
#     #     pred_boxes = outputs['pred_boxes'] # [B, 300, 4] (cxcywh)
#     #     quality_scores = outputs['quality_score'] # [B, 300, 1]
#     #     device = pred_boxes.device
#     #     B, N, _ = pred_boxes.shape
#     #     memory_map = feats[-2]   
#     #     _, _, m_h, m_w = memory_map.shape

#     #     if self.training:
#     #         # 1. 匹配：确定 300 个框里谁是正样本
#     #         outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
#     #         indices = self.matcher(outputs_without_aux, targets)["indices"]
            
#     #         # 2. 计算全局 num_boxes
#     #         num_boxes = sum(len(t["labels"]) for t in targets)
#     #         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
#     #         if is_dist_available_and_initialized():
#     #             torch.distributed.all_reduce(num_boxes)
#     #         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#     #         # 3. 构造全量 RoIs [B*N+gt, 5] 
#     #         gt_boxes_list = [t["boxes"] for t in targets] # 每个元素是 [Mi, 4]
#     #         num_gts_per_batch = [len(b) for b in gt_boxes_list]
#     #         all_gt_boxes = torch.cat(gt_boxes_list, dim=0) # [Total_GT, 4]
#     #         flat_gt_boxes_xyxy = box_cxcywh_to_xyxy(all_gt_boxes).clamp(0.0, 1.0)
#     #         num_gts = all_gt_boxes.shape[0]

#     #         flat_boxes_cxcywh = pred_boxes.reshape(-1, 4)
#     #         flat_pred_boxes_xyxy = box_cxcywh_to_xyxy(flat_boxes_cxcywh)
#     #         x1, y1, x2, y2 = flat_pred_boxes_xyxy.unbind(-1)
#     #         x2 = torch.max(x2, x1 + 1e-4)
#     #         y2 = torch.max(y2, y1 + 1e-4)
#     #         # 哪怕 x1, x2 都被按回了 1.0，roi_align 也能处理（会采样边界像素）
#     #         flat_pred_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)

#     #         all_boxes_xyxy = torch.cat([flat_pred_boxes_xyxy, flat_gt_boxes_xyxy], dim=0)
#     #         # print(f"ROIs max: {flat_pred_boxes_xyxy.max()}, min: {flat_pred_boxes_xyxy.min()}, has_nan: {torch.isnan(flat_pred_boxes_xyxy).any()}")
#     #         # print(f"ROIs max: {flat_gt_boxes_xyxy.max()}, min: {flat_gt_boxes_xyxy.min()}, has_nan: {torch.isnan(flat_gt_boxes_xyxy).any()}")

#     #         # 坐标缩放
#     #         rois_xyxy = all_boxes_xyxy.clone()
#     #         rois_xyxy[:, [0, 2]] *= m_w
#     #         rois_xyxy[:, [1, 3]] *= m_h
            
#     #         # 构造 batch_indices: [0,0...0, 1,1...1, ...]
#     #         pred_batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1, 1)
#     #         gt_batch_idx = torch.cat([torch.full((n, 1), i, device=device) for i, n in enumerate(num_gts_per_batch)], dim=0)
#     #         batch_idx = torch.cat([pred_batch_idx.float(), gt_batch_idx.float()], dim=0)
#     #         flatten_rois = torch.cat([batch_idx.float(), rois_xyxy], dim=1)

#     #         # 4. 执行 VPE 和分类头
#     #         # 特别注意：flatten_feats 将用于全量分类，但对比学习只取正样本
#     #         flatten_feats = self.vpe(memory_map, flatten_rois, None) 
#     #         flatten_vpe_logits = self.cls_head(flatten_feats) # [B*N+gt, 10]

#     #         #构造质量分数 (Quality Scores)
#     #         flat_pred_quality = quality_scores.reshape(-1, 1)
#     #         gt_quality = torch.full((all_gt_boxes.shape[0], 1), 0.93, device=device)
#     #         combined_quality = torch.cat([flat_pred_quality, gt_quality], dim=0)

#     #         # 5. 构造全量 Labels 和 IoUs 矩阵
#     #         flatten_labels = torch.full((B, N), -1, dtype=torch.long, device=device)
#     #         flatten_ious = torch.zeros((B, N), device=device)
            
#     #         for i in range(B):
#     #             src_idx, tgt_idx = indices[i]
#     #             if len(src_idx) > 0:
#     #                 flatten_labels[i, src_idx] = targets[i]["labels"][tgt_idx]
#     #                 # 计算正样本 IoU
#     #                 p_boxes = flat_pred_boxes_xyxy.view(B, N, 4)[i, src_idx]
#     #                 t_boxes = box_cxcywh_to_xyxy(targets[i]["boxes"][tgt_idx])
#     #                 # 使用 diag 快速获取匹配对的 IoU
#     #                 ious = torch.diag(box_iou(p_boxes, t_boxes)[0])
#     #                 flatten_ious[i, src_idx] = ious
            
#     #         gt_labels = torch.cat([t["labels"] for t in targets], dim=0)
#     #         combined_labels = torch.cat([flatten_labels.reshape(-1), gt_labels], dim=0)
#     #         gt_ious = torch.ones(num_gts, device=device)
#     #         combined_ious = torch.cat([flatten_ious.reshape(-1), gt_ious], dim=0)

#     #         # 6. 对比学习分支 (仅对正样本)
#     #         pos_mask = (combined_labels != -1).reshape(-1)
#     #         pos_feats = flatten_feats[pos_mask]
#     #         matched_sim_logits = None
#     #         if pos_feats.shape[0] > 0:
#     #             v_proj = F.normalize(self.v_to_t_projection(pos_feats), dim=-1)
#     #             t_norm = F.normalize(self.target_text_feats, dim=-1)
#     #             matched_sim_logits = torch.matmul(v_proj, t_norm.t()) / 0.07

#     #         # grpo_data = self.sample_grpo_features(memory_map, outputs, indices, targets, G=32)
#     #         return {
#     #             "matched_vpe_logits": flatten_vpe_logits,
#     #             "matched_labels": combined_labels,
#     #             "matched_ious": combined_ious,
#     #             "matched_quality": combined_quality,
#     #             "matched_sim_logits": matched_sim_logits,
#     #             "num_boxes": num_boxes + num_gts,
#     #             # **grpo_data
#     #         }
#     #     else:
#     #         full_batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1, 1)
#     #         flat_boxes = pred_boxes.reshape(-1, 4)
            
#     #         boxes_xyxy = box_cxcywh_to_xyxy(flat_boxes)
#     #         boxes_xyxy[:, [0, 2]] *= m_w
#     #         boxes_xyxy[:, [1, 3]] *= m_h
#     #         infer_rois = torch.cat([full_batch_idx.float(), boxes_xyxy], dim=1)
            
#     #         # 得到 VPE 分支的预测分数
#     #         box_features = self.vpe(memory_map, infer_rois, pred_boxes)
#     #         new_logits = self.cls_head(box_features).view(B, N, -1) + quality_scores

#     #         #  获取 D-FINE 原生 Logits (已包含类别分数 + 质量分数)
#     #         orig_logits = outputs['pred_logits']

#     #         # 让 VPE 作为一个“视觉修正项”存在
#     #         w = 1 
#     #         combined_logits = (1 - w) * orig_logits + w * new_logits
            
#     #         # 5. 更新并返回
#     #         outputs['pred_logits'] = combined_logits
#     #         return outputs

#     # #用VPE预测的类别，然后匹配  (再修改一下，加上原本预测类别的output)   发现这里的GT没有+output——已解决
#     def forward(self, feats, outputs, targets=None):
#         pred_boxes = outputs['pred_boxes'] # [B, 300, 4] (cxcywh)
#         quality_scores = outputs['quality_score'] # [B, 300, 1]
#         output = outputs['output']
#         device = pred_boxes.device
#         B, N, _ = pred_boxes.shape
#         memory_map = feats[-2] 
#         _, _, m_h, m_w = memory_map.shape

#         # --- 步骤 1: 无论训练还是推理，先计算 pred_boxes 的 VPE Logits ---
#         # 构造推理/匹配用的 RoIs
#         pred_batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1, 1)
#         flat_boxes_cxcywh = pred_boxes.reshape(-1, 4)
#         flat_pred_boxes_xyxy = box_cxcywh_to_xyxy(flat_boxes_cxcywh)
        
#         # 缩放坐标到特征图尺度
#         rois_pred_xyxy = flat_pred_boxes_xyxy.clone()
#         rois_pred_xyxy[:, [0, 2]] *= m_w
#         rois_pred_xyxy[:, [1, 3]] *= m_h
#         flatten_rois_pred = torch.cat([pred_batch_idx.float(), rois_pred_xyxy], dim=1)

#         # 提取特征并分类
#         pred_vpe_feats = self.vpe(memory_map, flatten_rois_pred, None) 
#         # [B, N, Num_Classes]
#         pred_vpe_logits = self.cls_head(pred_vpe_feats).view(B, N, -1) 
#         # pred_vpe_logits = self.cls_head(pred_vpe_feats + output.reshape(-1, 256)).view(B, N, -1) 
        
#         # fusion_feat = self.fusion(output.reshape(-1, 256), pred_vpe_logits)
#         # pred_vpe_logits = self.cls_head(fusion_feat).view(B, N, -1) 

#         # 融合质量分数作为最终匹配/预测的依据
#         refined_logits = pred_vpe_logits + quality_scores 

#         if self.training:
#             # --- 步骤 2: 使用重新预测的 Logits 进行匹配 ---
#             outputs_for_matcher = {
#                 'pred_logits': refined_logits, # 使用 VPE 修正后的分数
#                 'pred_boxes': pred_boxes
#             }
#             indices = self.matcher(outputs_for_matcher, targets)["indices"]
            
#             # 计算 num_boxes (保持原逻辑)
#             num_boxes = sum(len(t["labels"]) for t in targets)
#             num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
#             if is_dist_available_and_initialized():
#                 torch.distributed.all_reduce(num_boxes)
#             num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#             # --- 步骤 3: 构造 GT RoIs 并与 Pred Feats 合并 ---
#             gt_boxes_list = [t["boxes"] for t in targets]
#             num_gts_per_batch = [len(b) for b in gt_boxes_list]
#             all_gt_boxes = torch.cat(gt_boxes_list, dim=0)
#             flat_gt_boxes_xyxy = box_cxcywh_to_xyxy(all_gt_boxes).clamp(0.0, 1.0)
#             num_gts = all_gt_boxes.shape[0]

#             rois_gt_xyxy = flat_gt_boxes_xyxy.clone()
#             rois_gt_xyxy[:, [0, 2]] *= m_w
#             rois_gt_xyxy[:, [1, 3]] *= m_h
            
#             gt_batch_idx = torch.cat([torch.full((n, 1), i, device=device) for i, n in enumerate(num_gts_per_batch)], dim=0)
#             flatten_rois_gt = torch.cat([gt_batch_idx.float(), rois_gt_xyxy], dim=1)

#             flatten_labels = torch.full((B, N), -1, dtype=torch.long, device=device)
#             flatten_ious = torch.zeros((B, N), device=device)
#             gt_output_list = []
#             for i in range(B):
#                 src_idx, tgt_idx = indices[i]
#                 if len(src_idx) > 0:
#                     flatten_labels[i, src_idx] = targets[i]["labels"][tgt_idx]
#                     p_boxes = flat_pred_boxes_xyxy.view(B, N, 4)[i, src_idx]
#                     t_boxes = box_cxcywh_to_xyxy(targets[i]["boxes"][tgt_idx])
#                     ious = torch.diag(box_iou(p_boxes, t_boxes)[0])
#                     # ious = torch.diag(generalized_box_iou(p_boxes, t_boxes))
#                     flatten_ious[i, src_idx] = ious
#                     gt_output_list.append(output[i, src_idx]) 
#             gt_output = torch.cat(gt_output_list, dim=0)

#             # 提取 GT 特征
#             gt_vpe_feats = self.vpe(memory_map, flatten_rois_gt, None)
#             gt_vpe_logits = self.cls_head(gt_vpe_feats)
#             # gt_vpe_logits = self.cls_head(gt_vpe_feats+ gt_output)

#             # gt_fusion_feat = self.fusion(gt_output, gt_vpe_feats)
#             # gt_vpe_logits = self.cls_head(gt_fusion_feat)

#             # 合并所有特征和 Logits 用于后续 Loss 计算
#             # flatten_feats: [B*N + num_gts, C]
#             # flatten_feats = torch.cat([pred_vpe_feats, gt_vpe_feats], dim=0)
#             # flatten_vpe_logits = torch.cat([pred_vpe_logits.reshape(-1, pred_vpe_logits.shape[-1]), gt_vpe_logits], dim=0)
#             flatten_feats = pred_vpe_feats
#             flatten_vpe_logits = pred_vpe_logits.reshape(-1, pred_vpe_logits.shape[-1])
            

#             # 质量分数构造
#             flat_pred_quality = quality_scores.reshape(-1, 1)
#             gt_quality = torch.full((num_gts, 1), 0.95, device=device)
#             # combined_quality = torch.cat([flat_pred_quality, gt_quality], dim=0)
#             combined_quality = flat_pred_quality

#             # --- 步骤 4: 构造 Labels 和 IoUs ---
#             gt_labels = torch.cat([t["labels"] for t in targets], dim=0)
#             # combined_labels = torch.cat([flatten_labels.reshape(-1), gt_labels], dim=0)
#             combined_labels = flatten_labels.reshape(-1)
#             gt_ious = torch.ones(num_gts, device=device)
#             # combined_ious = torch.cat([flatten_ious.reshape(-1), gt_ious], dim=0)
#             combined_ious = flatten_ious.reshape(-1)

#             # --- 步骤 5: 对比学习 (仅正样本) ---
#             # pos_mask = (combined_labels != -1).reshape(-1)
#             # pos_feats = flatten_feats[pos_mask]
#             # matched_sim_logits = None
#             # if pos_feats.shape[0] > 0:
#             #     v_proj = F.normalize(self.v_to_t_projection(pos_feats), dim=-1)
#             #     t_norm = F.normalize(self.target_text_feats, dim=-1)
#             #     matched_sim_logits = torch.matmul(v_proj, t_norm.t()) / 0.07
            
#             #----对比学习----
#             sim_logits = None
#             v_proj = F.normalize(self.v_to_t_projection(flatten_feats), dim=-1)
#             t_norm = F.normalize(self.target_text_feats, dim=-1)
#             sim_logits = torch.matmul(v_proj, t_norm.t()) / 0.07
    

#             # grpo_data = self.sample_grpo_features(memory_map, outputs, indices, targets, G=32)
            
#             return {
#                 "matched_vpe_logits": flatten_vpe_logits,
#                 "matched_labels": combined_labels,
#                 "matched_ious": combined_ious,
#                 "matched_quality": combined_quality,
#                 # "matched_sim_logits": matched_sim_logits,
#                 "sim_logits": sim_logits,
#                 "num_boxes": num_boxes,
#                 # "num_boxes": num_boxes + num_gts,
#                 # **grpo_data
#             }
#         else:
#             # 推理模式：直接使用前面算好的 refined_logits
#             outputs['pred_logits'] = refined_logits
#             return outputs


#     #(这里的分类没有加上output)
#     @torch.no_grad()
#     def predict_refine(self, feats, post_results, w=1):
#         """
#         直接利用归一化坐标进行特征提取，无需 orig_target_sizes
#         """
#         memory_map = feats[-2] # [B, C, H, W]
#         B, C, m_h, m_w = memory_map.shape
#         device = memory_map.device

#         all_rois = []
#         all_boxes_norm = []
#         all_qualities = []
        
#         for i, res in enumerate(post_results):
#             # 1. 获取归一化框 [num_top, 4] (cx, cy, w, h)
#             b_norm = res['boxes_norm']
#             det_quality = res['quality_score']  # [300, 1]
            
#             # 2. 转换为 xyxy 归一化格式
#             b_norm_xyxy = box_cxcywh_to_xyxy(b_norm)
            
#             # 3. 缩放到特征图的像素尺度
#             # 直接乘以特征图的长宽，这比用原图尺度去缩放要准得多，且不依赖外部尺寸
#             b_feat_coords = b_norm_xyxy.clone()

#             b_feat_coords[:, [0, 2]] *= m_w
#             b_feat_coords[:, [1, 3]] *= m_h
            
#             # 4. 构造 RoI 格式
#             batch_idx = torch.full((len(b_feat_coords), 1), i, device=device)
#             roi = torch.cat([batch_idx, b_feat_coords], dim=1)
            
#             all_rois.append(roi)
#             all_boxes_norm.append(b_norm)
#             all_qualities.append(det_quality)

#         # 批量执行 VPE
#         flatten_rois = torch.cat(all_rois, dim=0) 
#         flatten_boxes_norm = torch.stack(all_boxes_norm, dim=0) 

#         # 这里的 spatial_scale=1.0，因为坐标已经是特征图像素单位了
#         box_features = self.vpe(memory_map, flatten_rois, flatten_boxes_norm)
#         new_logits = self.cls_head(box_features).view(B, -1, self.num_classes) # [B, num_top, C]
        
#         fused_all_qualities = torch.stack(all_qualities, dim=0)
#         new_scores = torch.sigmoid(new_logits + fused_all_qualities.detach())
        
#         for i in range(B):
#             #epoch 21  26.0
#             labels = post_results[i]['labels']
#             # 提取对应类别的 VPE 分数
#             vpe_scores = new_scores[i].gather(1, labels.unsqueeze(-1)).squeeze(-1)
#             # 融合分数
#             post_results[i]['scores'] = (1 - w) * post_results[i]['scores'] + w * vpe_scores
            
#             #epoch 21 0.05
#             # fused_scores = new_scores[i]
#             # vpe_max_scores, vpe_max_labels = fused_scores.max(dim=-1)
#             # post_results[i]['scores'] = vpe_max_scores
#             # post_results[i]['labels'] = vpe_max_labels

#             #epoch 21  28.4
#             # refined_full_scores = new_scores[i] * post_results[i]['scores'].unsqueeze(-1)
#             # vpe_max_scores, vpe_max_labels = refined_full_scores.max(dim=-1)
#             # post_results[i]['scores'] = vpe_max_scores
#             # post_results[i]['labels'] = vpe_max_labels
            
#             # post_results[i].pop('boxes_norm', None)
#             # post_results[i].pop('query_index', None)
#         return post_results

#     def get_losses(self, outputs, ref_cls_outputs=None):
#         num_boxes = outputs["num_boxes"]
#         losses = {}
#         # 如果当前 batch 一个正样本都没有，返回 0
#         if outputs["matched_labels"] is None:
#             zero = torch.tensor(0.0, device=self.cls_head.weight.device)
#             return {"loss_vfl": zero, "loss_contrast": zero}

#         losses.update(self.loss_labels_matched(outputs, num_boxes))
        
#         # 只针对正样本的 Contrastive Loss
#         # losses.update(self.loss_contrast_matched(outputs, num_boxes))
        
#         # if "grpo_logits" in outputs:
#         #     grpo_loss_dict = self.grpo_loss_v1(outputs, num_boxes, ref_grpo_logits=ref_cls_outputs)
#         #     losses.update(grpo_loss_dict)

#         losses = {
#             k: losses[k] * self.weight_dict[k] for k in losses if k in self.weight_dict
#         }
#         losses = {k + "_vpe_cls": v for k, v in losses.items()}

#         return losses

#     #全量训练
#     def loss_labels_matched(self, outputs, num_boxes):
#         src_logits = outputs["matched_vpe_logits"] # [4800, 10]
#         target_labels = outputs["matched_labels"] # [4800]
#         ious = outputs["matched_ious"]             # [4800]
#         quality = outputs["matched_quality"]       # [4800, 1]

#         if quality.dim() == 1:
#             quality = quality.unsqueeze(-1)
#         fused_logits = src_logits + quality.detach()

#         target_classes = target_labels.clone()
#         target_classes[target_classes == -1] = self.num_classes 
#         # target 形状为 [4800, 10]
#         target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()

#         # 构造 Target Score (IoU 软标签)
#         target_score = target * ious.view(-1, 1)

#         pred_score = F.sigmoid(fused_logits).detach()
#         weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

#         loss = F.binary_cross_entropy_with_logits(
#             fused_logits, target_score, weight=weight, reduction="none"
#         )

#         return {"loss_cls": loss.sum() / num_boxes}

#     # def loss_labels_matched(self, outputs, num_boxes):
#     #     src_logits = outputs["matched_vpe_logits"]
#     #     target_labels = outputs["matched_labels"]
#     #     ious = outputs["matched_ious"]
#     #     quality = outputs["matched_quality"]

#     #     fused_logits = src_logits + quality

#     #     # 先处理类别，负样本(-1)在这里会变成全0
#     #     target_classes = target_labels.clone()
#     #     target_classes[target_classes == -1] = self.num_classes # 设为临时类别
#     #     target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()

#     #     # 2. 构造 Target Score [K, 10] (IoU 软标签)
#     #     target_score = torch.zeros_like(target)
#     #     pos_mask = (target_labels != -1)
#     #     if pos_mask.any():
#     #         target_score[pos_mask] = target[pos_mask] * ious.unsqueeze(-1)

#     #     pred_score = F.sigmoid(fused_logits).detach()

#     #     weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
#     #     # target_score = target_score.pow(self.gamma)
#     #     # if self.mal_alpha != None:
#     #     #     weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
#     #     # else:
#     #     #     weight = pred_score.pow(self.gamma) * (1 - target) + target

#     #     # 计算带权重的 BCE
#     #     loss = F.binary_cross_entropy_with_logits(
#     #         fused_logits, target_score, weight=weight, reduction="none"
#     #     )

#     #     return {"loss_cls": loss.sum() / num_boxes}

#     def loss_contrast_matched(self, outputs, num_boxes):
#         # sim_logits = outputs["matched_sim_logits"] # 只有正样本 [Total_Pos, 10]
#         sim_logits = outputs["sim_logits"] # [Total_All, 10]
#         all_targets = outputs["matched_labels"]    # 包含正负样本 [Total_All]
#         pos_mask = (all_targets != -1)
#         neg_mask = ~pos_mask
#         pos_targets = all_targets[pos_mask] # [Total_Pos]

#         # sim_logits 已经是 [N, 10]，pos_targets 是 [N]，每个值是 0-9
#         # loss = F.cross_entropy(sim_logits, pos_targets, reduction="sum")
#         loss_pos = F.cross_entropy(sim_logits[pos_mask], pos_targets, reduction="sum")

#         #希望背景特征与任何文本类别的相似度都尽可能低
#         if neg_mask.any():
#             # 这本质上是在最小化背景样本被分到任何一类的概率
#             loss_neg = torch.logsumexp(sim_logits[neg_mask], dim=1).mean()
#         else:
#             loss_neg = 0.0
            
#         # 使用 num_boxes 进行归一化
#         # return {"loss_contrast": loss / num_boxes}
#         return {"loss_contrast": (loss_pos + 0.5 * loss_neg) / num_boxes}

#     def grpo_loss_v1(self, grpo_data, num_boxes, ref_grpo_logits=None):

#         """
#         Args:
#             grpo_data: sample_grpo_features 返回的字典
#                 - grpo_logits: [Total_M, G, C]
#                 - grpo_ious:   [Total_M, G]
#                 - grpo_labels: [Total_M]
#             num_boxes: 归一化因子
#             ref_grpo_logits: 参考模型的 Logits, [Total_M * G, C]
#         """
#         final_losses = {}
#         logits_g = grpo_data["grpo_logits"]
    
#         if logits_g is None or logits_g.numel() == 0:
#             return {"loss_rl": logits_g.sum() * 0.0} # 保持梯度流的 0 损失

#         Total_M, G, C = logits_g.shape
#         device = logits_g.device

#         grpo_advantage_weight = 1
#         grpo_beta = 0.03
#         epsilon = 1e-9

#         #准备标签 [Total_M, G]
#         tgt_labels = grpo_data["grpo_labels"].view(-1, 1).expand(-1, G)
#         curr_ious = grpo_data["grpo_ious"] # [Total_M, G]
#         grpo_cost_bbox = grpo_data["grpo_cost_bbox"]

#         #计算 Reward (分类准确性 + IoU 拉动力)
#         # 计算每个采样框的负交叉熵作为原始 Reward
#         # individual_loss = F.cross_entropy(
#         #     logits_g.reshape(-1, C),
#         #     tgt_labels.reshape(-1),
#         #     reduction='none'
#         # ).view(Total_M, G)
        
#         # individual_loss = F.binary_cross_entropy_with_logits(
#         #     logits_g, 
#         #     F.one_hot(tgt_labels, num_classes=C).float(), 
#         #     reduction='none'
#         # ).sum(dim=-1)
       
#         # # Reward 设计：分类越准越好，IoU 越高越好
#         # # rewards = -individual_loss + 2.0 * curr_ious
#         # rewards = -individual_loss

#         #负（类别cost+IOU）作为奖励
#         prob = torch.sigmoid(logits_g)
#         tgt_prob = prob.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)  # [Total_M, G]
#         neg_cost_class = (
#             (1 - 0.25) * (tgt_prob**2.0) * (-(1 - tgt_prob + 1e-8).log())
#         )
#         pos_cost_class = (
#             0.25 * ((1 - tgt_prob) ** 2.0) * (-(tgt_prob + 1e-8).log())
#         )
#         cost = 2*(pos_cost_class - neg_cost_class) - 2*curr_ious + 5*grpo_cost_bbox
#         cost = torch.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=-1e6)
#         rewards = -cost

#         # rewards = compute_robust_grpo_reward(logits_g, grpo_data["grpo_labels"], grpo_data["grpo_ious"])

#         # 组内标准化 (Advantage) - 核心 GRPO 逻辑
#         mean_r = rewards.mean(dim=1, keepdim=True)
#         std_r = rewards.std(dim=1, keepdim=True, unbiased=False)
#         advantages = (rewards - mean_r) / (std_r + epsilon) # [Total_M, G]
#         advantages = torch.clamp(advantages, -1.5, 1.5).detach()

#         curr_probs = None
#         #  计算当前策略的 log 概率 [Total_M, G]
#         # logp = F.logsigmoid(logits_g)
#         logp = F.log_softmax(logits_g, dim=-1)
#         logp = logp.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)
#         curr_probs = logp.exp()
#         if ref_grpo_logits is not None:
#             ref_grpo_logits = ref_grpo_logits.view(Total_M, G, C)
#             ref_logp = F.log_softmax(ref_grpo_logits.detach(), dim=-1)
#             # ref_logp = F.logsigmoid(ref_grpo_logits.detach())
#             ref_logp = ref_logp.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)

#             p = logp.exp()
#             pref = ref_logp.exp()
#             curr_probs = p / (pref + epsilon)
#             kl = (pref + epsilon) / (p + epsilon) - torch.log(pref + epsilon) + torch.log(p + epsilon) - 1
#             # kl = torch.exp(ref_logp) * (ref_logp - logp)
#         else:
#             kl = torch.zeros_like(logp)

#         # #logits_g 维度应该是 [Total_M, G, C]
#         # max_logits_g, _ = logits_g.max(dim=-1) 
#         # logp = F.logsigmoid(max_logits_g) # 维度变为 [Total_M, G]

#         # if ref_grpo_logits is not None:
#         #     ref_grpo_logits = ref_grpo_logits.view(Total_M, G, C)            
#         #     ref_max_logits, _ = ref_grpo_logits.detach().max(dim=-1)
#         #     ref_logp = F.logsigmoid(ref_max_logits)
            
#         #     p = logp.exp()
#         #     pref = ref_logp.exp()
            
#         #     # 这里的计算逻辑是：对比“这个位置是物体的概率”在两个模型间的偏移
#         #     kl = (pref + epsilon) / (p + epsilon) - torch.log(pref + epsilon) + torch.log(p + epsilon) - 1
#         # else:
#         #     kl = torch.zeros_like(logp)

#         # 计算 Policy Loss
#         # GRPO 的核心：优势函数越大，logp 应该越大；KL 散度作为惩罚
#         element_loss = -(logp * advantages.detach()) * grpo_advantage_weight + grpo_beta * kl

#         # print("1",advantages)
#         # print("2",kl)
#         # print("3",element_loss)

#         #  归一化与返回
#         # loss_rl = element_loss.mean(dim=1).sum() * G / num_boxes
#         loss_rl = element_loss.sum() / (num_boxes + 1e-6)
#         final_losses["loss_rl"] = loss_rl

#         return final_losses 

#     @torch.no_grad()
#     def compute_robust_grpo_reward(logits_g, target_labels, curr_ious):
#         """
#         逻辑：在组内强制制造差异化，让分类得分的顺序必须对齐 IoU 的顺序
#         """
#         M, G = curr_ious.shape
#         device = logits_g.device
        
#         # 计算分类概率 
#         probs = torch.sigmoid(logits_g)
#         tgt_probs = probs.gather(2, target_labels.view(M, 1, 1).expand(-1, G, 1)).squeeze(-1)
        
#         # IoU 排名奖励 
#         # 将 IoU 转换为 0-1 之间的排名得分，消除 IoU 数值太接近的问题
#         # iou_rank: [M, G], 排名第一的框值为 1.0, 排名最后的为 0.0
#         iou_rank = torch.argsort(torch.argsort(curr_ious, dim=1, descending=False), dim=1).float() / (G - 1)
        
#         #  计算“得分-IoU”一致性奖励
#         # 如果一个框 IoU 排名高，但 prob 排名低，给予惩罚
#         prob_rank = torch.argsort(torch.argsort(tgt_probs, dim=1, descending=False), dim=1).float() / (G - 1)
#         consistency_reward = 1.0 - torch.abs(iou_rank - prob_rank)
        
#         # 4. 组合最终 Reward
#         # 我们不仅看 prob 的绝对值，更看它在组内的相对排名
#         reward = 1.0 * tgt_probs + 1.0 * iou_rank + 1.0 * consistency_reward
        
#         return reward
   
#     def sample_grpo_features(self, memory_map, outputs, indices, targets, G=32):
#         """
#         无循环批量化 GRPO 采样
#         """
#         device = memory_map.device
#         B, _, m_h, m_w = memory_map.shape
        
#         # 1. 提取所有图片的正样本索引，合并为全 Batch 展平索引
#         # batch_idx_map: [Total_M], src_idx_map: [Total_M], tgt_idx_map: [Total_M]
#         batch_idx_map = torch.cat([torch.full((len(indices[i][0]),), i, device=device) for i in range(B)])
#         src_idx_map = torch.cat([indices[i][0].to(device) for i in range(B)])
#         tgt_idx_map = torch.cat([indices[i][1].to(device) for i in range(B)])

#         if len(src_idx_map) == 0:
#             return {"grpo_logits": None, "grpo_ious": None, "grpo_labels": None}

#         Total_M = len(src_idx_map)

#         # 2. 批量采样偏移量 [Total_M, 4, G]
#         # 从 pred_corners 提取对应的分布
#         dist_prob = F.softmax(outputs['pred_corners'][batch_idx_map, src_idx_map].float(), dim=-1)
#         # 展平进行 multinomial: [Total_M * 4, reg_max + 1] -> [Total_M * 4, G]
#         sampled_indices = torch.multinomial(dist_prob.reshape(-1, self.reg_max + 1), G, replacement=True)
#         sampled_offsets = sampled_indices.view(Total_M, 4, G)

#         # 3. 转换为像素尺度的 RoIs [Total_M * G, 5]
#         # 提取对应的参考点 [Total_M, 2]
#         ref_points = outputs['ref_points'][batch_idx_map, src_idx_map]
        
#         rois_g, bboxes_g_cxcywh = self.convert_to_rois_batch(
#             ref_points, sampled_offsets, batch_idx_map, G, m_h, m_w
#         )

#         # 4. 批量 VPE 推理 [Total_M * G, num_classes]
#         # memory_map 依然是 [B, C, H, W]，rois_g 里的 batch_idx 会自动对应
#         bboxes_g_normalized = bboxes_g_cxcywh.view(Total_M, G, 4)
#         box_features = self.vpe(memory_map, rois_g, bboxes_g_normalized)  #[Total_M*G, C]

#         # 取对应 query feature
#         matched_queries = outputs["output"][batch_idx_map, src_idx_map]  # [Total_M,256]

#         # 扩展到 G
#         matched_queries_expand = (
#             matched_queries.unsqueeze(1)
#             .expand(-1, G, -1)
#             .reshape(-1, matched_queries.shape[-1])
#         )  # [Total_M*G,256]

#         grpo_logits = self.cls_head(box_features+ matched_queries_expand).view(Total_M, G, -1)
#         # fusion_feat = self.fusion(matched_queries_expand, box_features)
#         # grpo_logits = self.cls_head(fusion_feat).view(Total_M, G, -1)

#         # 5. 计算奖励 IoU [Total_M, G]
#         # 获取对应的 GT 框并扩展
#         all_tgt_labels = torch.cat([targets[i]["labels"][indices[i][1]] for i in range(B)])
#         all_tgt_boxes = torch.cat([targets[i]["boxes"][indices[i][1]] for i in range(B)]) # [Total_M, 4]
        
#         target_boxes_expand = all_tgt_boxes.unsqueeze(1).expand(-1, G, -1).reshape(-1, 4)
#         sampled_boxes_flat = bboxes_g_cxcywh.view(-1, 4)

#         #L1损失
#         cost_bbox = torch.cdist(sampled_boxes_flat, target_boxes_expand, p=1)
#         grpo_cost_bbox = cost_bbox.diag().view(Total_M, G)  # 对角线就是每个采样框对应的GT
        
#         # 计算成对 IoU 并取对角线
#         ious = generalized_box_iou(box_cxcywh_to_xyxy(sampled_boxes_flat), 
#                        box_cxcywh_to_xyxy(target_boxes_expand))
#         grpo_ious = torch.diag(ious).view(Total_M, G)
#         num_iou_gt_05 = (grpo_ious > 0.5).sum()

#         return {
#             "grpo_logits": grpo_logits,
#             "grpo_ious": grpo_ious,
#             "grpo_cost_bbox": grpo_cost_bbox,
#             "grpo_labels": all_tgt_labels,
#             "grpo_feats": box_features+matched_queries_expand   #输入参考模型
#         }
 
#     def convert_to_rois_batch(self, ref_points, sampled_offsets, batch_idx_map, G, m_h, m_w):
#         """
#         ref_points: [Total_M, 2]
#         sampled_offsets: [Total_M, 4, G]
#         batch_idx_map: [Total_M] 记录每个正样本属于哪张图
#         """
#         Total_M = ref_points.shape[0]

#         # 1. 投影偏移量值
#         flat_offsets = sampled_offsets.flatten() # [Total_M * 4 * G]
#         dist_values = self.project[flat_offsets]
#         dist = dist_values.view(Total_M, 4, G).permute(0, 2, 1) # [Total_M, G, 4]
        
#         # 2. 扩展参考点并转换
#         # 限制偏移量最大不超过 0.5 (即半张图)
#         dist = torch.clamp(dist, min=-0.5, max=0.5)
#         ref_points_g = ref_points.unsqueeze(1).expand(-1, G, -1) # [Total_M, G, 2]
#         bboxes_g_cxcywh = distance2bbox(ref_points_g, dist, self.reg_scale)

#         # 4. 准备 RoIAlign 输入 [Total_M * G, 5]
#         bboxes_g_xyxy = box_cxcywh_to_xyxy(bboxes_g_cxcywh)

#         x1, y1, x2, y2 = bboxes_g_xyxy.unbind(-1)
#         x2 = torch.max(x2, x1 + 1e-4)
#         y2 = torch.max(y2, y1 + 1e-4)
#         # 哪怕 x1, x2 都被按回了 1.0，roi_align 也能处理（会采样边界像素）
#         bboxes_g_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)
#         bboxes_g_cxcywh = box_xyxy_to_cxcywh(bboxes_g_xyxy)

#         bboxes_g_xyxy[..., [0, 2]] *= m_w
#         bboxes_g_xyxy[..., [1, 3]] *= m_h
        
#         # 构造对应的 batch 索引
#         # batch_idx_map 是 [Total_M], 扩展到 [Total_M, G]
#         full_batch_idx = batch_idx_map.unsqueeze(1).expand(-1, G).reshape(-1, 1)
        
#         rois = torch.cat([full_batch_idx.float(), bboxes_g_xyxy.reshape(-1, 4)], dim=-1)
        
#         return rois, bboxes_g_cxcywh

#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

# class VisualPromptEncoder(nn.Module):
#     def __init__(self, hidden_dim, nhead=4, num_levels=1):
#         super().__init__()
#         self.hidden_dim = hidden_dim
        
#         # 坐标编码层：将 BBox (x, y, w, h) 映射到高维空间
#         self.visual_bbox_encoding = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim), # *2 是因为 sine 包含 sin/cos
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
        
#         # 为了轻量化，保留 ROI 思想，但在后面加入 Self-Attention 建模
#         self.roi_size = 7
#         self.visual_net = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1)
#         )

#         # 全局关系建模：让不同的 Visual Prompt 之间产生交互
#         self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        
#         # 前馈网络与归一化
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )

#     def forward(self, memory_map, rois, bboxes_normalized):
#         """
#         Args:
#             memory_map: [B, C, H, W] 特征图
#             rois: [B*N, 5] (batch_idx, x1, y1, x2, y2) 像素尺度坐标
#             bboxes_normalized: [B, N, 4] (cx, cy, w, h) 归一化坐标，用于生成 Positional Embedding
#         """
#         B, N, _ = bboxes_normalized.shape
        
#         # ---  视觉特征提取 (ROI-based) ---
#         roi_feat = roi_align(memory_map, rois, output_size=self.roi_size, spatial_scale=1.0, aligned=True)
#         visual_features = self.visual_net(roi_feat).view(B, N, self.hidden_dim) # [B, N, C]

#         # --- 坐标位置编码 (Position Embedding) ---
#         # pos_embed: 使用 sine 编码将 [cx, cy, w, h] 映射到 hidden_dim
#         pos_embed = self.gen_sine_embed_for_bbox(bboxes_normalized) # [B, N, C]
#         query_pos = self.visual_bbox_encoding(pos_embed)

#         # 这里的 query = visual_features + query_pos
#         q = k = visual_features + query_pos
#         v = visual_features
        
#         # 自注意力建模
#         attn_output, _ = self.self_attn(q, k, value=v)
#         x = self.norm1(visual_features + attn_output)

#         # --- FFN ---
#         x = x + self.ffn(x)
#         fused_feat = self.norm2(x)

#         return fused_feat # [B, N, C]

#     def gen_sine_embed_for_bbox(self, bboxes, temperature=10000):
#         """
#         Args:
#             bboxes: [B, N, 4] 归一化的 (cx, cy, w, h)
#         Returns:
#             pos_embed: [B, N, hidden_dim * 2] 用于送入 visual_bbox_encoding
#         """

#         num_pos_feats = self.hidden_dim // 2 # 映射后的单分量维度 (sin + cos)
        
#         # 计算频率尺度
#         dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32, device=bboxes.device)
#         dim_t = temperature ** (2 * dim_t / (num_pos_feats // 2))

#         # 将 bboxes 扩展并除以频率
#         # bboxes[..., None] 形状: [B, N, 4, 1] ,dim_t 形状: [num_pos_feats // 2]
#         pos = bboxes[..., None] * 2 * torch.pi / dim_t # 归一化到 2pi 周期

#         # 形状变为 [B, N, 4, num_pos_feats // 2, 2] -> [B, N, 4, num_pos_feats]
#         pos_embed = torch.stack((pos.sin(), pos.cos()), dim=-1).flatten(2)
        
#         # 对应你 init 里的 nn.Linear(hidden_dim * 2, hidden_dim)
#         return pos_embed

# class SimpleVisualPromptEncoder(nn.Module):

#     def __init__(self, hidden_dim):
#         super().__init__()
#         # 投影层：将 D-FINE 的特征维度对齐（如果 hidden_dim 不一致）
#         self.reduction = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
#         # 特征提炼
#         self.refine = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(32, hidden_dim), 
#             nn.ReLU(inplace=True),
#         )
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.ln = nn.LayerNorm(hidden_dim)

#     def forward(self, memory_map, rois, bboxes_normalized):
#         # print(f"ROIs max: {rois.max()}, min: {rois.min()}, has_nan: {torch.isnan(rois).any()}")
#         x = roi_align(memory_map, rois, output_size=7, spatial_scale=1.0, aligned=True)
        
#         #局部感知
#         x = self.reduction(x)
#         # print(f"Reduction Weight Max: {self.reduction.weight.abs().max()}")
#         # if torch.isnan(self.reduction.weight).any():
#         #     print("!!! Reduction weight is already NaN !!!")
#         x = x + self.refine(x) # 残差连接：保留原始特征，只学习修正
#         #转化为向量
#         x = self.pool(x).flatten(1) # [B*N, C]
#         # 最终归一化，为分类头准备
#         x = self.ln(x) 
#         return x

# #微微改动，但效果更差
# class SimpleVisualPromptEncoder(nn.Module):
#     def __init__(self, hidden_dim, roi_size=7):
#         super().__init__()
#         self.roi_size = roi_size

#         # channel align
#         self.reduction = nn.Conv2d(hidden_dim, hidden_dim, 1)

#         # spatial refinement
#         self.refine = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
#             nn.GroupNorm(16, hidden_dim),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
#             nn.GroupNorm(16, hidden_dim),
#             nn.ReLU(inplace=True),
#         )

#         # spatial compression
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         # semantic projection
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.ln = nn.LayerNorm(hidden_dim)

#     def forward(self, memory_map, rois, bboxes_normalized=None):

#         x = roi_align(
#             memory_map,
#             rois,
#             output_size=self.roi_size,
#             spatial_scale=1.0,
#             aligned=True
#         )  # [N, C, 7, 7]

#         x = self.reduction(x)
#         x = x + self.refine(x)
#         x = self.pool(x).flatten(1)
#         x = self.fc(x)
#         x = self.ln(x)
#         return x

# class ClassificationHead(nn.Module):
#     def __init__(self, hidden_dim, num_classes):
#         super().__init__()
#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x):
#         return self.head(x)   

# class ClassificationHead(nn.Module):
#     def __init__(self, hidden_dim, num_classes):
#         super().__init__()
#         self.head = nn.Linear(hidden_dim, num_classes)
        
#         prior_prob = 0.01
#         # 根据 sigmoid 反函数计算 bias: bias = -log((1 - p) / p)
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
        
#         # 初始化权重为很小的随机值
#         nn.init.normal_(self.head.weight, std=0.01)
#         # 初始化偏置，使模型初始输出的概率接近 0.01
#         nn.init.constant_(self.head.bias, bias_value)

#     def forward(self, x):
#         return self.head(x)

# class ROIQueryFusion(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.Sigmoid()
#         )
#         self.proj = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.LayerNorm(hidden_dim)
#         )
#     def forward(self, query_feat, roi_feat):
#         x = torch.cat([query_feat, roi_feat], dim=-1)
#         gate = self.gate(x)
#         fused = query_feat + gate * roi_feat
#         fused = self.proj(fused)
#         return fused
