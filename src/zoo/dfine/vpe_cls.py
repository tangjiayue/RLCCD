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

        # # ========== 自适应特征清创参数==========
        # self.use_debridement=True                  # 是否启用
        # self.debridement_xi_range=(0.001, 0.01)          # 扰动幅度范围
        # self.debridement_sample_ratio=0.15          # 采样比例
        # self.debridement_tau=0.1                  # 温度参数
        # self.debridement_reinit_interval=9360        # 重新初始化分类层的间隔  478*20

        # ========== 门控融合参数 (Gating Fusion) ==========
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        self.fusion_cls_head = ClassificationHead(hidden_dim, num_classes)

        # ==================== 类别原型库 ====================
        # 可学习或随机初始化的全局原型
        self.register_buffer("class_prototypes", torch.randn(num_classes, hidden_dim))
        nn.init.normal_(self.class_prototypes, std=0.02)
        # 初始化时归一化
        with torch.no_grad():
            self.class_prototypes.copy_(F.normalize(self.class_prototypes, dim=-1))
        # 动量更新系数
        self.proto_momentum = 0.9

    def forward(self, feats, outputs, targets=None):
        #========开始
        pred_boxes = outputs['pred_boxes'] # [B, 300, 4] (cxcywh)
        quality_scores = outputs['quality_score'] # [B, 300, 1]
        query_feats = outputs["output"] # [B, N, C] D-FINE 解码器输出的 query 特征
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

                #生成投影向量 fr (全连接网络生成)
                fr = self.gate_proj(query_feats) # [B, N, C]                
                Gr = F.sigmoid(fr) # [B, N, C]

                # 融合特征: query + weight * Gr * VPE
                fusion_feats = query_feats + 0.5 * Gr * layer_feats
                fusion_logits = self.fusion_cls_head(fusion_feats)
            else:
                # 中间层使用辅助分类头
                layer_logits = self.cls_head_aux(layer_feats)
            
            all_layer_logits.append(layer_logits)
       
        # # 融合质量分数作为最终匹配/预测的依据
        # refined_logits = pred_vpe_logits + quality_scores
        refined_logits = fusion_logits + quality_scores # 直接使用融合后的分数进行匹配和预测
        refined_logits1 = pred_vpe_logits + quality_scores

        if self.training:
            # #===========================匈牙利匹配===========================
            # outputs_for_matcher = {
            #     'pred_logits': refined_logits, # 使用 VPE 修正后的分数
            #     'pred_boxes': pred_boxes
            # }
            # indices = self.matcher(outputs_for_matcher, targets)["indices"]
            
            outputs_for_matcher = {
                'pred_logits': fusion_logits + quality_scores, 
                'pred_boxes': pred_boxes
            }
            fusion_indices = self.matcher(outputs_for_matcher, targets)["indices"]
            outputs['fusion_logits'] = fusion_logits + quality_scores
            outputs['fusion_feats'] = fusion_feats          # [B, N, C]

            #=====================iou匹配===========================
            # iou一对多匹配版本
            indices, neg_indices = rcnn_iou_match(
                pred_boxes=pred_boxes,
                targets=targets,
                pos_threshold=0.6,   
            )

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

            # #==============自适应特征清创================
            # loss_debridement = torch.tensor(0.0, device=device)
    
            # if self.use_debridement and targets is not None:
            #     # 采样并减少框特征
            #     # result = self.sample_and_reduce_box_features(
            #     #     pred_vpe_feats,   # [B, N, C]
            #     #     pred_vpe_logits,  # [B, N, C]
            #     #     quality_scores,
            #     #     targets,
            #     #     indices
            #     # )
            #     result = self.sample_and_reduce_box_features(
            #         fusion_feats,   # [B, N, C]
            #         fusion_logits,  # [B, N, C]
            #         quality_scores,
            #         targets,
            #         fusion_indices
            #     )
                
            #     if result[0] is not None:
            #         (reduced_features, sampled_labels, misclass_classes, xi, 
            #         (sampled_batch_idx, sampled_query_idx)) = result
                    
            #         # 获取原始特征
            #         # original_features = pred_vpe_feats[sampled_batch_idx, sampled_query_idx]
            #         original_features = fusion_feats[sampled_batch_idx, sampled_query_idx]
                    
            #         # 计算对比损失
            #         loss_debridement = self.compute_debridement_contrastive_loss(
            #             original_features, reduced_features, sampled_labels
            #         )
                    
            #         # 每20步重新初始化分类层
            #         self.reinitialize_classification_layer()
            
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
                # "loss_debridement": loss_debridement,

                "layer_matched_aux": layer_matched_aux,
                "fusion_indices": fusion_indices,
                "targets": targets,

                **contrast_data,
                "num_boxes": self._get_num_boxes(targets, device),
                # **grpo_data,
                "outputs":outputs,  #联调用
                "feats": feats  #多尺度特征
            }
        else:
            # 推理模式：直接使用前面算好的 refined_logits
            # outputs['pred_logits'] = refined_logits
            outputs['pred_logits'] = refined_logits
            outputs['vpe_logits'] = refined_logits1      

            
            if targets is not None:
                # ================= [推理阶段特征与混淆矩阵收集] =================
                try:
                    fake_targets = []
                    for t in targets:
                        boxes_xyxy_norm = t["boxes"] / 640.                     
                        boxes_cxcywh_norm = box_xyxy_to_cxcywh(boxes_xyxy_norm)
                        fake_targets.append({"labels": t["labels"], "boxes": boxes_cxcywh_norm})
       
                    outputs_for_matcher = {
                        'pred_logits': refined_logits, 
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
                    flatten_feats = fusion_feats.reshape(B*N, -1)
                    
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

    def sample_and_reduce_box_features(self, box_features, classifier_logits, quality_score, targets, indices):
        """
        在框特征空间进行采样和减少
        Args:
            box_features: [B, N, C] VPE 提取的框特征
            classifier_logits: [B, N, C] 分类器 logits
            quality_score: [B, N, 1] 质量分数
            targets: list of dict 包含 labels
            indices: 匹配结果 [(src_idx, tgt_idx), ...]
        Returns:
            reduced_features: [M, C] 减少后的框特征
            sampled_labels: [M] 原始标签
            misclass_classes: [M] 误分类目标类别
            xi: [M] 扰动幅度
            sampled_positions: (batch_indices, query_indices)
        """
        B, N, C = box_features.shape
        device = box_features.device
        
        # 收集所有正样本框的特征和标签
        all_pos_features = []
        all_pos_labels = []
        all_pos_batch_idx = []
        all_pos_query_idx = []
        all_pos_quality = []
        
        for i in range(B):
            src_idx, tgt_idx = indices[i]
            if len(src_idx) > 0:
                pos_feats = box_features[i, src_idx]  # [K, C]
                pos_labels = targets[i]["labels"][tgt_idx]  # [K]
                pos_quality = quality_score[i, src_idx]  # [K, 1]
                
                all_pos_features.append(pos_feats)
                all_pos_labels.append(pos_labels)
                all_pos_batch_idx.extend([i] * len(src_idx))
                all_pos_query_idx.extend(src_idx.cpu().tolist())
                all_pos_quality.append(pos_quality) 
        
        if len(all_pos_features) == 0:
            return None, None, None, None, None
        
        all_pos_features = torch.cat(all_pos_features, dim=0)  # [Total_Pos, C]
        all_pos_labels = torch.cat(all_pos_labels, dim=0)      # [Total_Pos]
        all_pos_batch_idx = torch.tensor(all_pos_batch_idx, device=device)
        all_pos_query_idx = torch.tensor(all_pos_query_idx, device=device)
        all_pos_quality = torch.cat(all_pos_quality, dim=0) # [Total_Pos, 1]
        
        # 计算采样概率（基于分类器对真实类别的置信度）
        # 获取每个正样本对真实类别的预测概率
        pos_logits = (classifier_logits)[all_pos_batch_idx, all_pos_query_idx]  # [Total_Pos, C]
        pos_logits = pos_logits + all_pos_quality  # 融合质量分数
        pos_probs = torch.sigmoid(pos_logits)  # [Total_Pos, C]
        correct_probs = pos_probs[torch.arange(len(all_pos_labels)), all_pos_labels]
        
        # 采样概率 = 1 - 正确概率（容易误分类的样本采样概率高）
        sampling_probs = 1 - correct_probs
        sampling_probs = sampling_probs / (sampling_probs.sum() + 1e-8)
        
        # 采样
        radio = self.debridement_sample_ratio
        num_samples = max(1, int(len(all_pos_features) * radio))
        sampled_indices_in_pos = torch.multinomial(sampling_probs, min(num_samples, len(all_pos_features)), replacement=False)
        
        sampled_features = all_pos_features[sampled_indices_in_pos]
        sampled_labels = all_pos_labels[sampled_indices_in_pos]
        sampled_batch_idx = all_pos_batch_idx[sampled_indices_in_pos]
        sampled_query_idx = all_pos_query_idx[sampled_indices_in_pos]
        sampled_quality = all_pos_quality[sampled_indices_in_pos]   # [M, 1]
        
        # 获取误分类目标类别（模型最可能误分类的类别）
        sampled_logits = pos_logits[sampled_indices_in_pos]
        sampled_probs = torch.sigmoid(sampled_logits)
        masked_probs = sampled_probs.clone()
        masked_probs[torch.arange(len(sampled_labels)), sampled_labels] = 0
        misclass_classes = masked_probs.argmax(dim=-1)
        
        # 随机扰动幅度
        xi = torch.empty(len(sampled_features), device=device).uniform_(
            self.debridement_xi_range[0], self.debridement_xi_range[1]
        )
        
        # 在特征空间进行梯度扰动
        feat_cloned = sampled_features.clone().detach().requires_grad_(True)
        
        # 通过分类头（只用分类头，不需要完整前向）
        # logits = self.cls_head(feat_cloned)
        logits = self.fusion_cls_head(feat_cloned)
        misclass_one_hot = F.one_hot(misclass_classes, num_classes=self.num_classes).float()

        quality_expanded = sampled_quality.expand(-1, logits.shape[-1])  # [M, num_classes]
        logits = logits + quality_expanded

        # BCE Loss（不使用 weight 和 target_score，只用硬标签）
        loss = F.binary_cross_entropy_with_logits(logits, misclass_one_hot, reduction='none')
        loss = loss.sum(dim=-1)  # [M]
        
        # 计算梯度
        gradients = torch.autograd.grad(loss.sum(), feat_cloned, create_graph=False, retain_graph=False)[0]
        
        # 特征扰动
        with torch.no_grad():
            reduced_features = feat_cloned + xi.view(-1, 1) * torch.sign(gradients)
        
        reduced_features = reduced_features.detach()

        return (reduced_features, sampled_labels, misclass_classes, xi, 
                (sampled_batch_idx, sampled_query_idx))

        #修改了，加了quality_score
   
    def compute_debridement_contrastive_loss(self, original_features, reduced_features, labels):
        """
        100% 对齐论文 Adaptive Feature Debridement Loss (Eq.3)
        """
        if original_features is None or reduced_features is None:
            return torch.tensor(0.0, device=original_features.device)
        if len(labels) <= 1:
            return torch.tensor(0.0, device=original_features.device)

        v_i = F.normalize(original_features, dim=-1)
        v_i_prime = F.normalize(reduced_features, dim=-1)
        tau = self.debridement_tau

        # 相似度矩阵
        sim = torch.matmul(v_i, v_i.T) / tau  # [M, M]

        # 正样本 mask
        labels = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(v_i.device)
        pos_mask.fill_diagonal_(False)  # 自己不算

        # # ======================
        # # 论文核心：margin 加到 所有 样本上！！
        # # ======================
        # margin = 1.0 - torch.sum(v_i * v_i_prime, dim=-1, keepdim=True)  # [M, 1]
        # sim = sim + margin 

        # Margin 只加到正样本对上
        margin = 1.0 - torch.sum(v_i * v_i_prime, dim=-1)  # [M]
        margin = margin.unsqueeze(1)  # [M, 1]
        sim = sim + margin * pos_mask  # 只加到正样本对

        # 分子：正样本
        pos_exp = torch.exp(sim) * pos_mask
        pos_sum = pos_exp.sum(dim=1)  # [M]

        # 分母：所有样本
        denominator = torch.exp(sim).sum(dim=1)  # [M]

        # 有效样本
        valid = pos_sum > 0
        if not valid.any():
            return torch.tensor(0.0, device=v_i.device)

        loss = -torch.log(pos_sum[valid] / (denominator[valid] + 1e-8))
        return loss.mean()   

    def reinitialize_classification_layer(self):
        """
        按指定间隔重新初始化分类层（论文设置）
        """
        self.debridement_call_counter = getattr(self, 'debridement_call_counter', 0) + 1
        
        # 使用参数，而不是硬编码 20
        if self.debridement_call_counter % self.debridement_reinit_interval == 0:
            # nn.init.normal_(self.cls_head.head.weight, std=0.01)
            nn.init.normal_(self.fusion_cls_head.head.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            # nn.init.constant_(self.cls_head.head.bias, bias_value)
            nn.init.constant_(self.fusion_cls_head.head.bias, bias_value)
            print(f"[Debridement] Step {self.debridement_call_counter}: Reinitialized classification layer")


    def get_losses(self, outputs,  **kwargs):
        num_boxes = outputs["num_boxes"]
        losses = {}
        if self.training:
            # 主分支
            losses_main = self.loss_labels_matched_branch(outputs, num_boxes, branch="main")
            losses.update(losses_main)

            losses.update(self.loss_labels_vfl(outputs["outputs"],outputs["targets"],outputs["fusion_indices"], num_boxes))

            # Contrastive Loss
            losses.update(self.loss_contrast_matched(outputs, num_boxes))

            # S_ref = kwargs.get("S_ref", None)
            # losses.update(self.loss_grqa(outputs, num_boxes, S_ref=S_ref))
            # losses.update(self.loss_img_prototype(outputs))

            #没用
            # if "loss_debridement" in outputs:
            #     losses["loss_debridement"] = outputs["loss_debridement"]

            # if "grpo_logits" in outputs:
            #     ref_cls_outputs = kwargs.get("ref_cls_outputs", None)
            #     # grpo_loss_dict = self.grpo_loss_v1(outputs, num_boxes, ref_grpo_logits=ref_cls_outputs)
            #     grpo_loss_dict = self.grpo_loss_v2(outputs, num_boxes, ref_grpo_logits=ref_cls_outputs)
            #     losses.update(grpo_loss_dict)

            losses = {
                k: losses[k] * self.weight_dict[k] for k in losses if k in self.weight_dict
            }
            losses = {k + "_vpe_cls": v for k, v in losses.items()}

        else:
            # 主分支
            losses_main = self.loss_labels_matched_branch(outputs, num_boxes, branch="main")
            losses.update(losses_main)
            losses = {
                k: losses[k] * self.weight_dict[k] for k in losses if k in self.weight_dict
            }
            losses = {k + "_test_vpe_cls": v for k, v in losses.items()}
        
        return losses

   
    def loss_grqa(self, outputs, num_boxes, S_ref=None):
        feats = outputs["outputs"].get("fusion_feats", None)
        if feats is None or feats.numel() == 0:
            return {"loss_rl": torch.tensor(0.0, device=self.class_prototypes.device, requires_grad=True)}
        
        B, N, C = feats.shape
        K = B * N
        Q_L = feats.view(K, C)
        P = self.class_prototypes.detach()
        
        # 1. 查询-原型对齐奖励
        Q = F.normalize(Q_L, p=2, dim=-1)
        P_norm = F.normalize(P, p=2, dim=-1)
        
        # 计算查询与全局原型相似度矩阵 S
        S_theta = torch.matmul(Q, P_norm.t()) # [K, num_classes]
        
        # 选取相似度最高的类别作为匹配类别，对应相似度作为奖励值
        r_i, c_i = S_theta.max(dim=-1) # [K], [K]
        
        # 2. 组相对优势 (借鉴 GRPO)
        A_i = torch.zeros_like(r_i)
        for c in range(self.num_classes):
            mask = (c_i == c)
            num_in_group = mask.sum()
            if num_in_group > 1:
                group_rewards = r_i[mask]
                mean_g = group_rewards.mean()
                std_g = group_rewards.std(unbiased=False)
                A_i[mask] = (group_rewards - mean_g) / (std_g + 1e-8)
            elif num_in_group == 1:
                A_i[mask] = 0.0  # 单独一个样本算作平均表现，无优势
                
        # 3. GRPO 式裁剪与 KL 稳定正则
        pi_theta = F.softmax(S_theta, dim=-1)
        pi_theta_c = pi_theta.gather(-1, c_i.unsqueeze(-1)).squeeze(-1)
        
        if S_ref is not None:
            if S_ref.dim() > 2:
                S_ref = S_ref.view(K, -1)
            pi_ref = F.softmax(S_ref, dim=-1)
            pi_ref_c = pi_ref.gather(-1, c_i.unsqueeze(-1)).squeeze(-1)
        else:
            pi_ref_c = pi_theta_c.detach()
            
        # 重要性比率
        rho_i = pi_theta_c / (pi_ref_c + 1e-8)
        
        epsilon = 0.1  # 裁剪边界
        surrogate1 = rho_i * A_i
        surrogate2 = torch.clamp(rho_i, 1 - epsilon, 1 + epsilon) * A_i
        loss_gr = -torch.min(surrogate1, surrogate2).mean()
        
        # 前向 KL 散度约束 (Forward KL)
        ratio = pi_ref_c / (pi_theta_c + 1e-8)
        kl_div = (ratio - torch.log(ratio + 1e-8) - 1.0).mean()
        
        beta = 0.001  # 正则化系数
        loss_grqa = loss_gr + beta * kl_div
        
        return {"loss_rl": loss_grqa}
    
    def loss_img_prototype(self, outputs):
        """L_img: 约束单图原型与全局原型距离，减小类内特征方差，并进行EMA更新"""
        if "gt_feats" not in outputs or "gt_labels" not in outputs:
            return {"loss_img_proto": torch.tensor(0.0, device=self.class_prototypes.device)}

        gt_feats = outputs["gt_feats"]                    # [N_gt, D]
        gt_labels = outputs["gt_labels"]                  # [N_gt]
        
        if gt_labels.numel() == 0:
            return {"loss_img_proto": torch.tensor(0.0, device=self.class_prototypes.device)}

        # 过滤背景
        valid_mask = gt_labels != -1
        gt_labels_valid = gt_labels[valid_mask]
        gt_feats_valid = gt_feats[valid_mask]
        
        if gt_labels_valid.numel() == 0:
            return {"loss_img_proto": torch.tensor(0.0, device=self.class_prototypes.device)}

        unique_c = torch.unique(gt_labels_valid)
        loss_img = torch.tensor(0.0, device=self.class_prototypes.device)

        alpha = self.proto_momentum

        for c in unique_c:
            mask = (gt_labels_valid == c)
            feats_c = gt_feats_valid[mask]
            
            # 1. 均值特征并做 L2 归一化 (f_c)
            fc = feats_c.mean(dim=0)
            fc = F.normalize(fc, dim=-1)
            
            # 2. 计算损失： || f_c - P_c ||_2^2
            # P_c 需要 detach，以免更新全局原型时破坏网络图梯度
            Pc = self.class_prototypes[c].detach()
            loss_img += torch.sum((fc - Pc) ** 2)

            # 3. EMA 无梯度更新全局原型库
            if self.training:
                with torch.no_grad():
                    new_Pc = alpha * self.class_prototypes[c] + (1.0 - alpha) * fc.detach()
                    self.class_prototypes[c] = F.normalize(new_Pc, dim=-1)

        # 依据批次中包含的类数计算平均损失
        loss_img = loss_img / max(len(unique_c), 1)

        return {"loss_img_proto": loss_img}

    #vfl
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
        # target 形状为 [4800, 10]
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()

        # 构造 Target Score (IoU 软标签)
        target_score = target * ious.view(-1, 1).pow(0.5)
        # target_score = target * ious.view(-1, 1)

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

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs["fusion_logits"]  #[B,N,C]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype).pow(0.5)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()

        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_fusion_cls": loss}


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
   
    def grpo_loss_v2(self, grpo_data, num_boxes, ref_grpo_logits=None):
        """
        改进版 GRPO 损失，旨在提升 mAP。
        核心思想：利用采样的 IoU 梯度来塑造更好的置信度分布。
        """
        final_losses = {}
        logits_g = grpo_data["grpo_logits"]

        if logits_g is None or logits_g.numel() == 0:
            return {"loss_rl": logits_g.sum() * 0.0}

        Total_M, G, C = logits_g.shape
        device = logits_g.device

        # 超参数
        grpo_advantage_weight = 1.0
        grpo_beta = 0.01
        epsilon = 1e-8

        # 数据准备
        tgt_labels = grpo_data["grpo_labels"].view(-1, 1).expand(-1, G)
        curr_ious = grpo_data["grpo_ious"].clamp(0.0, 1.0) # [Total_M, G]

        # ==================== 新核心：基于 IoU 的奖励设计 ====================
        # 1. 基础奖励：IoU，但加入“信心门槛”概念
        # 高 IoU (>= 0.5) 给予正向奖励，低 IoU (< 0.5) 给予惩罚
        # 这模拟了 AP 计算中的 TP/FP 逻辑
        threshold = 0.5
        reward_base = torch.where(curr_ious >= threshold, curr_ious, -curr_ious)

        # 2. (可选) 增强奖励：强调高 IoU 与低 IoU 的差异
        # 让 IoU=0.9 的奖励显著高于 IoU=0.6，加速学习
        # reward_enhanced = reward_base * (curr_ious ** 2) # 平方增强高 IoU

        # 使用基础奖励
        rewards = reward_base

        # ==================== 标准 GRPO 流程 ====================
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = (rewards - mean_r) / (std_r + epsilon)
        advantages = torch.clamp(advantages, -2, 2).detach()

        # 获取目标类的 logits: [Total_M, G]
        tgt_logits = logits_g.gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)

        # 使用稳定的 NLL (Negative Log-Likelihood)
        # 如果 IoU > 0.5，我们希望 p 接近 1，所以 target 是 1
        # 如果 IoU <= 0.5，我们希望 p 接近 0，所以 target 是 0
        # 这样构建一个“软”目标，引导模型学习 IoU 与置信度的关系
        soft_targets = (curr_ious > threshold).float() # [Total_M, G]
        nll_loss = F.binary_cross_entropy_with_logits(tgt_logits, soft_targets, reduction='none')

        # KL 散度 (保持不变，用于稳定性)
        kl = torch.zeros_like(nll_loss)
        if ref_grpo_logits is not None:
            ref_grpo_logits = ref_grpo_logits.view(Total_M, G, C)
            ref_tgt_logits = ref_grpo_logits.detach().gather(2, tgt_labels.unsqueeze(-1)).squeeze(-1)
            p = torch.sigmoid(tgt_logits).clamp(epsilon, 1 - epsilon)
            pref = torch.sigmoid(ref_tgt_logits).clamp(epsilon, 1 - epsilon)
            kl = pref * (torch.log(pref + epsilon) - torch.log(p + epsilon))

        # ==================== 组合损失 ====================
        # 核心：使用优势函数来调制 NLL 损失
        # 优势为正 (好样本) -> 降低 NLL (让模型更确信预测)
        # 优势为负 (坏样本) -> 提升 NLL (让模型降低确信度)
        policy_loss = advantages.detach() * nll_loss # 注意符号！
        total_loss = policy_loss * grpo_advantage_weight + grpo_beta * kl

        loss_rl = total_loss.mean()
        final_losses["loss_rl"] = loss_rl

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

        matched_queries = outputs["output"][batch_idx_map, src_idx_map]  # [Total_M, C]
        matched_queries_expand = matched_queries.unsqueeze(1).expand(-1, G, -1).reshape(-1, matched_queries.shape[-1])

        # fused_feats = box_features + matched_queries_expand
        fused_feats = fused_feats
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
            generalized_box_iou(
                box_cxcywh_to_xyxy(samp_flat),
                box_cxcywh_to_xyxy(tgt_expand),
            )
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
        # multi_scale_feats = self.freq_enhancer(multi_scale_feats)

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


