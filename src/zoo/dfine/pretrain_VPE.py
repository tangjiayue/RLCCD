# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision

# from .vpe_cls import VisualPromptEncoder, ClassificationHead
# from .box_ops import box_cxcywh_to_xyxy
# from ...core import register

# __all__ = [
#     "VPEPretrain",
#     ]


# @register()
# class VPEPretrain(nn.Module):
#     __inject__ = [
#         "backbone",
#         "encoder",
#     ]

#     def __init__(
#         self,
#         backbone: nn.Module,
#         encoder: nn.Module,
#         hidden_dim = 256, #encoder输出一致
#         num_classes = 10,
#         alpha=0.75,
#         gamma=2.0,
#         text_feats_path="/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/text10_feats.pt",
#     ):
#         super().__init__()
#         self.backbone = backbone
#         self.encoder = encoder
#         self.VPEPretrainWrapper = VPEPretrainWrapper(hidden_dim, num_classes)

#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.gamma = gamma

#         # 加载预先编码好的文本特征 [10, 512] 或 [10, 768]
#         checkpoint = torch.load(text_feats_path, map_location='cpu')
#         text_feats = checkpoint["text_feats"] # 假设维度是 [10, D]
        
#         # 建立一个投影层，将 VPE 的 hidden_dim (256) 映射到文本特征维度 (如 CLIP 的 512)
#         text_dim = text_feats.shape[1]
#         self.v_to_t_projection = nn.Linear(hidden_dim, text_dim)
        
#         # 将文本特征注册为 buffer (不参与梯度更新)
#         self.register_buffer("target_text_feats", text_feats)

#     def forward(self, x, targets=None, box_fmt="cxcywh"):
#         x = self.backbone(x)
#         x = self.encoder(x)
#         logits, labels_final, fused_feat = self.VPEPretrainWrapper(x, targets, box_fmt=box_fmt)

#         if logits is None:
#             return {"loss_vpe_cls": x[0].sum() * 0}, None, None

#         target_one_hot = F.one_hot(labels_final.long(), num_classes=self.num_classes).float() #[N,10]
#         loss_focal = torchvision.ops.sigmoid_focal_loss(
#             logits, 
#             target_one_hot, 
#             alpha=self.alpha, 
#             gamma=self.gamma, 
#             reduction="mean"
#         )

#         v_feats_proj = self.v_to_t_projection(fused_feat)
#         v_feats_proj = F.normalize(v_feats_proj, dim=-1) # [N, text_dim]
        
#         #归一化文本特征 (buffer 里的特征)
#         t_feats_norm = F.normalize(self.target_text_feats, dim=-1) # [10, text_dim]
        
#         # 计算相似度矩阵 [N, 10]
#         # 每个框对 10 个文本类别的相似度得分
#         sim_logits = torch.matmul(v_feats_proj, t_feats_norm.t()) / 0.07 # 0.07 是温度系数
        
#         # 计算交叉熵损失
#         # 目标是：第 i 个框应该对应 labels_final[i] 那个类别的文本
#         loss_contrast = F.cross_entropy(sim_logits, labels_final.long())

#         # w_focal=20.0, w_contrast=1.0
#         # print(f"loss_focal:{loss_focal},loss_contrast:{loss_contrast}")
#         total_loss = (loss_focal * 100.0) + (loss_contrast * 0.5)

#         return {"total_loss": total_loss * self.num_classes}, logits, labels_final

# class VPEPretrainWrapper(nn.Module):
#     def __init__(self, hidden_dim, num_classes):
#         super().__init__()
#         self.vpe = VisualPromptEncoder(hidden_dim)  
#         self.cls_head = ClassificationHead(hidden_dim, num_classes)

#     def forward(self, memory_map, targets, box_fmt="cxcywh"):
#         # 1. 提取特征图信息
#         memory_map = memory_map[-1]  # [B, C, H, W]
#         B, C, m_h, m_w = memory_map.shape
#         device = memory_map.device

#         # 获取每张图的 boxes 数量
#         boxes_per_image = [len(t["boxes"]) for t in targets]
#         total_boxes = sum(boxes_per_image)

#         if total_boxes == 0:
#             return None, None, None

#         all_logits = []
#         all_fused_feats = []
#         all_labels = []

#         for i in range(B):
#             num_boxes = boxes_per_image[i]
#             if num_boxes == 0:
#                 continue

#             # --- A. 准备当前图片的输入 ---
#             cur_boxes_norm = targets[i]["boxes"] # [Ni, 4]
#             cur_labels = targets[i]["labels"]     # [Ni]
            
#             # 坐标转换与缩放
#             if box_fmt == "cxcywh":
#                 cur_boxes_xyxy = box_cxcywh_to_xyxy(cur_boxes_norm)
#             else:
#                 cur_boxes_xyxy = cur_boxes_norm
            
#             # 缩放至特征图尺度
#             scale_vec = torch.tensor([m_w, m_h, m_w, m_h], device=device, dtype=cur_boxes_norm.dtype)
#             cur_boxes_scaled = cur_boxes_xyxy * scale_vec

#             # 构造 ROI [Ni, 5] (batch_index 在这里传 0，因为 memory_map 我们只切片传一张图)
#             batch_idx_vec = torch.zeros((num_boxes, 1), device=device, dtype=cur_boxes_norm.dtype)
#             cur_rois = torch.cat([batch_idx_vec, cur_boxes_scaled], dim=1)

#             # --- B. 严谨交互 (仅针对当前图 memory_map[i:i+1] 和其对应的框) ---
#             # 关键：这里 vpe 看到的 B=1，Self-Attention 范围限制在 Ni 内部
#             cur_fused_feat = self.vpe(
#                 memory_map[i:i+1], 
#                 cur_rois, 
#                 cur_boxes_norm.unsqueeze(0) # 扩展为 [1, Ni, 4] 匹配 vpe 输入要求
#             ) # 输出: [1, Ni, C]
            
#             # --- C. 计算预测 ---
#             cur_fused_feat = cur_fused_feat.squeeze(0) # [Ni, C]
#             cur_logits = self.cls_head(cur_fused_feat) # [Ni, num_classes]

#             all_logits.append(cur_logits)
#             all_fused_feats.append(cur_fused_feat)
#             all_labels.append(cur_labels)

#         # 合并整个 Batch 的结果供 Loss 计算
#         logits_final = torch.cat(all_logits, dim=0)       # [Total_N, num_classes]
#         labels_final = torch.cat(all_labels, dim=0)       # [Total_N]
#         fused_feat_final = torch.cat(all_fused_feats, dim=0) # [Total_N, C]
        
#         return logits_final, labels_final, fused_feat_final