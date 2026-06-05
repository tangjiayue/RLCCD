""" "
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision
from typing import List, Dict

torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples", "save_grpo_samples"]

def save_samples(samples: torch.Tensor, targets: List[Dict], output_dir: str, split: str, normalized: bool, box_fmt: str):
    '''
    normalized: whether the boxes are normalized to [0, 1]
    box_fmt: 'xyxy', 'xywh', 'cxcywh', D-FINE uses 'cxcywh' for training, 'xyxy' for validation
    '''
    from torchvision.transforms.functional import to_pil_image
    from torchvision.ops import box_convert
    from pathlib import Path
    from PIL import ImageDraw, ImageFont
    import os

    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)
    # Predefined colors (standard color names recognized by PIL)
    BOX_COLORS = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "lime", "pink",
        "teal", "lavender", "brown", "beige", "maroon",
        "navy", "olive", "coral", "turquoise", "gold"
    ]

    LABEL_TEXT_COLOR = "white"

    font = ImageFont.load_default()
    font.size = 32

    for i, (sample, target) in enumerate(zip(samples, targets)):
        sample_visualization = sample.clone().cpu()
        target_boxes = target["boxes"].clone().cpu()
        target_labels = target["labels"].clone().cpu()
        target_image_id = target["image_id"].item()
        target_image_path = target["image_path"]
        target_image_path_stem = Path(target_image_path).stem

        sample_visualization = to_pil_image(sample_visualization)
        sample_visualization_w, sample_visualization_h = sample_visualization.size

        # normalized to pixel space
        if normalized:
            target_boxes[:, 0] = target_boxes[:, 0] * sample_visualization_w
            target_boxes[:, 2] = target_boxes[:, 2] * sample_visualization_w
            target_boxes[:, 1] = target_boxes[:, 1] * sample_visualization_h
            target_boxes[:, 3] = target_boxes[:, 3] * sample_visualization_h

        # any box format -> xyxy
        target_boxes = box_convert(target_boxes, in_fmt=box_fmt, out_fmt="xyxy")

        # clip to image size
        target_boxes[:, 0] = torch.clamp(target_boxes[:, 0], 0, sample_visualization_w)
        target_boxes[:, 1] = torch.clamp(target_boxes[:, 1], 0, sample_visualization_h)
        target_boxes[:, 2] = torch.clamp(target_boxes[:, 2], 0, sample_visualization_w)
        target_boxes[:, 3] = torch.clamp(target_boxes[:, 3], 0, sample_visualization_h)

        target_boxes = target_boxes.numpy().astype(np.int32)
        target_labels = target_labels.numpy().astype(np.int32)

        draw = ImageDraw.Draw(sample_visualization)

        # draw target boxes
        for box, label in zip(target_boxes, target_labels):
            x1, y1, x2, y2 = box

            # Select color based on class ID
            box_color = BOX_COLORS[int(label) % len(BOX_COLORS)]

            # Draw box (thick)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

            label_text = f"{label}"

            # Measure text size
            text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:4]

            # Draw text background
            padding = 2
            draw.rectangle(
                [x1, y1 - text_height - padding * 2, x1 + text_width + padding * 2, y1],
                fill=box_color
            )

            # Draw text (LABEL_TEXT_COLOR)
            draw.text((x1 + padding, y1 - text_height - padding), label_text,
                     fill=LABEL_TEXT_COLOR, font=font)

        save_path = Path(output_dir) / f"{split}_samples" / f"{target_image_id}_{target_image_path_stem}.webp"
        sample_visualization.save(save_path)

def save_grpo_samples(samples: torch.Tensor, targets: List[Dict], output: Dict, 
                      output_dir: str, split: str, num_vis_samples: int = 8):
    '''
    samples: 图像张量 [B, 3, H, W]
    output: 模型输出字典，包含 "dec_out_grpo_bboxes" [B, N, G, 4] 和 "pred_boxes" [B, N, 4]
    num_vis_samples: 每个 Query 随机选多少个采样框进行可视化
    '''

    from torchvision.transforms.functional import to_pil_image
    from torchvision.ops import box_convert
    from pathlib import Path
    from PIL import ImageDraw, Image
    import os
    os.makedirs(Path(output_dir) / Path(f"{split}_grpo_vis"), exist_ok=True)
    
    # 获取数据
    grpo_bboxes = output["dec_out_grpo_bboxes"].detach().cpu() # [B, N, G, 4] (cxcywh)
    # pred_bboxes = output["pred_boxes"].detach().cpu()         # [B, N, 4] (cxcywh)
    B, N, G, _ = grpo_bboxes.shape

    for i, (sample, target) in enumerate(zip(samples, targets)):
        # 1. 准备基础图像
        img_pil = to_pil_image(sample.cpu()).convert("RGBA")
        w, h = img_pil.size
        
        # 创建一个透明层用于绘制采样框（处理半透明效果）
        overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_base = ImageDraw.Draw(img_pil)

        # --- A. 绘制 Target (红色) ---
        tgt_boxes = target["boxes"].clone().cpu()
        if len(tgt_boxes) > 0:
            # D-FINE 训练通常是 cxcywh 0-1
            tgt_boxes = box_convert(tgt_boxes, in_fmt='cxcywh', out_fmt='xyxy')
            tgt_boxes[:, [0, 2]] *= w
            tgt_boxes[:, [1, 3]] *= h
            for box in tgt_boxes:
                draw_base.rectangle(box.tolist(), outline="red", width=1)

        # --- B. 绘制 GRPO 采样框 (蓝色，半透明) ---
        # 随机从 G 个采样中选出 num_vis_samples 个
        indices = torch.randperm(G)[:num_vis_samples]
        selected_grpo = grpo_bboxes[i, :, indices, :] # [N, num_vis, 4]
        
        # 展平所有 Query 的采样框进行绘制
        selected_grpo = selected_grpo.reshape(-1, 4) 
        selected_grpo = box_convert(selected_grpo, in_fmt='cxcywh', out_fmt='xyxy')
        selected_grpo[:, [0, 2]] *= w
        selected_grpo[:, [1, 3]] *= h

        for box in selected_grpo:
            # 使用半透明蓝色 (RGBA: 0, 100, 255, 100)
            draw_overlay.rectangle(box.tolist(), outline=(0, 150, 255, 120), width=1)

        # --- C. 绘制 Pred 预测框 (绿色) ---
        # p_boxes = pred_bboxes[i] # [N, 4]
        # p_boxes = box_convert(p_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        # p_boxes[:, [0, 2]] *= w
        # p_boxes[:, [1, 3]] *= h
        
        # for box in p_boxes:
        #     # 过滤掉置信度极低的框（可选，如果你的 pred_boxes 已经选过 TopK 就不用加）
        #     draw_base.rectangle(box.tolist(), outline="lime", width=2)

        # 合并图层
        img_pil = Image.alpha_composite(img_pil, overlay).convert("RGB")

        # 4. 保存
        target_image_id = target.get("image_id", i)
        save_path = Path(output_dir) / f"{split}_grpo_vis" / f"grpo_{target_image_id}.jpg"
        img_pil.save(save_path)
    print(f"已保存 GRPO 可视化结果到: {Path(output_dir) / f'{split}_grpo_vis'}")

def show_sample(sample):
    """for coco dataset/dataloader"""
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()



import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torchvision.ops import box_convert
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import torchvision  # 添加这个导入

class InferenceVisualizer:
    """
    推理时可视化工具（支持 PIL）
    输出英文类别名称，只显示全局置信度前 K 个预测框
    """
    
    # 类别名称
    CLASS_NAMES = [
        "normal",                    # 0
        "ascus",                     # 1
        "asch",                      # 2
        "lsil",                      # 3
        "hsil",              # 4
        "agc",     # 5
        "vag",                 # 6
        "mon",                   # 7
        "dys", # 8
        "ec"                         # 9
    ]
    
    # 每类固定颜色（RGB格式）
    CLASS_COLORS = [
        (220, 50, 50),    # 0: normal - 深红
        (50, 200, 50),    # 1: ascus - 鲜绿（深色）
        (50, 50, 220),    # 2: asch - 深蓝
        (200, 200, 50),   # 3: lsil - 橄榄黄
        (200, 50, 200),   # 4: hsil_scc_omn - 紫罗兰
        (50, 200, 200),   # 5: agc_adenocarcinoma_em - 深青
        (180, 100, 20),   # 6: vaginalis - 深橙色
        (180, 50, 180),   # 7: monilia - 深紫红
        (50, 150, 220),   # 8: dysbacteriosis_herpes_act - 天蓝
        (150, 150, 50),   # 9: ec - 橄榄绿
    ]
    
    def __init__(
        self,
        score_threshold: float = 0.05,
        top_k: int = 10,
        line_width: int = 2,
        font_size: int = 12,
    ):
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.line_width = line_width
        self.font_size = font_size
        
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            self.font = ImageFont.load_default()
    
    @staticmethod
    def tensor_to_pil(image_tensor):
        """将 tensor 转换为 PIL Image（与 save_samples 保持一致）"""
        if isinstance(image_tensor, torch.Tensor):
            # to_pil_image 会自动处理 [0,1] 或 [0,255] 范围的 tensor
            return to_pil_image(image_tensor.cpu())
        return image_tensor
    
    def draw_pred_boxes_pil(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Image.Image:
        """使用 PIL 绘制预测框"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        
        # 按置信度排序，取前 top_k 个
        if len(scores) > self.top_k:
            idx = np.argsort(scores)[::-1][:self.top_k]
            boxes = boxes[idx]
            scores = scores[idx]
            labels = labels[idx]
        
        for box, score, label in zip(boxes, scores, labels):
            if score < self.score_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            
            color = self.CLASS_COLORS[label % len(self.CLASS_COLORS)]
            class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else str(label)
            text = f"{class_name}: {score:.2f}"
            
            # 绘制边框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)
            
            # 测量文字大小
            bbox = draw.textbbox((x1, y1), text, font=self.font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # 绘制文字背景
            draw.rectangle(
                [x1, y1 - text_h - 4, x1 + text_w + 4, y1],
                fill=color
            )
            
            # 绘制文字
            draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=self.font)
        
        return img_copy
    
    def draw_gt_boxes_pil(
        self,
        image: Image.Image,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> Image.Image:
        """使用 PIL 绘制 GT 框"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        gt_color = (30, 100, 200)  # 深蓝色
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = map(int, box)
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            
            draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=self.line_width)
            
            class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else str(label)
            text = f"GT: {class_name}"
            
            bbox = draw.textbbox((x1, y1), text, font=self.font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=gt_color)
            draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=self.font)
        
        return img_copy
    
    def draw_gt_and_pred_combined_pil(
        self,
        image: Image.Image,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> Image.Image:
        """在一张图上同时绘制 GT 框和预测框"""
        # 先画预测框
        img_copy = self.draw_pred_boxes_pil(image, pred_boxes, pred_scores, pred_labels)
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        gt_color = (30, 100, 200)  # 深蓝色
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = map(int, box)
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            
            # GT 框用虚线或更粗的线（PIL 不支持虚线，用粗线代替）
            draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=self.line_width + 1)
            
            class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else str(label)
            text = f"GT: {class_name}"
            
            bbox = draw.textbbox((x1, y1), text, font=self.font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=gt_color)
            draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=self.font)
        
        return img_copy
    
    # #创建对比图：左侧预测框，右侧GT框
    # def create_comparison_image_pil(
    #     self,
    #     image: Image.Image,
    #     pred_boxes: np.ndarray,
    #     pred_scores: np.ndarray,
    #     pred_labels: np.ndarray,
    #     gt_boxes: Optional[np.ndarray] = None,
    #     gt_labels: Optional[np.ndarray] = None,
    # ) -> Image.Image:
    #     """创建对比图：左侧预测框，右侧GT框"""
    #     w, h = image.size
        
    #     left_img = self.draw_pred_boxes_pil(image.copy(), pred_boxes, pred_scores, pred_labels)
        
    #     if gt_boxes is not None:
    #         right_img = self.draw_gt_boxes_pil(image.copy(), gt_boxes, gt_labels)
    #         comparison = Image.new('RGB', (w * 2, h))
    #         comparison.paste(left_img, (0, 0))
    #         comparison.paste(right_img, (w, 0))
    #         return comparison
    #     else:
    #         return left_img
    
    def create_comparison_image_pil(
        self,
        image: Image.Image,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: Optional[np.ndarray] = None,
        gt_labels: Optional[np.ndarray] = None,
        iou_threshold: float = 0.5,
    ) -> Image.Image:
        """创建对比图：左侧预测框，右侧GT框，并在图上显示召回率和精确率"""

        w, h = image.size
        # =========================
        # 关键：先按置信度取 top_k（与画图一致）
        # =========================
        if len(pred_scores) > self.top_k:
            idx = np.argsort(pred_scores)[::-1][:self.top_k]
            pred_boxes = pred_boxes[idx]
            pred_scores = pred_scores[idx]
            pred_labels = pred_labels[idx]

        # =========================
        # 计算 Precision / Recall
        # =========================
        precision = 0.0
        recall = 0.0
        tp = fp = fn = 0

        if gt_boxes is not None and len(gt_boxes) > 0 and len(pred_boxes) > 0:
            
            # 初始化匹配记录
            gt_matched = [False] * len(gt_boxes)
            pred_matched = [False] * len(pred_boxes)
            
            # 计算所有 IoU 矩阵
            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
            for i, pred_box in enumerate(pred_boxes):
                for j, gt_box in enumerate(gt_boxes):
                    iou_matrix[i, j] = self._compute_iou(pred_box, gt_box)
            
            # 贪心匹配：按 IoU 从高到低排序
            matches = []
            for i in range(len(pred_boxes)):
                for j in range(len(gt_boxes)):
                    if iou_matrix[i, j] >= iou_threshold:
                        matches.append((iou_matrix[i, j], i, j))
            
            # 按 IoU 降序排序
            matches.sort(key=lambda x: x[0], reverse=True)
            
            # 执行匹配
            for _, pred_idx, gt_idx in matches:
                if not pred_matched[pred_idx] and not gt_matched[gt_idx]:
                    pred_matched[pred_idx] = True
                    gt_matched[gt_idx] = True
            
            tp = sum(pred_matched)
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - sum(gt_matched)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
        elif gt_boxes is not None and len(gt_boxes) > 0 and len(pred_boxes) == 0:
            # 没有预测框，但有待检目标
            tp = 0
            fp = 0
            fn = len(gt_boxes)
            precision = 0.0
            recall = 0.0
            
        elif gt_boxes is not None and len(gt_boxes) == 0 and len(pred_boxes) > 0:
            # 有预测框，但无待检目标
            tp = 0
            fp = len(pred_boxes)
            fn = 0
            precision = 0.0
            recall = 0.0

        # =========================
        # 左图：预测
        # =========================
        left_img = self.draw_pred_boxes_pil(
            image.copy(),
            pred_boxes,
            pred_scores,
            pred_labels
        )

        # =========================
        # 无GT直接返回
        # =========================
        if gt_boxes is None:
            return left_img

        # =========================
        # 右图：GT
        # =========================
        right_img = self.draw_gt_boxes_pil(
            image.copy(),
            gt_boxes,
            gt_labels
        )

        # 拼接左右图
        comparison = Image.new("RGB", (w * 2, h))
        comparison.paste(left_img, (0, 0))
        comparison.paste(right_img, (w, 0))

        draw = ImageDraw.Draw(comparison)

        # =========================
        # 字体
        # =========================
        try:
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                18
            )

            metric_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                16
            )

        except:
            title_font = ImageFont.load_default()
            metric_font = ImageFont.load_default()

        # =========================
        # 标题背景
        # =========================
        draw.rectangle((0, 0, 180, 30), fill=(0, 0, 0))
        draw.rectangle((w, 0, w + 180, 30), fill=(0, 0, 0))

        # 标题
        draw.text(
            (10, 5),
            "Predictions",
            fill=(255, 0, 0),   # 红色
            font=title_font
        )

        draw.text(
            (w + 10, 5),
            "Ground Truth",
            fill=(0, 255, 0),   # 绿色
            font=title_font
        )

        # =========================
        # 指标文字
        # =========================
        metric_text = (
            f"Precision: {precision:.2%} | "
            f"Recall: {recall:.2%} | "
            f"TP:{tp} FP:{fp} FN:{fn}"
        )

        # 计算文字大小
        bbox = draw.textbbox(
            (0, 0),
            metric_text,
            font=metric_font
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = w * 2 - text_width - 15
        text_y = 5

        padding = 6

        # =========================
        # 黑色背景框
        # =========================
        draw.rectangle(
            (
                text_x - padding,
                text_y - padding,
                text_x + text_width + padding,
                text_y + text_height + padding
            ),
            fill=(0, 0, 0)
        )

        # =========================
        # 根据 Recall 动态颜色（召回率低更危险）
        # =========================
        if recall < 0.5:
            metric_color = (255, 0, 0)       # 红（漏检严重）
        elif recall < 0.8:
            metric_color = (255, 165, 0)     # 橙
        else:
            metric_color = (0, 255, 0)       # 绿

        # =========================
        # 绘制指标文字
        # =========================
        draw.text(
            (text_x, text_y),
            metric_text,
            fill=metric_color,
            font=metric_font
        )

        return comparison


    def _compute_iou(self, box1, box2):
        """计算两个框的 IoU（输入 xyxy 格式）"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)

    def visualize_batch(
        self,
        images: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        save_dir: str,
        batch_idx: int = 0,
        max_images: int = 8,
        gt_targets: Optional[List[Dict]] = None,
        draw_mode: str = "pred_only",
        random_sample: bool = True,
        nms_threshold: float = 0.5,
        conf_threshold: float = 0.05,
    ):
        """批量可视化推理结果"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        pred_scores = torch.sigmoid(pred_logits)
        
        batch_total = images.shape[0]
        num_to_vis = min(batch_total, max_images)  
        if random_sample:
            indices = np.random.choice(batch_total, num_to_vis, replace=False)
        else:
            indices = range(min(batch_total, max_images))

        for idx_in_batch in indices:
            i = idx_in_batch
            
            # 关键：直接转换，不做任何归一化/反归一化
            img_pil = self.tensor_to_pil(images[i])
            w, h = img_pil.size
            
            # 获取预测结果
            scores_i, labels_i = pred_scores[i].max(dim=-1)
            scores_i = scores_i.flatten()
            labels_i = labels_i.flatten()
            boxes_i = pred_boxes[i]
            
            # 预测框是归一化的 [0,1]，需要转换到像素坐标
            boxes_xyxy = box_convert(boxes_i, in_fmt='cxcywh', out_fmt='xyxy')
            boxes_xyxy[:, [0, 2]] *= w
            boxes_xyxy[:, [1, 3]] *= h
            boxes_xyxy = boxes_xyxy.cpu().numpy().astype(np.int32)
            scores_i = scores_i.cpu().numpy()
            labels_i = labels_i.cpu().numpy()
            
            # NMS 过滤
            keep_conf = scores_i >= conf_threshold
            boxes_filtered = boxes_xyxy[keep_conf]
            scores_filtered = scores_i[keep_conf]
            labels_filtered = labels_i[keep_conf]
            
            if len(boxes_filtered) > 0:
                boxes_tensor = torch.tensor(boxes_filtered).float()
                scores_tensor = torch.tensor(scores_filtered).float()
                labels_tensor = torch.tensor(labels_filtered).long()
                
                keep_nms = torchvision.ops.batched_nms(
                    boxes_tensor, scores_tensor, labels_tensor, nms_threshold
                )
                
                final_boxes = boxes_filtered[keep_nms.cpu().numpy()]
                final_scores = scores_filtered[keep_nms.cpu().numpy()]
                final_labels = labels_filtered[keep_nms.cpu().numpy()]
            else:
                final_boxes = boxes_filtered
                final_scores = scores_filtered
                final_labels = labels_filtered
            
            # GT boxes 已经是像素坐标，不需要转换
            gt_boxes = None
            gt_labels = None
            if gt_targets is not None and i < len(gt_targets):
                gt = gt_targets[i]
                if 'boxes' in gt:
                    gt_boxes = gt['boxes'].cpu().numpy()
                    # GT boxes 已经是像素坐标，直接使用
                    gt_boxes = gt_boxes.astype(np.int32)
                    gt_labels = gt['labels'].cpu().numpy() if 'labels' in gt else None
            
            img_id = gt_targets[i].get("image_id", i) if gt_targets else i
            
            # 绘制并保存
            if draw_mode == "pred_only":
                vis = self.draw_pred_boxes_pil(img_pil.copy(), final_boxes, final_scores, final_labels)
                filename = f"img{img_id}_pred_nms{nms_threshold}.jpg"
            elif draw_mode == "gt_only" and gt_boxes is not None:
                vis = self.draw_gt_boxes_pil(img_pil.copy(), gt_boxes, gt_labels)
                filename = f"img{img_id}_gt.jpg"
            elif draw_mode == "combined" and gt_boxes is not None:
                vis = self.draw_gt_and_pred_combined_pil(
                    img_pil.copy(), final_boxes, final_scores, final_labels, gt_boxes, gt_labels
                )
                filename = f"img{img_id}_combined_nms{nms_threshold}.jpg"
            elif draw_mode == "compare" and gt_boxes is not None:
                vis = self.create_comparison_image_pil(
                    img_pil.copy(), final_boxes, final_scores, final_labels, gt_boxes, gt_labels
                )
                filename = f"img{img_id}_compare_nms{nms_threshold}.jpg"
            else:
                continue
            
            vis.save(str(save_path / filename), quality=95)
    
    def save_detection_json(self, images, outputs, save_path, image_ids=None, score_threshold=0.05):
        """保存检测结果为 JSON"""
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        pred_scores = torch.sigmoid(pred_logits)
        
        results = []
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            img = self.tensor_to_pil(images[i])
            w, h = img.size
            
            scores_i, labels_i = pred_scores[i].max(dim=-1)
            scores_i = scores_i.flatten()
            labels_i = labels_i.flatten()
            boxes_i = pred_boxes[i]
            
            boxes_xyxy = box_convert(boxes_i, in_fmt='cxcywh', out_fmt='xyxy')
            boxes_xyxy[:, [0, 2]] *= w
            boxes_xyxy[:, [1, 3]] *= h
            
            keep = scores_i > score_threshold
            boxes_xyxy = boxes_xyxy[keep].cpu().numpy()
            scores_i = scores_i[keep].cpu().numpy()
            labels_i = labels_i[keep].cpu().numpy()
            
            img_id = image_ids[i] if image_ids else i
            
            for box, score, label in zip(boxes_xyxy, scores_i, labels_i):
                x1, y1, x2, y2 = box
                results.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "category_name": self.CLASS_NAMES[int(label)] if int(label) < len(self.CLASS_NAMES) else str(int(label)),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n保存检测结果到: {save_path}")
        return results

# ==================== 使用示例 ====================
def visualize_inference_example(model, samples, targets, output_dir, epoch=0):
    """
    在你的推理代码中集成
    """
    # 1. 初始化可视化器（全局置信度前10）
    visualizer = InferenceVisualizer(
        score_threshold=0.05,  # 低于0.05不显示
        top_k=10,              # 只显示前10个
        line_width=2,
    )
    
    # 2. 模型推理
    model.eval()
    with torch.no_grad():
        outputs, _ = model.module.sample(samples)  # 根据你的模型调整
    
    # 3. 可视化（多种模式可选）
    save_dir = f"{output_dir}/inference_vis_epoch{epoch}"
    
    # 模式1：只显示预测框
    visualizer.visualize_batch(
        images=samples,
        outputs=outputs,
        save_dir=save_dir,
        batch_idx=0,
        max_images=8,
        draw_mode="pred_only"
    )
    
    # 模式2：如果有GT，可以同时显示（并排对比）
    if targets is not None:
        visualizer.visualize_batch(
            images=samples,
            outputs=outputs,
            save_dir=save_dir,
            batch_idx=0,
            max_images=8,
            gt_targets=targets,
            draw_mode="compare"  # 或 "combined"
        )
    
    # 4. 保存 JSON 结果
    visualizer.save_detection_json(
        images=samples,
        outputs=outputs,
        save_path=f"{save_dir}/detections.json",
        score_threshold=0.1,
    )
    
    return visualizer

    