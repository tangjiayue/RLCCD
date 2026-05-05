"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Dict, Iterable, List
import gc
import os

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples, save_grpo_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes, AdvancedMetricsCalculator
from .tpr_fpr_eval import evaluate_tpr_fpr_from_gt_preds, generate_and_plot_roc


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    freeze_for_vpe_and_cls(model)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)

    postprocessor = kwargs.get("postprocessor", None)
    ref_module = kwargs.get("ref_module", None)
    cfg = kwargs.get("cfg", None)
    ref_vc = kwargs.get("ref_vc", None)

    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)
    accum_steps = cfg.yaml_cfg["train_dataloader"].get("accum_steps", 1)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=False):
                outputs, outputs1= model(samples, targets=targets)

            #可视化GRPO采样情况
            if "dec_out_grpo_bboxes" in outputs1 and outputs1["dec_out_grpo_bboxes"] is not None:
                if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():          
                        save_grpo_samples(
                            samples=samples, 
                            targets=targets, 
                            output=outputs1, 
                            output_dir=output_dir, 
                            split="train_grpo", 
                            num_vis_samples=12  # 采样中随机抽 12 个显示，避免画面太乱
                        )
            
            if cfg.grpo_finetune:
                if ref_module is not None:
                    with torch.no_grad():
                        ref_outputs = ref_module(samples, targets=targets)
            else:
                ref_outputs = None

            ref_cls_outputs = None
            if cfg.grpo_cls:
                ref_vc = kwargs.get("ref_vc", None)  # <- 从 train_one_epoch kwargs 取
                if (
                    ref_vc is not None
                    and outputs1.get("grpo_boxes", None) is not None
                    and outputs1.get("grpo_batch_idx", None) is not None
                ):
                    with torch.no_grad():
                        grpo_boxes = outputs1["grpo_boxes"]          # [Total_M*G, 4]
                        grpo_bidx = outputs1["grpo_batch_idx"]       # [Total_M*G]
                        B = samples.shape[0]

                        # 旧 VPE 需要 multi_scale_feats：你需要在 model forward 里把它塞进 outputs1
                        ref_ms_feats = outputs1.get("vpe_multi_scale_feats", None)
                        if ref_ms_feats is None:
                            raise RuntimeError(
                                "cfg.grpo_cls=True 但 outputs1 缺少 'vpe_multi_scale_feats'。"
                                "请在 model.forward / VisualClassifier.sample_grpo_features 调用处把 multi_scale_feats 写入 outputs1。"
                            )

                        # pad 成 [B, Nmax, 4] + mask，供 ref_vc.vpe 使用
                        counts = torch.bincount(grpo_bidx, minlength=B)
                        max_n = int(counts.max().item()) if counts.numel() > 0 else 0
                        if max_n == 0:
                            ref_cls_outputs = None
                        else:
                            boxes_pad = grpo_boxes.new_zeros((B, max_n, 4))
                            mask_pad = torch.zeros((B, max_n), device=grpo_boxes.device, dtype=torch.bool)

                            for bi in range(B):
                                n = int(counts[bi].item())
                                if n == 0:
                                    continue
                                sel = (grpo_bidx == bi)
                                boxes_b = grpo_boxes[sel]
                                boxes_pad[bi, :n] = boxes_b
                                mask_pad[bi, :n] = True

                            # 旧 VPE + 旧分类头 输出 ref logits
                            ref_vc = ref_vc.to(grpo_boxes.device)
                            ref_vc.eval()

                            ref_vpe_feats_padded = ref_vc.vpe(
                                reference_boxes=boxes_pad,
                                multi_scale_feats=ref_ms_feats,
                            )  # [B, max_n, C]
                            ref_box_feats = ref_vpe_feats_padded[mask_pad]  # [Total_M*G, C]

                            ref_logits_flat = ref_vc.cls_head(ref_box_feats)  # [Total_M*G, num_classes]
                            ref_cls_outputs = ref_logits_flat.detach()
                else:
                    ref_cls_outputs = None
            else:
                ref_cls_outputs = None


            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                # loss_dict = criterion(outputs, targets, ref_outputs=ref_outputs, cfg=cfg, ref_cls_outputs=None,**metas)
                loss_dict = model.module.get_losses(outputs1, ref_cls_outputs=ref_cls_outputs)
                # loss_dict.update(criterion(outputs, targets, ref_outputs=ref_outputs, cfg=cfg, ref_cls_outputs=None,**metas))
                
                # loss_dict2 = criterion(outputs1["outputs"], targets, ref_outputs=ref_outputs, cfg=cfg, ref_cls_outputs=None,**metas)
                # loss_dict.update(loss_dict2)

            loss_raw = sum(loss_dict.values())
            loss = loss_raw / accum_steps 

            # for name, param in model.module.VisualClassifier.vpe.named_parameters():
            #     if param.grad is not None:
            #         print(f"[GRAD] {name}: {param.grad.norm().item():.6f}")
            #     else:
            #         print(f"[GRAD] {name}: None")
            scaler.scale(loss).backward()   

            if (i + 1) % accum_steps == 0:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            del outputs, loss_raw, loss
            # if ref_cls_outputs is not None:
            #     del ref_cls_outputs

        else:
            outputs, outputs1 = model(samples, targets=targets)

            if cfg.grpo_finetune:
                if ref_module is not None:
                    with torch.no_grad():
                        ref_outputs = ref_module(samples, targets=targets)
            else:
                ref_outputs = None

            ref_cls_outputs = None
            if cfg.grpo_cls:
                ref_vc = kwargs.get("ref_vc", None)  # <- 从 train_one_epoch kwargs 取
                if (
                    ref_vc is not None
                    and outputs1.get("grpo_boxes", None) is not None
                    and outputs1.get("grpo_batch_idx", None) is not None
                ):
                    with torch.no_grad():
                        grpo_boxes = outputs1["grpo_boxes"]          # [Total_M*G, 4]
                        grpo_bidx = outputs1["grpo_batch_idx"]       # [Total_M*G]
                        B = samples.shape[0]

                        # 旧 VPE 需要 multi_scale_feats：你需要在 model forward 里把它塞进 outputs1
                        ref_ms_feats = outputs1.get("vpe_multi_scale_feats", None)
                        if ref_ms_feats is None:
                            raise RuntimeError(
                                "cfg.grpo_cls=True 但 outputs1 缺少 'vpe_multi_scale_feats'。"
                                "请在 model.forward / VisualClassifier.sample_grpo_features 调用处把 multi_scale_feats 写入 outputs1。"
                            )

                        # pad 成 [B, Nmax, 4] + mask，供 ref_vc.vpe 使用
                        counts = torch.bincount(grpo_bidx, minlength=B)
                        max_n = int(counts.max().item()) if counts.numel() > 0 else 0
                        if max_n == 0:
                            ref_cls_outputs = None
                        else:
                            boxes_pad = grpo_boxes.new_zeros((B, max_n, 4))
                            mask_pad = torch.zeros((B, max_n), device=grpo_boxes.device, dtype=torch.bool)

                            for bi in range(B):
                                n = int(counts[bi].item())
                                if n == 0:
                                    continue
                                sel = (grpo_bidx == bi)
                                boxes_b = grpo_boxes[sel]
                                boxes_pad[bi, :n] = boxes_b
                                mask_pad[bi, :n] = True

                            # 旧 VPE + 旧分类头 输出 ref logits
                            ref_vc = ref_vc.to(grpo_boxes.device)
                            ref_vc.eval()

                            ref_vpe_feats_padded = ref_vc.vpe(
                                reference_boxes=boxes_pad,
                                multi_scale_feats=ref_ms_feats,
                            )  # [B, max_n, C]
                            ref_box_feats = ref_vpe_feats_padded[mask_pad]  # [Total_M*G, C]

                            ref_logits_flat = ref_vc.cls_head(ref_box_feats)  # [Total_M*G, num_classes]
                            ref_cls_outputs = ref_logits_flat.detach()
                else:
                    ref_cls_outputs = None
            else:
                ref_cls_outputs = None

            # loss_dict = criterion(outputs, targets, ref_outputs=ref_outputs, cfg=cfg, ref_cls_outputs=None,**metas)
            loss_dict = model.module.get_losses(outputs1)
            # loss_dict.update(criterion(outputs, targets, ref_outputs=ref_outputs, cfg=cfg, ref_cls_outputs=None,**metas))
            # loss_dict2 = criterion(outputs1["outputs"], targets, ref_outputs=ref_outputs, cfg=cfg, ref_cls_outputs=None,**metas)
            # loss_dict.update(loss_dict2)

            loss_raw: torch.Tensor = sum(loss_dict.values())
            loss = loss_raw / accum_steps            
            loss.backward()
            
            if (i + 1) % accum_steps == 0:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()
                optimizer.zero_grad()

            del outputs, loss_raw, loss

        # #可视化GRPO采样情况
        # if "dec_out_grpo_bboxes" in outputs1 and outputs1["dec_out_grpo_bboxes"] is not None:
        #     if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():          
        #             save_grpo_samples(
        #                 samples=samples, 
        #                 targets=targets, 
        #                 output=outputs1, 
        #                 output_dir=output_dir, 
        #                 split="train_grpo", 
        #                 num_vis_samples=12  # 采样中随机抽 12 个显示，避免画面太乱
        #             )
         
        if (i + 1) % accum_steps == 0:
            # ema
            if ema is not None:
                ema.update(model)

            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())
        del loss_dict

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

        gc.collect()
        torch.cuda.empty_cache()

    if use_wandb:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)}
        )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # ========================== [绘制特征分布图] ==========================
    if dist_utils.is_main_process():
        try:
            from ..zoo.dfine.plot_distribution import epoch_visualizer
            # 使用引擎传入的 output_dir 作为保存路径
            save_path = output_dir if output_dir is not None else "./output"
            analysis_dir = os.path.join(save_path, "train_visualizations")
            os.makedirs(analysis_dir, exist_ok=True)
            epoch_visualizer.plot_and_clear(save_dir=analysis_dir, epoch=epoch)
        except Exception as e:
            print(f"绘制特征分布图失败: {e}")
    # =========================================================================

    # print("DEBUG WEIGHT:", model.module.VisualClassifier.cls_head.head.weight.mean().item())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = coco_evaluator.iou_types
   
    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    m_output_dir = kwargs.get("m_output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    # ==================== 初始化可视化器 ====================
    # 只在主进程且指定 epoch 时初始化（比如每 5 个 epoch 可视化一次）
    visualize_this_epoch = (epoch % 5 == 0) or (epoch == -1)  # 每 5 个 epoch 可视化一次，可以调整
    visualizer = None
    # print(dist_utils.is_main_process(), visualize_this_epoch, m_output_dir is not None)
    if dist_utils.is_main_process() and visualize_this_epoch and m_output_dir is not None:
        from ..misc.visualizer import InferenceVisualizer  
        visualizer = InferenceVisualizer(score_threshold=0., top_k=20)
        print(f"\n[可视化] 初始化可视化器，epoch={epoch}")

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")


        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        outputs, feats = model.module.sample(samples, targets=targets)
        results = postprocessor(outputs, orig_target_sizes)
        # results = model.module.predict_refine(feats, results, 1)

        # if m_output_dir is not None and dist_utils.is_main_process():          
        #     save_grpo_samples(
        #         samples=samples, 
        #         targets=targets, 
        #         output=outputs, 
        #         output_dir=m_output_dir, 
        #         split="pred", 
        #         num_vis_samples=12  # 随机抽 12 个显示，避免画面太乱
        #     )

        # ==================== 可视化推理结果 ====================
        # 只在主进程、前几个 batch、且可视化器已初始化时执行
        if dist_utils.is_main_process() and visualizer is not None and i < 2:  # 只可视化前 2 个 batch
            try:
                save_dir = os.path.join(m_output_dir, "inference_vis", f"epoch_{epoch}")                
                # 并排对比（需要 targets）
                visualizer.visualize_batch(
                    images=samples,
                    outputs=outputs,
                    save_dir=save_dir,
                    batch_idx=i,
                    max_images=4,
                    gt_targets=targets,
                    draw_mode="compare",
                    nms_threshold=1
                )
                
                print(f"[可视化] 已保存 batch {i} 的可视化结果到 {save_dir}")
            except Exception as e:
                print(f"[可视化] 警告: 可视化失败: {e}")
        # ========================================================

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # validator format for metrics
        for idx, (target, result) in enumerate(zip(targets, results)):
            labels = target["labels"]
            boxes = target["boxes"]

            # 过滤掉 label == 0
            keep = labels != -1

            gt.append(
                {
                    "boxes": scale_boxes(
                        boxes[keep],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": labels[keep],
                }
            )
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            preds.append(
                {"boxes": result["boxes"], "labels": labels, "scores": result["scores"]}
            )

    #Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    # calculator = AdvancedMetricsCalculator(device=device)
    
    # # 执行计算 (内部自动处理分布式汇聚)
    # metrics = calculator.compute(gt_list=gt, preds_list=preds)
    
    # # 仅在主进程输出结果
    # if dist_utils.is_main_process():
    #     print("\n" + "="*50)
    #     print(f"{'Metric':<20} | {'Value':<10}")
    #     print("-" * 50)
    #     print(f"{'mAP @[.50:.95]':<20} | {metrics['mAP_50_95']:<10.4f}")
    #     print(f"{'mAP @ .50':<20} | {metrics['mAP_50']:<10.4f}")
    #     print(f"{'mAP @ .75':<20} | {metrics['mAP_75']:<10.4f}")
    #     print("-" * 50)
    #     print(f"{'Max-F1 (Avg)':<20} | {metrics['Max-F1 (Avg)']:<10.4f}")
    #     print(f"{'Precision':<20} | {metrics['Precision']:<10.4f}")
    #     print(f"{'Recall':<20} | {metrics['Recall']:<10.4f}")
    #     print(f"{'F1-Score':<20} | {metrics['F1-Score']:<10.4f}")
    #     print("="*50 + "\n")

    #     # WandB 记录
    #     if use_wandb:
    #         wandb.log({
    #             "Eval/mAP_50_95": metrics['mAP_50_95'],
    #             "Eval/mAP_50": metrics['mAP_50'],
    #             "Eval/Precision": metrics['Precision'],
    #             "Eval/Recall": metrics['Recall'],
    #             "Eval/F1": metrics['F1-Score'],
    #             "epoch": epoch
    #         })

    # # 同步进程
    # if dist.is_initialized():
    #     dist.barrier()

    # # ===== TPR/FPR @ IoU=0.5（独立模块）=====
    # tprfpr = evaluate_tpr_fpr_from_gt_preds(
    #     gt=gt,
    #     preds=preds,
    #     iou_thr=0.5,
    #     score_thr=0.0,
    #     per_class=True,
    #     ignore_label=0,  # 与你当前“过滤 label==0”一致
    # )
    # print("[TPR/FPR] settings:", tprfpr["settings"])
    # print("[TPR/FPR] overall:", tprfpr["overall"])
    # if dist_utils.is_main_process():
    #     import os
    #     roc_save_dir = output_dir if output_dir is not None else str(m_output_dir)
    #     os.makedirs(roc_save_dir, exist_ok=True)
    #     roc_save_path = os.path.join(roc_save_dir, f"roc_curve_iou05/roc_curve_iou05_epoch_{epoch}.png")

    #     generate_and_plot_roc(
    #         gt=gt, 
    #         preds=preds, 
    #         iou_thr=0.5, 
    #         ignore_label=0, 
    #         save_path=roc_save_path 
    #     )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    if coco_evaluator is not None and dist_utils.is_main_process():
        # 获取 bbox 评估结果对象
        stats_obj = coco_evaluator.coco_eval["bbox"]
        cat_ids = stats_obj.params.catIds
        
        # precision 矩阵维度: [IoU阈值, Recall阈值, 类别, 面积范围, 最大检测数]
        # IoU阈值 0-9 对应 0.50:0.95
        # 面积范围 0 对应 'all'
        precision = stats_obj.eval['precision']

        # 用于计算排除 label 0 后的平均值
        valid_aps_50_95 = []
        valid_aps_50 = []
        
        print("\n" + "="*50)
        print(f"{'CatID':<10} | {'AP@50:95':<12} | {'AP@50':<10}")
        print("-" * 50)
        
        for i, cat_id in enumerate(cat_ids):
            # 计算该类在 0.50:0.95 上的平均 AP
            # 我们对前两个维度（IoU 和 Recall）求平均
            ap_50_95 = precision[:, :, i, 0, -1].mean()
            
            # 计算该类在 0.50 阈值下的 AP (IoU 维度的索引 0)
            ap_50 = precision[0, :, i, 0, -1].mean()
            
            print(f"{cat_id:<10} | {ap_50_95:<12.4f} | {ap_50:<10.4f}")

            #排除 label 0 ---
            if cat_id != 0:
                valid_aps_50_95.append(ap_50_95)
                valid_aps_50.append(ap_50)
        
        print("="*50 + "\n")

        # 计算并打印排除 label 0 后的平均值 (mAP)
        if len(valid_aps_50_95) > 0:
            mean_ap_50_95 = sum(valid_aps_50_95) / len(valid_aps_50_95)
            mean_ap_50 = sum(valid_aps_50) / len(valid_aps_50)
            
            print(f"{'mAP(excl.0)':<10} | {mean_ap_50_95:<12.4f} | {mean_ap_50:<10.4f}")
        else:
            print("No labels other than 0 found.")
            
        print("="*50 + "\n")

    # 保存检测结果
    if dist_utils.is_main_process():
        # 保存为JSON格式
        json_path = dist_utils.save_detection_results(
            gt=gt,
            preds=preds,
            save_dir= str(m_output_dir) +"/detection_results",
            epoch=epoch,
        )

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    del gt,preds,res,outputs,results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================== [绘制验证集推理特征分布图] ==========================
    if dist_utils.is_main_process():
        try:
            from ..zoo.dfine.plot_distribution import epoch_visualizer
            save_path = m_output_dir if m_output_dir is not None else "./output"
            analysis_dir = os.path.join(save_path, "val_visualizations")
            os.makedirs(analysis_dir, exist_ok=True)
            epoch_visualizer.plot_and_clear(save_dir=analysis_dir, epoch=f"{epoch}_eval")
        except Exception as e:
            print(f"绘制验证集分布图失败: {e}")
    # ======================================================================================

    return stats, coco_evaluator

def freeze_for_vpe_and_cls(model):
    # 递归获取原始模型
    root_m = unwrap_model(model)

    # 冻结所有参数
    for p in root_m.parameters():
        p.requires_grad = False

    for name, module in root_m.named_modules():
        if "VisualClassifier" in name:
            continue
            
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
            module.eval()
            module.training = False

    root_m.VisualClassifier.train()
    #  解冻 VPE 相关的参数 (包括分类头)
    for p in root_m.VisualClassifier.parameters():
        p.requires_grad = True


    # #只冻结BN层，其他层都参与训练
    # for name, module in root_m.named_modules():
    #     if "VisualClassifier" in name:
    #         continue
            
    #     if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.BatchNorm1d)):
    #         module.eval()
    #         module.training = False

    #         if hasattr(module, 'weight') and module.weight is not None:
    #             module.weight.requires_grad = False
    #         if hasattr(module, 'bias') and module.bias is not None:
    #             module.bias.requires_grad = False

def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model