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
import torch.distributed as dist

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
from .validator import Validator, scale_boxes


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
    # freeze_for_vpe_and_cls(model)

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
    ref_cls = kwargs.get("ref_cls", None)
    old_module = kwargs.get("old_module", None)

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

            vis_outputs = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in outputs.items()
            }
            
            if cfg.grpo_finetune:
                if ref_module is not None:
                    with torch.no_grad():
                        ref_outputs = ref_module(samples, targets=targets)
            else:
                ref_outputs = None

            ref_cls_outputs = None
            old_cls_outputs = None

            # if cfg.grpo_cls:
            #     if (outputs1.get("grpo_boxes", None) is not None and outputs1.get("grpo_batch_idx", None) is not None):
            #         with torch.no_grad():
            #             grpo_boxes = outputs1["grpo_boxes"]          # [Total_M*G, 4]
            #             grpo_bidx = outputs1["grpo_batch_idx"]       # [Total_M*G]
            #             B = samples.shape[0]

            #             # 旧 VPE 需要 multi_scale_feats：你需要在 model forward 里把它塞进 outputs1
            #             ref_ms_feats = outputs1.get("vpe_multi_scale_feats", None)
            #             if ref_ms_feats is None:
            #                 raise RuntimeError(
            #                     "cfg.grpo_cls=True 但 outputs1 缺少 'vpe_multi_scale_feats'。"
            #                     "请在 model.forward / VisualClassifier.sample_grpo_features 调用处把 multi_scale_feats 写入 outputs1。"
            #                 )

            #             # pad 成 [B, Nmax, 4] + mask，供 ref_vc.vpe 使用
            #             counts = torch.bincount(grpo_bidx, minlength=B)
            #             max_n = int(counts.max().item()) if counts.numel() > 0 else 0
            #             if max_n == 0:
            #                 ref_cls_outputs = None
            #                 old_cls_outputs = None
            #             else:
            #                 boxes_pad = grpo_boxes.new_zeros((B, max_n, 4))
            #                 mask_pad = torch.zeros((B, max_n), device=grpo_boxes.device, dtype=torch.bool)

            #                 for bi in range(B):
            #                     n = int(counts[bi].item())
            #                     if n == 0:
            #                         continue
            #                     sel = (grpo_bidx == bi)
            #                     boxes_b = grpo_boxes[sel]
            #                     boxes_pad[bi, :n] = boxes_b
            #                     mask_pad[bi, :n] = True

            #                 if ref_vc is not None:
            #                     with torch.no_grad():
            #                         # 旧 VPE + 旧分类头 输出 ref logits
            #                         ref_vc = ref_vc.to(grpo_boxes.device)
            #                         ref_vc.eval()

            #                         ref_vpe_feats_padded = ref_vc.vpe(
            #                             reference_boxes=boxes_pad,
            #                             multi_scale_feats=ref_ms_feats,
            #                         )  # [B, max_n, C]
            #                         if isinstance(ref_vpe_feats_padded, list):
            #                             ref_vpe_feats_padded = ref_vpe_feats_padded[-1]
            #                         ref_box_feats = ref_vpe_feats_padded[mask_pad.bool()]  # [Total_M*G, C]

            #                         ref_logits_flat = ref_vc.cls_head(ref_box_feats)  # [Total_M*G, num_classes]
            #                         ref_cls_outputs = ref_logits_flat.detach()
            #                 else:
            #                     ref_cls_outputs = None

            #                 if old_module is not None:
            #                     with torch.no_grad():
            #                         # 旧 VPE + 旧分类头 输出 ref logits
            #                         old_module = old_module.to(grpo_boxes.device)
            #                         old_module.eval()

            #                         old_vpe_feats_padded = old_module.vpe(
            #                             reference_boxes=boxes_pad,
            #                             multi_scale_feats=ref_ms_feats,
            #                         )  # [B, max_n, C]
            #                         if isinstance(old_vpe_feats_padded, list):
            #                             old_vpe_feats_padded = old_vpe_feats_padded[-1]
            #                         old_box_feats = old_vpe_feats_padded[mask_pad.bool()]  # [Total_M*G, C]

            #                         old_logits_flat = old_module.cls_head(old_box_feats)  # [Total_M*G, num_classes]
            #                         old_cls_outputs = old_logits_flat.detach()
            #                 else:
            #                     old_cls_outputs = None
            #     else:
            #         ref_cls_outputs = None
            #         old_cls_outputs = None

            S_ref = None
            if old_module is not None:
                with torch.no_grad():
                    with torch.autocast(device_type=str(device), cache_enabled=False):
                        old_feats = outputs1["feats"]
                        old_outputs = outputs1["outputs"]
                        old_targets = outputs1["targets"]
                        old_result = old_module(old_feats, old_outputs, targets=old_targets)
                        S_ref = old_result["pred_logits"]

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
                loss_dict = model.module.get_losses(outputs1, ref_cls_outputs=ref_cls_outputs, S_ref=S_ref)
                loss_dict2 = criterion(outputs1["outputs"], targets, ref_outputs=ref_outputs, cfg=cfg, ref_cls_outputs=None,**metas)
                loss_dict.update(loss_dict2)
                
            loss_raw = sum(loss_dict.values())
            loss = loss_raw / accum_steps 
            scaler.scale(loss).backward()   

            if (i + 1) % accum_steps == 0:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            del outputs, loss_raw, loss


        else:
            outputs, outputs1 = model(samples, targets=targets)
            vis_outputs = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in outputs.items()
            }
            if cfg.grpo_finetune:
                if ref_module is not None:
                    with torch.no_grad():
                        ref_outputs = ref_module(samples, targets=targets)
            else:
                ref_outputs = None

            if cfg.grpo_cls:
                if ref_cls is not None:
                    with torch.no_grad():
                        ref_cls = ref_cls.to(outputs1["grpo_feats"].dtype)
                        ref_cls_outputs = ref_cls(outputs1["grpo_feats"])
                        ref_cls_outputs = ref_cls_outputs.detach()   # [Total_M, Class]
            else:
                ref_cls_outputs = None

            # loss_dict = criterion(outputs, targets, ref_outputs=ref_outputs, cfg=cfg, ref_cls_outputs=None,**metas)
            loss_dict = model.module.get_losses(outputs1)
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


        #可视化GRPO采样情况
        if "dec_out_grpo_bboxes" in vis_outputs and vis_outputs["dec_out_grpo_bboxes"] is not None:
            if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():          
                    save_grpo_samples(
                        samples=samples, 
                        targets=targets, 
                        output=vis_outputs, 
                        output_dir=output_dir, 
                        split="train_grpo", 
                        num_vis_samples=12  # 从 64 个采样中随机抽 12 个显示，避免画面太乱
                    )
         
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

        # gc.collect()
        # torch.cuda.empty_cache()

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
            epoch_visualizer.plot_and_clear(save_dir=save_path, epoch=epoch)
        except Exception as e:
            print(f"绘制特征分布图失败: {e}")
    # =========================================================================

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

    # # ==================== 初始化可视化器 ====================
    # # 只在主进程且指定 epoch 时初始化（比如每 5 个 epoch 可视化一次）
    # visualize_this_epoch = (epoch % 5 == 0) or (epoch == -1)  # 每 5 个 epoch 可视化一次，可以调整
    # visualizer = None
    # # print(dist_utils.is_main_process(), visualize_this_epoch, m_output_dir is not None)
    # if dist_utils.is_main_process() and visualize_this_epoch and m_output_dir is not None:
    #     from ..misc.visualizer import InferenceVisualizer  
    #     visualizer = InferenceVisualizer(score_threshold=0., top_k=20)
    #     print(f"\n[可视化] 初始化可视化器，epoch={epoch}")

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

        # if output_dir is not None and dist_utils.is_main_process():          
        #     save_grpo_samples(
        #         samples=samples, 
        #         targets=targets, 
        #         output=outputs, 
        #         output_dir=output_dir, 
        #         split="pred", 
        #         num_vis_samples=12  # 随机抽 12 个显示，避免画面太乱
        #     )

        #  # ==================== 可视化推理结果 ====================
        # # 只在主进程、前几个 batch、且可视化器已初始化时执行
        # if dist_utils.is_main_process() and visualizer is not None and i < 2:  # 只可视化前 2 个 batch
        #     try:
        #         save_dir = os.path.join(m_output_dir, "inference_vis", f"epoch_{epoch}")                
        #         # 并排对比（需要 targets）
        #         visualizer.visualize_batch(
        #             images=samples,
        #             outputs=outputs,
        #             save_dir=save_dir,
        #             batch_idx=i,
        #             max_images=4,
        #             gt_targets=targets,
        #             draw_mode="compare",
        #             nms_threshold=0.5
        #         )
                
        #         print(f"[可视化] 已保存 batch {i} 的可视化结果到 {save_dir}")
        #     except Exception as e:
        #         print(f"[可视化] 警告: 可视化失败: {e}")
        # # ========================================================

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
            
            # # =========================================
            # # save coco predictions
            # # =========================================
            # if dist_utils.is_main_process():
            #     import json

            #     coco_results = []
            #     for image_id, output in res.items():
            #         labels = output["labels"].cpu().numpy()
            #         boxes = output["boxes"].cpu().numpy()
            #         scores = output["scores"].cpu().numpy()
            #         for lab, box, score in zip(labels, boxes, scores):
            #             x1, y1, x2, y2 = box
            #             coco_results.append({
            #                 "image_id": int(image_id),
            #                 "category_id": int(lab),
            #                 "bbox": [
            #                     float(x1),
            #                     float(y1),
            #                     float(x2 - x1),
            #                     float(y2 - y1)
            #                 ],
            #                 "score": float(score)
            #             })

            #     os.makedirs(str(m_output_dir) + "/detection_results", exist_ok=True)
            #     path = str(m_output_dir) + "/detection_results/" + f"epoch{epoch}-predictions.json"

            #     with open(path, "w") as f:
            #         json.dump(coco_results, f)
            #     print(f"saved prediction json: {path}")

            #     from tidecv import TIDE, datasets
            #     tide = TIDE()
            #     tide.evaluate_range(
            #         datasets.COCO("/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/val5000-cocolike-cat10.json"),
            #         datasets.COCOResult(path),
            #         mode=TIDE.BOX
            #     )
            #     tide.summarize()

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

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds, conf_thresh=0).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()      
        coco_evaluator.summarize()

        stats_obj = coco_evaluator.coco_eval["bbox"]    # 获取 bbox 评估结果对象
        cat_ids = stats_obj.params.catIds
        precision = stats_obj.eval['precision']

        # 用于计算排除 label 0 后的平均值
        valid_aps_50_95 = []
        valid_aps_50 = []
        for i, cat_id in enumerate(cat_ids):
            ap_50_95 = precision[:, :, i, 0, -1].mean()
            ap_50 = precision[0, :, i, 0, -1].mean()
            
            if cat_id != 0:  # 排除类别0
                valid_aps_50_95.append(ap_50_95)
                valid_aps_50.append(ap_50)

        # 计算排除类别0后的 mAP（所有进程都计算）
        if len(valid_aps_50_95) > 0:
            mean_ap_50_95 = sum(valid_aps_50_95) / len(valid_aps_50_95)
            mean_ap_50 = sum(valid_aps_50) / len(valid_aps_50)
        else:
            mean_ap_50_95 = 0.0
            mean_ap_50 = 0.0

        if dist_utils.is_main_process():
            print("\n" + "="*50)
            print(f"{'CatID':<10} | {'AP@50:95':<12} | {'AP@50':<10}")
            print("-" * 50)
            
            for i, cat_id in enumerate(cat_ids):
                ap_50_95 = precision[:, :, i, 0, -1].mean()
                ap_50 = precision[0, :, i, 0, -1].mean()
                print(f"{cat_id:<10} | {ap_50_95:<12.4f} | {ap_50:<10.4f}")
            
            print("="*50)
            print(f"{'mAP(excl.0)':<10} | {mean_ap_50_95:<12.4f} | {mean_ap_50:<10.4f}")
            print("="*50 + "\n")


    # # 保存检测结果
    # if dist_utils.is_dist_available_and_initialized():
    #     # 收集所有rank的gt和preds
    #     all_gt = gather_all_ranks_data(gt)
    #     all_preds = gather_all_ranks_data(preds)
        
    #     if dist_utils.is_main_process():
    #         # 只有主进程保存汇总结果
    #         json_path = dist_utils.save_detection_results(
    #             gt=all_gt,
    #             preds=all_preds,
    #             save_dir=str(m_output_dir) + "/detection_results",
    #             epoch=epoch,
    #         )

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["bbox_mean_ap_50"] = [mean_ap_50]

            # stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
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
            save_path = output_dir if output_dir is not None else "./output"
            epoch_visualizer.plot_and_clear(save_dir=save_path, epoch=f"{epoch}_eval")
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

def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def gather_all_ranks_data(data_list):
    """收集所有rank的数据到主进程（支持任意Python对象）"""
    if not dist.is_initialized():
        return data_list
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 创建存储所有rank数据的列表
    gathered_data = [None] * world_size
    
    # 使用 all_gather_object 收集所有数据
    dist.all_gather_object(gathered_data, data_list)
    
    # 展平所有数据
    if rank == 0:
        all_data = []
        for r_data in gathered_data:
            all_data.extend(r_data)
        return all_data
    else:
        return None
