"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import atexit
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from datetime import datetime
import json

# from torch.utils.data.dataloader import DataLoader
from ..data import DataLoader


def setup_distributed(
    print_rank: int = 0,
    print_method: str = "builtin",
    seed: int = None,
):
    """
    env setup
    args:
        print_rank,
        print_method, (builtin, rich)
        seed,
    """
    try:
        # https://pytorch.org/docs/stable/elastic/run.html
        RANK = int(os.getenv("RANK", -1))
        LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
        WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

        # torch.distributed.init_process_group(backend=backend, init_method='env://')
        torch.distributed.init_process_group(init_method="env://")
        torch.distributed.barrier()

        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        enabled_dist = True
        if get_rank() == print_rank:
            print("Initialized distributed mode...")

    except Exception:
        enabled_dist = False
        print("Not init distributed mode.")

    setup_print(get_rank() == print_rank, method=print_method)
    if seed is not None:
        setup_seed(seed)

    return enabled_dist


def setup_print(is_main, method="builtin"):
    """This function disables printing when not in master process"""
    import builtins as __builtin__

    if method == "builtin":
        builtin_print = __builtin__.print

    elif method == "rich":
        import rich

        builtin_print = rich.print

    else:
        raise AttributeError("")

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


@atexit.register
def cleanup():
    """cleanup distributed environment"""
    if is_dist_available_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def warp_model(
    model: torch.nn.Module,
    sync_bn: bool = False,
    dist_mode: str = "ddp",
    find_unused_parameters: bool = True,
    compile: bool = False,
    compile_mode: str = "reduce-overhead",
    **kwargs,
):
    if is_dist_available_and_initialized():
        rank = get_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        print(f"find_unused_parameters:{find_unused_parameters}")
        if dist_mode == "dp":
            model = DP(model, device_ids=[rank], output_device=rank)
        elif dist_mode == "ddp":
            model = DDP(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            raise AttributeError("")

    if compile:
        model = torch.compile(model, mode=compile_mode)

    return model


def de_model(model):
    return de_parallel(de_complie(model))


def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(
            loader.dataset,
            loader.batch_size,
            sampler=sampler,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            num_workers=loader.num_workers,
        )
    return loader


def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (
        torch.nn.parallel.DataParallel,
        torch.nn.parallel.DistributedDataParallel,
    )


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    """
    Args
        data dict: input, {k: v, ...}
        avg bool: true
    """
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)

        if avg is True:
            values /= world_size

        return {k: v for k, v in zip(keys, values)}


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    torch.distributed.all_gather_object(data_list, data)
    return data_list


def sync_time():
    """sync_time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()


def setup_seed(seed: int, deterministic=False):
    """setup_seed for reproducibility
    torch.manual_seed(3407) is all you need. https://arxiv.org/abs/2109.08203
    """
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # memory will be large when setting deterministic to True
    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True


# for torch.compile
def check_compile():
    import warnings

    import torch

    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower " "than expected."
        )
    return gpu_ok


def is_compile(model):
    import torch._dynamo

    return type(model) in (torch._dynamo.OptimizedModule,)


def de_complie(model):
    return model._orig_mod if is_compile(model) else model


def save_detection_results(gt, preds, save_dir="./detection_results", 
                          epoch=0):
    """
    保存检测结果到文件
    
    Args:
        gt: 真实框列表，每个元素包含"boxes"和"labels"
        preds: 预测框列表，每个元素包含"boxes"、"labels"和"scores"
        save_dir: 保存目录
        epoch: 当前epoch
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名（包含时间戳避免覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"detection_results_epoch{epoch}_{timestamp}.json"
    json_path = os.path.join(save_dir, json_filename)
    
    # 转换数据为可序列化的格式
    detection_data = []
    
    for img_idx, (gt_item, pred_item) in enumerate(zip(gt, preds)):
        # 获取真实框数据
        gt_boxes = gt_item["boxes"].cpu().numpy() if hasattr(gt_item["boxes"], 'cpu') else gt_item["boxes"]
        gt_labels = gt_item["labels"].cpu().numpy() if hasattr(gt_item["labels"], 'cpu') else gt_item["labels"]
        pred_boxes = pred_item["boxes"].cpu().numpy() if hasattr(pred_item["boxes"], 'cpu') else pred_item["boxes"]
        pred_labels = pred_item["labels"].cpu().numpy() if hasattr(pred_item["labels"], 'cpu') else pred_item["labels"]
        pred_scores = pred_item["scores"].cpu().numpy() if hasattr(pred_item["scores"], 'cpu') else pred_item["scores"]
        
        # 转换真实框信息
        gt_list = []
        for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            gt_info = {
                "gt_id": i,
                "bbox": box.tolist() if hasattr(box, 'tolist') else box,
                "label": int(label),
            }
            gt_list.append(gt_info)
        
        # 转换预测框信息
        pred_list = []
        for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            pred_info = {
                "pred_id": i,
                "bbox": box.tolist() if hasattr(box, 'tolist') else box,
                "label": int(label),
                "score": float(score)
            }
            pred_list.append(pred_info)
        
        # 图像级别的检测结果
        img_result = {
            "image_id": img_idx,
            "ground_truths": gt_list,
            "predictions": pred_list,
            "statistics": {
                "num_gt": len(gt_list),
                "num_pred": len(pred_list)
            }
        }
        
        detection_data.append(img_result)
    
    # 计算总体统计
    total_stats = {
        "epoch": epoch,
        "timestamp": timestamp,
        "total_images": len(detection_data),
        "total_gt_boxes": sum(item["statistics"]["num_gt"] for item in detection_data),
        "total_pred_boxes": sum(item["statistics"]["num_pred"] for item in detection_data),
    }
    
    # 创建最终结果结构
    final_result = {
        "metadata": total_stats,
        "detections": detection_data
    }
    
    # 保存为JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False, 
                  default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"检测结果已保存到: {json_path}")
    print(f"统计信息: 图像数={total_stats['total_images']}, "
          f"真实框数={total_stats['total_gt_boxes']}, "
          f"预测框数={total_stats['total_pred_boxes']}")
    
    return json_path     
        # 获取预测框数据
   
