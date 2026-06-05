"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time
import torch.distributed as dist
import os

import torch
import copy
from copy import deepcopy


from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def pretrain(self):
        self.train()
        args = self.cfg
        device = self.device

        freeze_for_vpe_pretrain(self.model) 
        model_wo_ddp = unwrap_model(self.model) 

        # vpe_ckpt_path = "/root/userfolder/Projects/RLCCD/output/dfine_hgnetv2_m_ccd/VPE/vpe_best.pth"
        # cls_head_path = "/root/userfolder/Projects/RLCCD/output/dfine_hgnetv2_m_ccd/VPE/cls_head_best.pth"
        # if os.path.exists(vpe_ckpt_path):
        #     state_dict = torch.load(vpe_ckpt_path, map_location='cpu')
        #     model_wo_ddp.VPEPretrainWrapper.vpe.load_state_dict(state_dict)
        #     print(f"Successfully loaded VPE checkpoint from {vpe_ckpt_path}")
        #     del state_dict
        # else:
        #     print(f"VPE checkpoint not found at {vpe_ckpt_path}, skipping loading.")

        # if os.path.exists(cls_head_path):
        #     state_dict = torch.load(cls_head_path, map_location='cpu')
        #     model_wo_ddp.VPEPretrainWrapper.cls_head.load_state_dict(state_dict)
        #     print(f"Successfully loaded VPE checkpoint from {cls_head_path}")
        #     del state_dict
        # else:
        #     print(f"VPE checkpoint not found at {cls_head_path}, skipping loading.")


        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
        
        print(f"[*] Pretrain focus: VPE module. Trainable params: {sum(p.numel() for p in trainable_params)}")
        print("-" * 30 + " Start VPE Pretraining " + "-" * 30)

        best_acc = 0.0
        start_time = time.time()

        
        # #评估
        # val_acc = self.validate_vpe(self.val_dataloader, device) 
        # print(f"Val_Acc: {val_acc:.4f}")
        
        # --- 训练循环 ---
        num_epochs = 30
        for epoch in range(num_epochs):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            epoch_loss = 0.0
            correct_total = 0
            sample_total = 0

            for i, (samples, targets) in enumerate(self.train_dataloader):
                if i % 10 == 0:
                    if dist_utils.is_dist_available_and_initialized():
                        torch.cuda.synchronize() # 确保所有卡都跑完了再清理
                    torch.cuda.empty_cache()
                samples = samples.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                # 混合精度前向传播
                with torch.autocast(device_type=str(device), cache_enabled=True):
                    # 调用 VPEPretrain 的 forward (包含 Backbone -> Encoder -> VPE -> Loss)
                    loss_dict, logits, labels = self.model(samples, targets)                  
                    loss = loss_dict.get("total_loss")

                # 反向传播更新
                if loss is not None and torch.isfinite(loss):
                    optimizer.zero_grad()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    continue

                # --- 实时评估逻辑 ---
                with torch.no_grad():
                    if logits is not None:
                        preds = logits.detach().sigmoid().argmax(dim=1)
                        # 计算当前进程的正确数和总数
                        curr_correct = (preds == labels).sum()
                        curr_total = torch.tensor(labels.size(0), device=device)
                        
                        # 如果是 DDP，聚合所有进程的数据以显示真实的 Batch 状态
                        if dist_utils.is_dist_available_and_initialized():
                            # 聚合 loss, correct, total
                            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                            dist.all_reduce(curr_correct, op=dist.ReduceOp.SUM)
                            dist.all_reduce(curr_total, op=dist.ReduceOp.SUM)
                            
                            # 获取平均 loss (总和除以进程数)
                            world_size = dist.get_world_size()
                            display_loss = loss.item() / world_size
                        else:
                            display_loss = loss.item()
                        
                        batch_acc = curr_correct.item() / curr_total.item()
                        
                        # 累计用于 Epoch 统计
                        epoch_loss += display_loss
                        correct_total += curr_correct.item()
                        sample_total += curr_total.item()

                        # 每隔一定步数打印 Batch 结果 (避免日志刷屏)
                        if i % 10 == 0 and dist_utils.is_main_process():
                            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(self.train_dataloader)}] "
                                  f"Loss: {display_loss:.4f} Batch_Acc: {batch_acc:.4f}")

                # 释放显存引用
                del loss_dict, logits, labels

            #评估
            val_acc = self.validate_vpe(self.val_dataloader, device) 

            # --- Epoch 总结与权重保存 ---
            if dist_utils.is_main_process():
                avg_loss = epoch_loss / len(self.train_dataloader)
                epoch_acc = correct_total / sample_total if sample_total > 0 else 0.0
                print(f"\n>> Epoch {epoch} Final Result: Loss={avg_loss:.4f}, Acc={epoch_acc:.4f}, Val_Acc: {val_acc:.4f}")

                if self.output_dir:
                    vpe_dir = self.output_dir / "VPE"
                    vpe_dir.mkdir(parents=True, exist_ok=True)

                    vpe_state = model_wo_ddp.VPEPretrainWrapper.vpe.state_dict()
                    cls_head_state = model_wo_ddp.VPEPretrainWrapper.cls_head.state_dict()
                    torch.save(vpe_state, vpe_dir / "vpe_latest.pth")
                    torch.save(cls_head_state, vpe_dir / "cls_head_latest.pth")
                    
                    # 保存表现最好的权重
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(vpe_state, vpe_dir / "vpe_best.pth")
                        torch.save(cls_head_state, vpe_dir / "cls_head_best.pth")
                        print(f"[*] New Best Acc: {best_acc:.4f}, saved 'vpe_wrapper_best.pth' and 'cls_head_best.pth'")

        total_time = time.time() - start_time
        print(f"Pretraining Finished. Total time: {str(datetime.timedelta(seconds=int(total_time)))}")
        print(f"Best VPE Accuracy recorded: {best_acc:.4f}")
    
    @torch.no_grad()
    def validate_vpe(self, val_dataloader, device):
        self.model.eval() # 切换到评估模式
        correct_total = 0
        sample_total = 0
        
        for samples, targets in val_dataloader:
            samples = samples.to(device)
            w, h = samples.shape[-2], samples.shape[-1]
            new_targets = []
            for t in targets:
                t_new = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                
                # --- 归一化 boxes ---
                # 验证集 boxes 格式为 [x1, y1, x2, y2] 的绝对像素值
                denom = torch.tensor([w, h, w, h], device=device, dtype=torch.float32)
                t_new["boxes"] = t_new["boxes"] / denom
                # ----------------------------
                
                new_targets.append(t_new)

            _, logits, labels = self.model(samples, new_targets, box_fmt="xyxy")
            
            if logits is not None:
                preds = logits.sigmoid().argmax(dim=1)
                correct_total += (preds == labels).sum().item()
                sample_total += labels.size(0)

        # DDP 环境下同步结果
        if dist_utils.is_dist_available_and_initialized():
            metrics = torch.tensor([correct_total, sample_total], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            correct_total, sample_total = metrics[0].item(), metrics[1].item()

        val_acc = correct_total / sample_total if sample_total > 0 else 0.0
        self.model.train() # 切换回训练模式
        return val_acc

    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        model_wo_ddp = unwrap_model(self.model) 
        
        # vpe_ckpt_path = "/root/userfolder/Projects/RLCCD/weight/VPE/vpe_best.pth"       
        # if os.path.exists(vpe_ckpt_path):
        #     state_dict = torch.load(vpe_ckpt_path, map_location='cpu')
        #     model_wo_ddp.VisualClassifier.vpe.load_state_dict(state_dict)
        #     print(f"Successfully loaded VPE checkpoint from {vpe_ckpt_path}")
        #     del state_dict
        # else:
        #     print(f"VPE checkpoint not found at {vpe_ckpt_path}, skipping loading.")

        # cls_head_path = "/root/userfolder/Projects/RLCCD/weight/VPE/cls_head_best.pth"
        # if os.path.exists(cls_head_path):
        #     state_dict = torch.load(cls_head_path, map_location='cpu')
        #     model_wo_ddp.VisualClassifier.cls_head.load_state_dict(state_dict)
        #     print(f"Successfully loaded cls checkpoint from {cls_head_path}")
        #     del state_dict
        # else:
        #     print(f"cls checkpoint not found at {cls_head_path}, skipping loading.")
        

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        if self.last_epoch > 0:
            module = self.ema if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb,
                m_output_dir=self.output_dir,
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        # load_visual_classifier_from_model2(self.model, "/root/userfolder/Projects/RLCCD/output/dfine_hgnetv2_m_ccd/6_1/34/best_stg1.pth", device=self.device)

        if args.yaml_cfg["grpo_finetune"]:
            print("-" * 42 + "GRPO finetuning"+ "-" * 43)
            self.ref_module = copy.deepcopy(self.model.module)
        else:
            self.ref_module = None

        if args.yaml_cfg.get("grpo_cls", False):
            print("-" * 30 + " Initializing GRPO Reference VisualClassifier " + "-" * 30)

            #初始模型作为参考模型
            curr_vc = self.model.module.VisualClassifier if hasattr(self.model, "module") else self.model.VisualClassifier
            self.ref_vc = deepcopy(curr_vc)  # 直接拷贝整个 VC: vpe + cls_head + 其它子模块

            self.ref_vc.eval()
            for p in self.ref_vc.parameters():
                p.requires_grad = False

            print("GRPO Reference VisualClassifier locked and loaded.")
        else:
            self.ref_vc = None

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            old_model = self.ema if self.ema else self.model
            old_vc = self.model.module.VisualClassifier if hasattr(old_model, "module") else self.model.VisualClassifier
            old_module = deepcopy(old_vc)
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
                postprocessor=self.postprocessor,
                ref_module=self.ref_module,
                cfg=self.cfg,
                ref_vc=self.ref_vc,
                old_module=old_module,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: LR = {current_lr}")

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
                m_output_dir=self.output_dir,
            )

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                        else:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")  # global best

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {
                        "epoch": -1,
                    }
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        # load_visual_classifier_from_model2(self.model, "/root/userfolder/Projects/RLCCD/output/dfine_hgnetv2_m_ccd/6_1/35/best_stg1.pth", device=self.device)
        module = self.ema if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
            m_output_dir=self.output_dir,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return


def freeze_for_vpe_pretrain(model):
    # 递归获取原始模型
    m = unwrap_model(model)
    model.train()

    # 冻结所有参数
    for p in m.parameters():
        p.requires_grad = False

    #  解冻 VPE 相关的参数 (包括分类头)
    for p in m.VPEPretrainWrapper.parameters():
        p.requires_grad = True
    for p in m.v_to_t_projection.parameters():
        p.requires_grad = True
    
    print("Optimization target: Only VisualClassifier is trainable.")


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def load_visual_classifier_from_model2(model, model2_path, device='cuda'):
    """
    从模型2加载 VisualClassifier 权重到现有模型
    
    Args:
        model: 已经加载了模型1权重的模型实例
        model2_path: 模型2的权重文件路径
        device: 设备
    
    Returns:
        model: 更新了 VisualClassifier 权重的模型
    """
    # 获取当前模型的 VisualClassifier
    curr_model = model.module if hasattr(model, 'module') else model
    curr_vc = curr_model.VisualClassifier
    
    # 加载模型2的权重
    checkpoint = torch.load(model2_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    
    # 提取并去除 VisualClassifier 前缀
    new_state_dict = {}
    prefix = "VisualClassifier."
    prefix_len = len(prefix)
    
    for k, v in state_dict.items():
        if k.startswith(prefix):
            # 去除 'VisualClassifier.' 前缀
            new_k = k[prefix_len:]
            new_state_dict[new_k] = v
    
    # 加载到 VisualClassifier（strict=False 允许缺失键）
    msg = curr_vc.load_state_dict(new_state_dict, strict=False)
    
    print(f"VisualClassifier Load Status: {msg}")
    print(f"Loaded {len(new_state_dict)} parameters from model2")
    
    return model
 
