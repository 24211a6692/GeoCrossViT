"""
trainers/trainer.py

This module contains the primary training engine for GeoCrossViT. It configures 
the training loop, mixed precision (AMP), gradient clipping, early stopping, 
checkpointing, and logging (TensorBoard and print summaries).

TODO:
- Support optional distributed data parallel (DDP) scaling.
"""

import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    # Fallback placeholder if TensorBoard is not installed
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from configs.config import GeoCrossViTConfig
from trainers.validate import Validator
from utils.logger import setup_logger

logger = setup_logger(name="GeoCrossTrainer", log_file="trainer.log")


class GeoCrossTrainer:
    """
    Core orchestrator managing training epochs, validation cycles, checkpoints, 
    early stopping, and mixed precision.
    """
    def __init__(
        self,
        config: GeoCrossViTConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        criterion_ce: nn.Module,
        criterion_info: nn.Module
    ) -> None:
        """
        Args:
            config (GeoCrossViTConfig): Global configuration instance.
            model (nn.Module): The integrated GeoCrossViT model.
            train_loader (DataLoader): Training DataLoader.
            val_loader (DataLoader): Validation DataLoader.
            optimizer (Optimizer): PyTorch optimizer.
            lr_scheduler (Scheduler): CosineAnnealingLR scheduler.
            criterion_ce (nn.Module): CrossEntropy loss module.
            criterion_info (nn.Module): Supervised InfoNCE loss module.
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion_ce = criterion_ce
        self.criterion_info = criterion_info

        # Hardware Setup
        self.device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Config parameters
        self.epochs = config.train.epochs
        self.lambda_info = config.train.lambda_info
        self.patience = config.train.patience
        self.grad_clip = config.train.grad_clip
        self.checkpoint_dir = config.train.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Mixed Precision Setup (AMP is only enabled on CUDA devices)
        self.use_amp = config.train.use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        logger.info(f"Mixed Precision (AMP) enabled: {self.use_amp} on device: {self.device}")

        # Metrics Tracking Setup
        self.best_metric_name = config.train.best_metric_name.lower()
        if self.best_metric_name not in ["val_acc", "val_loss"]:
            raise ValueError(f"Invalid best_metric_name '{self.best_metric_name}'. Must be 'val_acc' or 'val_loss'.")
        
        self.best_metric_val = float("inf") if self.best_metric_name == "val_loss" else -1.0
        self.patience_counter = 0

        # Tensorboard Logging
        self.use_tb = config.train.use_tensorboard
        if self.use_tb:
            tb_log_dir = os.path.join(config.train.log_dir, "tb_logs")
            self.writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info(f"TensorBoard logging initialized at: {tb_log_dir}")
        else:
            self.writer = SummaryWriter()

        # Instantiate Validator
        self.validator = Validator(
            config=config,
            model=model,
            val_loader=val_loader,
            criterion_ce=criterion_ce,
            criterion_info=criterion_info,
            device=self.device
        )

    def _is_better_metric(self, current: float, best: float) -> bool:
        """
        Determines whether the current metric is better than the previous best.
        """
        if self.best_metric_name == "val_loss":
            return current < best
        return current > best

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Executes one complete epoch of training.
        """
        self.model.train()
        
        total_loss = 0.0
        total_ce = 0.0
        total_info = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            fundus = batch["fundus"].to(self.device)
            oct_scan = batch["oct"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed-precision autocast
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(fundus, oct_scan, mode="train")
                logits = outputs["logits"]
                proj_emb = outputs["projection_embedding"]

                ce_loss = self.criterion_ce(logits, labels)
                info_out = self.criterion_info(proj_emb, labels)
                info_loss = info_out["loss"]

                loss = ce_loss + self.lambda_info * info_loss

            # Backward pass and scaling
            self.scaler.scale(loss).backward()

            # Gradient Clipping (must unscale before clipping parameters)
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            # Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate batch stats
            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_ce += ce_loss.item() * batch_size
            total_info += info_loss.item() * batch_size

            # Track accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

        # Averages
        epoch_loss = total_loss / total_samples
        epoch_ce = total_ce / total_samples
        epoch_info = total_info / total_samples
        epoch_acc = correct_predictions / total_samples

        return {
            "train_loss": epoch_loss,
            "train_ce_loss": epoch_ce,
            "train_info_loss": epoch_info,
            "train_acc": epoch_acc
        }

    def fit(self) -> None:
        """
        Orchestrates full train/validation sequence over epochs.
        """
        logger.info("Starting GeoCrossViT model fit...")
        
        for epoch in range(1, self.epochs + 1):
            # 1. Train
            train_metrics = self.train_epoch(epoch)
            
            # 2. Validate (disabled gradients inside)
            val_metrics = self.validator.validate(epoch)

            # 3. Retrieve current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 4. Step LR Scheduler
            self.lr_scheduler.step()

            # 5. Log progress summaries
            msg = (
                f"Epoch [{epoch}/{self.epochs}] | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} (CE: {train_metrics['train_ce_loss']:.4f}, InfoNCE: {train_metrics['train_info_loss']:.4f}) | "
                f"Train Acc: {train_metrics['train_acc']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} (CE: {val_metrics['val_ce_loss']:.4f}, InfoNCE: {val_metrics['val_info_loss']:.4f}) | "
                f"Val Acc: {val_metrics['val_acc']:.4f}"
            )
            logger.info(msg)
            print(msg)

            # Log to TensorBoard
            if self.use_tb:
                self.writer.add_scalar("Loss/Train_Total", train_metrics["train_loss"], epoch)
                self.writer.add_scalar("Loss/Train_CE", train_metrics["train_ce_loss"], epoch)
                self.writer.add_scalar("Loss/Train_InfoNCE", train_metrics["train_info_loss"], epoch)
                self.writer.add_scalar("Accuracy/Train", train_metrics["train_acc"], epoch)
                
                self.writer.add_scalar("Loss/Val_Total", val_metrics["val_loss"], epoch)
                self.writer.add_scalar("Loss/Val_CE", val_metrics["val_ce_loss"], epoch)
                self.writer.add_scalar("Loss/Val_InfoNCE", val_metrics["val_info_loss"], epoch)
                self.writer.add_scalar("Accuracy/Val", val_metrics["val_acc"], epoch)
                
                self.writer.add_scalar("Train/Learning_Rate", current_lr, epoch)

            # 6. Checkpoint Saving Logic
            val_metric = val_metrics[self.best_metric_name]
            
            # Save latest checkpoint dictionary (full training state)
            latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
            full_checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_metric_val": self.best_metric_val,
                "best_metric_name": self.best_metric_name
            }
            torch.save(full_checkpoint, latest_path)

            # Check if this is the best epoch
            if self._is_better_metric(val_metric, self.best_metric_val):
                self.best_metric_val = val_metric
                self.patience_counter = 0
                
                # Save best weights only
                best_weights_path = os.path.join(self.checkpoint_dir, "best_model_weights.pth")
                torch.save(self.model.state_dict(), best_weights_path)
                logger.info(f"--> Saved new best model weights based on {self.best_metric_name}: {val_metric:.4f}")
            else:
                self.patience_counter += 1

            # 7. Early Stopping Checks
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without {self.best_metric_name} improvement.")
                print(f"Early stopping triggered after {self.patience} epochs without {self.best_metric_name} improvement.")
                break

        if self.use_tb:
            self.writer.close()
