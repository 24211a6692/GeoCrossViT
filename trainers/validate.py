"""
trainers/validate.py

This module contains the Validator class, responsible for running model verification 
and evaluation cycles over validation or testing datasets under a gradient-free 
context.

TODO:
- Support dynamic metric tracking plots generation (confusion matrices, ROC curves).
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from configs.config import GeoCrossViTConfig


class Validator:
    """
    Validation executor module. Computes evaluation metrics (Accuracy, CE Loss, InfoNCE Loss).
    Runs under torch.no_grad() context.
    """
    def __init__(
        self,
        config: GeoCrossViTConfig,
        model: nn.Module,
        val_loader: DataLoader,
        criterion_ce: nn.Module,
        criterion_info: nn.Module,
        device: torch.device
    ) -> None:
        """
        Args:
            config (GeoCrossViTConfig): Global configuration instance.
            model (nn.Module): The integrated GeoCrossViT model.
            val_loader (DataLoader): DataLoader for validation dataset.
            criterion_ce (nn.Module): CrossEntropy loss module.
            criterion_info (nn.Module): Supervised InfoNCE loss module.
            device (torch.device): Target hardware device.
        """
        self.config = config
        self.model = model
        self.val_loader = val_loader
        self.criterion_ce = criterion_ce
        self.criterion_info = criterion_info
        self.device = device
        self.lambda_info = config.train.lambda_info

    def validate(self, epoch: int) -> Dict[str, Any]:
        """
        Evaluates model performance on the validation dataset.
        
        Args:
            epoch (int): Current epoch number (used for logging purposes).
            
        Returns:
            Dict[str, Any] containing validation metrics:
                - "val_loss": Total joint validation loss.
                - "val_ce_loss": Pure CrossEntropy loss.
                - "val_info_loss": Pure Supervised InfoNCE loss.
                - "val_acc": Validation accuracy.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_ce = 0.0
        total_info = 0.0
        correct_predictions = 0
        total_samples = 0

        # Disable gradient calculations during validation
        with torch.no_grad():
            for batch in self.val_loader:
                fundus = batch["fundus"].to(self.device)
                oct_scan = batch["oct"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass in train mode to extract logits and projection embeddings
                outputs = self.model(fundus, oct_scan, mode="train")
                
                logits = outputs["logits"]
                proj_emb = outputs["projection_embedding"]

                # 1. Compute loss values
                ce_loss = self.criterion_ce(logits, labels)
                
                info_out = self.criterion_info(proj_emb, labels)
                info_loss = info_out["loss"]
                
                joint_loss = ce_loss + self.lambda_info * info_loss

                # Accumulate metrics
                batch_size = labels.shape[0]
                total_loss += joint_loss.item() * batch_size
                total_ce += ce_loss.item() * batch_size
                total_info += info_loss.item() * batch_size

                # 2. Track accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += batch_size

        # Compute dataset-wide metrics averages
        if total_samples > 0:
            val_loss = total_loss / total_samples
            val_ce = total_ce / total_samples
            val_info = total_info / total_samples
            val_acc = correct_predictions / total_samples
        else:
            val_loss, val_ce, val_info, val_acc = 0.0, 0.0, 0.0, 0.0

        return {
            "val_loss": val_loss,
            "val_ce_loss": val_ce,
            "val_info_loss": val_info,
            "val_acc": val_acc
        }
