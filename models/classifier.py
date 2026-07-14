"""
models/classifier.py

This module implements the downstream DiseaseClassifier for GeoCrossViT, which 
takes a pooled multi-modal feature representation and predicts disease category logits.

TODO:
- Support optional multi-head setups for clinical multi-label settings.
"""

from typing import Dict
import torch
import torch.nn as nn
from configs.config import ModelConfig


class DiseaseClassifier(nn.Module):
    """
    Classification head mapping pooled multi-modal feature representations to diagnostic class logits.
    """
    def __init__(self, config: ModelConfig) -> None:
        """
        Args:
            config (ModelConfig): Model configuration parameters.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_classes = config.num_classes
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, pooled_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pooled_feature (torch.Tensor): Pooled joint representation of shape [B, embed_dim].
            
        Returns:
            Dict[str, torch.Tensor]:
                - "logits": Logits tensor of shape [B, num_classes].
        """
        logits = self.classifier(pooled_feature)  # Shape [B, num_classes]
        return {
            "logits": logits
        }
