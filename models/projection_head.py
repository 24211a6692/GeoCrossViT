"""
models/projection_head.py

This module implements the Projection Head for GeoCrossViT. It pools joint token 
sequences using a configurable pooling strategy ("mean" or "cls"), maps them into 
a low-dimensional space via an MLP projection, and applies L2 normalization for 
multimodal contrastive pretraining.

TODO:
- Support optional batch normalization or layer normalization options.
"""

from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import ModelConfig


class ProjectionHead(nn.Module):
    """
    MLP Projection Head mapping ViT token representations to a contrastive latent space.
    Supports configurable token pooling strategies.
    """
    def __init__(self, config: ModelConfig, pooling_strategy: str = "mean") -> None:
        """
        Args:
            config (ModelConfig): Model hyperparameters.
            pooling_strategy (str): Pooling strategy, either "mean" or "cls".
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.projection_dim = config.projection_dim
        
        self.pooling_strategy = pooling_strategy.lower()
        if self.pooling_strategy not in ["mean", "cls"]:
            raise ValueError(f"Invalid pooling_strategy '{self.pooling_strategy}'. Must be 'mean' or 'cls'.")
            
        # MLP Layer structures
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embed_dim, self.projection_dim)
        )

    def forward(self, joint_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            joint_tokens (torch.Tensor): Joint token tensor of shape [B, N, embed_dim].
            
        Returns:
            Dict[str, torch.Tensor]:
                - "projection_embedding": L2-normalized embeddings of shape [B, projection_dim].
                - "projection_features": Embeddings before normalization of shape [B, projection_dim].
                - "pooled_feature": Features after pooling of shape [B, embed_dim].
        """
        # Step 1: Configurable Pooling Strategy
        if self.pooling_strategy == "mean":
            pooled_feature = joint_tokens.mean(dim=1)  # Shape [B, embed_dim]
        elif self.pooling_strategy == "cls":
            pooled_feature = joint_tokens[:, 0, :]    # Shape [B, embed_dim]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")

        # Step 2: MLP Mapping
        projection_features = self.mlp(pooled_feature)  # Shape [B, projection_dim]

        # Step 3: L2 Normalization
        projection_embedding = F.normalize(projection_features, p=2, dim=-1)  # Shape [B, projection_dim]

        return {
            "projection_embedding": projection_embedding,
            "projection_features": projection_features,
            "pooled_feature": pooled_feature
        }
