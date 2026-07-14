"""
losses/info_nce.py

This module contains the InfoNCE (Information Noise-Contrastive Estimation) loss 
implementation, used for self-supervised contrastive learning to align features 
extracted from different modalities (Fundus and OCT).

TODO:
- Support distributed multi-GPU contrastive loss gathering (gather_from_all_gpus).
- Implement dynamic temperature scaling or learnable temperature parameters.
- Add support for hard-negative mining within batches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE loss computing similarity scores between modality pairs.
    """
    def __init__(self, temperature: float = 0.07) -> None:
        """
        Args:
            temperature (float): Scaling factor for cosine similarity.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss between two modalities.
        
        Args:
            features_a (torch.Tensor): Projected features of shape (B, projection_dim).
            features_b (torch.Tensor): Projected features of shape (B, projection_dim).
            
        Returns:
            torch.Tensor: Scalar loss tensor.
            
        TODO:
            - Normalize features prior to similarity matrix calculation.
            - Construct targets for diagonal matchings.
            - Compute cross-entropy loss over similarities.
        """
        # Normalize representations
        features_a = F.normalize(features_a, p=2, dim=1)
        features_b = F.normalize(features_b, p=2, dim=1)
        
        # Batch size
        batch_size = features_a.shape[0]
        
        # Placeholders
        # similarity matrices: shape (B, B)
        # targets: shape (B,) - labels representing matches on the diagonal
        dummy_loss = torch.tensor(0.0, device=features_a.device, requires_grad=True)
        
        return dummy_loss
