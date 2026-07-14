"""
losses/info_nce.py

This module contains the Supervised InfoNCE Loss implementation. It computes a 
supervised contrastive loss (SupCon) that clusters samples from the same diagnostic 
class while pushing samples from different diagnostic classes apart. It supports 
multiple positive pairs in a batch, dynamically filters singleton anchors, and handles 
empty positive conditions gracefully.

TODO:
- Support optional dual-modality pretraining loss balancing (Fundus-OCT cross alignment).
- Support distributed multi-GPU gather operations inside _compute_similarity_matrix.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


class SupervisedInfoNCELoss(nn.Module):
    """
    Supervised InfoNCE Loss (SupCon) supporting multiple positive pairs and 
    modular similarity calculation for future distributed training.
    """
    def __init__(self, temperature: float = 0.07) -> None:
        """
        Args:
            temperature (float): Contrastive temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature

    def _compute_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity matrix. Exposing this as a modular helper 
        method allows distributed environments to override it with gathered features.
        
        Args:
            embeddings (torch.Tensor): L2-normalized embeddings, shape [B, D].
            
        Returns:
            torch.Tensor: Similarity matrix of shape [B, B].
        """
        # Cosine similarity is a simple dot product for L2-normalized vectors
        return torch.matmul(embeddings, embeddings.T)

    def forward(
        self,
        projection_embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Computes the supervised contrastive loss.
        
        Args:
            projection_embeddings (torch.Tensor): Normalized embeddings of shape [B, D].
            labels (torch.Tensor): Ground-truth labels of shape [B].
            
        Returns:
            Dict[str, Any]:
                - "loss": Scalar contrastive loss tensor.
                - "similarity_matrix": Raw cosine similarity matrix of shape [B, B].
                - "num_positive_pairs": Total positive pair combinations in the batch.
                - "temperature": The scale factor used.
        """
        device = projection_embeddings.device
        batch_size = projection_embeddings.shape[0]

        # 1. Compute raw similarity matrix
        similarity_matrix = self._compute_similarity_matrix(projection_embeddings)  # Shape [B, B]
        
        # 2. Scale similarities by temperature to get logits
        logits = similarity_matrix / self.temperature  # Shape [B, B]
        
        # 3. Create self-exclusion mask (excluding diagonal)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        non_self_mask = ~self_mask  # Shape [B, B]
        
        # 4. Create positive pairs mask (same label, excluding self)
        labels_col = labels.view(-1, 1)
        pos_mask = torch.eq(labels_col, labels_col.T) & non_self_mask  # Shape [B, B]
        
        num_positive_pairs = int(pos_mask.sum().item())

        # 5. Gating: Handle batches containing no positive pairs
        if num_positive_pairs == 0:
            # Returns a zero tensor connected to the computational graph to preserve gradient flow
            loss = projection_embeddings.sum() * 0.0
            return {
                "loss": loss,
                "similarity_matrix": similarity_matrix,
                "num_positive_pairs": 0,
                "temperature": self.temperature
            }

        # 6. Apply numerical stability correction (subtract max logit per row)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stable = logits - logits_max.detach()  # Shape [B, B]
        
        # 7. Compute exp logits and sum them excluding self-comparisons (denominator)
        exp_logits = torch.exp(logits_stable) * non_self_mask.float()
        sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)  # Shape [B, 1]
        
        # Log probability computation
        log_prob = logits_stable - torch.log(sum_exp_logits + 1e-8)  # Shape [B, B]

        # 8. Compute SupCon loss over anchors with at least one other positive sample (valid anchors)
        pos_count = pos_mask.float().sum(dim=1)  # Shape [B]
        valid_anchors_mask = pos_count > 0       # Shape [B]
        
        # Compute mean log prob per positive pair for each anchor
        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / (pos_count + 1e-8)  # Shape [B]

        # Average over valid anchors
        loss = -mean_log_prob_pos[valid_anchors_mask].mean()

        return {
            "loss": loss,
            "similarity_matrix": similarity_matrix,
            "num_positive_pairs": num_positive_pairs,
            "temperature": self.temperature
        }
