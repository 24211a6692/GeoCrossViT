"""
models/attention_visualizer.py

This module contains class/methods to extract, format, and visualize self-attention 
and cross-attention matrices from the Vision Transformer encoders and cross-modal 
fusion layers.

TODO:
- Implement rollout attention visualization (e.g. Attention Rollout).
- Create patch-to-image mapping functions to project patch tokens back to image space.
- Integrate heatmap plotting using matplotlib or seaborn.
"""

from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class AttentionVisualizer:
    """
    Utility class to extract attention weights and construct overlay heatmaps 
    for retinal scans, highlighting focus areas during cross-modal fusion.
    """
    def __init__(self, patch_size: int = 16, img_size: int = 224) -> None:
        """
        Args:
            patch_size (int): Size of the patches (default: 16).
            img_size (int): Size of the input images (default: 224).
        """
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size

    def extract_attention_weights(self, model: nn.Module, sample_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Hooks into the model or accesses registered attention logs to extract weights.
        
        Args:
            model (nn.Module): Instantiated GeoCrossViT model.
            sample_batch (dict): Batch of inputs.
            
        Returns:
            torch.Tensor: Attention weights of shape (B, num_heads, N_queries, M_keys).
            
        TODO:
            - Register forward hooks to capture multi-head attention weights during inference.
        """
        # Placeholder tensor
        batch_size = next(iter(sample_batch.values())).shape[0]
        num_patches = self.grid_size ** 2
        # Stub attention matrix: query-to-key attention
        dummy_attn = torch.ones((batch_size, 8, num_patches, num_patches)) / num_patches
        return dummy_attn

    def generate_attention_map(
        self, 
        attention_weights: torch.Tensor, 
        target_patch_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Reshapes patch attention weights into 2D heatmaps.
        
        Args:
            attention_weights (torch.Tensor): Attention map from a specific layer (B, N, N) or (B, N_queries, M_keys).
            target_patch_idx (int, optional): Specific query patch index to visualize attention for.
            
        Returns:
            np.ndarray: Rescaled 2D heatmap matrix of shape (H, W).
            
        TODO:
            - Resize heatmap back to (img_size, img_size) using interpolation.
            - Handle cross-modal alignment projections.
        """
        # Placeholder heatmap creation
        heatmap = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        # Create a simple dummy gradient heatmap
        x = np.linspace(-3, 3, self.img_size)
        y = np.linspace(-3, 3, self.img_size)
        xx, yy = np.meshgrid(x, y)
        heatmap = np.exp(-0.5 * (xx**2 + yy**2))
        
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    def plot_attention_overlay(
        self, 
        image: np.ndarray, 
        attention_map: np.ndarray, 
        alpha: float = 0.5,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Overlays the computed attention heatmap onto the input retinal image.
        
        Args:
            image (np.ndarray): Original image array of shape (H, W, 3).
            attention_map (np.ndarray): Attention heatmap of shape (H, W).
            alpha (float): Transparency factor for the overlay.
            save_path (str, optional): Target directory/file to export the plot.
            
        Returns:
            plt.Figure: Matplotlib figure containing the overlay plot.
            
        TODO:
            - Implement colormap mapping and bounding box overlay.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # Placeholders
        ax.imshow(image)
        ax.imshow(attention_map, cmap='jet', alpha=alpha)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        return fig
