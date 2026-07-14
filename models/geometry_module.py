"""
models/geometry_module.py

This module contains the Geometry-Aware Local Topology Module (GALTM). 
GALTM processes Vision Transformer patch tokens using depthwise/pointwise 
convolutions, stochastic depth (DropPath), and a lightweight spatial attention 
mechanism to capture spatial and topological relationships between patches.

TODO:
- Support non-square patch structures dynamically using factor matching.
- Add configuration parameters for convolution kernel size and strides.
"""

import math
from typing import Dict, Tuple, Union, Optional
import torch
import torch.nn as nn
from configs.config import ModelConfig

# Try to import DropPath from timm, fallback to custom implementation if unavailable
try:
    from timm.layers import DropPath
except ImportError:
    class DropPath(nn.Module):
        """
        Stochastic Depth (DropPath) layer.
        """
        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # Binarize
            return x.div(keep_prob) * random_tensor


class SpatialAttentionModule(nn.Module):
    """
    Lightweight Spatial Attention Module.
    Generates a 2D spatial attention map by pooling channels and applying convolutions.
    """
    def __init__(self, kernel_size: int = 7) -> None:
        """
        Args:
            kernel_size (int): Convolution kernel size for extracting spatial correlations.
        """
        super().__init__()
        padding = kernel_size // 2
        # Inputs to conv are average-pooled and max-pooled channel maps (2 channels)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Feature maps of shape [B, C, H, W].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - out: Weighted feature maps of shape [B, C, H, W].
                - attn_map: Spatial attention map of shape [B, 1, H, W].
        """
        # Average pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Shape [B, 1, H, W]
        # Max pooling along the channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Shape [B, 1, H, W]
        
        # Concatenate and pass to convolutional projection
        combined = torch.cat([avg_out, max_out], dim=1)  # Shape [B, 2, H, W]
        attn_map = self.sigmoid(self.conv(combined))  # Shape [B, 1, H, W]
        
        # Multiply features with attention map
        out = x * attn_map
        return out, attn_map


class GeometryAwareModule(nn.Module):
    """
    Geometry-Aware Local Topology Module (GALTM).
    Applies spatial feature extraction and alignment on ViT patch tokens.
    
    NOTE: This module does NOT accept CLS tokens. The orchestration model preserves 
    the CLS token separately and reattaches it downstream after GALTM execution.
    """
    def __init__(self, config: ModelConfig) -> None:
        """
        Args:
            config (ModelConfig): Model configuration class.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        
        # Conv Block: Depthwise (3x3) -> Pointwise (1x1) -> GELU
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=3,
            padding=1,
            groups=self.embed_dim,
            bias=True
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=1,
            bias=True
        )
        self.act = nn.GELU()

        # Regularizations
        self.drop_path = DropPath(config.drop_path_rate)
        self.layernorm = nn.LayerNorm(self.embed_dim)
        
        # Spatial Attention Module
        self.spatial_attention = SpatialAttentionModule(kernel_size=7)

    def forward(self, patch_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            patch_tokens (torch.Tensor): Vision Transformer patch tokens, shape [B, N, embed_dim].
            
        Returns:
            Dict[str, torch.Tensor]:
                - "geometry_tokens": Fused local topology features of shape [B, N, embed_dim].
                - "attention_map": Spatial attention map of shape [B, 1, grid_h, grid_w].
        """
        batch_size, seq_len, embed_dim = patch_tokens.shape
        
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embedding dimension {self.embed_dim}, but got {embed_dim}")

        # Compute spatial grid dimensions dynamically
        grid_h = int(math.isqrt(seq_len))
        grid_w = seq_len // grid_h
        if grid_h * grid_w != seq_len:
            raise ValueError(
                f"Patch sequence length {seq_len} is not a perfect square grid representation (e.g. HxW)."
            )

        # Step 1: Reshape [B, N, embed_dim] -> [B, embed_dim, grid_h, grid_w]
        # Transpose sequence and channel dims to prepare for spatial convolutions
        x = patch_tokens.transpose(1, 2).reshape(batch_size, embed_dim, grid_h, grid_w)  # Shape [B, C, H, W]

        # Step 2: Apply Conv block
        conv_out = self.depthwise_conv(x)  # Shape [B, C, H, W]
        conv_out = self.pointwise_conv(conv_out)  # Shape [B, C, H, W]
        conv_out = self.act(conv_out)  # Shape [B, C, H, W]

        # Step 3: Residual Connection with DropPath
        x = x + self.drop_path(conv_out)  # Shape [B, C, H, W]

        # Step 4: Apply Spatial Attention Module
        x, attn_map = self.spatial_attention(x)  # x: [B, C, H, W], attn_map: [B, 1, H, W]

        # Step 5 & 6: Flatten back to [B, N, embed_dim] and apply LayerNorm over embedding dimension
        x = x.flatten(2).transpose(1, 2)  # Reshape back to [B, N, embed_dim]
        geometry_tokens = self.layernorm(x)  # Shape [B, N, embed_dim]

        return {
            "geometry_tokens": geometry_tokens,
            "attention_map": attn_map
        }
