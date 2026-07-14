"""
models/cross_modal_transformer.py

This module implements the Cross-Modal Transformer fusion block for GeoCrossViT. 
It performs bidirectional cross-attention (Fundus enhanced by attending to OCT, 
and OCT enhanced by attending to Fundus) across variable sequence lengths. It uses 
learnable modality embeddings, residual connections, layer normalization, FFN layers, 
an adaptive gating mechanism, and a final projection MLP to yield a fused joint token sequence.

TODO:
- Support optional relative positional embeddings inside the attention block.
- Add support for cross-attention masking.
"""

from typing import Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
from configs.config import ModelConfig


class CrossModalTransformerBlock(nn.Module):
    """
    A single bidirectional cross-modal transformer block containing:
    - Fundus-to-OCT cross-attention (Fundus attends to OCT)
    - OCT-to-Fundus cross-attention (OCT attends to Fundus)
    - Residual connections, LayerNorms, and Feed-Forward Networks for both branches.
    """
    def __init__(self, config: ModelConfig) -> None:
        """
        Args:
            config (ModelConfig): Model configuration class.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        
        # 1. Fundus Branch Cross-Attention (Q: Fundus, K/V: OCT)
        self.fundus_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.fundus_norm1 = nn.LayerNorm(self.embed_dim)
        self.fundus_norm2 = nn.LayerNorm(self.embed_dim)
        
        # Fundus Feed-Forward Network
        self.fundus_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * config.mlp_ratio)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(int(self.embed_dim * config.mlp_ratio), self.embed_dim),
            nn.Dropout(config.dropout)
        )
        
        # 2. OCT Branch Cross-Attention (Q: OCT, K/V: Fundus)
        self.oct_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.oct_norm1 = nn.LayerNorm(self.embed_dim)
        self.oct_norm2 = nn.LayerNorm(self.embed_dim)
        
        # OCT Feed-Forward Network
        self.oct_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * config.mlp_ratio)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(int(self.embed_dim * config.mlp_ratio), self.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(
        self, 
        fundus_tokens: torch.Tensor, 
        oct_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            fundus_tokens (torch.Tensor): Fundus tokens of shape [B, N, embed_dim].
            oct_tokens (torch.Tensor): OCT tokens of shape [B, N, embed_dim].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - fundus_out: Enhanced Fundus tokens of shape [B, N, embed_dim].
                - oct_out: Enhanced OCT tokens of shape [B, N, embed_dim].
                - fundus_attn_w: Attention weights map from Fundus-to-OCT, shape [B, num_heads, N, N].
                - oct_attn_w: Attention weights map from OCT-to-Fundus, shape [B, num_heads, N, N].
        """
        # Pre-LN implementation for stable gradient propagation
        
        # --- Fundus Branch (Q = Fundus, K/V = OCT) ---
        fundus_normed = self.fundus_norm1(fundus_tokens)
        oct_normed = self.fundus_norm1(oct_tokens)
        
        # Cross Attention
        fundus_attn_out, fundus_attn_w = self.fundus_cross_attn(
            query=fundus_normed,
            key=oct_normed,
            value=oct_normed,
            average_attn_weights=False
        )
        fundus_x = fundus_tokens + fundus_attn_out  # Residual connection
        fundus_out = fundus_x + self.fundus_mlp(self.fundus_norm2(fundus_x))  # FFN + Residual

        # --- OCT Branch (Q = OCT, K/V = Fundus) ---
        # Cross Attention
        oct_attn_out, oct_attn_w = self.oct_cross_attn(
            query=oct_normed,
            key=fundus_normed,
            value=fundus_normed,
            average_attn_weights=False
        )
        oct_x = oct_tokens + oct_attn_out  # Residual connection
        oct_out = oct_x + self.oct_mlp(self.oct_norm2(oct_x))  # FFN + Residual

        return fundus_out, oct_out, fundus_attn_w, oct_attn_w


class GeoCrossModalFusion(nn.Module):
    """
    Main Cross-Modal Transformer fusion block stacking multiple cross-attention blocks 
    and integrating modality embeddings, adaptive gating, and projection networks.
    """
    def __init__(self, config: ModelConfig) -> None:
        """
        Args:
            config (ModelConfig): Model configuration class.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        
        # 1. Learnable Modality Embeddings
        self.fundus_modality_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.oct_modality_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Initialize embeddings with small values
        nn.init.normal_(self.fundus_modality_emb, std=0.02)
        nn.init.normal_(self.oct_modality_emb, std=0.02)

        # 2. Transformer layers stack
        self.layers = nn.ModuleList([
            CrossModalTransformerBlock(config) for _ in range(config.depth)
        ])

        # 3. Lightweight Gating Mechanism
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * self.embed_dim, 2 * self.embed_dim),
            nn.Sigmoid()
        )

        # 4. Final Joint Projection MLP
        self.projection_mlp = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(
        self,
        fundus_tokens: torch.Tensor,
        oct_tokens: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Fuses Fundus and OCT tokens bidirectionally and combines them with adaptive gating.
        
        Args:
            fundus_tokens (torch.Tensor): Fundus tokens of shape [B, N, embed_dim].
            oct_tokens (torch.Tensor): OCT tokens of shape [B, N, embed_dim].
            
        Returns:
            Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
                - "joint_tokens": Joint representation of shape [B, N, embed_dim].
                - "fundus_tokens": Enhanced Fundus tokens of shape [B, N, embed_dim].
                - "oct_tokens": Enhanced OCT tokens of shape [B, N, embed_dim].
                - "fundus_attention": List of Fundus attention weights from all layers.
                - "oct_attention": List of OCT attention weights from all layers.
        """
        # 1. Apply learnable modality embeddings before first transformer block
        fundus_x = fundus_tokens + self.fundus_modality_emb
        oct_x = oct_tokens + self.oct_modality_emb

        fundus_attns = []
        oct_attns = []

        # 2. Pass through stacked cross-attention blocks
        for layer in self.layers:
            fundus_x, oct_x, f_attn, o_attn = layer(fundus_x, oct_x)
            fundus_attns.append(f_attn)
            oct_attns.append(o_attn)

        # 3. Concatenate enhanced representations: shape [B, N, 2 * embed_dim]
        concat_features = torch.cat([fundus_x, oct_x], dim=-1)

        # 4. Lightweight Gating Module
        gate_weights = self.gate_proj(concat_features)  # Shape [B, N, 2 * embed_dim]
        gated_concat = concat_features * gate_weights   # Shape [B, N, 2 * embed_dim]

        # 5. Projection MLP -> Joint Tokens: shape [B, N, embed_dim]
        joint_tokens = self.projection_mlp(gated_concat)

        return {
            "joint_tokens": joint_tokens,
            "fundus_tokens": fundus_x,
            "oct_tokens": oct_x,
            "fundus_attention": fundus_attns,
            "oct_attention": oct_attns
        }
