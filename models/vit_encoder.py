"""
models/vit_encoder.py

This module implements a reusable, production-ready Vision Transformer (ViT) encoder 
based on the HuggingFace `google/vit-base-patch16-224-in21k` backbone.

DESIGN NOTE:
The Geometry Module in GeoCrossViT is designed to consume ONLY the patch tokens 
returned by this encoder to operate on localized spatial structures. The CLS token 
is preserved separately by the orchestration model and is reattached after 
geometry processing is completed. Downstream cross-modal attention blocks then 
consume the reconstructed full token sequence.

TODO:
- Support loading alternative HuggingFace ViT backbones dynamically from model config.
- Implement gradient checkpointing for memory-efficient training.
"""

from typing import Dict, Any, Union, Optional, Tuple
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from configs.config import ModelConfig


class ViTEncoder(nn.Module):
    """
    Reusable Vision Transformer encoder module powered by HuggingFace Transformers.
    Can be instantiated independently for different imaging modalities.
    """
    def __init__(self, config: ModelConfig) -> None:
        """
        Args:
            config (ModelConfig): Complete configuration instance containing model properties.
        """
        super().__init__()
        self.config = config
        self.vit_name = config.vit_name
        self.output_attentions = config.output_attentions

        # 1. Load pretrained HuggingFace ViT Model
        if config.pretrained:
            self.vit = ViTModel.from_pretrained(
                self.vit_name,
                add_pooling_layer=False,
                output_attentions=self.output_attentions
            )
        else:
            # Load with random initialization using matching configuration
            hf_config = ViTConfig.from_pretrained(
                self.vit_name,
                add_pooling_layer=False,
                output_attentions=self.output_attentions
            )
            self.vit = ViTModel(hf_config)

        # 2. Expose embedding dimension attribute
        self._embed_dim = self.vit.config.hidden_size

        # 3. Apply parameter freezing if configured
        if config.freeze_backbone:
            self._apply_freeze_policy()

    @property
    def embed_dim(self) -> int:
        """
        Public property exposing the feature embedding dimension (e.g. 768).
        """
        return self._embed_dim

    def _apply_freeze_policy(self) -> None:
        """
        Freezes every parameter inside the Vision Transformer backbone except 
        LayerNorm parameters, as per requirements.
        """
        frozen_count = 0
        trainable_count = 0
        for name, param in self.vit.named_parameters():
            if "layernorm" in name.lower():
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        print(f"Applied freeze policy: {frozen_count} params frozen, {trainable_count} LayerNorm params left trainable.")

    def forward(
        self, 
        x: torch.Tensor,
        output_attentions: Optional[bool] = None
    ) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """
        Extracts patch embeddings and CLS tokens from the input.
        
        Args:
            x (torch.Tensor): Input image batch of shape (B, C, H, W).
            output_attentions (bool, optional): Overrides self.output_attentions if provided.
            
        Returns:
            Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
                - "cls_token": The global classification token of shape (B, 1, embed_dim).
                - "patch_tokens": Local patch feature representations of shape (B, 196, embed_dim).
                                  NOTE: The Geometry Module must consume ONLY these patch tokens.
                - "full_sequence": The complete sequence (CLS + patches) of shape (B, 197, embed_dim).
                - "attentions": (Optional) Tuple of attention maps returned if output_attentions=True.
        """
        out_attns = output_attentions if output_attentions is not None else self.output_attentions

        # HF expects shape [B, C, H, W]
        outputs = self.vit(x, output_attentions=out_attns)
        
        # full_sequence shape: [B, 197, embed_dim]
        full_sequence = outputs.last_hidden_state
        
        # Slice representations
        cls_token = full_sequence[:, 0:1, :]       # Shape: [B, 1, embed_dim]
        patch_tokens = full_sequence[:, 1:, :]      # Shape: [B, 196, embed_dim]
        
        result = {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "full_sequence": full_sequence
        }

        if out_attns:
            # Tuple of length num_layers, each of shape (B, num_heads, 197, 197)
            result["attentions"] = outputs.attentions
            
        return result
