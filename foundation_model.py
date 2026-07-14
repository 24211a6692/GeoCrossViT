"""
foundation_model.py

This module contains the GeoCrossViT model class which integrates the Vision Transformer 
encoders, the Geometry-Aware Local Topology Modules (GALTM), the Cross-Modal 
Transformer fusion block, the Projection Head, and the DiseaseClassifier 
into a single end-to-end multi-modal deep learning model.

It supports three forward execution modes:
- "backbone": returns backbone-level outputs only (encoders, geometry, and cross-attention tokens).
- "train" (default): returns the full set of outputs including joint tokens, intermediate features, and head outputs.
- "inference": returns only inference-relevant outputs (logits, probabilities, predictions, and projection embeddings).
"""

from typing import Dict, List, Union, Tuple, Optional, Any
import torch
import torch.nn as nn
from configs.config import ModelConfig
from models.vit_encoder import ViTEncoder
from models.geometry_module import GeometryAwareModule
from models.cross_modal_transformer import GeoCrossModalFusion
from models.projection_head import ProjectionHead
from models.classifier import DiseaseClassifier


class GeoCrossViT(nn.Module):
    """
    Geometry-Aware Cross-Modal Vision Transformer (GeoCrossViT) full pipeline.
    Links ViT encoders, local topology, cross-modal attention, and task heads.
    """
    def __init__(self, config: ModelConfig) -> None:
        """
        Args:
            config (ModelConfig): Model architecture configurations.
        """
        super().__init__()
        self.config = config
        
        # 1. Vision Transformer Encoders (independent weights)
        self.fundus_encoder = ViTEncoder(config)
        self.oct_encoder = ViTEncoder(config)
        
        # 2. Geometry-Aware Local Topology Modules (GALTM)
        self.fundus_geometry = GeometryAwareModule(config)
        self.oct_geometry = GeometryAwareModule(config)
        
        # 3. Bidirectional Cross-Modal Fusion
        self.cross_modal_fusion = GeoCrossModalFusion(config)

        # 4. Contrastive Projection Head (for pretraining / representation learning)
        self.projection_head = ProjectionHead(config, pooling_strategy="mean")

        # 5. Downstream Disease Classifier
        self.classifier = DiseaseClassifier(config)

    def forward(
        self,
        fundus_image: torch.Tensor,
        oct_image: torch.Tensor,
        mode: str = "train"
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Runs the end-to-end forward pass through model components.
        
        Args:
            fundus_image (torch.Tensor): Fundus image of shape [B, 3, 224, 224].
            oct_image (torch.Tensor): OCT image of shape [B, 3, 224, 224].
            mode (str): Execution mode. Must be one of 'train', 'backbone', or 'inference'.
            
        Returns:
            Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
                Outputs matching selected mode requirements.
        """
        mode = mode.lower()
        if mode not in ["train", "backbone", "inference"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of 'train', 'backbone', or 'inference'.")

        # --- Backbone Feed-Forward ---
        # 1. Modality-specific encoding
        fundus_enc_out = self.fundus_encoder(fundus_image)
        oct_enc_out = self.oct_encoder(oct_image)

        fundus_cls = fundus_enc_out["cls_token"]              # Shape: [B, 1, embed_dim]
        oct_cls = oct_enc_out["cls_token"]                  # Shape: [B, 1, embed_dim]
        
        fundus_patches = fundus_enc_out["patch_tokens"]      # Shape: [B, 196, embed_dim]
        oct_patches = oct_enc_out["patch_tokens"]          # Shape: [B, 196, embed_dim]

        fundus_full = fundus_enc_out["full_sequence"]        # Shape: [B, 197, embed_dim]
        oct_full = oct_enc_out["full_sequence"]            # Shape: [B, 197, embed_dim]

        # 2. Geometry Module processing (consumes ONLY patch tokens)
        fundus_geom_out = self.fundus_geometry(fundus_patches)
        oct_geom_out = self.oct_geometry(oct_patches)

        fundus_geom_tokens = fundus_geom_out["geometry_tokens"]  # Shape: [B, 196, embed_dim]
        oct_geom_tokens = oct_geom_out["geometry_tokens"]      # Shape: [B, 196, embed_dim]

        geom_attn_fundus = fundus_geom_out["attention_map"]    # Shape: [B, 1, grid_h, grid_w]
        geom_attn_oct = oct_geom_out["attention_map"]        # Shape: [B, 1, grid_h, grid_w]

        # 3. Fusing representations via Cross-Modal Transformer
        fusion_out = self.cross_modal_fusion(fundus_geom_tokens, oct_geom_tokens)
        joint_tokens = fusion_out["joint_tokens"]

        # Backbone Outputs Dictionary
        backbone_dict = {
            "fundus_cls": fundus_cls,
            "oct_cls": oct_cls,
            "fundus_patch_tokens": fundus_patches,
            "oct_patch_tokens": oct_patches,
            "fundus_full_sequence": fundus_full,
            "oct_full_sequence": oct_full,
            "fundus_geometry_tokens": fundus_geom_tokens,
            "oct_geometry_tokens": oct_geom_tokens,
            "joint_tokens": joint_tokens,
            "fundus_attention": fusion_out["fundus_attention"],
            "oct_attention": fusion_out["oct_attention"],
            "geometry_attention_fundus": geom_attn_fundus,
            "geometry_attention_oct": geom_attn_oct
        }

        if mode == "backbone":
            return backbone_dict

        # --- Task Heads Feed-Forward ---
        # 4. Projection Head (takes joint tokens)
        proj_out = self.projection_head(joint_tokens)
        pooled_feature = proj_out["pooled_feature"]
        projection_features = proj_out["projection_features"]
        projection_embedding = proj_out["projection_embedding"]

        # 5. Classifier Head (takes pooled features)
        class_out = self.classifier(pooled_feature)
        logits = class_out["logits"]

        if mode == "train":
            # Return full set of backbone + head outputs
            backbone_dict.update({
                "pooled_feature": pooled_feature,
                "projection_features": projection_features,
                "projection_embedding": projection_embedding,
                "logits": logits
            })
            return backbone_dict

        elif mode == "inference":
            # Return only inference-relevant outputs
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            return {
                "logits": logits,
                "probabilities": probs,
                "predictions": preds,
                "projection_embedding": projection_embedding
            }
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
