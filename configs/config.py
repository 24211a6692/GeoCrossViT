"""
configs/config.py

This module contains configuration classes and hyperparameter schemas for the 
Geometry-Aware Cross-Modal Vision Transformer (GeoCrossViT) project.
It uses Python's standard `dataclasses` to provide strict typing and ease of use.

TODO:
- Add CLI parser support or OmegaConf/YAML integration.
- Define specific modal dimension sizes and Transformer hyperparameter ranges.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for GeoCrossViT model architecture."""
    # Encoder parameters
    vit_name: str = "google/vit-base-patch16-224-in21k"
    pretrained: bool = True
    img_size: int = 224
    in_chans: int = 3
    freeze_backbone: bool = False
    output_attentions: bool = False
    
    # Geometry Module parameters
    coord_dim: int = 2
    geom_hidden_dim: int = 128
    drop_path_rate: float = 0.1
    
    # Cross-modal Transformer parameters
    embed_dim: int = 768
    num_heads: int = 8
    depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    
    # Projection & Classifier parameters
    projection_dim: int = 256
    num_classes: int = 5  # e.g., Diabetic Retinopathy stages


@dataclass
class DatasetConfig:
    """Configuration for data loading and preprocessing."""
    csv_path: str = "data/metadata.csv"
    img_dir: str = "data/images/"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class TrainConfig:
    """Configuration for the training loop."""
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 10
    temperature: float = 0.07  # InfoNCE temperature
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    seed: int = 42
    
    # Training pipeline configurations
    lambda_info: float = 0.3
    patience: int = 10
    use_amp: bool = True
    use_tensorboard: bool = True
    best_metric_name: str = "val_acc"  # "val_acc" or "val_loss"
    grad_clip: float = 1.0


@dataclass
class GeoCrossViTConfig:
    """Global configuration wrapper."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "GeoCrossViTConfig":
        """
        Loads configuration from a YAML file.
        
        TODO: Implement YAML parser conversion using PyYAML or OmegaConf.
        """
        # Placeholder implementation
        print(f"Loading config from {yaml_path} (stub)...")
        return cls()
