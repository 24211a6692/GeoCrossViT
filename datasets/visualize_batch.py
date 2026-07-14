"""
datasets/visualize_batch.py

This module contains utility functions to visualize batch samples of Fundus and OCT 
image pairs, including their diagnostic labels and patient IDs, helping to visually 
verify coordinated augmentations and data loading pipeline.

TODO:
- Support overlaying segmentation maps or bounding boxes if added.
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch


def unnormalize(tensor: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """
    Unnormalizes an ImageNet-normalized image tensor and converts to numpy HWC format.
    """
    # Clone to avoid changing source tensor
    img = tensor.clone().detach().cpu()
    
    # Unnormalize channels: img = img * std + mean
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
        
    # Clip to [0, 1] range and transpose to (H, W, C)
    img = torch.clamp(img, 0.0, 1.0)
    return img.numpy().transpose(1, 2, 0)


def create_batch_grid_figure(
    batch: Dict[str, Any], 
    max_samples: int = 4
) -> plt.Figure:
    """
    Helper function to generate a Matplotlib Figure displaying Fundus and OCT pairs.
    """
    fundus_batch = batch["fundus"]
    oct_batch = batch["oct"]
    labels = batch["label"]
    patient_ids = batch["patient_id"]
    sample_ids = batch["sample_id"]

    batch_size = fundus_batch.shape[0]
    num_display = min(batch_size, max_samples)

    # ImageNet stats for unnormalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    fig, axes = plt.subplots(2, num_display, figsize=(4 * num_display, 8))
    
    # Handle single column edge cases for subplots shape
    if num_display == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(num_display):
        # 1. Unnormalize and plot Fundus
        fundus_img = unnormalize(fundus_batch[i], mean, std)
        axes[0, i].imshow(fundus_img)
        axes[0, i].set_title(f"Fundus | Label: {labels[i].item()}")
        axes[0, i].axis("off")

        # 2. Unnormalize and plot OCT
        oct_img = unnormalize(oct_batch[i], mean, std)
        axes[1, i].imshow(oct_img)
        axes[1, i].set_title(f"OCT | Patient: {patient_ids[i]}\nID: {sample_ids[i]}")
        axes[1, i].axis("off")

    plt.tight_layout()
    return fig


def visualize_batch(batch: Dict[str, Any], max_samples: int = 4) -> None:
    """
    Plots a grid of Fundus and OCT image pairs from a batch, displaying their 
    corresponding labels and IDs, then displays the plot.
    
    Args:
        batch (Dict[str, Any]): Batch dictionary returned from DataLoader.
        max_samples (int): Max number of samples to visualize in grid.
    """
    fig = create_batch_grid_figure(batch, max_samples)
    plt.show()
    plt.close(fig)


def save_batch_grid(
    batch: Dict[str, Any], 
    filepath: str, 
    max_samples: int = 4
) -> None:
    """
    Plots a grid of Fundus and OCT image pairs from a batch, and saves the plot to a file.
    
    Args:
        batch (Dict[str, Any]): Batch dictionary returned from DataLoader.
        filepath (str): Path where the output image will be saved.
        max_samples (int): Max number of samples to visualize in grid.
    """
    fig = create_batch_grid_figure(batch, max_samples)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Batch grid successfully saved to: {filepath}")
