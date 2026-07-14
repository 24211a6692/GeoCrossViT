"""
utils/checkpoint.py

This module contains utilities to save and load model checkpoints, including 
optimizers, schedulers, and metrics history.

TODO:
- Support cloud storage uploads (e.g. AWS S3 or Google Cloud Storage) if needed.
- Support pruning old checkpoints to save disk space.
"""

import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str = "checkpoints",
    filename: str = "checkpoint.pth"
) -> str:
    """
    Saves a checkpoint containing the model weights, optimizer, and epoch information.
    
    Args:
        state (Dict[str, Any]): Dictionary containing 'model_state_dict', 'optimizer_state_dict', etc.
        checkpoint_dir (str): Folder destination.
        filename (str): Name of the file.
        
    Returns:
        str: Absolute filepath of the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    
    # Save using PyTorch utility
    torch.save(state, filepath)
    print(f"Checkpoint successfully saved to {filepath}")
    return filepath


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Loads model and optimizer states from a saved checkpoint dictionary.
    
    Args:
        filepath (str): Path to the saved checkpoint file.
        model (nn.Module): The model to load weights into.
        optimizer (Optimizer, optional): The optimizer to restore state.
        device (torch.device, optional): Device to map state location.
        
    Returns:
        Dict[str, Any]: The complete checkpoint dictionary (e.g. metadata, best metrics).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint file found at {filepath}")
        
    map_location = device if device else torch.device("cpu")
    checkpoint = torch.load(filepath, map_location=map_location)
    
    # Load states
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    print(f"Checkpoint successfully loaded from {filepath}")
    return checkpoint
