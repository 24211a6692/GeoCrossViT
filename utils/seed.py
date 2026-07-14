"""
utils/seed.py

This module contains utility functions to set random seeds across random, 
numpy, and PyTorch (including CUDA backend) to guarantee determinism in experiments.

TODO:
- Test determinism reproducibility on diverse CUDA setups (multi-GPU, different architectures).
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic algorithms (might reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"Random seed set to {seed} for reproducibility.")
