"""
datasets/transforms.py

This module contains custom data transformation classes. It implements 
coordinated spatial transformations (shared flips, crops, and rotations) 
across multiple imaging modalities (Fundus and OCT) using torchvision's 
functional API to ensure geographic and spatial alignment, while applying 
modality-specific intensity modifications like ColorJitter only to the Fundus.

TODO:
- Support custom ranges for rotation and crop size from configuration.
- Support additional modal-specific intensity operations (e.g., contrast stretching).
"""

import random
from typing import Tuple, Dict, Any, Union
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class CoordinatedGeoTransforms:
    """
    Applies coordinated spatial transformations (Resizing, Random Flips, Rotations, 
    and Crops) to paired Fundus and OCT images, preserving spatial alignment, 
    while applying ColorJitter solely to the Fundus image.
    """
    def __init__(
        self,
        img_size: int = 224,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        degrees: float = 15.0,
        color_jitter_params: Dict[str, float] = None,
        is_training: bool = True
    ) -> None:
        """
        Args:
            img_size (int): Target square dimension for output images.
            crop_scale (Tuple[float, float]): Scale range for random cropped resizing.
            degrees (float): Maximum range of rotation angles in degrees.
            color_jitter_params (dict): Custom brightness, contrast, saturation, hue values.
            is_training (bool): If True, applies random augmentations. Otherwise, uses deterministic validation transforms.
        """
        self.img_size = img_size
        self.crop_scale = crop_scale
        self.degrees = degrees
        self.is_training = is_training

        # Initialize color jitter only for Fundus
        cj_defaults = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}
        if color_jitter_params is not None:
            cj_defaults.update(color_jitter_params)
        self.color_jitter = T.ColorJitter(**cj_defaults)

        # ImageNet normalization statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _apply_coordinated_spatial_transforms(
        self, 
        fundus: Image.Image, 
        oct_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Generates random spatial parameters and applies identical spatial operations 
        to both Fundus and OCT inputs.
        """
        # 1. Random Resized Crop parameter estimation
        if self.crop_scale != (1.0, 1.0):
            # get_params returns i, j, h, w
            i, j, h, w = T.RandomResizedCrop.get_params(
                fundus, 
                scale=self.crop_scale, 
                ratio=(0.75, 1.33)
            )
            fundus = TF.crop(fundus, i, j, h, w)
            oct_img = TF.crop(oct_img, i, j, h, w)

        # Resize to final dimensions
        fundus = TF.resize(fundus, [self.img_size, self.img_size])
        oct_img = TF.resize(oct_img, [self.img_size, self.img_size])

        # 2. Coordinated Horizontal Flip
        if random.random() > 0.5:
            fundus = TF.hflip(fundus)
            oct_img = TF.hflip(oct_img)

        # 3. Coordinated Vertical Flip
        if random.random() > 0.5:
            fundus = TF.vflip(fundus)
            oct_img = TF.vflip(oct_img)

        # 4. Coordinated Rotation
        angle = random.uniform(-self.degrees, self.degrees)
        fundus = TF.rotate(fundus, angle)
        oct_img = TF.rotate(oct_img, angle)

        return fundus, oct_img

    def __call__(
        self, 
        fundus: Image.Image, 
        oct_img: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies coordinated spatial and independent intensity augmentations to a sample.
        
        Args:
            fundus (PIL.Image): Fundus image.
            oct_img (PIL.Image): OCT scan image (grayscale or RGB).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed Fundus and OCT tensors.
        """
        # Ensure correct format - convert OCT from Grayscale to RGB by duplicating channels
        if oct_img.mode != "RGB":
            oct_img = oct_img.convert("RGB")
        if fundus.mode != "RGB":
            fundus = fundus.convert("RGB")

        if self.is_training:
            # Coordinated spatial updates
            fundus, oct_img = self._apply_coordinated_spatial_transforms(fundus, oct_img)
            # Modality-specific intensity changes
            fundus = self.color_jitter(fundus)
        else:
            # Deterministic resize for evaluation/testing
            fundus = TF.resize(fundus, [self.img_size, self.img_size])
            oct_img = TF.resize(oct_img, [self.img_size, self.img_size])

        # Convert to tensor
        fundus_tensor = TF.to_tensor(fundus)
        oct_tensor = TF.to_tensor(oct_img)

        # Normalize with ImageNet stats
        fundus_tensor = TF.normalize(fundus_tensor, mean=self.mean, std=self.std)
        oct_tensor = TF.normalize(oct_tensor, mean=self.mean, std=self.std)

        return fundus_tensor, oct_tensor
