"""
datasets/dataset.py

This module contains the custom PyTorch Dataset implementation for loading paired 
multi-modal retinal scans (Fundus and OCT images) from the OLIVES dataset.

TODO:
- Support on-the-fly image alignment or cropping checks.
- Add support for loading 3D OCT volumes (multi-slice B-scans).
"""

import os
from typing import Dict, Union, Optional, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.logger import setup_logger

# Initialize local module logger
logger = setup_logger(name="GeoCrossDataset", log_file="dataset.log")


class GeoCrossDataset(Dataset):
    """
    Multimodal Dataset class for loading paired Fundus and OCT images 
    linked to patient records and diagnostic classes.
    """
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[Any] = None,
        split: str = "train"
    ) -> None:
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            img_dir (str): Base folder containing target image files.
            transform (callable, optional): Custom transform class mapping (Fundus, OCT) -> (Tensor, Tensor).
            split (str): Targeted data split. Must be one of 'train', 'val', or 'test'.
        """
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.transform = transform
        self.split = split.lower()

        # Validate split input
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split '{self.split}'. Must be one of 'train', 'val', or 'test'.")

        self.data_df = self._load_and_filter_metadata()

    def _load_and_filter_metadata(self) -> pd.DataFrame:
        """
        Loads the CSV metadata file and filters rows based on target split.
        
        Required columns: patient_id, fundus_path, oct_path, label, split
        """
        if not os.path.exists(self.csv_path):
            logger.error(f"Metadata CSV file not found at path: {self.csv_path}")
            raise FileNotFoundError(f"Metadata CSV file not found at path: {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV file {self.csv_path}: {e}")
            raise e

        # Validate columns
        required_cols = ["patient_id", "fundus_path", "oct_path", "label", "split"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            msg = f"Missing required columns in CSV: {missing_cols}. Found: {df.columns.tolist()}"
            logger.error(msg)
            raise ValueError(msg)

        # Filter by split (case-insensitive)
        filtered_df = df[df["split"].str.lower() == self.split].reset_index(drop=True)
        
        logger.info(f"Loaded {len(filtered_df)} samples for split: {self.split}")
        return filtered_df

    def _resolve_image_path(self, relative_path: str) -> str:
        """
        Resolves absolute image path based on base img_dir.
        """
        # If path is already absolute, return it
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.img_dir, relative_path)

    def __len__(self) -> int:
        """Returns the total number of samples in this split."""
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        """
        Loads and returns a single multimodal sample containing paired images, labels, 
        patient context, and unique sample indexing.
        
        Returns:
            Dict[str, Any]:
                - "fundus": Transformed 3-channel Fundus tensor.
                - "oct": Transformed 3-channel OCT tensor.
                - "label": Diagnostic classification target label (int).
                - "patient_id": Patient identifier string (str).
                - "sample_id": Unique sample identifier (str).
        """
        row = self.data_df.iloc[idx]
        
        patient_id = str(row["patient_id"])
        label = int(row["label"])
        # Use row index combined with patient_id as unique sample_id
        sample_id = f"{patient_id}_idx_{idx}"

        fundus_raw_path = self._resolve_image_path(str(row["fundus_path"]))
        oct_raw_path = self._resolve_image_path(str(row["oct_path"]))

        # Check files existence
        if not os.path.exists(fundus_raw_path):
            logger.error(f"Fundus image not found: {fundus_raw_path} at index {idx}")
            raise FileNotFoundError(f"Fundus image not found: {fundus_raw_path}")
            
        if not os.path.exists(oct_raw_path):
            logger.error(f"OCT image not found: {oct_raw_path} at index {idx}")
            raise FileNotFoundError(f"OCT image not found: {oct_raw_path}")

        # Load images
        try:
            fundus_img = Image.open(fundus_raw_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open or convert Fundus image {fundus_raw_path}: {e}")
            raise IOError(f"Corrupted Fundus file: {fundus_raw_path}. Error: {e}")

        try:
            oct_img = Image.open(oct_raw_path)
        except Exception as e:
            logger.error(f"Failed to open OCT image {oct_raw_path}: {e}")
            raise IOError(f"Corrupted OCT file: {oct_raw_path}. Error: {e}")

        # Apply spatial/intensity transformations
        if self.transform:
            try:
                fundus_tensor, oct_tensor = self.transform(fundus_img, oct_img)
            except Exception as e:
                logger.error(f"Transform application failed on sample index {idx}: {e}")
                raise RuntimeError(f"Transform failure: {e}")
        else:
            # Fallback basic transformations to output tensors if no transform provided
            to_tensor = T.ToTensor()
            fundus_tensor = to_tensor(fundus_img.convert("RGB"))
            oct_tensor = to_tensor(oct_img.convert("RGB"))

        return {
            "fundus": fundus_tensor,
            "oct": oct_tensor,
            "label": label,
            "patient_id": patient_id,
            "sample_id": sample_id
        }
