"""
datasets/datamodule.py

This module contains the RetinalDataModule class which initializes the custom 
GeoCrossDataset and setups training, validation, and testing DataLoaders with 
coordinated spatial transformations.

TODO:
- Support class weight estimation for handling class imbalances.
- Integrate custom collation to support varying input sizes or missing modalities.
"""

from typing import Optional
from torch.utils.data import DataLoader

from configs.config import GeoCrossViTConfig
from datasets.olives_dataset import OlivesDataset
from datasets.transforms import CoordinatedGeoTransforms


class RetinalDataModule:
    """
    Data module orchestrator instantiating dataset loaders for train, validation, 
    and test phases using configurations.
    """
    def __init__(self, config: GeoCrossViTConfig) -> None:
        """
        Args:
            config (GeoCrossViTConfig): Global config configuration.
        """
        self.config = config
        
        self.train_dataset: Optional[OlivesDataset] = None
        self.val_dataset: Optional[OlivesDataset] = None
        self.test_dataset: Optional[OlivesDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Prepares datasets for loading based on target stage ('fit' or 'test').
        
        Args:
            stage (str, optional): Target stage, e.g. 'fit', 'test', or None.
        """
        # Create coordinated geo transforms
        train_transform = CoordinatedGeoTransforms(
            img_size=self.config.model.img_size,
            is_training=True
        )
        val_transform = CoordinatedGeoTransforms(
            img_size=self.config.model.img_size,
            is_training=False
        )

        if stage == "fit" or stage is None:
            self.train_dataset = OlivesDataset(
                config=self.config,
                transform=train_transform,
                split="train"
            )
            self.val_dataset = OlivesDataset(
                config=self.config,
                transform=val_transform,
                split="val"
            )

        if stage == "test" or stage is None:
            self.test_dataset = OlivesDataset(
                config=self.config,
                transform=val_transform,
                split="test"
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for the training set.
        """
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not set up. Call setup('fit') first.")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            persistent_workers=self.config.dataset.persistent_workers if self.config.dataset.num_workers > 0 else False
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for the validation set.
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not set up. Call setup('fit') first.")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            persistent_workers=self.config.dataset.persistent_workers if self.config.dataset.num_workers > 0 else False
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for the test set.
        """
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not set up. Call setup('test') first.")
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            persistent_workers=self.config.dataset.persistent_workers if self.config.dataset.num_workers > 0 else False
        )
