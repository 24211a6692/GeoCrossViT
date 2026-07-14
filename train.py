"""
train.py

This is the top-level script to initialize components and launch pretraining 
or downstream classification fine-tuning experiments for GeoCrossViT.

TODO:
- Setup parser arguments (argparse) to configure runs from terminal.
- Support resuming from checkpoints automatically.
- Setup wandb or logging backend parameters dynamically.
"""

import argparse
import sys
import torch
from configs.config import GeoCrossViTConfig
from datasets.datamodule import RetinalDataModule
from foundation_model import GeoCrossViT
from trainers.trainer import GeoCrossTrainer
from utils.logger import setup_logger
from utils.seed import set_seed


def main() -> None:
    """
    Orchestrates the pretraining or classification fine-tuning cycle.
    """
    # 1. Parse CLI inputs (Placeholder)
    parser = argparse.ArgumentParser(description="GeoCrossViT Training Entrypoint")
    parser.add_argument("--config", type=str, default="configs/config.py", help="Path to config file")
    parser.add_argument("--mode", type=str, default="classify", choices=["pretrain", "classify"])
    args = parser.parse_args()

    # 2. Setup seed & logs
    set_seed(42)
    logger = setup_logger(name="GeoCrossViT_Train", log_file="train.log")
    logger.info("Initializing GeoCrossViT training pipeline.")

    # 3. Load configurations
    config = GeoCrossViTConfig()
    logger.info(f"Loaded config parameters: {config}")

    # 4. Initialize Data Module
    data_module = RetinalDataModule(config)
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    logger.info("Dataset and DataLoaders initialized successfully.")

    # 5. Initialize Model
    model = GeoCrossViT(config.model)
    logger.info("GeoCrossViT Model components built successfully.")

    # 6. Initialize Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay
    )
    
    # 7. Initialize Trainer
    trainer = GeoCrossTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=None
    )

    # 8. Start Fitting
    logger.info(f"Starting model fit in '{args.mode}' mode...")
    trainer.fit()
    logger.info("Fitting process finished successfully.")


if __name__ == "__main__":
    main()
