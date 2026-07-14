"""
utils/logger.py

This module configures logging for the GeoCrossViT training and validation steps, 
outputting structured information to console and file.

TODO:
- Support JSON lines format or CSV logs format.
- Integrate logging with mlflow or Weights & Biases (wandb).
"""

import logging
import os
from typing import Optional


def setup_logger(
    name: str = "GeoCrossViT",
    log_dir: str = "logs",
    log_file: str = "train.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Initializes a structured logger.
    
    Args:
        name (str): Logger name.
        log_dir (str): Folder containing log file.
        log_file (str): Log filename.
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if already setup
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s"
        )
        
        # Stream Handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # File Handler
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to create file logger handler: {e}")
            
    return logger
