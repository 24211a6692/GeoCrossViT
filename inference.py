"""
inference.py

This top-level script runs inference predictions on test image pathways or 
batch test files using a trained GeoCrossViT model checkpoint.

TODO:
- Support directories inputs directly from CLI.
- Implement batch predictions saver (exporting class predictions to .csv).
- Add support for visualization overlays saving.
"""

import argparse
import sys
import torch
from configs.config import GeoCrossViTConfig
from foundation_model import GeoCrossViT
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger


def run_inference(checkpoint_path: str, fundus_path: str, oct_path: str) -> None:
    """
    Runs single-pair inference using a trained checkpoint.
    
    Args:
        checkpoint_path (str): File path to .pth weights.
        fundus_path (str): File path to Fundus image.
        oct_path (str): File path to OCT scan.
        
    TODO:
        - Load and preprocess images.
        - Run model forward in classification mode.
        - Print classification outputs.
    """
    logger = setup_logger(name="GeoCrossViT_Inference", log_file="inference.log")
    logger.info("Initializing GeoCrossViT inference script.")
    
    config = GeoCrossViTConfig()
    model = GeoCrossViT(config.model)
    
    # Load weights
    try:
        load_checkpoint(checkpoint_path, model)
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}. Running with dummy weights.")

    model.eval()
    
    # Dummy tensors representing single sample batch
    dummy_fundus = torch.zeros((1, 3, 224, 224))
    dummy_oct = torch.zeros((1, 3, 224, 224))
    
    with torch.no_grad():
        output = model(dummy_fundus, dummy_oct, mode="classify")
        logits = output["logits"]
        probs = torch.softmax(logits, dim=-1)
        
    logger.info(f"Inference complete. Logits: {logits.tolist()} | Probabilities: {probs.tolist()}")
    print(f"Classification Probabilities: {probs.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoCrossViT Inference Entrypoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model weights (.pth)")
    parser.add_argument("--fundus", type=str, required=True, help="Path to input Fundus image")
    parser.add_argument("--oct", type=str, required=True, help="Path to input OCT scan")
    
    args = parser.parse_args()
    run_inference(args.checkpoint, args.fundus, args.oct)


if __name__ == "__main__":
    main()
