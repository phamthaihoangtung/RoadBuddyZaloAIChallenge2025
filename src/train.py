import torch
import numpy as np
import argparse
from utils.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train model for Road Buddy Challenge")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    args = parser.parse_args()
    
    # Load config from the specified path
    config = load_config(args.config)

    print("Training script placeholder.")
    # TODO: Implement training loop

if __name__ == "__main__":
    main()
