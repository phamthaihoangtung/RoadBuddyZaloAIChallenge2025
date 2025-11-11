#!/bin/bash

# Install Unsloth and dependencies for training
# Usage: bash scripts/install_unsloth.sh

echo "Installing Unsloth and dependencies..."
echo "=========================================="

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: No conda environment detected."
    echo "It's recommended to activate your conda environment first:"
    echo "  conda activate road-buddy"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "Installing Unsloth..."
pip install unsloth

echo ""
echo "Installing additional dependencies..."
pip install --no-deps bitsandbytes accelerate peft trl cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0"

echo ""
echo "=========================================="
echo "Installation complete!"
echo ""
echo "To verify installation, run:"
echo "  python -c 'from unsloth import FastVisionModel; print(\"Unsloth installed successfully!\")'"
echo ""
echo "To start training:"
echo "  bash scripts/train_unsloth.sh"
