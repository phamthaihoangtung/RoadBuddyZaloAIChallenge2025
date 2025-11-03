# Road Buddy Challenge - Zalo AI Challenge 2025

This project contains code and resources for training deep learning models for the Road Buddy Challenge in Zalo AI Challenge 2025.

## Environment Setup

This project uses [conda](https://docs.conda.io/en/latest/miniconda.html) for environment management. To set up the environment:

```bash
conda env create -f environment.yml
```
```bash
conda activate road-buddy
```

## FlashAttention Installation

FlashAttention cannot be installed directly via environment.yml. After activating your environment, install it manually:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

Refer to the official instructions for troubleshooting and advanced installation: https://github.com/Dao-AILab/flash-attention#installation

## Structure
- `src/` - Main source code
- `models/` - Saved models
- `data/` - Datasets
- `scripts/` - Utility scripts
- `configs/` - Configuration files
