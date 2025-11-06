# Road Buddy Challenge - Zalo AI Challenge 2025

This project contains code and resources for training deep learning models for the Road Buddy Challenge in Zalo AI Challenge 2025.

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- CUDA-compatible GPU 

## Environment Setup

### Conda Environment

This project uses Conda for environment management. To set up the environment:

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate road-buddy
```

<!-- ### FlashAttention Installation

FlashAttention must be installed manually after the conda environment is set up:

```bash
# Make sure the road-buddy environment is activated
conda activate road-buddy

# Install flash-attn
pip install flash-attn==2.7.3 --no-build-isolation
``` -->

<!-- Refer to the official instructions for troubleshooting and advanced installation: https://github.com/Dao-AILab/flash-attention#installation -->


### HuggingFace Access Token

To access models from HuggingFace, you need to set up your access token:

1. Create a `.env` file in the project root:
```bash
cp .env.example .env
```

2. Add your HuggingFace token to the `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

You can get your token from https://huggingface.co/settings/tokens

**Note:** The `.env` file is already included in `.gitignore` to prevent committing sensitive tokens.

## Structure
- `src/` - Main source code
- `models/` - Saved models
- `data/` - Datasets
- `scripts/` - Utility scripts
- `configs/` - Configuration files
