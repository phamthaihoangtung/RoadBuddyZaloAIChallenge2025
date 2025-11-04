# Road Buddy Challenge - Zalo AI Challenge 2025

This project contains code and resources for training deep learning models for the Road Buddy Challenge in Zalo AI Challenge 2025.

## Environment Setup


### Python & uv Environment

This project now uses [uv](https://github.com/astral-sh/uv) and `pyproject.toml` for environment management. To set up the environment:

```bash
pip install uv
cd <project_path>
uv venv
uv pip install -e .
```

### FlashAttention Installation

FlashAttention cannot be installed directly via requirements.txt. After activating your environment, install it manually:

```bash
uv pip install flash-attn==2.7.3 --no-build-isolation
```

Refer to the official instructions for troubleshooting and advanced installation: https://github.com/Dao-AILab/flash-attention#installation

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
