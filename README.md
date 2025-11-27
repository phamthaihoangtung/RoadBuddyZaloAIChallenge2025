# Road Buddy Challenge - Zalo AI Challenge 2025

This project contains code and resources for training deep learning models for the Road Buddy Challenge in Zalo AI Challenge 2025.

## Project Approach

This project employs several key techniques to fine-tune a Vision Language Model (VLM) for the video question-answering task:

1.  **LoRA for Efficient Fine-Tuning**: We use Low-Rank Adaptation (LoRA) to efficiently fine-tune the VLM. This allows for faster training with significantly lower memory requirements compared to full model fine-tuning, making it feasible to train on consumer-grade GPUs.

2.  **External Knowledge Injection**: To enhance the model's understanding of traffic rules, images of relevant traffic signs are provided as additional context during both training and inference. The model is prompted to refer to these signs when answering questions, effectively injecting external knowledge.

3.  **Support Frame Prediction**: The model is explicitly trained to identify and output the key frames from the video that support its answer. This is achieved by prepending the ground-truth support frame numbers to the answer during training, encouraging the model to learn a form of visual grounding and reasoning.

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

## Usage

### Training

To start training the model, run the training script:

```bash
bash scripts/train.sh
```

This script executes `src/train.py` with the default configuration `configs/config_unsloth.yaml`. You can customize training parameters (e.g., model, learning rate, epochs) by editing this file or creating a new one and running:

```bash
python src/train.py --config path/to/your_config.yaml
```

### Inference

To generate predictions for the test set, run the inference script:

```bash
bash scripts/infer_unsloth.sh
```

This script runs `src/infer.py` with the `configs/config_unsloth_infer.yaml` configuration. The script will produce a `submission.csv` file in the project root.

To use a different model or configuration for inference, you can modify the script or run `src/infer.py` directly:

```bash
python src/infer.py --config path/to/your_inference_config.yaml
```

## Structure
- `src/` - Main source code
- `models/` - Saved models
- `data/` - Datasets
- `scripts/` - Utility scripts
- `configs/` - Configuration files
