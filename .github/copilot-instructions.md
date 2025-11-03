# Copilot Instructions for Road Buddy Challenge (Zalo AI Challenge 2025)

## Project Overview
- This repository is for training and inference of deep learning models for video-based question answering in the Road Buddy Challenge.
- Main components:
  - `src/`: Core scripts for training (`train.py`) and inference (`infer.py`).
  - `data/`: Contains datasets, public test samples, and video download utilities.
  - `configs/`: Configuration files (e.g., `config.yaml`) for model and data paths.
  - `models/`: Directory for saving trained models.
  - `scripts/`: Utility scripts, e.g., `infer.sh` for running inference.

## Environment & Dependencies
- Use Conda for environment management:
  - Create environment: `conda env create -f environment.yml`
  - Activate: `conda activate road-buddy`
- **FlashAttention** must be installed manually after environment setup:
  - `pip install flash-attn==2.7.3 --no-build-isolation`
- PyTorch and TorchVision are installed with CUDA 11.8 wheels (see `environment.yml`).

## Data & Workflow
- Datasets are in `data/train/` and `data/public_test/`.
- Download videos using `data/download_videos.py` (see script for usage).
- Inference uses `src/infer.py` and is typically run via `scripts/infer.sh` (uses `uv run src/infer.py`).
- Model and processor are loaded from HuggingFace (`DAMO-NLP-SG/VideoLLaMA3-7B` by default, configurable in `configs/config.yaml`).
- Inference expects video files and questions from JSON (see `data/public_test/public_test.json`).

## Patterns & Conventions
- Configuration is loaded from `configs/config.yaml`.
- Inference script (`src/infer.py`) uses a conversation format for input, with video and text question.
- Model inference uses `flash_attention_2` and expects CUDA-enabled hardware.
- Responses are printed for each test item; batch size and number of items can be adjusted in config or code.
- Utility scripts may use `uv` for running Python scripts (see `scripts/infer.sh`).

## Tips for AI Agents
- Always check and update `configs/config.yaml` for paths and model settings.
- Ensure all required videos are downloaded before running inference.
- For new models or data formats, update the config and relevant scripts accordingly.
- Use the provided directory structure for new scripts, models, or data.
- Reference `README.md` for setup and troubleshooting steps.

## Example: Running Inference
```bash
conda activate road-buddy
pip install flash-attn==2.7.3 --no-build-isolation
uv run src/infer.py
```

## Key Files
- `src/infer.py`, `src/train.py`, `configs/config.yaml`, `data/download_videos.py`, `scripts/infer.sh`, `environment.yml`, `README.md`

---
_If any section is unclear or missing, please provide feedback for improvement._
