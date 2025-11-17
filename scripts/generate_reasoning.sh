#!/bin/bash

# This script runs the reasoning generation for the dataset.
# It uses `uv` to run the python script, similar to other scripts in this project.
#
# Usage:
#   ./scripts/generate_reasoning.sh [input_path] [output_path] [batch_size]
#
# Example (with defaults):
#   chmod +x ./scripts/generate_reasoning.sh
#   ./scripts/generate_reasoning.sh
#
# Example (with custom paths):
#   ./scripts/generate_reasoning.sh data/public_test/public_test.json data/public_test/public_test_with_reasoning.json 4

# Set default values
DATA_PATH=${1:-"data/train/train.json"}
OUTPUT_PATH=${2:-"data/train/train_thinking.json"}
BATCH_SIZE=${3:-1}

echo "Starting reasoning generation..."
echo "Input data: $DATA_PATH"
echo "Output file: $OUTPUT_PATH"
echo "Batch size: $BATCH_SIZE"

# Ensure you have activated the conda environment first:
# conda activate road-buddy

python src/reasoning_generating.py \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE"

echo "Reasoning generation finished."
