#!/bin/bash

# CA-TTS Metrics Evaluation Script
# Usage: Set environment variables and run this script

# --- Environment Setup ---
# export CUDA_VISIBLE_DEVICES=0
# export LD_LIBRARY_PATH=/path/to/your/cudnn/lib  # Optional: if needed

# --- Configuration Parameters ---
TASK_NAME="${TASK_NAME:-math-vision}"
JSON_PATH="${JSON_PATH:?Error: JSON_PATH environment variable not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Print configuration
echo "--- CA-TTS Metrics Evaluation ---"
echo "Task: $TASK_NAME"
echo "Input: $JSON_PATH"
echo "----------------------------------"

# Run Python script
python "${PROJECT_ROOT}/run_metrics.py" \
  --task_type "$TASK_NAME" \
  --json_file "$JSON_PATH"

