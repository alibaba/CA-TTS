#!/bin/bash

# CA-TTS Benchmarking Script
# Usage: Set environment variables and run this script

# --- Environment Setup ---
# export CUDA_VISIBLE_DEVICES=0,1
# export LD_LIBRARY_PATH=/path/to/your/cudnn/lib  # Optional: if needed
# export EXPERT_API_KEY='your_api_key'
# export GOOGLE_API_KEY='your_google_api_key'

# --- 1. Configuration Parameters ---
TASK_NAME="${TASK_NAME:-math-vision}"
MODEL_PATH="${MODEL_PATH:?Error: MODEL_PATH environment variable not set}"

GPU_IDS="${GPU_IDS:-0}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-1}"

# Experiment variables
MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
ADD_NOISE="${ADD_NOISE:-False}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
VERIFY_MODE="${VERIFY_MODE:-True}"
VERIFY_TAU="${VERIFY_TAU:-0.5}"

# --- 2. Dynamic Output Path Construction ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BASE_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
mkdir -p "$BASE_DIR"

BASE_NAME="${MODEL_FAMILY}_${TASK_NAME}"

NOISE_LOG_PATH="${NOISE_LOG_PATH:-${BASE_DIR}/noise_log.json}"

EXPERT_MODEL="${EXPERT_MODEL:-gemini-2.5-pro}"

# Noise suffix
if [ "$ADD_NOISE" = "True" ]; then
    NOISE_SUFFIX="noise"
else
    NOISE_SUFFIX="text"
fi

# Verify suffix
if [ "$VERIFY_MODE" = "True" ]; then
    TAU_FILENAME=$(echo $VERIFY_TAU | tr '.' 'p')
    VERIFY_SUFFIX="verify_tau_${TAU_FILENAME}"
else
    VERIFY_SUFFIX="noverify"
fi

# Build final filename
FILENAME="${BASE_NAME}_${NOISE_SUFFIX}_sample_${NUM_SAMPLES}_${VERIFY_SUFFIX}_${EXPERT_MODEL}.json"
OUTPUT_JSON_PATH="${BASE_DIR}/${FILENAME}"

# Print configuration
echo "--- CA-TTS Benchmark Configuration ---"
echo "Task: $TASK_NAME"
echo "Model: $MODEL_PATH"
echo "Noise: $ADD_NOISE ($NOISE_SUFFIX)"
echo "Samples: $NUM_SAMPLES"
echo "Verify: $VERIFY_MODE (Tau: $VERIFY_TAU)"
echo "Expert Model: $EXPERT_MODEL"
echo "--------------------------------------"
echo "Output Path: $OUTPUT_JSON_PATH"
echo "--------------------------------------"

# --- 3. Run Python Script ---
python "${PROJECT_ROOT}/run_benchmark.py" \
    --task_name "$TASK_NAME" \
    --output "$OUTPUT_JSON_PATH" \
    --model "$MODEL_PATH" \
    --gpu_ids "$GPU_IDS" \
    --model_family "$MODEL_FAMILY" \
    --processes_per_gpu "$PROCESSES_PER_GPU" \
    --add_noise "$ADD_NOISE" \
    --num_samples "$NUM_SAMPLES" \
    --verify_mode "$VERIFY_MODE" \
    --verify_tau "$VERIFY_TAU" \
    --noise_log_path "$NOISE_LOG_PATH" \
    --verify_expert_model "$EXPERT_MODEL"
