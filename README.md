# CA-TTS: Confidence-Aware Test-Time Scaling for MLLMs

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)](https://cvpr.thecvf.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"Linking Perception, Confidence and Accuracy in MLLMs"** (CVPR 2026).

**Authors:** Yuetian Du<sup>1*</sup>, Yucheng Wang<sup>1*</sup>, Rongyu Zhang<sup>1</sup>, Zhijie Xu<sup>4</sup>, Boyu Yang<sup>2</sup>, Ming Kong<sup>1</sup>, Jie Liu<sup>3†</sup>, Qiang Zhu<sup>1†</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>Alibaba Group, <sup>3</sup>City University of Hong Kong, <sup>4</sup>University of Michigan

<sup>*</sup>Equal contribution, <sup>†</sup>Corresponding authors

## Abstract

Recent advances in Multi-modal Large Language Models (MLLMs) have predominantly focused on enhancing visual perception to improve accuracy. However, a critical question remains unexplored: Do models know when they do not know? Through a probing experiment, we reveal a severe confidence miscalibration problem in MLLMs. To address this, we propose Confidence-Driven Reinforcement Learning (CDRL), which uses original-noise image pairs and a novel confidence-based reward to enhance perceptual sensitivity and robustly calibrate the model's confidence. Beyond training benefits, calibrated confidence enables more effective test-time scaling as a free lunch. We further propose Confidence-Aware Test-Time Scaling (CA-TTS), which dynamically coordinates Self-Consistency, Self-Reflection, and Visual Self-Check modules guided by confidence signals. An Expert Model acts in multiple roles (e.g., Planner, Critic, Voter) to schedule these modules and provide external verification. Our integrated framework establishes new state-of-the-art results with consistent 8.8% gains across four benchmarks. More ablation studies demonstrate the effectiveness of each module and scaling superiority.

## Architecture

```
Input (Image + Question)
         |
         v
   Expert Planner (M^Planner_expert)
         |
         v
   Scheduling Order π
         |
         v
   +------------------+------------------+------------------+
   |                  |                  |                  |
   v                  v                  v                  v
Self-Consistency  Self-Reflection   Self-Check
   (M_sc)            (M_sr)            (M_sk)
   |                  |                  |
   +------------------+------------------+
                      |
                      v
            Shared Voting (V_final)
                      |
                      v
                 Final Answer
```

## Key Features

- **Confidence-Aware Voting**: Multiple confidence metrics (mean, tail, bottom-window, min-window)
- **Expert Planning**: LLM-based adaptive module scheduling
- **Visual Self-Check**: Contrastive decoding with diffusion noise
- **Self-Reflection**: Expert critique for low-confidence answers
- **Modular Design**: Fully decoupled, order-insensitive modules

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/CA-TTS.git
cd CA-TTS

# Create conda environment
conda env create -f environment.yml
conda activate catts

# Or install via pip
pip install -r requirements.txt
```

## Project Structure

```
CA-TTS/
├── ca_tts/                        # Python package
│   ├── __init__.py
│   ├── outputs.py
│   ├── core/
│   │   ├── algorithms.py          # Voting & confidence algorithms
│   │   ├── generation.py          # Model inference
│   │   ├── prompts.py             # Prompt templates
│   │   ├── strategies.py          # High-level strategies
│   │   └── verifier.py            # Expert verification
│   └── planner/
│       ├── __init__.py
│       └── planner.py             # Expert Planner module
├── configs/
│   ├── default.yaml               # Default hyperparameters
│   └── tasks/                     # Task-specific configs
├── scripts/
│   ├── run_benchmark.sh
│   └── run_metrics.sh
├── tools/
│   └── noise_visualization.py     # Noise analysis tools
├── data/                          # Dataset directory
├── run_benchmark.py               # Main entry point
├── run_metrics.py                 # Evaluation metrics
├── environment.yml
├── requirements.txt
├── README.md
└── LICENSE
```

## Quick Start

### 1. Prepare Datasets

Download the required datasets and place them in the `data/` directory. See [data/README.md](data/README.md) for details.

### 2. Set Environment Variables

```bash
# For expert model API access
export EXPERT_API_KEY="your_api_key_here"
export EXPERT_API_BASE_URL="https://api.openai.com/v1"  # Optional

# For Gemini API (if using Gemini as expert)
export GOOGLE_API_KEY="your_google_api_key"
```

### 3. Run Baseline Evaluation

```bash
python run_benchmark.py \
  --task_name math-vista \
  --model_path /path/to/qwen2.5-vl-7b \
  --dataset_path ./data/testmini.json \
  --num_samples 8 \
  --gpu_ids 0
```

### 4. Run with Expert Planner

```bash
python run_benchmark.py \
  --task_name math-vista \
  --model_path /path/to/qwen2.5-vl-7b \
  --dataset_path ./data/testmini.json \
  --num_samples 8 \
  --verify_mode True \
  --planner_enabled True \
  --expert_model gemini-2.5-pro \
  --gpu_ids 0
```

## Configuration

All hyperparameters can be configured via YAML files or command-line arguments.

### Main Config (`configs/default.yaml`)

```yaml
model:
  path: /path/to/model
  family: "qwen"  # or "internvl"

self_consistency:
  num_samples: 8
  add_noise: false

verification:
  enabled: true
  expert_model: "gemini-2.5-pro"

planner:
  enabled: true
  mode: "llm"
```

### Task Config (`configs/tasks/math_vista.yaml`)

```yaml
task_name: "math-vista"
dataset_path: ./data/testmini.json
prompt_template: "zero"
```

## Supported Models

- Qwen2.5-VL-7B / 72B
- InternVL3-8B
- Other Hugging Face Transformers-compatible vision-language models

## Evaluation

After running benchmarks, evaluate the results:

```bash
python run_metrics.py \
  --result_file ./results/math_vista_results.json \
  --task_name math-vista
```

## Citation

```bibtex
@misc{du2026linkingperceptionconfidenceaccuracy,
      title={Linking Perception, Confidence and Accuracy in MLLMs}, 
      author={Yuetian Du and Yucheng Wang and Rongyu Zhang and Zhijie Xu and Boyu Yang and Ming Kong and Jie Liu and Qiang Zhu},
      year={2026},
      eprint={2603.12149},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.12149}, 
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon research in test-time scaling, self-consistency, and multimodal reasoning. We thank the open-source community for their foundational contributions.
