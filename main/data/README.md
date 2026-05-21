# Dataset Setup

This directory should contain the benchmark datasets used for evaluation.

## Supported Datasets

### MathVista
- Download from: [MathVista GitHub](https://github.com/lupantech/MathVista)
- Place `testmini.json` in this directory

### MathVision
- Download from: [MathVision Dataset](https://mathvision-cuhk.github.io/)
- Place `testmini_convert.json` in this directory

### MMMU
- Download from: [MMMU GitHub](https://github.com/MMMU-Benchmark/MMMU)
- Follow their instructions for dataset preparation

### MMStar
- Download from: [MMStar GitHub](https://github.com/MMStar-Benchmark/MMStar)
- Follow their instructions for dataset preparation

## Directory Structure

After downloading, your structure should look like:

```
data/
├── README.md
├── testmini.json              # MathVista
├── testmini_convert.json      # MathVision
├── MMMU/                      # MMMU dataset
└── mmstar/                    # MMStar dataset
```

## Configuration

Update the `dataset_path` in the corresponding task config file:
- `configs/tasks/math_vista.yaml`
- `configs/tasks/math_vision.yaml`
- `configs/tasks/mmmu.yaml`
- `configs/tasks/mmstar.yaml`
