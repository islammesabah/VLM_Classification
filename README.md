# Assessing Tree-Based Reasoning in Vision Language Models (VLMs)

This repository investigates whether hierarchical, tree-based reasoning can improve image classification performance in Vision Language Models (VLMs). It provides a framework to compare standard zero-shot classification with a structured, decision-tree-based approach across multiple datasets and VLMs.

## What This Repo Does

- Implements both zero-shot and tree-based inference for image classification
- Supports multiple VLMs (e.g., GPT-4o, Qwen-VL-Max, Llama-3.2-Vision)
- Works with CIFAR-10 and GTSRB (traffic sign) datasets
- Stores all experiment results in SQLite databases for reproducibility and analysis

## How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API keys

Set your API keys as environment variables for the models you want to use:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ALIBABA_API_KEY="your-alibaba-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

### 3. Download datasets (if needed)

```bash
python datasets/download_CIFAR.py
python datasets/download_GTSRB.py
```

### 4. Run inference

Run zero-shot or tree-based inference with your chosen model and dataset:

```bash
python main.py --model gpt-4o --dataset_name GTSRB
python main.py --model gpt-4o --dataset_name GTSRB --include_memory
```

Additional options:

- `--include_description 1` : Add class descriptions
- `--include_zero_shot_label` : Include zero-shot label in tree input
- `--RunId 0` : Select prompt variant

## Results and Reproducibility

All results from model runs are stored in the SQLite databases:

- `results/cifar10.db` (CIFAR-10 experiments)
- `results/gtsrb.db` (GTSRB experiments)

These databases contain the outputs for all models and configurations, enabling full reproducibility of the reported results.

## Citation

If you use this code or results, please cite the associated paper (comming soon).
