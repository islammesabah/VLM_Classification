# Assessing Tree-Based Reasoning in Vision Language Models (VLMs)

This project implements and evaluates tree-based reasoning approaches for image classification using Vision Language Models (VLMs). The system compares traditional zero-shot classification with hierarchical tree-based inference to assess the effectiveness of structured reasoning in VLM performance.

## 🎯 Project Overview

The project evaluates whether tree-based reasoning can improve VLM classification performance by:
- Implementing hierarchical decision trees for image classification
- Comparing zero-shot vs. tree-based inference approaches
- Supporting multiple datasets (CIFAR-10, GTSRB Traffic Signs)
- Testing various VLM models (GPT-4o, Qwen-VL-Max, Llama-3.2-Vision)

## 📁 Project Structure

```
VLM_Classification/
├── main.py                 # Main execution script
├── config.json            # Configuration for datasets and models
├── requirements.txt       # Python dependencies
├── main_run.sh           # Shell script for batch execution
├── models/
│   ├── vlm_model.py      # VLM client implementations
│   └── openrouter_providers.json
├── datasets/
│   ├── data_loader.py    # Dataset loading and processing
│   ├── data_cleaning.py  # Data preprocessing utilities
│   ├── download_CIFAR.py # CIFAR-10 download script
│   ├── download_GTSRB.py # GTSRB download script
│   ├── CIFAR-10/         # CIFAR-10 dataset
│   └── GTSRB/           # GTSRB traffic signs dataset
├── tree_generation/
│   ├── gtsrb_tree.py     # GTSRB decision tree implementation
│   ├── cifar_tree.py     # CIFAR-10 decision tree implementation
│   ├── gtsrb-tree.txt    # GTSRB tree structure
│   └── cifar10-tree.txt  # CIFAR-10 tree structure
├── prompts/
│   ├── CIFAR-10/         # CIFAR-10 prompts and descriptions
│   ├── GTSRB/           # GTSRB prompts
│   └── include_description.txt
├── results/
│   ├── cifar10.db       # CIFAR-10 results database
│   └── gtsrb.db         # GTSRB results database
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd VLM_Classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Set up your API keys for the VLM providers:

```bash
# For OpenAI GPT-4o
export OPENAI_API_KEY="your-openai-api-key"

# For Alibaba Qwen-VL-Max
export ALIBABA_API_KEY="your-alibaba-api-key"

# For OpenRouter (Llama-3.2-Vision)
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

### 3. Download Datasets

```bash
# Download CIFAR-10
python datasets/download_CIFAR.py

# Download GTSRB
python datasets/download_GTSRB.py
```

### 4. Run Experiments

```bash
# Basic zero-shot inference
python main.py --model gpt-4o --dataset_name GTSRB

# Tree-based inference
python main.py --model gpt-4o --dataset_name GTSRB --include_memory

# With additional features
python main.py --model gpt-4o --dataset_name GTSRB \
    --include_memory \
    --include_description 1 \
    --include_zero_shot_label \
    --RunId 0
```

## 🔧 Core Components

### 1. Main Execution (`main.py`)

The main script orchestrates the entire inference pipeline:

- **Argument Parsing**: Configures experiment parameters
- **Configuration Loading**: Loads dataset and model settings
- **Inference Execution**: Runs zero-shot and tree-based inference
- **Result Storage**: Saves results to SQLite databases

Key classes:
- `InferenceConfig`: Manages experiment configuration
- `DatabaseManager`: Handles result storage and retrieval
- `VLMInference`: Core inference engine

### 2. VLM Models (`models/vlm_model.py`)

Supports multiple VLM providers:
- **GPT-4o** (OpenAI): High-performance multimodal model
- **Qwen-VL-Max** (Alibaba): Advanced vision-language model
- **Llama-3.2-11B-Vision** (Meta): Open-source vision model

### 3. Dataset Loaders (`datasets/data_loader.py`)

Abstract interface for dataset handling:
- `DatasetLoader`: Base class defining the interface
- `CIFARLoader`: CIFAR-10 specific implementation
- `GTSRBLoader`: GTSRB traffic signs implementation

### 4. Tree Generation (`tree_generation/`)

Implements hierarchical decision trees for structured reasoning:

#### GTSRB Tree (`gtsrb_tree.py`)
Organizes traffic signs into logical categories:
- **Warnings**: Triangle signs with various hazard types
- **Speed Limits**: Circular signs with numerical limits
- **Prohibitions**: Red circular signs with restrictions
- **Mandatory**: Blue circular signs with directions

#### CIFAR-10 Tree (`cifar_tree.py`)
Categorizes natural images by:
- **Animals**: bird, cat, deer, dog, frog, horse
- **Vehicles**: airplane, automobile, ship, truck

### 5. Configuration (`config.json`)

Centralized configuration for:
- Dataset paths and parameters
- Model settings and API configurations
- Inference parameters (temperature, max tokens)
- Database file locations

## 🧪 Experiment Parameters

### Model Options
- `gpt-4o`: OpenAI's GPT-4 with vision
- `qwen-vl-max`: Alibaba's Qwen-VL-Max
- `meta-llama/llama-3.2-11b-vision-instruct`: Meta's Llama-3.2-Vision

### Dataset Options
- `GTSRB`: German Traffic Sign Recognition Benchmark (43 classes)
- `CIFAR-10`: Natural image classification (10 classes)

### Inference Modes
- `--disable_tree`: Run only zero-shot inference
- `--include_memory`: Include conversation history in tree inference
- `--include_description`: Add class descriptions (0=off, 1=short, 2=long)
- `--include_zero_shot_label`: Include zero-shot predictions in tree input
- `--RunId`: Select different prompt variations (0-9)

## 📊 Results and Analysis

Results are stored in SQLite databases:
- `results/cifar10.db`: CIFAR-10 experiment results
- `results/gtsrb.db`: GTSRB experiment results

Each database contains:
- Image paths and metadata
- Zero-shot predictions and confidence
- Tree-based predictions and reasoning paths
- Model configurations and parameters

## 🔬 Research Context

This project investigates whether structured reasoning can improve VLM performance by:

1. **Hierarchical Classification**: Using decision trees to break down complex classification tasks
2. **Contextual Reasoning**: Incorporating previous decisions and descriptions
3. **Comparative Analysis**: Evaluating tree-based vs. zero-shot approaches
4. **Multi-Modal Evaluation**: Testing across different datasets and model architectures

## 📝 Usage Examples

### Basic Zero-Shot Classification
```bash
python main.py --model gpt-4o --dataset_name GTSRB
```

### Tree-Based Inference with Memory
```bash
python main.py --model gpt-4o --dataset_name GTSRB \
    --include_memory \
    --include_description 1
```

### Comparative Analysis
```bash
# Run multiple configurations
python main.py --model gpt-4o --dataset_name GTSRB --RunId 0
python main.py --model gpt-4o --dataset_name GTSRB --RunId 1
python main.py --model qwen-vl-max --dataset_name CIFAR-10
```

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 📚 References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- [Vision Language Models](https://arxiv.org/abs/2302.00923)
