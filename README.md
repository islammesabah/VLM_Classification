# Assessing Tree-Based Reasoning in Vision Language Models (VLMs)

This repository investigates whether hierarchical, tree-based reasoning can improve image classification performance in Vision Language Models (VLMs). It provides a framework to compare standard zero-shot classification with a structured, decision-tree-based approach across multiple datasets and VLMs.

## What This Repo Does

- Implements both zero-shot and tree-based inference for image classification
- Supports multiple VLMs (e.g., GPT-4o, Qwen-VL-Max, Llama-3.2-Vision)
- Works with CIFAR-10 and GTSRB (traffic sign) datasets
- Stores all experiment results in SQLite databases for reproducibility and analysis

## How to Use

### 1. Install dependencies

python conda environment is recommended.
Install miniconda or anaconda from [here](https://www.anaconda.com/docs/getting-started/miniconda/install)
then init conda in your shell:
```bash
conda init
```

Create a new conda environment for this project:
```bash
conda create --name treevlm python=3.8
conda activate treevlm # Windows
source activate treevlm # Bash
```

Then install the dependencies:
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

or use a `.env` file which contains the API keys in the following format:

```plaintext
OPENAI_API_KEY="your-openai-api-key"
ALIBABA_API_KEY="your-alibaba-api-key"
OPENROUTER_API_KEY="your-openrouter-api-key"
```

### 3. Download datasets (if needed)

```bash
python datasets/download_CIFAR.py
python datasets/download_GTSRB.py
```
Note: if you want to save the images to be able to view them, you can add --save_images to python datasets/download_CIFAR.py
### 4. Run inference

We have made it easy to run all the experiments all at once by running:
```
python run_all_experiments.py
```
It will run all the combinations of models, datasets, and configurations that we have included in the paper.

If you want to run specialized experiments, you can do so by specifying the model, dataset, and other parameters directly.

Run zero-shot or tree-based inference for example:

```bash
python main.py --model gpt-4o --dataset_name GTSRB
```
or
```
python main.py --model gpt-4o --dataset_name GTSRB --include_memory
```

Argument options:
| Argument                    | Description                                                                                                                | Type / Values                                                                                                       | Default |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|---------|
| `--model`                   | Model to use for inference.                                                                                                | `gpt-4o`, `qwen-vl-max`, `meta-llama/llama-3.2-11b-vision-instruct`                                                  | —       |
| `--dataset_name`            | Dataset to run inference on.                                                                                                | `GTSRB`, `CIFAR-10`                                                                                                 | `GTSRB` |
| `--include_description`     | Include image and class descriptions in inference. `1` to include in tree-based, `2` for zero-shot (option 2 requires `--disable_tree`).      | `0` (disabled), `1` (tree-based), `2` (zero-shot)                                                                    | `0`     |
| `--disable_tree`            | Run only zero-shot inference.                               | Flag                                                                                                                 | False   |
| `--include_memory`          | Include memory of previous questions and answers in tree-based inference only.                                                                 | Flag                                                                                                                 | False   |
| `--temperature`             | Temperature setting for model generation.                                                                                  | Float                                                                                                  | `0.7`   |
| `--RunId`                   | Select prompt variant from 10 zero-shot prompt options (only for zero-shot inference).                                     | Integer from `0` to `9`                                                                                              | `0`     |


## Results and Reproducibility

All results from model runs are stored in the SQLite databases:

- `results/cifar10.db` (CIFAR-10 experiments)
- `results/gtsrb.db` (GTSRB experiments)

These databases contain the outputs for all models and configurations.

## Citation

If you use this code or results, please cite the associated paper (coming soon).
