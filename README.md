# Assessing Tree-Based Reasoning in Vision Language Models (VLMs)

This repository investigates whether hierarchical, tree-based reasoning can improve image classification performance in Vision Language Models (VLMs). It provides a framework to compare standard zero-shot classification with a structured, decision-tree-based approach across multiple datasets and VLMs.

## What This Repo Does

- Implements both zero-shot and tree-based inference for image classification.
- Supports multiple VLMs (e.g., GPT-4o, Qwen-VL-Max, Llama-3.2-Vision).
- Works with CIFAR-10, GTSRB (traffic sign), MNIST, and MSCOCO datasets.
- Stores all experiment results in SQLite databases for reproducibility and analysis.

## How to Use

### 1. Install Dependencies

Using a Python `conda` environment is recommended. Install Miniconda or Anaconda from [here](https://www.anaconda.com/docs/getting-started/miniconda/install), then initialize `conda` in your shell:

```bash
conda init
```

Create a new `conda` environment for this project:

```bash
conda create --name treevlm python=3.9
conda activate treevlm  # For Windows
source activate treevlm  # For Bash
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Set your API keys as environment variables for the models you want to use:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ALIBABA_API_KEY="your-alibaba-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

Alternatively, use a `.env` file containing the API keys in the following format:

```plaintext
OPENAI_API_KEY="your-openai-api-key"
ALIBABA_API_KEY="your-alibaba-api-key"
OPENROUTER_API_KEY="your-openrouter-api-key"
```

### 3. Download Datasets

Download the required datasets using the provided scripts:

```bash
python datasets/download_CIFAR.py
python datasets/download_GTSRB.py
```

To save the images for viewing, add the `--save_images` flag:

```bash
python datasets/download_CIFAR.py --save_images
```

For MNIST:

```bash
cd datasets/MNIST

curl -L -o mnist-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
unzip mnist-dataset.zip

python extract_transform_mnist.py
```

For MSCOCO:

```bash
cd datasets/MSCOCO

curl -L -O http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

curl -L -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

python extract_transform_mscoco.py 'train2014' --num_categories 1 --max_samples 1000 --DATASET_ANNOTATIONS_PATH 'annotations/' --random_seed 42 --copy_images --output_dir './'
```

You should have the following structure for MSCOCO:

```
datasets/MSCOCO
├── annotations
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── person_keypoints_train2014.json
│   └── person_keypoints_val2014.json
├── train2014
│   ├── COCO_train2014_000000000009.jpg
│   ...
│
└── extract_transform_mscoco.py
```

### 4. Run Inference

Run all experiments at once:

```bash
python run_all_experiments.py
```

This will execute all combinations of models, datasets, and configurations included in the paper.

To run specialized experiments, specify the model, dataset, and other parameters directly. For example:

```bash
python main.py --model gpt-4o --dataset_name GTSRB
```

or

```bash
python main.py --model gpt-4o --dataset_name GTSRB --include_memory
```

#### Argument Options:

| Argument                | Description                                                                                                                | Type / Values                                                                                                       | Default |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|---------|
| `--model`               | Model to use for inference.                                                                                                | `gpt-4o`, `qwen-vl-max`, `meta-llama/llama-3.2-11b-vision-instruct`                                                  | —       |
| `--dataset_name`        | Dataset to run inference on.                                                                                                | `GTSRB`, `CIFAR-10`, `MNIST`, `MSCOCO`                                                                              | `GTSRB` |
| `--include_description` | Include image and class descriptions in inference. `1` to include in tree-based, `2` for zero-shot (requires `--disable_tree`). | `0` (disabled), `1` (tree-based), `2` (zero-shot)                                                                    | `0`     |
| `--disable_tree`        | Run only zero-shot inference.                                                                                               | Flag                                                                                                                 | False   |
| `--include_memory`      | Include memory of previous questions and answers in tree-based inference only.                                              | Flag                                                                                                                 | False   |
| `--temperature`         | Temperature setting for model generation.                                                                                  | Float                                                                                                                | `0.7`   |
| `--RunId`               | Select prompt variant from 10 zero-shot prompt options (only for zero-shot inference).                                     | Integer from `0` to `9`                                                                                              | `0`     |

### 5. Results and Reproducibility

All results from model runs are stored in SQLite databases:

- `results/cifar10.db` (CIFAR-10 experiments)
- `results/gtsrb.db` (GTSRB experiments)
- `results/mnist.db` (MNIST experiments)
- `results/mscoco.db` (MSCOCO experiments)

These databases contain the outputs for all models and configurations.

### 6. Additional Notes

- **Image Descriptions**: The `image_descriptions` directory contains the response of each model for each image to the image description prompt. This allows reusing the responses without re-querying the model.