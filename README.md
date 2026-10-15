# MYOPIA: Understanding the Limits of Tree-Based Reasoning in Vision-Language Model

This repository investigates whether hierarchical, tree-based reasoning can improve image classification performance in Vision Language Models (VLMs). It provides a framework to compare standard zero-shot classification with a structured, decision-tree-based approach across multiple datasets and VLMs.

## What This Repo Does

- Implements both zero-shot and tree-based inference for image classification.
- Supports multiple VLMs (e.g., GPT-4o, Qwen-VL-Max, Llama-3.2-Vision).
- Works with CIFAR-10, GTSRB (traffic sign), MNIST, and MSCOCO datasets.
- Stores all experiment results in SQLite databases for reproducibility and analysis.
- Includes CLIP-based baselines for comparison (`evaluation/baselines.py`): plain zero-shot, CHiLS-style hierarchical classification, and description-based classification. These run locally on an open-weight CLIP checkpoint and require no API keys.
- Includes an error-analysis script (`evaluation/tree_error_analysis.py`) that reconstructs per-depth and per-question decision accuracy from stored tree-traversal paths.

## License

The code, decision trees, prompt templates, and evaluation harness in this repository are released under the [MIT License](LICENSE). This does **not** apply to the underlying datasets, which remain under their own original licenses and terms (see below) — you are responsible for complying with those terms when downloading and using the datasets themselves.

## Dataset Licensing & Provenance

| Dataset                                                | License / terms                                                                                                                                                                                                                 | Notes                                                                                                                                                          |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [GTSRB](https://benchmark.ini.rub.de/)                  | Free to use for research; citation requested ([Stallkamp et al., 2012](https://doi.org/10.1016/j.neunet.2012.02.016)). An archival Zenodo mirror additionally lists it under CC BY 4.0.                                          | May incidentally contain identifiable people/vehicles in the background.                                                                                       |
| [MNIST](http://yann.lecun.com/exdb/mnist/)              | CC BY-SA 3.0                                                                                                                                                                                                                    | Derived from NIST data by LeCun and Cortes.                                                                                                                    |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | No formal license published; the authors request only that the accompanying technical report be cited. We use it strictly for non-commercial academic research, consistent with standard practice in the ML research community. |                                                                                                                                                                |
| [MSCOCO](https://cocodataset.org/)                      | Annotations: CC BY 4.0. Images: not owned by the COCO Consortium; each remains under its original Flickr uploader's license.                                                                                                    | This repo redistributes only annotations, class labels, and derived sample-ID lists as part of the benchmark artifact — not the raw MSCOCO images themselves. |

This work uses only these publicly available benchmark datasets and involves no collection of new human-subject or personally identifiable data.

**Intended use.** The released decision trees, prompt templates, and evaluation harness are intended for research on structured/hierarchical reasoning in VLMs. They are not intended as a stand-alone certification suite for deploying any specific model — see the Ethics and Broader Impact section of the paper for a fuller discussion of this and related known limitations (tree quality reflects a single human reviewer's judgment; trees are tailored to their originating dataset's label set).

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

Copy `.env.example` to `.env` and fill in the variables for the models you
want to run:

| Variable | Needed for |
| --- | --- |
| `OPENROUTER_API_KEY` | GPT-4o and LLaMA-3.2-11B-Vision (OpenRouter); automated tree generation with `--backend openrouter` |
| `ALIBABA_API_KEY` | Qwen-VL-Max (DashScope) |
| `OLLAMA_BASE_URL` | the self-hosted `llama3.2-vision:11b-instruct-fp16` endpoint, e.g. `http://localhost:11434/v1` |
| `OPENAI_API_KEY` | automated tree generation with `--backend openai` only |

The main harness never reads `OPENAI_API_KEY`.

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

`run_all_experiments.py` enumerates the paper's full experiment grid (4
datasets x 3 models x conditions x two temperatures, plus the 10-prompt
zero-shot sweep) and launches the `main.py` commands locally or via SLURM:

```bash
python run_all_experiments.py
```

The paper reports the completed subset of that grid; cells below 90% coverage
are suppressed downstream by `evaluation/paper_table.py --min-coverage`.

To run specialized experiments, specify the model, dataset, and other parameters directly. For example:

```bash
python main.py --model gpt-4o --dataset_name GTSRB
```

or

```bash
python main.py --model gpt-4o --dataset_name GTSRB --include_memory
```

#### Argument Options:

| Argument                  | Description                                                                                                                          | Type / Values                                                             | Default   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- | --------- |
| `--model`               | Model to use for inference.                                                                                                          | `gpt-4o`, `qwen-vl-max`, `meta-llama/llama-3.2-11b-vision-instruct` (OpenRouter), `llama3.2-vision:11b-instruct-fp16` (self-hosted Ollama) | —        |
| `--dataset_name`        | Dataset to run inference on.                                                                                                         | `GTSRB`, `CIFAR-10`, `MNIST`, `MSCOCO`                            | `GTSRB` |
| `--include_description` | Include image and class descriptions in inference.`1` to include in tree-based, `2` for zero-shot (requires `--disable_tree`). | `0` (disabled), `1` (tree-based), `2` (zero-shot)                   | `0`     |
| `--disable_tree`        | Run only zero-shot inference.                                                                                                        | Flag                                                                      | False     |
| `--include_memory`      | Include memory of previous questions and answers in tree-based inference only.                                                       | Flag                                                                      | False     |
| `--temperature`         | Temperature setting for model generation.                                                                                            | Float                                                                     | `0.7`   |
| `--RunId`               | Select prompt variant from 10 zero-shot prompt options (only for zero-shot inference).                                               | Integer from`0` to `9`                                                | `0`     |

### 5. Results and Reproducibility

All results from model runs are stored in SQLite databases:

- `results/cifar10.db` (CIFAR-10 experiments)
- `results/gtsrb.db` (GTSRB experiments)
- `results/mnist.db` (MNIST experiments)
- `results/mscoco.db` (MSCOCO experiments)

These databases contain the outputs for all models and configurations.

**GTSRB merged database.** GPT-4o's primary GTSRB condition was re-run with
tree-path logging after 667 early rows stored free-text traversals; the two
sources are combined by:

```bash
python results/build_gtsrb_merged.py   # gtsrb.db + gtsrb_gpt4o_paths.db -> gtsrb_merged.db
```

Run this once before any analysis script — `bootstrap_ci.py`,
`tree_error_analysis.py`, `oracle_extraction_analysis.py`,
`tree_quality_audit.py`, `agreement_table.py`, and `cost_frontier.py` read
`results/gtsrb_merged.db`. All analysis queries additionally go through
`evaluation/db_utils.py`'s `ANSWERS` view, which de-duplicates GTSRB legacy
rows; skipping it inflates GTSRB denominators from 901 to up to 915.

**Reproducing the paper's tables and figures** (offline, from the databases —
no API key):

| Paper artifact | Command |
| --- | --- |
| Table 1 (+ supplement S2) | `python evaluation/paper_table.py -o results/tab_zs_vs_tree_extended.tex` / `python evaluation/supp_table.py` |
| Table 2 + class-level counts | `python evaluation/agreement_table.py` |
| Table 3 | `python evaluation/baselines.py` (local CLIP; needs torch) |
| Tables 4-5 | `python evaluation/grounding_tables.py` |
| Table 6 + Figure 2 | `python evaluation/oracle_extraction_analysis.py && python evaluation/cost_frontier.py` |
| Path-selection depth profile | `python evaluation/tree_error_analysis.py` |
| Bootstrap + Holm (S3) | `python evaluation/bootstrap_ci.py` |
| Tree-quality audit (S8-S9, regressions) | `python evaluation/tree_quality_audit.py` |
| Supplement document | `python evaluation/supplementary.py` then `pdflatex` twice on `results/paper/supplementary.tex` |

The CoT baseline (`evaluation/cot_baseline.py`) and the probe scripts
(`global_list_probe.py`, `constrained_choice_probe.py`, `variance_probe.py`,
`*_knowledge_verification.py`) collect new data via paid API calls; their
stored outputs under `results/` are what the offline table scripts read. The
paper's CoT rows used `--n_images 100 --max_calls 100` per dataset-model pair.

### 6. Additional Notes

- **Image Descriptions**: The `image_descriptions` directory contains the response of each model for each image to the image description prompt. This allows reusing the responses without re-querying the model.
- **CLIP-based baselines**: `evaluation/baselines.py` requires `torch` and `transformers` in addition to `requirements.txt`, and no API keys — it runs entirely on a local CLIP checkpoint (CPU, CUDA, or Apple Silicon MPS). Results are written to `results/baselines/`.
- **Tree error analysis**: `evaluation/tree_error_analysis.py` requires no additional dependencies beyond `requirements.txt` and reads directly from the existing result databases. Results are written to `results/tree_error_analysis/`.
- **Image-conditioned knowledge verification**: `evaluation/image_grounded_knowledge_verification.py` replicates the label-conditioned knowledge-verification protocol but shows the model the actual image instead. Makes real, paid VLM API calls (needs `OPENROUTER_API_KEY`) and hard-stops at a `--max_cost` budget. Results are written to `results/image_grounded_verification/`.
- **Tree generation on a new label set**: `tree_generation/automated_tree_generation/generate_tree.py` runs the LLM generator/evaluator pipeline on any dataset directory with a `class_names.json`, demonstrated on 20 CUB-200-2011 species (`datasets/CUB200Demo/`). Must be run from `tree_generation/automated_tree_generation/` (matching its sibling scripts' relative-path convention); makes real LLM calls (needs `OPENROUTER_API_KEY`) and hard-stops at a `--max_calls` budget.

## Maintenance

This is a research-project repository maintained on a best-effort basis. Issue reporting and contact details (for bugs, data or licensing concerns, or questions about reproducing a result) are omitted from this anonymous review copy and will be added to the public release.
