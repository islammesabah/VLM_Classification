# MYOPIA: Understanding the Limits of Tree-Based Reasoning in Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2610.21573-b31b1b.svg)](https://arxiv.org/abs/2610.21573)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-myopia--benchmark-yellow)](https://huggingface.co/datasets/IslamMesabah/myopia-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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
conda create --name treevlm python=3.12
conda activate treevlm  # For Windows
source activate treevlm  # For Bash
```

Python 3.9 or newer works (the code uses PEP 585 builtin generics); 3.12 is
what the shipped results were produced and re-verified on.

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Copy `.env.example` to `.env` and fill in the variables for the models you
want to run:

| Variable | Needed for |
| --- | --- |
| `OPENROUTER_API_KEY` | GPT-4o (OpenRouter); automated tree generation with `--provider openrouter` (the default) |
| `ALIBABA_API_KEY` | Qwen-VL-Max (DashScope) |
| `OPENAI_API_KEY` | automated tree generation with `--provider openai` only |
| `OPENROUTER_BASE_URL` | optional; overrides the GPT-4o endpoint. Default `https://openrouter.ai/api/v1` |
| `ALIBABA_BASE_URL` | optional; overrides the Qwen-VL-Max endpoint. Default `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| `OLLAMA_BASE_URL` | **required** for `llama3.2-vision:11b-instruct-fp16`; no default. e.g. `http://localhost:11434/v1` |

The main harness never reads `OPENAI_API_KEY`.

### Prompts

No prompt wording lives in the code. Every prompt is an asset under `prompts/`,
loaded through `prompt_library.py`:

| File | Contents |
| --- | --- |
| `templates.json` | The 12 named prompt templates the harness sends, with `{placeholder}` fields |
| `<DATASET>/zs_prompts.json` | The ten zero-shot paraphrases (index = `RunId`) |
| `<DATASET>/classes_description.json` | Per-class descriptions for the with-descriptions conditions |
| `describe_image.txt` | The image-description prompt used when `--include_description` is set |
| `clip_superclass_prompts.json` | CHiLS root-branch captions per dataset |
| `prompts_generator.json`, `prompts_evaluator.json` | System prompts for the tree-generation pipeline |

Editing a string in `templates.json` changes the experiment, so keep it in sync
with Sec 4.3 of the paper. The directory is found relative to `prompt_library.py`,
so scripts run from any working directory; set `MYOPIA_PROMPTS_DIR` to override.

### 3. Download Datasets

Each dataset has one script that downloads it **and reduces it to the fixed
evaluation subset** listed in `sample_ids/`. Everything in one go:

```bash
python datasets/download.py
```

Or individually, or a subset:

```bash
python datasets/download_GTSRB.py
python datasets/download_CIFAR.py
python datasets/download_MNIST.py
python datasets/download_MSCOCO.py

python datasets/download.py --dataset MNIST GTSRB
```

Each script writes what its loader reads: `class_names.json` (from
`datasets/class_names/`), the sample-id list under the name `config.json`
expects, and the filtered images.

| Dataset | Subset | On disk after the run |
| --- | --- | --- |
| GTSRB | 901 frames | `Training/` pruned to the sampled frames, `901_samples.txt` |
| CIFAR-10 | 1000 images | raw `images/data_batch_*` + `1000_samples.txt` (the loader filters in memory) |
| MNIST | 1000 images | `sampled_images/`, `image_to_class_mapping.json`, `sampled_images_metadata.json`, `sampled_image_ids.txt` |
| MSCOCO | 850 images | `filtered_1_category_images/` pruned to the subset, `850_filtered_1_category_image_ids.txt` |

Notes:

- Re-running is cheap: anything already on disk is skipped.
- Paths resolve relative to `datasets/`, so the scripts work from any working directory.
- If a listed sample id is missing from the download, the script stops with an
  error rather than silently producing a short subset.
- MSCOCO deletes the unfiltered `train2014/` source after filtering (~13 GB);
  pass `--keep_source` to retain it.
- MNIST comes from the public S3 mirror of the original IDX files, so no Kaggle
  account is needed; `--url_base` overrides the host.
- `download_CIFAR.py --save_images` additionally writes every CIFAR image as a
  PNG; the harness does not need them.

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
| `--model`               | Model to use for inference.                                                                                                          | `gpt-4o`, `qwen-vl-max`, `llama3.2-vision:11b-instruct-fp16` (self-hosted Ollama) | —        |
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

These databases contain the outputs for all models and configurations. Their
paths come from `config.json`; `paper_table.py` and `supp_table.py` resolve
their inputs through it.

`bootstrap_ci.py`, `tree_error_analysis.py`, `oracle_extraction_analysis.py`,
`tree_quality_audit.py`, `agreement_table.py` and `cost_frontier.py` hardcode
the same four paths rather than going through `config.json`, so a fresh
checkout must have the databases staged into `results/` before they run.

**GTSRB provenance.** GPT-4o's primary GTSRB condition was collected twice:
the first pass stored a free-text answer instead of the JSON traversal path
for 667 of its 901 images, so it was re-run with path logging intact. The
shipped `gtsrb.db` takes GPT-4o's primary-condition rows from the re-run and
every other row from the original pass — each row comes from exactly one run.
No merge step is needed at analysis time. See the "Known caveats" section of
the top-level `README.md` for the full account.

All analysis queries go through `evaluation/db_utils.py`'s `ANSWERS` view,
which de-duplicates GTSRB legacy `Data/GTSRB/all_sequences_training/` rows;
skipping it inflates GTSRB denominators from 901 to up to 915.

**Reproducing the paper's tables and figures** (offline, from the databases —
no API key):

| Paper artifact | Command |
| --- | --- |
| Table 1 (+ supplement S2) | `python evaluation/paper_table.py -o results/tab_zs_vs_tree_extended.tex` / `python evaluation/supp_table.py` |
| Table 2 + class-level counts | `python evaluation/agreement_table.py` |
| Table 3 | `python evaluation/baselines.py` (local CLIP; needs torch) |
| Tables 4-5 | `python evaluation/grounding_tables.py` |
| Table 6 + cost-frontier figure | `python evaluation/oracle_extraction_analysis.py && python evaluation/cost_frontier.py` |
| Path-selection depth profile | `python evaluation/tree_error_analysis.py` |
| Bootstrap + Holm (S3) | `python evaluation/bootstrap_ci.py` |
| Tree-quality audit (S8-S9, regressions) | `python evaluation/tree_quality_audit.py` |
| Supplement Table S1 (all conditions, with n) | `python evaluation/supp_table.py -o results/tab_supp_full.tex` |

One ordering constraint: `oracle_extraction_analysis.py` must run before
`cost_frontier.py`, which reads
`results/tree_error_analysis/oracle_failure_breakdown_primary_condition.csv`.
Run every command from this directory (`src/`) — all paths above are relative
to it.

The scripts above regenerate the numbers behind each section of the paper's
supplementary material — see the archive's top-level `README.md` for the
section-to-script mapping.

The `tables/` LaTeX and the CSVs under `data/` in the archive
are byte-for-byte the outputs of these commands on the shipped databases.

The CoT baseline (`evaluation/cot_baseline.py`) and the probe scripts
(`global_list_probe.py`, `constrained_choice_probe.py`, `variance_probe.py`,
`*_knowledge_verification.py`) collect new data via paid API calls; their
stored outputs under `results/` are what the offline table scripts read. The
paper's CoT rows used `--n_images 100 --max_calls 100` per dataset-model pair.

### 6. Additional Notes

- **Image descriptions**: the `--include_description` conditions first ask the model to describe the image, then feed that text into classification. Each description is cached on disk as it is produced, under `datasets/<DATASET>/image_descriptions/` (created at run time by `main.py`), keyed by image and model, so re-running a condition reuses the stored descriptions instead of paying for them again. The cache lives inside the dataset directories and is therefore **not** part of this archive — it is rebuilt on demand, and only for Tier 2 collection runs.
- **CLIP-based baselines**: `evaluation/baselines.py` requires `torch` and `transformers` in addition to `requirements.txt`, and no API keys — it runs entirely on a local CLIP checkpoint (CPU, CUDA, or Apple Silicon MPS). Results are written to `results/baselines/`.
- **Tree error analysis**: `evaluation/tree_error_analysis.py` requires no additional dependencies beyond `requirements.txt` and reads directly from the existing result databases. Results are written to `results/tree_error_analysis/`.
- **Image-conditioned knowledge verification**: `evaluation/image_grounded_knowledge_verification.py` replicates the label-conditioned knowledge-verification protocol but shows the model the actual image instead. Makes real, paid VLM API calls (needs `OPENROUTER_API_KEY`) and hard-stops at a `--max_cost` budget. Results are written to `results/image_grounded_verification/`.
- **Tree generation on a new label set**: `tree_generation/automated_tree_generation/generate_tree.py` runs the LLM generator/evaluator pipeline on any directory containing a `class_names.json` (`--dataset_path`). It must be run from `tree_generation/automated_tree_generation/`, matching its sibling scripts' relative-path convention; it makes real LLM calls (needs `OPENROUTER_API_KEY`, or `OPENAI_API_KEY` with `--provider openai`) and hard-stops at a `--max_calls` budget. The shipped demo covers 20 CUB-200-2011 species; no image data is needed, only the label set, so reproducing it is:

  ```bash
  mkdir -p datasets/CUB200Demo
  cp datasets/class_names/cub200_demo_class_names.json datasets/CUB200Demo/class_names.json
  cd tree_generation/automated_tree_generation
  python generate_tree.py --dataset_path ../../datasets/CUB200Demo \
      --max_calls 40 --output_prefix cub200_demo_tree
  ```

  The result of that run is shipped as `cub200_demo_tree.json` next to the
  script (18 decision nodes, 20 leaves, deepest question at level 5), with a
  human-readable dump in `trees/cub200_demo_tree.txt`. Regenerating it will
  not reproduce the same tree verbatim: the questions come from live LLM
  calls.

## Maintenance

This is a research-project repository maintained on a best-effort basis. Please report bugs, data or licensing concerns, and questions about reproducing a result via [GitHub Issues](https://github.com/islammesabah/VLM_Classification/issues).

## Citation

If you use this code or the MYOPIA benchmark, please cite:

```bibtex
@misc{mesabah2026myopia,
  title         = {MYOPIA: Understanding the Limits of Tree-Based Reasoning in Vision-Language Models},
  author        = {Mesabah, Islam and Elmansoury, Sary and Gro{\ss}mann, Gerrit and Neigel, Peter and Vollmer, Sebastian J.},
  year          = {2026},
  eprint        = {2610.21573},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2610.21573},
  url           = {https://arxiv.org/abs/2610.21573}
}
```
