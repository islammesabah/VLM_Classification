# MYOPIA — Reproducibility Archive

[![arXiv](https://img.shields.io/badge/arXiv-2610.21573-b31b1b.svg)](https://arxiv.org/abs/2610.21573)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-myopia--benchmark-yellow)](https://huggingface.co/datasets/IslamMesabah/myopia-benchmark)

This archive accompanies *MYOPIA: Understanding the Limits of Tree-Based
Reasoning in Vision-Language Models*
([arXiv:2610.21573](https://arxiv.org/abs/2610.21573)). Every number in the
main paper is reproducible from it: the analysis layer runs entirely offline
from the shipped databases and CSVs (no API key), and the collection layer
(paid API calls) is included for full-pipeline replication. The data behind
the supplementary material ships too, and its Tables S1, S2 and S8-S9
regenerate the same way (see "Reproduction matrix" below).

It is the frozen snapshot behind the paper's numbers. The repository root
holds the development copy of the same harness, which records the LLaMA
model under its OpenRouter name rather than the self-hosted FP16 name used
here; use this directory to reproduce the paper. The benchmark itself (trees,
prompts, sample ids, reference predictions) is also on the
[Hugging Face Hub](https://huggingface.co/datasets/IslamMesabah/myopia-benchmark).

## Archive layout

```
src/                         harness, tree pipeline, and analysis scripts
data/dbs/                    the four result databases (SQLite, uncompressed)
data/                        derived CSVs (probes, CoT, baselines, audit, ...)
trees/                       the four evaluated decision trees + CUB-200 demo
prompts/                     all prompt templates and class descriptions
sample_ids/                  fixed evaluation subsets (901/1000/1000/850 ids)
tables/                      LaTeX of the reported tables
figures/                     script-generated figures (cost frontier)
SUPPLEMENT_MANIFEST.json     contents, artifact-to-script map, licensing
```

Only the tables and figures that scripts generate ship here. The paper and
supplement text, and their hand-made figures, are not part of the repository.

## Setup

```bash
cd src
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras: `torch`+`transformers` only for the CLIP baselines
(`evaluation/baselines.py`). For data collection, copy
`.env.example` to `.env` and fill in the keys — the offline analysis needs
none of them.

**Stage the data** (once), from `src/`:

```bash
mkdir -p results
cp ../data/dbs/*.db results/
cp ../data/bootstrap_ci_holm.csv results/
for d in agreement baselines constrained_choice_probe cot_baseline \
         global_list_probe grounding image_grounded_verification \
         inference_cost label_conditioned_verification \
         tree_error_analysis tree_quality_audit; do
  cp -r ../data/$d results/
done
```

Two notes on the GTSRB database. First, `gtsrb.db` is the single GTSRB
database every script reads, and it already carries the re-run described
under "Known caveats" below: GPT-4o's primary GTSRB condition was collected
twice, and the 901 path-logged rows of the second pass are the ones in this
file. There is no separate merge step to run. Second, all queries go through
`evaluation/db_utils.py`'s deduplicating `ANSWERS` view (GTSRB legacy
`Data/GTSRB/all_sequences_training/` rows); without it GTSRB denominators
inflate from 901 to up to 915.

## Reproduction matrix

Tier 1 — offline, from the shipped data, run from `src/`:

| Paper artifact                                  | Command                                                                                                                                   | Output                                                      |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Table 1 (mean accuracy, T=0.7)                  | `python evaluation/paper_table.py -o results/tab_zs_vs_tree_extended.tex`                                                               | `results/tab_zs_vs_tree_extended.tex`                     |
| Table 2 + "72/129"-style class counts           | `python evaluation/agreement_table.py`                                                                                                  | `results/agreement/`                                      |
| Table 3 (CLIP baselines)                        | shipped`data/baselines/summary_accuracy.csv`; recompute with `python evaluation/baselines.py` (needs torch + dataset images)          | `results/baselines/`                                      |
| Table 4 (label vs image grounding)              | `python evaluation/grounding_tables.py`                                                                                                 | `results/grounding/`                                      |
| Table 5 (GTSRB accuracy by depth)               | same command                                                                                                                              | `results/grounding/`                                      |
| Table 6 (calls + unresolved rates)              | `python evaluation/oracle_extraction_analysis.py && python evaluation/cost_frontier.py`                                                 | `results/inference_cost/`                                 |
| Cost-frontier figure (accuracy vs calls)        | `python evaluation/cost_frontier.py`                                                                                                    | `results/cost_frontier.pdf`                               |
| Path-selection depth profile (Sec 5.3)          | `python evaluation/tree_error_analysis.py`                                                                                              | `results/tree_error_analysis/`                            |
| Paired bootstrap + Holm (Sec 5.1, S3)           | `python evaluation/bootstrap_ci.py`                                                                                                     | `results/bootstrap_ci_holm.csv`                           |
| Audit regressions incl. weighted spec (Sec 5.4) | `python evaluation/tree_quality_audit.py`                                                                                               | `results/tree_quality_audit/multivariate_regression*.csv` |
| Supplement Table S1 (every condition, with n)   | `python evaluation/supp_table.py -o results/tab_supp_full.tex`                                                                           | `results/tab_supp_full.tex`                               |

One ordering constraint: run `oracle_extraction_analysis.py` before
`cost_frontier.py`, which reads
`results/tree_error_analysis/oracle_failure_breakdown_primary_condition.csv`.

Everything in `tables/`, and every CSV under `data/`, is byte-for-byte the
output of these commands on the shipped databases.

For the supplementary material, where each section's numbers come from, and
what this archive can regenerate offline:

| Supplement section | Source | Regenerates offline? |
| --- | --- | --- |
| S1 conventions, S10 reproducibility | prose | n/a |
| S2 (Table S1) | `evaluation/supp_table.py` | yes |
| S3 (Table S2) | `evaluation/bootstrap_ci.py` | yes |
| S8-S9 (Tables S8-S9) | `evaluation/tree_quality_audit.py` | yes |
| S4-S5 (Tables S3-S4, Figure S1) | `RunId` rows in `data/dbs/` | data ships; table not rebuilt |
| S6 (Table S5) | `data/global_list_probe/results.csv` | data ships; table not rebuilt |
| S7 (Tables S6-S7) | `data/constrained_choice_probe/results.csv` | data ships; table not rebuilt |

The three "not rebuilt" rows were assembled by a script that is no longer
part of the archive. Their inputs are shipped in full, so the reported
figures can still be checked against the underlying rows by hand.
`global_list_probe.py` and `constrained_choice_probe.py` remain in the tree
but are Tier 2 collection scripts — they issue paid API calls to *gather*
that data, and do not recompute the tables from it.

The prompt sweep behind S4-S5 is not uniform across datasets, and S5 states
this as a scope limit. GTSRB and CIFAR-10 carry the full ten-paraphrase sweep
(`RunId` 0-9, whole dataset each) from `run_all_experiments.py`; that is the
only place a real accuracy distribution exists. MNIST and MSCOCO were not
affordable at that scale, so they carry only small fixed-subsample checks from
`src/evaluation/variance_probe.py` (MNIST `RunId` 0, 1, 3, 10; MSCOCO `RunId`
0-3), and their spread is computed on the images those runs share rather than
on a full pass. Tree traversal was run under `RunId` 0 alone in every dataset,
so no repeated-run replicate of a tree result exists at any scale.

Tier 2 — collection, paid API calls (OpenRouter / DashScope / self-hosted
Ollama), needs `.env` and the datasets:

| Data                                                        | Command                                                                                                                 |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Main grid (all conditions, both temperatures, prompt sweep) | `python run_all_experiments.py`                                                                                       |
| One cell                                                    | e.g.`python main.py --dataset_name GTSRB --model gpt-4o --temperature 0.7`                                            |
| CoT baseline (paper: n=100/pair)                            | `python evaluation/cot_baseline.py --n_images 100 --max_calls 100 ...`                                                |
| Grounding probes                                            | `evaluation/label_conditioned_knowledge_verification.py`, `evaluation/image_grounded_knowledge_verification.py`     |
| Format probes                                               | `evaluation/global_list_probe.py`, `evaluation/constrained_choice_probe.py`                                         |
| Tree generation demo (CUB-200)                              | `python generate_tree.py --dataset_path ... --max_calls ...`, run from `tree_generation/automated_tree_generation/` |

## Datasets

All four are automated and self-contained — one script each, no manual
download or post-processing step:

```bash
python datasets/download.py          # all four
python datasets/download_GTSRB.py    # or individually
```

Each script downloads its dataset and reduces it to the fixed evaluation
subset in `sample_ids/` (901 GTSRB / 1000 MNIST / 1000 CIFAR-10 / 850
single-category MSCOCO images), writing the `class_names.json` and metadata
its loader reads. Nothing is redrawn at run time; the identifier lists are
the subsets. Datasets are needed only for Tier 2 and for re-running the CLIP
baselines. See `src/README.md` for the per-dataset on-disk layout and flags.

## Trees

| Tree         | Internal | Leaves | Max question depth |
| ------------ | -------- | ------ | ------------------ |
| GTSRB        | 22       | 43     | 14                 |
| MNIST        | 8        | 10     | 3                  |
| CIFAR-10     | 9        | 10     | 3                  |
| MSCOCO       | 89       | 90     | 10                 |
| CUB-200 demo | 18       | 20     | 5                  |

Depth is that of the deepest question node, with the root at 0. The four
evaluated trees are defined in code (`src/tree_generation/*_tree.py`, loaded
via each dataset loader's `build_tree()`); the `trees/*.txt` files are
human-readable dumps of the same structures. The CUB-200 demo is not part of
the evaluation — it is the output of the generator-evaluator pipeline, and
ships as `src/tree_generation/automated_tree_generation/cub200_demo_tree.json`
with its dump in `trees/cub200_demo_tree.txt`. Full specifications
(branching, split entropy, unreached nodes) for the four evaluated trees are
in supplement Table S8.

## Known caveats

- Model responses come from hosted APIs: even T=0 is not bit-reproducible,
  and providers may retire or update checkpoints (the OpenRouter
  LLaMA-3.2-11B-Vision endpoint used for the main runs was retired
  mid-study and replaced by a self-hosted FP16 deployment of the same
  weights). Re-running collection reproduces the sign and ordering of the
  gaps, not exact percentages.
- **GTSRB, GPT-4o, primary condition was collected twice.** The first pass
  stored a free-text answer instead of the JSON traversal path for 667 of
  its 901 images, leaving only 234 usable traversals covering 25 of 43
  classes — and that surviving subset was biased, scoring 31.6% tree
  accuracy against 66.7% for the rest. GPT-4o was therefore re-run over all
  901 images with path logging intact; the re-run reproduces the original
  tree accuracy (57.71% vs 57.60%) while yielding usable paths for 901/901
  images across all 43 classes. The shipped `gtsrb.db` takes GPT-4o's
  primary-condition rows from that re-run and every other row from the
  original pass, so each row comes from exactly one run and no row mixes
  paths with class predictions. GPT-4o's other GTSRB configurations (other
  temperatures, prompt variants, RunIds) are untouched, since the re-run
  covered the primary condition only.
- The CUB-200 demo's "36 LLM calls / 18 decision nodes" figures came from
  the generation run's console output; the shipped `cub200_demo_tree.json`
  is the resulting tree itself.
- Accuracy counts unparseable answers and unlogged traversals as wrong
  (supplement S1); scoring only the parseable subset is not comparable.
- `prompts/MNIST/zs_prompts.json` index 2 was corrected in place after an
  earlier wording hardcoded the task as binary. The shipped `mnist.db`
  carries no RunId=2 rows; the validated re-run of that wording is stored
  under RunId=10. See `src/evaluation/variance_probe.py`.

## Licensing

The decision trees, prompts, generator-evaluator pipeline, analysis code,
and evaluation harness are released under the **MIT License**
(`src/LICENSE`). This grant does not extend to the datasets, which remain
governed by their original licenses and terms (see Section 9 of the paper
and `src/README.md`); the archive ships no dataset images — only
identifier lists and derived metadata.

## Citation

See [`CITATION.cff`](../CITATION.cff) or the BibTeX entry in the
[top-level README](../README.md#citation).
