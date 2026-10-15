#!/bin/bash
# Build the anonymous WACV reviewer supplement package.
# Usage (from the repository root): bash scripts/build_supplement.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/MYOPIA_Supplementary Material"
cd "$ROOT"

echo "== staging =="
rm -rf "$PKG/trees" "$PKG/prompts" "$PKG/tables" "$PKG/code" "$PKG/sample_ids" \
       "$PKG/data" "$PKG/supplementary_tex"
mkdir -p "$PKG/trees" "$PKG/prompts" "$PKG/tables" "$PKG/code" "$PKG/sample_ids" \
         "$PKG/data/dbs" "$PKG/supplementary_tex"

# --- supplement document: rendered PDF + renderable tex source ---
cp results/paper/supplementary.pdf "$PKG/MYOPIA_supplementary.pdf"
cp results/paper/supplementary.tex results/paper/zs_prompt_sweep.pdf "$PKG/supplementary_tex/"
printf 'Render with: pdflatex supplementary.tex  (run twice for references)\n' \
  > "$PKG/supplementary_tex/README.txt"

# --- full trees ---
cp tree_generation/gtsrb-tree.txt tree_generation/mnist-tree.txt \
   tree_generation/cifar10-tree.txt tree_generation/mscoco-tree.txt "$PKG/trees/"
cp tree_generation/automated_tree_generation/cub200_demo_tree.json \
   tree_generation/automated_tree_generation/cub200_demo_tree.txt "$PKG/trees/"

# --- complete prompts ---
for d in CIFAR-10 GTSRB MNIST MSCOCO; do
  mkdir -p "$PKG/prompts/$d"
  cp "prompts/$d/classes_description.json" "prompts/$d/zs_prompts.json" "$PKG/prompts/$d/"
done
cp prompts/describe_image.txt prompts/generate_prompts_zeroshot.py "$PKG/prompts/"
cp tree_generation/automated_tree_generation/prompts_generator.json \
   tree_generation/automated_tree_generation/prompts_evaluator.json "$PKG/prompts/"

# --- fixed image subsets ---
cp datasets/GTSRB/901_samples.txt "$PKG/sample_ids/gtsrb_901_samples.txt"
cp datasets/MNIST/sampled_image_ids.txt "$PKG/sample_ids/mnist_1000_samples.txt"
cp datasets/CIFAR-10/1000_samples.txt "$PKG/sample_ids/cifar10_1000_samples.txt"
cp datasets/MSCOCO/850_filtered_1_category_image_ids.txt "$PKG/sample_ids/mscoco_850_samples.txt"

# --- table sources (LaTeX + machine-readable) ---
cp results/tab_zs_vs_tree_extended.tex "$PKG/tables/"
cp results/paper/tab_supp_full.tex "$PKG/tables/" 2>/dev/null || true
cp results/agreement/tab_agreement.tex results/grounding/tab_grounding.tex \
   results/grounding/tab_gtsrb_depth.tex results/inference_cost/tab_inference_cost.tex "$PKG/tables/"

# --- code: harness, pipeline, analysis (cleaned working tree) ---
RSYNC_EX=(--exclude=__pycache__ --exclude='*.pyc' --exclude=.DS_Store
          --exclude='*.db' --exclude='.ipynb_checkpoints')
rsync -a "${RSYNC_EX[@]}" tree_generation/ "$PKG/code/tree_generation/"
rsync -a "${RSYNC_EX[@]}" evaluation/ "$PKG/code/evaluation/"
rsync -a "${RSYNC_EX[@]}" models/ "$PKG/code/models/"
mkdir -p "$PKG/code/datasets/MNIST" "$PKG/code/datasets/MSCOCO" "$PKG/code/results"
cp datasets/__init__.py datasets/data_loader.py datasets/download_CIFAR.py \
   datasets/download_GTSRB.py "$PKG/code/datasets/"
cp datasets/MNIST/extract_transform_mnist.py "$PKG/code/datasets/MNIST/"
cp datasets/MSCOCO/extract_transform_mscoco.py "$PKG/code/datasets/MSCOCO/"
cp results/initiate_db.py results/build_gtsrb_merged.py "$PKG/code/results/"
cp config.json requirements.txt main.py run_all_experiments.py LICENSE README.md \
   .env.example "$PKG/code/"

# --- data: everything needed to recompute every number offline ---
for db in gtsrb gtsrb_gpt4o_paths mnist cifar10 mscoco; do
  gzip -9 -c "results/$db.db" > "$PKG/data/dbs/$db.db.gz"
done
for d in cot_baseline baselines label_conditioned_verification \
         image_grounded_verification tree_quality_audit global_list_probe \
         constrained_choice_probe grounding agreement inference_cost; do
  rsync -a --exclude=.DS_Store "results/$d/" "$PKG/data/$d/"
done
cp results/bootstrap_ci_holm.csv "$PKG/data/"
mkdir -p "$PKG/data/tree_error_analysis"
cp results/tree_error_analysis/oracle_failure_breakdown_*.csv \
   results/tree_error_analysis/per_depth_accuracy_*.csv \
   results/tree_error_analysis/per_question_accuracy_*.csv \
   "$PKG/data/tree_error_analysis/" 2>/dev/null || true

find "$PKG" \( -name '__pycache__' -o -name '.DS_Store' -o -name '*.pyc' \) \
  -exec rm -rf {} + 2>/dev/null || true

echo "== manifest =="
python3 - "$PKG" <<'PYEOF'
import hashlib, json, os, sys
pkg = sys.argv[1]
files = {}
for dirpath, dirnames, filenames in os.walk(pkg):
    dirnames[:] = sorted(dirnames)
    for name in sorted(filenames):
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, pkg)
        if rel == 'SUPPLEMENT_MANIFEST.json':
            continue
        with open(path, 'rb') as fh:
            files[rel] = {'sha256': hashlib.sha256(fh.read()).hexdigest(),
                          'bytes': os.path.getsize(path)}
manifest_path = os.path.join(pkg, 'SUPPLEMENT_MANIFEST.json')
with open(manifest_path) as fh:
    manifest = json.load(fh)
manifest['files'] = files
manifest['n_files'] = len(files)
with open(manifest_path, 'w') as fh:
    json.dump(manifest, fh, indent=2)
print(f"{len(files)} files hashed into SUPPLEMENT_MANIFEST.json")
PYEOF

echo "== anonymity scan =="
if grep -rniE 'dfki|mesabah|islam|kaiserslautern|saarbr|github\.com/[a-z]|gitlab\.|@gmail|@dfki|/Users/[a-z]|/home/[a-z]|corresponding author' \
     "$PKG" --include='*.py' --include='*.md' --include='*.json' --include='*.txt' \
     --include='*.tex' --include='*.sh' --include='*.csv' \
   | grep -viE 'islamabad'; then
  echo "ANONYMITY SCAN FAILED (hits above)"; exit 1
fi
echo "clean"

echo "== size check =="
du -sh "$PKG"
SIZE=$(du -sm "$PKG" | cut -f1)
if [ "$SIZE" -gt 95 ]; then echo "PACKAGE TOO LARGE (${SIZE}MB > 95MB)"; exit 1; fi
echo "BUILD OK (${SIZE}MB)"
