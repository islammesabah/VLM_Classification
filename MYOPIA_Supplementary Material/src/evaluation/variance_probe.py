"""
Small, budget-bounded repeated-run zero-shot variance probe for MNIST/MSCOCO.

GTSRB/CIFAR-10 already have RunId 0-9 (10 zero-shot prompt variants, full
dataset each) giving a real accuracy distribution (Figure fig:prompt_sensitivity
in main.tex). MNIST/MSCOCO only have RunId=0 -- no variance/CI is possible for
them yet. Running the same full-dataset x 10-run design for MNIST/MSCOCO is not
affordable (remaining OpenRouter budget: ~$2.86), so this script reuses the
exact same zero-shot methodology (VLMInference.zero_shot_inference, from
main.py) but restricts each new RunId to a small, fixed subsample instead of
the full dataset.

LLaMA-3.2-11B's OpenRouter zero-shot endpoint is gone (see main.tex
Limitations), so it is intentionally excluded here: mixing self-hosted results
into the same RunId series as the OpenRouter-based RunId=0 rows would conflate
prompt-wording variance with inference-backend variance.

Usage:
    python evaluation/variance_probe.py --dataset MNIST --model gpt-4o \
        --run_ids 1 2 3 --n_images 20 --max_calls 50

Note: prompts/MNIST/zs_prompts.json index 2 (RunId=2) originally hardcoded
the task as binary ('0' or '1' only) instead of the actual ten digit
classes, causing spurious refusals on other digits; it has since been
corrected in place (see git history for the original text). DB rows tagged
RunId=2 from before the fix used the old broken prompt and should not be
reused for analysis. The corrected wording was validated under a temporary
RunId=10 before being folded back into index 2 -- those RunId=10 rows are
still valid (same prompt text) and remain in the database, they're just no
longer needed for future runs now that RunId=2 itself is fixed.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from main import InferenceConfig, VLMInference

load_dotenv()


def select_subsample(labels, n_images, seed=123):
    rng = random.Random(seed)
    indices = list(range(len(labels)))
    rng.shuffle(indices)
    # stratify lightly: walk the shuffled order, cap 2 per class, until n_images reached
    by_class_count = {}
    selected = []
    for idx in indices:
        cls = labels[idx]
        if by_class_count.get(cls, 0) >= 2:
            continue
        selected.append(idx)
        by_class_count[cls] = by_class_count.get(cls, 0) + 1
        if len(selected) >= n_images:
            break
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["MNIST", "MSCOCO"])
    parser.add_argument("--model", required=True, choices=["gpt-4o", "qwen-vl-max"])
    parser.add_argument("--run_ids", type=int, nargs="+", required=True)
    parser.add_argument("--n_images", type=int, default=20)
    parser.add_argument("--max_calls", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    config = InferenceConfig.from_file("config.json", args.dataset, args.model, args.temperature)
    engine = VLMInference(config)

    images, labels, class_names, additional_info = engine.dataset_loader.load_images(config)
    subsample_idx = select_subsample(labels, args.n_images)
    print(f"Selected {len(subsample_idx)} images (seed=123, class-capped) "
          f"out of {len(images)} total.", flush=True)

    n_calls = 0
    for run_id in args.run_ids:
        n_correct = 0
        n_done = 0
        for image_idx in subsample_idx:
            image_path, sequence = engine.dataset_loader.extract_image_info(image_idx, additional_info)
            if engine.db_manager.image_exists(image_path, config.model, False, 0, False,
                                                config.temperature, run_id):
                print(f"[RunId {run_id}] {image_path} already done, skipping.", flush=True)
                continue
            if n_calls >= args.max_calls:
                print(f"max_calls={args.max_calls} reached; stopping.", flush=True)
                engine.cleanup()
                return
            base64_image = engine._encode_image(images[image_idx])
            result = engine.zero_shot_inference(base64_image, class_names, config.to_dict(),
                                                 0, "", run_id)
            engine.db_manager.insert_result(
                class_label=labels[image_idx], sequence=sequence, image_path=image_path,
                model_name=config.model, tree_result=None, zero_shot_result=result.response,
                tree_class=-1, zero_shot_class=result.predicted_class,
                include_memory=False, include_description=0, include_zero_shot_label=False,
                temperature=config.temperature, RunId=run_id,
            )
            n_calls += 1
            n_done += 1
            correct = str(result.predicted_class) == str(labels[image_idx])
            n_correct += int(correct)
            print(f"[RunId {run_id}] [{n_done}/{len(subsample_idx)}] {image_path} "
                  f"pred={result.predicted_class} true={labels[image_idx]} correct={correct}",
                  flush=True)
        print(f"RunId {run_id} done: {n_correct}/{n_done} new calls this run "
              f"(cumulative calls used: {n_calls}/{args.max_calls})", flush=True)

    engine.cleanup()


if __name__ == "__main__":
    main()
