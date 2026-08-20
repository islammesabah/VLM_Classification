"""
Label-conditioned knowledge-verification probe (Plan Cycle 2, Phase B1).

Extends the paper's original GTSRB-only label-conditioned protocol (Section
"Decision Tree Knowledge Verification") to MNIST, CIFAR-10, and MSCOCO. The
original GTSRB run (once reported as 98.20% accuracy) was never saved as a
script (confirmed via repo/history search); this reimplements it from the
paper's own description and, run against GTSRB itself with `--dataset
GTSRB`, reproduces 79.59% macro-averaged accuracy
(results/label_conditioned_verification/gtsrb_gpt_4o_label_conditioned.csv)
-- the paper has been corrected to this figure. The model is given the
ground-truth class *name* (no image), asked to answer
each tree question along that class's root-to-leaf path in turn, and retains
the full question-answer history across the walk (conversation mode) --
matching main.py's `include_memory=True` tree_inference branch, with the
image swapped for a text statement of the class name. One walk per class (no
image, so no per-image sampling); accuracy is per-class (correct nodes /
total nodes on that class's path), then macro-averaged across classes to
match the paper's "per-class accuracy... then averaged" description.

Usage:
    python evaluation/label_conditioned_knowledge_verification.py \
        --dataset MSCOCO --max_cost 1.0 --max_calls 800
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from datasets.data_loader import CIFARLoader, GTSRBLoader, MNISTLoader, MSCOCOLoader
from main import InferenceConfig, VLMInference
from evaluation.image_grounded_knowledge_verification import path_to_class, extract_answer

with open("models/openrouter_providers.json", "r") as f:
    OPENROUTER_PROVIDERS = json.load(f)

OUT_DIR = Path("results/label_conditioned_verification")

LOADERS = {
    "GTSRB": GTSRBLoader,
    "MNIST": MNISTLoader,
    "CIFAR-10": CIFARLoader,
    "MSCOCO": MSCOCOLoader,
}


def run(dataset_name: str, max_cost: float, model: str = "gpt-4o", max_calls: int = None):
    config = InferenceConfig.from_file("config.json", dataset_name, model=model)
    loader = LOADERS[dataset_name]()
    _, _, class_names, _ = loader.load_images(config)
    tree_root = loader.build_tree()
    all_classes = sorted(tree_root.get_all_possible_classes())

    inference = VLMInference(config)
    total_cost = 0.0
    n_calls = 0
    records = []
    stopped_early = False

    def limit_reached():
        return total_cost >= max_cost or (max_calls is not None and n_calls >= max_calls)

    for target_class in all_classes:
        if limit_reached():
            stopped_early = True
            break
        node_path = path_to_class(tree_root, target_class)
        if not node_path:
            continue
        class_name = class_names.get(str(target_class), str(target_class))

        conversation_history = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        f"You will be asked a series of questions to classify an object "
                        f"belonging to the class: '{class_name}'. Answer each question based "
                        f"on this class's typical characteristics."
                    ),
                }],
            }
        ]

        node = tree_root
        for depth, (question, expected_answer) in enumerate(node_path):
            if limit_reached():
                stopped_early = True
                break
            prompt = f"{question} Choose one of these answers: {node.answers}."
            conversation_history.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

            payload = {
                "model": inference.model_client.model_name,
                "messages": conversation_history,
                "max_tokens": 30,
                "temperature": 0.7,
            }
            if inference.model_client.model_name == "meta-llama/llama-3.2-11b-vision-instruct":
                payload["extra_body"] = {
                    "provider": {"order": OPENROUTER_PROVIDERS},
                    "allow_fallbacks": False,
                }
            completion = inference.model_client.client.chat.completions.create(**payload)
            n_calls += 1
            call_cost = getattr(completion.usage, "cost", None) or 0.0
            total_cost += call_cost
            response_text = completion.choices[0].message.content
            predicted_answer = extract_answer(question, node.answers, response_text)
            correct = predicted_answer == expected_answer

            conversation_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_text or ""}],
            })
            records.append({
                "dataset": dataset_name, "model": model, "true_class": target_class,
                "class_name": class_name, "depth": depth, "question": question,
                "expected_answer": expected_answer, "predicted_answer": predicted_answer,
                "correct": correct, "call_cost": call_cost,
            })
            node = node.children[expected_answer]
            time.sleep(0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    model_slug = model.replace("/", "_").replace("-", "_")
    out_path = OUT_DIR / f"{dataset_name.replace('-', '').lower()}_{model_slug}_label_conditioned.csv"
    df.to_csv(out_path, index=False)

    if len(df):
        per_class_acc = df.groupby("true_class")["correct"].mean()
        macro_acc = per_class_acc.mean()
        n_full_100 = (per_class_acc == 1.0).sum()
    else:
        macro_acc, n_full_100 = float("nan"), 0
    print(f"\n[{dataset_name}] {len(df)} node-level decisions, {df['true_class'].nunique() if len(df) else 0} classes, "
          f"macro per-class accuracy={macro_acc:.4f} ({n_full_100} classes at 100%), "
          f"total spend=${total_cost:.4f} ({n_calls} calls)"
          f"{' (STOPPED EARLY: limit reached)' if stopped_early else ''}")
    return df, total_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(LOADERS.keys()), required=True)
    parser.add_argument("--max_cost", type=float, default=1.0)
    parser.add_argument("--max_calls", type=int, default=None)
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()
    run(args.dataset, args.max_cost, args.model, args.max_calls)


if __name__ == "__main__":
    main()
