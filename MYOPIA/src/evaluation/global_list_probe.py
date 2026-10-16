"""
Global-class-list tree-traversal variant (WACV E&D self-review, W4: "is the
ZS vs. Tree comparison fair?").

Zero-shot classification always shows the model the full label set at
inference time ("Classify the object. It should be only one of these
classes: {class_names_str}."). Tree-based classification instead shows only
the current node's local answer options, and the child subtree is picked by
that local choice alone -- the model never sees the global label list once
traversal starts. That is an information asymmetry, not just a difference in
reasoning strategy: a wrong node-level answer might partly reflect not
knowing which classes are still in play, not misreading the image.

This script re-runs tree traversal with the full global class list added to
every node's prompt (still a single greedy root-to-leaf path, no
backtracking) and compares it against the existing local-list traversal on
the same images, to see how much of the ZS-vs-Tree gap that asymmetry
accounts for.

Standalone (does not touch main.py or the shared results/{dataset}.db):
the existing DB schema has no tree-variant column, so writing this
condition's rows into it would collide with the existing primary key. Follows
the same standalone-walker, crash-resilient pattern as
evaluation/constrained_choice_probe.py, and reuses its subsample selection,
free-form answer extraction, and API-call helper directly.

Usage:
    python evaluation/global_list_probe.py --dataset GTSRB \
        --model gpt-4o --n_images 50 --max_calls 400
"""
import argparse
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompt_library
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from datasets.data_loader import CIFARLoader, GTSRBLoader, MNISTLoader, MSCOCOLoader
from main import InferenceConfig, VLMInference
from constrained_choice_probe import extract_free_form, select_subsample, _call

LOADERS = {
    "GTSRB": GTSRBLoader, "MNIST": MNISTLoader,
    "CIFAR-10": CIFARLoader, "MSCOCO": MSCOCOLoader,
}
OUT_DIR = Path("results/global_list_probe")
FIELDNAMES = ["dataset", "model", "condition", "image_index", "true_class",
              "depth", "question", "expected_answer", "predicted_answer",
              "correct", "resolved"]


def already_done():
    done = set()
    out_path = OUT_DIR / "results.csv"
    if out_path.exists():
        with open(out_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["dataset"], row["model"], row["condition"], row["image_index"]))
    return done


def append_rows(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "results.csv"
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def walk_local_list(inference, tree_root, base64_image, target_class):
    """Existing tree-traversal prompt format, unchanged (main.py's exact wording)."""
    rows, node = [], tree_root
    while node.question:
        prompt = prompt_library.render("tree_node", question=node.question, answers=node.answers)
        response = _call(inference, prompt, base64_image)
        answer = extract_free_form(node.answers, response)
        resolved = answer is not None and answer in node.children
        correct = resolved and target_class in node.children[answer].get_all_possible_classes()
        expected_answer = next((a for a in node.answers
                                 if target_class in node.children[a].get_all_possible_classes()), None)
        rows.append({"depth": len(rows), "question": node.question, "expected_answer": expected_answer,
                     "predicted_answer": answer, "resolved": resolved, "correct": correct})
        if not resolved:
            break
        node = node.children[answer]
        if not correct:
            break
    return rows


def walk_global_list(inference, tree_root, base64_image, target_class, class_names_str):
    """Same node question/answers, but prefaced with the full class list every step."""
    rows, node = [], tree_root
    while node.question:
        prompt = prompt_library.render("tree_node_global_list", class_names_str=class_names_str,
                                       question=node.question, answers=node.answers)
        response = _call(inference, prompt, base64_image)
        answer = extract_free_form(node.answers, response)
        resolved = answer is not None and answer in node.children
        correct = resolved and target_class in node.children[answer].get_all_possible_classes()
        expected_answer = next((a for a in node.answers
                                 if target_class in node.children[a].get_all_possible_classes()), None)
        rows.append({"depth": len(rows), "question": node.question, "expected_answer": expected_answer,
                     "predicted_answer": answer, "resolved": resolved, "correct": correct})
        if not resolved:
            break
        node = node.children[answer]
        if not correct:
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(LOADERS.keys()), required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--n_images", type=int, default=50)
    parser.add_argument("--max_calls", type=int, default=400)
    args = parser.parse_args()

    config = InferenceConfig.from_file("config.json", args.dataset, args.model)
    loader = LOADERS[args.dataset]()
    images, labels, class_names, additional_info = loader.load_images(config)
    labels = [int(l) for l in labels]
    tree_root = loader.build_tree()

    class_ids = sorted(int(k) for k in class_names.keys())
    class_names_str = ", ".join(f"{i}: {class_names[str(i)]}" for i in class_ids)

    subsample_idx = select_subsample(labels, args.n_images)
    inference = VLMInference(config)
    done = already_done()
    n_calls = 0

    conditions = [
        ("local_list", lambda img, tc: walk_local_list(inference, tree_root, img, tc)),
        ("global_list", lambda img, tc: walk_global_list(inference, tree_root, img, tc, class_names_str)),
    ]

    for condition, walk_fn in conditions:
        for image_idx in subsample_idx:
            key = (args.dataset, args.model, condition, str(image_idx))
            if key in done:
                continue
            if n_calls >= args.max_calls:
                print(f"max_calls={args.max_calls} reached; stopping.", flush=True)
                return
            target_class = labels[image_idx]
            base64_image = inference._encode_image(images[image_idx])
            step_rows = walk_fn(base64_image, target_class)
            n_calls += len(step_rows)
            rows = [{
                "dataset": args.dataset, "model": args.model, "condition": condition,
                "image_index": str(image_idx), "true_class": target_class, **r,
            } for r in step_rows]
            append_rows(rows)
            final_correct = step_rows[-1]["correct"] if step_rows else False
            print(f"[{condition}] image={image_idx} nodes={len(step_rows)} "
                  f"final_correct={final_correct} (calls used: {n_calls}/{args.max_calls})", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
