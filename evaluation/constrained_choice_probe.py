"""
Constrained-choice vs. free-form tree-traversal mini-experiment
(Plan Cycle 2, Phase B2).

Complements the post-hoc oracle re-parse in evaluation/oracle_extraction_analysis.py
(which asks "how much of the existing gap is a parsing artifact, retrospectively")
with a prospective test: on the same small image subsample, run tree traversal
twice per image -- once with the existing free-form prompt + regex extraction
(main.py's exact format), once with an explicit numbered-choice format that
asks for only the option number -- and compare accuracy and unresolved-rate.

Crash-resilient like evaluation/variance_probe.py: skips (image, condition)
pairs already in the output CSV on restart.

Usage:
    python evaluation/constrained_choice_probe.py --dataset GTSRB \
        --model gpt-4o --n_images 20 --max_calls 200
"""
import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from datasets.data_loader import CIFARLoader, GTSRBLoader, MNISTLoader, MSCOCOLoader
from main import InferenceConfig, VLMInference

LOADERS = {
    "GTSRB": GTSRBLoader, "MNIST": MNISTLoader,
    "CIFAR-10": CIFARLoader, "MSCOCO": MSCOCOLoader,
}
OUT_DIR = Path("results/constrained_choice_probe")
FIELDNAMES = ["dataset", "model", "condition", "image_index", "true_class",
              "depth", "question", "expected_answer", "predicted_answer",
              "correct", "resolved"]


def select_subsample(labels, n_images, seed=123):
    rng = random.Random(seed)
    indices = list(range(len(labels)))
    rng.shuffle(indices)
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


def extract_free_form(answers, response_text):
    if not response_text:
        return None
    sorted_answers = sorted(answers, key=len, reverse=True)
    pattern = "(?i)(" + "|".join(f"\\b{re.escape(a)}\\b" for a in sorted_answers) + ")"
    m = re.search(pattern.lower(), response_text.lower())
    if not m:
        return None
    return next((a for a in sorted_answers if a.lower() == m.group(1).lower()), None)


def extract_numbered(answers, response_text):
    """Expects a bare integer (1-indexed) selecting one of `answers`."""
    if not response_text:
        return None
    m = re.search(r"\b(\d+)\b", response_text)
    if not m:
        return None
    idx = int(m.group(1)) - 1
    if 0 <= idx < len(answers):
        return answers[idx]
    return None


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


def walk_free_form(inference, tree_root, base64_image, target_class):
    rows, node = [], tree_root
    while node.question:
        prompt = f"{node.question} Choose one of these answers: {node.answers}."
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


def walk_constrained(inference, tree_root, base64_image, target_class):
    rows, node = [], tree_root
    while node.question:
        options = "\n".join(f"{i+1}. {a}" for i, a in enumerate(node.answers))
        prompt = (f"{node.question}\nOptions:\n{options}\n"
                  f"Respond with ONLY the option number, nothing else.")
        response = _call(inference, prompt, base64_image)
        answer = extract_numbered(node.answers, response)
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


def _call(inference, prompt_text, base64_image):
    payload = {
        "model": inference.model_client.model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }],
        "max_tokens": 30,
        "temperature": 0.7,
    }
    completion = inference.model_client.client.chat.completions.create(**payload)
    time.sleep(0.5)
    return completion.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(LOADERS.keys()), required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--n_images", type=int, default=20)
    parser.add_argument("--max_calls", type=int, default=200)
    args = parser.parse_args()

    config = InferenceConfig.from_file("config.json", args.dataset, args.model)
    loader = LOADERS[args.dataset]()
    images, labels, class_names, additional_info = loader.load_images(config)
    labels = [int(l) for l in labels]
    tree_root = loader.build_tree()

    subsample_idx = select_subsample(labels, args.n_images)
    inference = VLMInference(config)
    done = already_done()
    n_calls = 0

    for condition, walk_fn in [("free_form", walk_free_form), ("constrained", walk_constrained)]:
        for image_idx in subsample_idx:
            key = (args.dataset, args.model, condition, str(image_idx))
            if key in done:
                continue
            if n_calls >= args.max_calls:
                print(f"max_calls={args.max_calls} reached; stopping.", flush=True)
                return
            target_class = labels[image_idx]
            base64_image = inference._encode_image(images[image_idx])
            step_rows = walk_fn(inference, tree_root, base64_image, target_class)
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
