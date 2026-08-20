"""
Image-conditioned knowledge-verification probe.

The paper's original knowledge-verification study (Section "Decision Tree
Knowledge Verification") gives the model the ground-truth class *label* and
asks it to answer each tree question along the path to that class, testing
class-attribute knowledge recall, not visual grounding. This script runs the
same protocol -- same tree, same path-per-class, same per-node accuracy
metric -- but shows the model the actual image instead of the label, so the
resulting accuracy is directly comparable and isolates the visual-grounding
component the original study could not measure.

Makes real, paid VLM API calls. Tracks exact spend via OpenRouter's
per-response `usage.cost` field and hard-stops at --max_cost.

Usage:
    python evaluation/image_grounded_knowledge_verification.py --dataset GTSRB \
        --samples_per_class 5 --max_cost 2.0
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from datasets.data_loader import CIFARLoader, GTSRBLoader, MNISTLoader, MSCOCOLoader
from main import InferenceConfig, VLMInference

with open("models/openrouter_providers.json", "r") as f:
    OPENROUTER_PROVIDERS = json.load(f)

OUT_DIR = Path("results/image_grounded_verification")

LOADERS = {
    "GTSRB": GTSRBLoader,
    "MNIST": MNISTLoader,
    "CIFAR-10": CIFARLoader,
    "MSCOCO": MSCOCOLoader,
}


def path_to_class(tree_root, target_class: int):
    """Walk the tree, at each node picking the child whose subtree contains
    target_class, returning [(question, expected_answer), ...] to the leaf."""
    path = []
    node = tree_root
    while node.question:
        chosen = None
        for answer, child in node.children.items():
            if target_class in child.get_all_possible_classes():
                chosen = (answer, child)
                break
        if chosen is None:
            break  # shouldn't happen for a valid target_class
        answer, child = chosen
        path.append((node.question, answer))
        node = child
    return path


def extract_answer(question: str, answers: list, response_text: str):
    if not response_text:
        return None
    sorted_answers = sorted(answers, key=len, reverse=True)
    pattern = "(?i)(" + "|".join(f"\\b{re.escape(a)}\\b" for a in sorted_answers) + ")"
    m = re.search(pattern.lower(), response_text.lower())
    if not m:
        return None
    return next((a for a in sorted_answers if a.lower() == m.group(1).lower()), None)


def sample_images_per_class(images, labels, samples_per_class: int, seed: int = 42):
    import random
    rng = random.Random(seed)
    by_class = {}
    for img, label in zip(images, labels):
        by_class.setdefault(int(label), []).append(img)
    sampled = {}
    for cls, imgs in by_class.items():
        k = min(samples_per_class, len(imgs))
        sampled[cls] = rng.sample(imgs, k)
    return sampled


def _load_mnist_via_torchvision(dataset_path: str):
    """Same fallback as evaluation/baselines.py: reconstructs the exact same
    1,000-image sample from torchvision's public MNIST mirror when the local
    sampled_images/*.png files were never materialized on disk."""
    from torchvision.datasets import MNIST as TVMNIST

    with open(f"{dataset_path}/sampled_image_ids.txt") as f:
        image_ids = [line.strip() for line in f if line.strip()]
    with open(f"{dataset_path}/image_to_class_mapping.json") as f:
        mapping = json.load(f)

    tv_ds = TVMNIST(root="/tmp/mnist_cache", download=True, train=True)
    images, labels = [], []
    for image_id in image_ids:
        img, _ = tv_ds[int(image_id)]
        images.append(img)
        labels.append(mapping[image_id])
    return images, labels


def run(dataset_name: str, samples_per_class: int, max_cost: float, model: str = "gpt-4o", max_calls: int = None, temperature: float = 0.7):
    """
    max_cost relies on the API response reporting usage.cost (OpenRouter does;
    DashScope/Alibaba does not, so it silently never trips for qwen-vl-max).
    max_calls is a provider-independent backstop: pass it explicitly whenever
    the provider's cost reporting is unverified, since max_cost alone is not a
    safety net in that case.
    """
    config = InferenceConfig.from_file("config.json", dataset_name, model=model)
    loader = LOADERS[dataset_name]()
    images, labels, class_names, _ = loader.load_images(config)
    if not images and dataset_name == "MNIST":
        print("[MNIST] Local sampled_images/*.png not found; reconstructing the "
              "same sample set from torchvision's MNIST mirror instead.")
        images, labels = _load_mnist_via_torchvision(config.dataset_path)
    labels = [int(l) for l in labels]
    tree_root = loader.build_tree()

    sampled = sample_images_per_class(images, labels, samples_per_class)

    inference = VLMInference(config)
    total_cost = 0.0
    n_calls = 0
    records = []
    stopped_early = False

    def limit_reached():
        return total_cost >= max_cost or (max_calls is not None and n_calls >= max_calls)

    for target_class, imgs in sampled.items():
        node_path = path_to_class(tree_root, target_class)
        if not node_path:
            continue
        for img in imgs:
            if limit_reached():
                stopped_early = True
                break
            base64_image = inference._encode_image(img)
            for depth, (question, expected_answer) in enumerate(node_path):
                if limit_reached():
                    stopped_early = True
                    break
                node = tree_root
                for q, a in node_path[:depth]:
                    node = node.children[a]
                prompt = f"{question} Choose one of these answers: {node.answers}."
                payload = {
                    "model": inference.model_client.model_name,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }],
                    "max_tokens": 30,
                    "temperature": temperature,
                }
                if inference.model_client.model_name == "meta-llama/llama-3.2-11b-vision-instruct":
                    # Provider pinning is a workaround curated specifically for this
                    # model's flaky OpenRouter providers; do not apply it to other
                    # meta-llama/ models, whose available providers differ.
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
                records.append({
                    "dataset": dataset_name, "model": model, "true_class": target_class,
                    "depth": depth, "question": question, "expected_answer": expected_answer,
                    "predicted_answer": predicted_answer, "correct": correct,
                    "call_cost": call_cost,
                })
                time.sleep(0.5)
        if stopped_early:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    model_slug = model.replace("/", "_").replace("-", "_")
    out_path = OUT_DIR / f"{dataset_name.replace('-', '').lower()}_{model_slug}_image_grounded.csv"
    df.to_csv(out_path, index=False)

    n = len(df)
    acc = df["correct"].mean() if n else float("nan")
    print(f"\n[{dataset_name}] {n} node-level decisions, {len(sampled)} classes, "
          f"accuracy={acc:.4f}, total spend=${total_cost:.4f} ({n_calls} calls)"
          f"{' (STOPPED EARLY: limit reached)' if stopped_early else ''}")
    return df, total_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(LOADERS.keys()), default="GTSRB")
    parser.add_argument("--samples_per_class", type=int, default=5)
    parser.add_argument("--max_cost", type=float, default=2.0)
    parser.add_argument("--max_calls", type=int, default=None,
                         help="Provider-independent call cap; required for providers "
                              "(e.g. DashScope) that don't report usage.cost")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    run(args.dataset, args.samples_per_class, args.max_cost, args.model, args.max_calls, args.temperature)


if __name__ == "__main__":
    main()
