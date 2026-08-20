"""
CLIP-based baselines for comparison against zero-shot and tree-based VLM
classification: plain CLIP zero-shot, CHiLS-style hierarchical classification,
and description-based classification (Menon & Vondrick style).

These are the three baselines flagged as missing across all three prior
rejection cycles of this paper. They run entirely locally (no paid API calls)
using an open-weight CLIP checkpoint, reusing the existing dataset loaders,
decision trees, and class-description JSON files already in the repo.

Usage:
    python evaluation/baselines.py [--dataset GTSRB] [--model openai/clip-vit-base-patch32]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.data_loader import CIFARLoader, GTSRBLoader, MNISTLoader, MSCOCOLoader
from main import InferenceConfig
from models.clip_model import CLIPWrapper

OUT_DIR = Path("results/baselines")
BATCH_SIZE = 32

LOADERS = {
    "GTSRB": GTSRBLoader,
    "MNIST": MNISTLoader,
    "CIFAR-10": CIFARLoader,
    "MSCOCO": MSCOCOLoader,
}


def _load_mnist_via_torchvision(dataset_path: str):
    """
    Fallback for local checkouts where datasets/MNIST/sampled_images/*.png were
    never materialized on disk: reconstruct the exact same 1,000-image sample
    (same image IDs, same labels) from the standard MNIST training set via
    torchvision's public mirror -- verified to index-match image_to_class_mapping.json
    exactly (no Kaggle credentials required).
    """
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


def load_dataset(dataset_name: str):
    """Reuse the existing InferenceConfig + DatasetLoader path so we evaluate
    the exact same sample sets as the VLM experiments."""
    config = InferenceConfig.from_file("config.json", dataset_name, model="gpt-4o")
    loader = LOADERS[dataset_name]()
    images, labels, class_names, _ = loader.load_images(config)
    if not images and dataset_name == "MNIST":
        print("[MNIST] Local sampled_images/*.png not found; reconstructing the "
              "same sample set from torchvision's MNIST mirror instead.")
        images, labels = _load_mnist_via_torchvision(config.dataset_path)
    labels = [int(l) for l in labels]
    return loader, images, labels, class_names


def encode_images_batched(clip: CLIPWrapper, images: list, batch_size: int = BATCH_SIZE) -> torch.Tensor:
    feats = []
    for i in range(0, len(images), batch_size):
        batch = [img.convert("RGB") if hasattr(img, "convert") else img for img in images[i:i + batch_size]]
        feats.append(clip.encode_images(batch).cpu())
    return torch.cat(feats, dim=0)


def evaluate(predictions: list, labels: list) -> float:
    correct = sum(int(p == l) for p, l in zip(predictions, labels))
    return correct / len(labels)


def run_clip_zero_shot(clip, images, labels, class_names, dataset_name):
    class_ids = sorted(int(k) for k in class_names.keys())
    prompts = [f"a photo of a {class_names[str(cid)]}" for cid in class_ids]
    text_feats = clip.encode_texts(prompts).cpu()

    image_feats = encode_images_batched(clip, images)
    sims = image_feats @ text_feats.T
    pred_idx = sims.argmax(dim=1).tolist()
    predictions = [class_ids[i] for i in pred_idx]

    acc = evaluate(predictions, labels)
    print(f"[{dataset_name}] CLIP zero-shot accuracy: {acc:.4f}")
    return predictions, acc


def run_description_based(clip, images, labels, class_names, dataset_name, description_path):
    with open(description_path, "r", encoding="utf-8") as f:
        descriptions = json.load(f).get("short_description", {})

    class_ids = sorted(int(k) for k in class_names.keys())
    prompts = [descriptions.get(str(cid), f"a photo of a {class_names[str(cid)]}") for cid in class_ids]
    text_feats = clip.encode_texts(prompts).cpu()

    image_feats = encode_images_batched(clip, images)
    sims = image_feats @ text_feats.T
    pred_idx = sims.argmax(dim=1).tolist()
    predictions = [class_ids[i] for i in pred_idx]

    acc = evaluate(predictions, labels)
    print(f"[{dataset_name}] Description-based accuracy: {acc:.4f}")
    return predictions, acc


# Short, CLIP-natural noun-phrase superclass prompts for each dataset's root
# split, keyed by the root branch answer. CLIP is trained on short captions
# ("a photo of a dog"), not full sentences quoting a question and answer, so
# the original "An image where the answer to the question '...' is '...'."
# template handicapped CHiLS relative to how CLIP was actually pretrained.
# These replace it with a phrase a human would plausibly caption the image
# with. Falls back to the original sentence template for any dataset (or
# root branch key) not covered here, so run_chils stays general-purpose.
ROOT_SUPERCLASS_PROMPTS = {
    "GTSRB": {
        "triangle": "a triangle-shaped traffic sign",
        "circle": "a circle-shaped traffic sign",
        "diamond": "a diamond-shaped traffic sign",
        "inverted-triangle": "an inverted-triangle-shaped traffic sign",
        "octagon": "an octagon-shaped traffic sign",
    },
    "MNIST": {
        "0": "a handwritten digit with no enclosed loop",
        "1": "a handwritten digit with one enclosed loop",
        "2": "a handwritten digit with two enclosed loops",
    },
    "CIFAR-10": {
        "yes": "a photo of an animal",
        "no": "a photo of a vehicle",
    },
    "MSCOCO": {
        "yes": "a photo of a person or an animal",
        "no": "a photo of an inanimate object",
    },
}


def run_chils(clip, images, labels, class_names, dataset_name, tree_root):
    """
    CHiLS-style hierarchical classification, adapted to reuse this
    benchmark's own decision trees as the coarse-to-fine label hierarchy:
    the tree root's branches become CLIP-embeddable "superclasses," each
    covering the leaf classes reachable beneath it. We first pick the
    top-1 superclass, then classify within its subclasses only.
    """
    superclasses = list(tree_root.children.items())  # [(branch_key, child_node), ...]
    dataset_prompts = ROOT_SUPERCLASS_PROMPTS.get(dataset_name, {})
    superclass_prompts = [
        dataset_prompts.get(key, f"An image where the answer to the question '{tree_root.question}' is '{key}'.")
        for key, _ in superclasses
    ]
    superclass_text_feats = clip.encode_texts(superclass_prompts).cpu()

    # subclass ids covered by each superclass
    superclass_subclasses = [sorted(child.get_all_possible_classes()) for _, child in superclasses]

    # pre-encode a "a photo of a {class_name}" prompt for every class once
    all_class_ids = sorted(int(k) for k in class_names.keys())
    all_prompts = [f"a photo of a {class_names[str(cid)]}" for cid in all_class_ids]
    all_text_feats = clip.encode_texts(all_prompts).cpu()
    class_id_to_row = {cid: i for i, cid in enumerate(all_class_ids)}

    image_feats = encode_images_batched(clip, images)

    superclass_sims = image_feats @ superclass_text_feats.T
    top_superclass = superclass_sims.argmax(dim=1).tolist()

    predictions = []
    for i, sc_idx in enumerate(top_superclass):
        subclass_ids = [cid for cid in superclass_subclasses[sc_idx] if cid in class_id_to_row]
        if not subclass_ids:
            predictions.append(-1)
            continue
        rows = [class_id_to_row[cid] for cid in subclass_ids]
        sub_feats = all_text_feats[rows]
        sims = image_feats[i:i + 1] @ sub_feats.T
        best = sims.argmax(dim=1).item()
        predictions.append(subclass_ids[best])

    acc = evaluate(predictions, labels)
    print(f"[{dataset_name}] CHiLS accuracy: {acc:.4f}")
    return predictions, acc


def run_dataset(dataset_name: str, clip: CLIPWrapper):
    print(f"\n=== {dataset_name} ===")
    loader, images, labels, class_names = load_dataset(dataset_name)
    if not images:
        print(f"[{dataset_name}] No images loaded (likely missing on disk locally) -- skipping.")
        return {}
    tree_root = loader.build_tree()
    description_path = f"prompts/{dataset_name}/classes_description.json"

    results = {}
    zs_pred, zs_acc = run_clip_zero_shot(clip, images, labels, class_names, dataset_name)
    results["clip_zero_shot"] = (zs_pred, zs_acc)

    if Path(description_path).exists():
        desc_pred, desc_acc = run_description_based(clip, images, labels, class_names, dataset_name, description_path)
        results["description_based"] = (desc_pred, desc_acc)

    chils_pred, chils_acc = run_chils(clip, images, labels, class_names, dataset_name, tree_root)
    results["chils"] = (chils_pred, chils_acc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, (preds, acc) in results.items():
        for label, pred in zip(labels, preds):
            rows.append({"dataset": dataset_name, "method": method, "true_class": label,
                         "predicted_class": pred, "correct": int(label == pred)})
    pd.DataFrame(rows).to_csv(OUT_DIR / f"{dataset_name.replace('-', '').lower()}_predictions.csv", index=False)

    return {method: acc for method, (_, acc) in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(LOADERS.keys()), default=None)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    clip = CLIPWrapper(model_name=args.model)
    print(f"Loaded {args.model} on {clip.device}")

    datasets = [args.dataset] if args.dataset else list(LOADERS.keys())
    summary = {}
    for name in datasets:
        summary[name] = run_dataset(name, clip)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(OUT_DIR / "summary_accuracy.csv")
    print("\n=== Summary (accuracy) ===")
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
