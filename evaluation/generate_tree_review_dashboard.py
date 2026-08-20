"""
Regenerates the Tree Review Bench dashboard (results/tree_review_dashboard.html)
used for second-annotator verification of the four decision trees (GTSRB,
MNIST, CIFAR-10, MSCOCO).

Walks each dataset's tree (via DatasetLoader.build_tree(), the exact same
tree used at inference time) into a plain JSON structure, pulls up to two
real sample images per class as small embedded thumbnails, and injects both
into evaluation/templates/tree_review_dashboard_template.html to produce a
single self-contained, offline-viewable HTML file.

Re-run this whenever a tree definition or a dataset's sample images change,
so the review dashboard stays in sync with what's actually used at inference
time.

Usage:
    python evaluation/generate_tree_review_dashboard.py
"""
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.data_loader import GTSRBLoader, MNISTLoader, CIFARLoader, MSCOCOLoader
from main import InferenceConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "evaluation" / "templates" / "tree_review_dashboard_template.html"
OUTPUT_PATH = REPO_ROOT / "results" / "tree_review_dashboard.html"

DATASETS = {
    "GTSRB": GTSRBLoader,
    "MNIST": MNISTLoader,
    "CIFAR-10": CIFARLoader,
    "MSCOCO": MSCOCOLoader,
}


def node_to_dict(node):
    if not node.question:
        return {"leaf": True, "classes": list(node.possible_classes)}
    return {
        "leaf": False,
        "question": node.question,
        "answers": list(node.answers),
        "children": {ans: node_to_dict(child) for ans, child in node.children.items()},
    }


def make_thumb(img, size=110, quality=70):
    pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.thumbnail((size, size))
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def extract_dataset(name, loader_cls):
    loader = loader_cls()
    tree_json = node_to_dict(loader.build_tree())

    config = InferenceConfig.from_file("config.json", name, model="gpt-4o")
    images, labels, class_names, _ = loader.load_images(config)
    labels = [int(l) for l in labels]

    by_class = {}
    for img, lbl in zip(images, labels):
        by_class.setdefault(lbl, []).append(img)

    thumbs = {str(cls_id): [make_thumb(im) for im in imgs[:2]] for cls_id, imgs in by_class.items()}
    return {"tree": tree_json, "class_names": class_names, "thumbs": thumbs}


def main():
    data = {name: extract_dataset(name, loader_cls) for name, loader_cls in DATASETS.items()}
    data_json = json.dumps(data)

    template = TEMPLATE_PATH.read_text()
    html = template.replace("__TREE_DATA_JSON__", data_json)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html)
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
