"""Download MNIST and reduce it to the paper's evaluation subset.

    python datasets/download_MNIST.py

Fetches the four canonical IDX files, then writes exactly the images named in
sample_ids/mnist_1000_samples.txt -- along with the class_names.json,
image_to_class_mapping.json and sampled_images_metadata.json that
MNISTLoader reads. Nothing outside the subset is written to disk.

The IDX files come from the public S3 mirror of the originals, so no Kaggle
account is needed; --url_base overrides the host.
"""
import argparse
import gzip
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _download_utils import (dataset_dir, download, install_class_names,
                             install_sample_ids)

DEFAULT_URL_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
IDX_FILES = {
    "train-images-idx3-ubyte.gz": "train-images.idx3-ubyte",
    "train-labels-idx1-ubyte.gz": "train-labels.idx1-ubyte",
    "t10k-images-idx3-ubyte.gz": "t10k-images.idx3-ubyte",
    "t10k-labels-idx1-ubyte.gz": "t10k-labels.idx1-ubyte",
}
IMAGES_DIR = "sampled_images"   # config.json images_dir
PREFIX = "mnist"                # file_name = {PREFIX}_{image_id}.png


def gunzip(src: Path, dest: Path) -> None:
    with gzip.open(src, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    src.unlink()


def read_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path.name}: expected image magic 2051, got {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path.name}: expected label magic 2049, got {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and filter the MNIST dataset.")
    parser.add_argument("--url_base", default=DEFAULT_URL_BASE,
                        help=f"Base URL for the IDX files (default: {DEFAULT_URL_BASE})")
    parser.add_argument("--force", action="store_true", help="Re-download and rebuild")
    args = parser.parse_args()

    target = dataset_dir("MNIST")
    print(f"MNIST -> {target}")

    for remote, local in IDX_FILES.items():
        final = target / local
        if final.exists() and not args.force:
            print(f"  [skip] {local} already present")
            continue
        gunzip(download(f"{args.url_base}/{remote}", target / remote, force=args.force), final)

    install_class_names("MNIST")
    sample_ids = install_sample_ids("MNIST")

    # image_id is the zero-padded index into the training split
    images = read_idx_images(target / IDX_FILES["train-images-idx3-ubyte.gz"])
    labels = read_idx_labels(target / IDX_FILES["train-labels-idx1-ubyte.gz"])
    print(f"  train split: {len(images)} images")

    out_dir = target / IMAGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata, mapping, missing = [], {}, []
    for image_id in sample_ids:
        index = int(image_id)
        if index >= len(images):
            missing.append(image_id)
            continue
        file_name = f"{PREFIX}_{image_id}.png"
        Image.fromarray(images[index]).save(out_dir / file_name)
        label = int(labels[index])
        mapping[image_id] = label
        metadata.append({"image_id": image_id, "file_name": file_name,
                         "label": label, "index": index})

    if missing:
        raise SystemExit(f"{len(missing)} sample ids fall outside the train split, "
                         f"e.g. {missing[:5]}")

    (target / "image_to_class_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (target / "sampled_images_metadata.json").write_text(
        json.dumps({"total_sampled": len(metadata), "images": metadata}, indent=2) + "\n")

    print(f"  wrote {len(metadata)} images to {IMAGES_DIR}/, "
          f"image_to_class_mapping.json, sampled_images_metadata.json")
    print("MNIST ready.")


if __name__ == "__main__":
    main()
