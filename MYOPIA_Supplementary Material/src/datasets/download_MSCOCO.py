"""Download MSCOCO (train2014) and reduce it to the paper's evaluation subset.

    python datasets/download_MSCOCO.py

Self-contained: fetches the train2014 images and the 2014 annotations, then
writes exactly the 850 images named in sample_ids/mscoco_850_samples.txt,
together with the class_names.json and filtered_1_category_images.json that
MSCOCOLoader reads.

Because the evaluation subset is fixed, no sampling happens here -- the ids
select the images and the annotations supply their single category. Each id is
verified to carry exactly one category, which is what makes the subset
single-category by construction.

This is a large download (~13 GB images + ~250 MB annotations); the filtering
step reclaims most of it. Parsing instances_train2014.json needs a few GB of
RAM.
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _download_utils import (dataset_dir, download, extract, install_class_names,
                             install_sample_ids)

IMAGES_URL = "http://images.cocodataset.org/zips/train2014.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
SOURCE_DIR = "train2014"
IMAGES_DIR = "filtered_1_category_images"   # config.json images_dir
ANNOTATION_FILE = "annotations/instances_train2014.json"


def build_subset(target: Path, sample_ids: list) -> list:
    """Metadata for the sampled ids, verifying each has exactly one category."""
    print(f"  reading {ANNOTATION_FILE} (large; this takes a moment) ...")
    with open(target / ANNOTATION_FILE) as f:
        coco = json.load(f)

    wanted = {int(sid) for sid in sample_ids}
    file_names = {img["id"]: img["file_name"] for img in coco["images"] if img["id"] in wanted}

    categories = defaultdict(set)
    for ann in coco["annotations"]:
        if ann["image_id"] in wanted:
            categories[ann["image_id"]].add(ann["category_id"])

    records, missing, multi = [], [], []
    for sid in sample_ids:
        image_id = int(sid)
        if image_id not in file_names:
            missing.append(sid)
            continue
        cats = sorted(categories.get(image_id, ()))
        if len(cats) != 1:
            multi.append((sid, cats))
            continue
        records.append({"image_id": image_id,
                        "file_name": file_names[image_id],
                        "category_ids": cats})

    if missing:
        raise SystemExit(f"{len(missing)} sampled ids are absent from train2014, "
                         f"e.g. {missing[:5]}")
    if multi:
        raise SystemExit(f"{len(multi)} sampled ids do not have exactly one category, "
                         f"e.g. {multi[:3]}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and filter the MSCOCO dataset.")
    parser.add_argument("--keep_source", action="store_true",
                        help=f"Keep the unfiltered {SOURCE_DIR}/ directory after filtering")
    args = parser.parse_args()

    target = dataset_dir("MSCOCO")
    print(f"MSCOCO -> {target}")

    if (target / SOURCE_DIR).is_dir():
        print(f"  [skip] {SOURCE_DIR}/ already present")
    else:
        extract(download(IMAGES_URL, target / "train2014.zip"), target)

    if (target / "annotations").is_dir():
        print("  [skip] annotations/ already present")
    else:
        extract(download(ANNOTATIONS_URL, target / "annotations_trainval2014.zip"), target)

    install_class_names("MSCOCO")
    sample_ids = install_sample_ids("MSCOCO")
    records = build_subset(target, sample_ids)

    out_dir = target / IMAGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        shutil.copy2(target / SOURCE_DIR / record["file_name"], out_dir / record["file_name"])

    (target / "filtered_1_category_images.json").write_text(
        json.dumps({"total_images": len(records), "images": records}, indent=2) + "\n")

    print(f"  wrote {len(records)} images to {IMAGES_DIR}/ "
          f"and filtered_1_category_images.json")

    if not args.keep_source:
        print(f"  removing the unfiltered {SOURCE_DIR}/ source (use --keep_source to retain)")
        shutil.rmtree(target / SOURCE_DIR, ignore_errors=True)

    print("MSCOCO ready.")


if __name__ == "__main__":
    main()
