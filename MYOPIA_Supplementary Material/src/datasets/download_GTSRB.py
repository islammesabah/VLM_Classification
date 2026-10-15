
"""Download GTSRB and reduce it to the paper's evaluation subset.

    python datasets/download_GTSRB.py

Fetches the fixed training archive, converts the .ppm frames to .png, writes
class_names.json, then keeps only the 901 frames named in
sample_ids/gtsrb_901_samples.txt.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _download_utils import (dataset_dir, download, extract, install_class_names,
                             install_sample_ids)

ARCHIVE_URL = ("https://sid.erda.dk/public/archives/"
               "daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip")

CLASS_NAMES = {
  "0": "20 Kph Limit",
  "1": "30 Kph Limit",
  "2": "50 Kph Limit",
  "3": "60 Kph Limit",
  "4": "70 Kph Limit",
  "5": "80 Kph Limit",
  "6": "End Of Restriction",
  "7": "100 Kph Limit",
  "8": "120 Kph Limit",
  "9": "No Cars Passing",
  "10": "No Trucks Passing",
  "11": "Right-Of-Way At Intersection",
  "12": "Priority Road",
  "13": "Yield",
  "14": "Stop",
  "15": "Empty Circle",
  "16": "No Truck Entry",
  "17": "No Entry",
  "18": "Exclamation Mark Warning",
  "19": "Left Curve Warning",
  "20": "Right Curve Warning",
  "21": "Double Curve Warning",
  "22": "Rough/Bumpy Road Warning",
  "23": "Slippery Road Warning",
  "24": "Merging/Narrow Lanes Warning",
  "25": "Construction/Road Work Warning",
  "26": "Traffic Light Warning",
  "27": "Pedestrian Warning",
  "28": "Child And Pedestrian Warning",
  "29": "Bicycle Warning",
  "30": "Ice/Snow Warning",
  "31": "Deer Warning",
  "32": "No Speed Limit",
  "33": "Right Arrow",
  "34": "Left Arrow",
  "35": "Forward Arrow",
  "36": "Forward-Right Arrow",
  "37": "Forward-Left Arrow",
  "38": "Keep-Right Arrow",
  "39": "Keep-Left Arrow",
  "40": "Circle Arrow",
  "41": "End Car Passing Ban",
  "42": "End Truck Passing Ban"
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and process the GTSRB dataset.")
    parser.parse_args()

    target = dataset_dir("GTSRB")
    print(f"GTSRB -> {target}")
    install_class_names("GTSRB")
    sample_ids = install_sample_ids("GTSRB")

    if (target / "Training").is_dir():
        print("  [skip] Training/ already present")
    else:
        # the archive contains a GTSRB/ top level, so extract into datasets/
        extract(download(ARCHIVE_URL, target.parent / "GTSRB-Training_fixed.zip"), target.parent)

    ppm_files = glob.glob(str(target / "**" / "*.ppm"), recursive=True)
    if ppm_files:
        print(f"  converting {len(ppm_files)} .ppm frames to .png ...")
        for img_path in ppm_files:
            Image.open(img_path).save(os.path.splitext(img_path)[0] + ".png", "PNG")
            os.remove(img_path)

    # keep only the sampled frames -- ids look like '00004/00039_00029'
    images_root = target / "Training"
    keep = {images_root / f"{sid}.png" for sid in sample_ids}
    present = set(images_root.glob("**/*.png"))
    missing = keep - present
    if missing:
        raise SystemExit(f"{len(missing)} sampled frames not found on disk, "
                         f"e.g. {sorted(str(m.relative_to(images_root)) for m in missing)[:5]}")
    for path in present - keep:
        path.unlink()
    for folder in sorted(images_root.glob("*"), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    print(f"  filtered to {len(keep)} sampled frames "
          f"({len(present) - len(keep)} others removed)")

    print("GTSRB ready.")


if __name__ == "__main__":
    main()