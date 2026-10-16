"""Shared plumbing for the per-dataset download scripts.

Every dataset is fetched the same way: resolve its directory relative to this
file (so scripts work from any working directory), stream the archive with a
progress bar, skip anything already on disk, extract, and write the
``class_names.json`` the loaders expect.
"""
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATASETS_DIR = Path(__file__).resolve().parent
CLASS_NAMES_DIR = DATASETS_DIR / "class_names"
REPO_ROOT = DATASETS_DIR.parent.parent

# Canonical class-name file and evaluation subset for each dataset. The sample
# id lists under sample_ids/ are the fixed subsets the paper reports on, so a
# download is only "done" once the dataset on disk has been filtered to them.
CLASS_NAMES_FILES = {
    "GTSRB": "gtsrb_class_names.json",
    "CIFAR-10": "cifar10_class_names.json",
    "MNIST": "mnist_class_names.json",
    "MSCOCO": "mscoc_class_names.json",
}
SAMPLE_ID_FILES = {
    "GTSRB": "gtsrb_901_samples.txt",
    "CIFAR-10": "cifar10_1000_samples.txt",
    "MNIST": "mnist_1000_samples.txt",
    "MSCOCO": "mscoco_850_samples.txt",
}
# Filename each dataset's config.json samples_path points at.
SAMPLES_PATH_NAMES = {
    "GTSRB": "901_samples.txt",
    "CIFAR-10": "1000_samples.txt",
    "MNIST": "sampled_image_ids.txt",
    "MSCOCO": "850_filtered_1_category_image_ids.txt",
}

# Directory name on disk for each dataset, matching config.json's Dataset_path.
DIR_NAMES = {
    "GTSRB": "GTSRB",
    "CIFAR-10": "CIFAR-10",
    "MNIST": "MNIST",
    "MSCOCO": "MSCOCO",
}


def dataset_dir(dataset: str, create: bool = True) -> Path:
    """Absolute path to a dataset's directory, independent of the caller's cwd."""
    if dataset not in DIR_NAMES:
        raise ValueError(f"Unknown dataset {dataset!r}. Known: {', '.join(DIR_NAMES)}")
    path = DATASETS_DIR / DIR_NAMES[dataset]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Stream `url` to `dest` with a progress bar. Existing files are kept."""
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"  [skip] {dest.name} already present")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"  {dest.name}"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    return dest


def extract(archive: Path, dest: Path, remove: bool = True) -> None:
    """Unpack a .zip or .tar.gz into `dest`, then delete the archive."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} ...")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(path=dest)
    else:
        raise ValueError(f"Don't know how to extract {archive.name}")
    if remove:
        archive.unlink()


def write_class_names(dataset: str, class_names: dict) -> Path:
    """Write the {id: name} map every DatasetLoader reads at startup."""
    path = dataset_dir(dataset) / "class_names.json"
    path.write_text(json.dumps(class_names, indent=2) + "\n")
    print(f"  wrote {path.name} ({len(class_names)} classes)")
    return path


def move_tree(src: Path, dest: Path) -> None:
    """Rename a directory, replacing any existing destination."""
    src, dest = Path(src), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    src.rename(dest)


def load_class_names(dataset: str) -> dict:
    """The curated {id: name} map shipped in datasets/class_names/."""
    path = CLASS_NAMES_DIR / CLASS_NAMES_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Missing class names for {dataset}: {path}")
    return json.loads(path.read_text())


def install_class_names(dataset: str) -> dict:
    """Copy the curated class names into the dataset dir, where loaders read them."""
    class_names = load_class_names(dataset)
    write_class_names(dataset, class_names)
    return class_names


def load_sample_ids(dataset: str) -> list:
    """The fixed evaluation subset for a dataset, in file order."""
    path = REPO_ROOT / "sample_ids" / SAMPLE_ID_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Missing sample ids for {dataset}: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def install_sample_ids(dataset: str) -> list:
    """Write the subset into the dataset dir under the name config.json expects."""
    ids = load_sample_ids(dataset)
    dest = dataset_dir(dataset) / SAMPLES_PATH_NAMES[dataset]
    dest.write_text("\n".join(ids) + "\n")
    print(f"  wrote {dest.name} ({len(ids)} ids)")
    return ids
