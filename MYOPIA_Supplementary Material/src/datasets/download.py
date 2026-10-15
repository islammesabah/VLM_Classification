"""One entry point for every dataset.

    python datasets/download.py                      # all four
    python datasets/download.py --dataset MNIST GTSRB

Each script downloads its dataset and reduces it to the fixed evaluation
subset in sample_ids/. This just runs them in turn and reports what succeeded.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = {
    "GTSRB": HERE / "download_GTSRB.py",
    "CIFAR-10": HERE / "download_CIFAR.py",
    "MNIST": HERE / "download_MNIST.py",
    "MSCOCO": HERE / "download_MSCOCO.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the benchmark datasets.")
    parser.add_argument("--dataset", nargs="+", choices=list(SCRIPTS), default=list(SCRIPTS),
                        help="Datasets to fetch (default: all four)")
    args = parser.parse_args()

    failed = []
    for name in args.dataset:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        if subprocess.run([sys.executable, str(SCRIPTS[name])]).returncode != 0:
            failed.append(name)
            print(f"!! {name} failed -- continuing with the rest")

    print(f"\n{'=' * 60}")
    done = [d for d in args.dataset if d not in failed]
    print(f"ready:  {', '.join(done) if done else 'none'}")
    if failed:
        print(f"failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
