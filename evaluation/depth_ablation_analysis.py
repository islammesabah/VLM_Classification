"""
Depth-ablation / error-propagation analysis (Plan Cycle 2, Phase C2), computed
entirely from data already collected this session (zero new API cost).

Reuses results/tree_error_analysis/oracle_classification_raw.csv (produced by
evaluation/oracle_extraction_analysis.py), which already labels each stored
tree-traversal path as "fully_resolved_correct" or not, and joins it against
each class's GROUND-TRUTH PATH LENGTH in the current tree definition. Grouping
by path length and computing the full-path success rate per group tests the
error-propagation hypothesis directly: if each node has an independent,
roughly constant per-node error rate p_correct, full-path accuracy at length L
should decay close to p_correct^L. Comparing observed vs. this naive
independence baseline shows whether errors compound faster or slower than
independent per-node failures would predict.

Usage:
    python evaluation/depth_ablation_analysis.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.data_loader import CIFARLoader, GTSRBLoader, MNISTLoader, MSCOCOLoader

LOADERS = {
    "GTSRB": GTSRBLoader, "MNIST": MNISTLoader,
    "CIFAR-10": CIFARLoader, "MSCOCO": MSCOCOLoader,
}
RAW_PATH = Path("results/tree_error_analysis/oracle_classification_raw.csv")
OUT_DIR = Path("results/tree_error_analysis")


def ground_truth_path_length(tree_root, target_class):
    node, length = tree_root, 0
    while node.question:
        chosen = None
        for answer, child in node.children.items():
            if target_class in child.get_all_possible_classes():
                chosen = child
                break
        if chosen is None:
            return None
        node = chosen
        length += 1
    return length


def main():
    df = pd.read_csv(RAW_PATH)
    primary = df[
        (df["temperature"] == 0.7) & (df["include_memory"] == 0)
        & (df["include_description"] == 0) & (df["run_id"] == 0)
    ].copy()

    for name, loader_cls in LOADERS.items():
        tree_root = loader_cls().build_tree()
        classes = sorted(tree_root.get_all_possible_classes())
        lengths = [ground_truth_path_length(tree_root, c) for c in classes]
        lengths = [l for l in lengths if l is not None]
        dist = pd.Series(lengths).value_counts().sort_index()
        print(f"\n=== {name}: ground-truth path-length distribution across {len(lengths)} classes ===")
        print(dist.to_string())

        sub = primary[primary["dataset"] == name]
        if sub.empty:
            continue
        full_success = (sub["category"] == "fully_resolved_correct").mean()
        depth_csv = OUT_DIR / "per_depth_accuracy_primary_condition.csv"
        if depth_csv.exists():
            depth_df = pd.read_csv(depth_csv)
            depth_df = depth_df[depth_df["dataset"] == name]
            if not depth_df.empty:
                p_node = (depth_df["n_correct"].sum() / depth_df["n_total"].sum())
                mean_len = sum(lengths) / len(lengths)
                naive_pred = p_node ** mean_len
                print(f"observed full-path success (primary condition, pooled models) = {full_success:.4f}")
                print(f"mean per-node accuracy p_node = {p_node:.4f}, mean path length = {mean_len:.2f}")
                print(f"naive independence prediction p_node^mean_length = {naive_pred:.4f}")
                ratio = full_success / naive_pred if naive_pred else float("nan")
                direction = "errors compound FASTER than independent" if full_success < naive_pred else "errors compound SLOWER than independent"
                print(f"observed / naive ratio = {ratio:.3f} ({direction})")


if __name__ == "__main__":
    main()
