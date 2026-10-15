"""
Inference-cost analysis: Table 6 (mean/max VLM calls per tree-classified
image and unresolved-path rates before/after oracle re-parsing) and Figure 2
(GPT-4o accuracy vs mean calls per image).

Calls per image are computed from the stored traversals: every visited node is
one VLM call, and LLM_output_with_tree holds the JSON list of visited nodes.
Zero-shot and CoT are one call per image by construction; CLIP zero-shot makes
no VLM calls and is plotted at calls=0 for reference. Unresolved rates come
from results/tree_error_analysis/oracle_failure_breakdown_primary_condition.csv
(run evaluation/oracle_extraction_analysis.py first). Accuracy inputs: the
answer databases (ZS/Tree), results/cot_baseline/ (CoT), and
results/baselines/summary_accuracy.csv (CLIP). No API access is needed.

Usage (from the repository root):
    python evaluation/cost_frontier.py [--wide] [-o results/cost_frontier.pdf]
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_utils import ANSWERS

OUT_PATH = Path("results/cost_frontier.pdf")
TABLE_DIR = Path("results/inference_cost")
ORACLE_CSV = Path("results/tree_error_analysis/oracle_failure_breakdown_primary_condition.csv")
DBS = {
    "GTSRB": "results/gtsrb_merged.db",
    "MNIST": "results/mnist.db",
    "CIFAR-10": "results/cifar10.db",
    "MSCOCO": "results/mscoco.db",
}
MODELS = ["gpt-4o", "meta-llama/llama-3.2-11b-vision-instruct", "qwen-vl-max"]
COT_CSV = {ds: Path(f"results/cot_baseline/{ds.replace('-', '').lower()}_gpt_4o_cot.csv")
           for ds in DBS}
BASELINES_CSV = Path("results/baselines/summary_accuracy.csv")

PRIMARY = """
      AND CAST(Include_memory AS INTEGER) = 0
      AND CAST(Include_description AS INTEGER) = 0
      AND temperature = 0.7
      AND CAST(RunId AS INTEGER) = 0
"""

METHOD_STYLE = {
    "CLIP ZS": dict(marker="D", color="tab:gray"),
    "ZS": dict(marker="o", color="tab:blue"),
    "CoT": dict(marker="^", color="tab:green"),
    "Tree": dict(marker="s", color="tab:red"),
}


def tree_calls(conn, models):
    """Per-image call counts (visited nodes) for the primary condition."""
    sql = f"""
        SELECT LLM_name, LLM_output_with_tree FROM {ANSWERS}
        WHERE LLM_name IN ({','.join('?' * len(models))}) {PRIMARY}
          AND LLM_output_with_tree IS NOT NULL
    """
    counts, skipped = [], 0
    for name, raw in conn.execute(sql, models):
        try:
            path = json.loads(raw)
            assert isinstance(path, list) and path
        except (json.JSONDecodeError, AssertionError):
            skipped += 1
            continue
        counts.append((name, len(path)))
    return counts, skipped


def accuracy(conn, model, column):
    n, correct = conn.execute(f"""
        SELECT COUNT(*),
               SUM(CAST(Class AS INTEGER) = CAST(LLM_output_{column}_tree_class AS INTEGER))
        FROM {ANSWERS} WHERE LLM_name = ? {PRIMARY}
    """, (model,)).fetchone()
    return None if not n else 100.0 * correct / n


def build_data():
    """(figure_points, table6_rows); every number computed, none hand-entered."""
    clip = pd.read_csv(BASELINES_CSV, index_col=0)
    points, table6 = {}, []
    for dataset, db in DBS.items():
        conn = sqlite3.connect(db)
        counts, skipped = tree_calls(conn, MODELS)
        pooled = [c for _, c in counts]
        gpt = [c for name, c in counts if name == "gpt-4o"]
        if skipped:
            print(f"[{dataset}] {skipped} rows had no parseable traversal; excluded from calls")

        cot = pd.read_csv(COT_CSV[dataset])
        cot_acc = 100.0 * (cot["correct"].astype(str).str.lower() == "true").mean()
        points[dataset] = {
            "CLIP ZS": (0.0, 100.0 * clip.loc[dataset, "clip_zero_shot"]),
            "ZS": (1.0, accuracy(conn, "gpt-4o", "without")),
            "CoT": (1.0, cot_acc),
            "Tree": (sum(gpt) / len(gpt), accuracy(conn, "gpt-4o", "with")),
        }
        table6.append({
            "dataset": dataset,
            "mean_calls": sum(pooled) / len(pooled),
            "max_calls": max(pooled),
            "mean_calls_gpt4o": sum(gpt) / len(gpt),
            "n_traversals": len(pooled),
        })
        conn.close()

    oracle = pd.read_csv(ORACLE_CSV).set_index("dataset")
    for row in table6:
        row["unresolved_pct"] = 100.0 * oracle.loc[row["dataset"], "orig_unresolved_rate"]
        row["oracle_unresolved_pct"] = 100.0 * oracle.loc[row["dataset"], "oracle_unresolved_rate"]
    return points, pd.DataFrame(table6)


def tex_table6(t6):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Mean and maximum API calls per tree-classified image and "
        r"unresolved-path rates before and after oracle re-parsing, pooled across models.}",
        r"\label{tab:inference_cost}", r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"\textbf{Dataset} & \textbf{Mean calls} & \textbf{Max calls} & "
        r"\textbf{Unresolved} & \textbf{Oracle} \\",
        r"\midrule",
    ]
    for _, r in t6.iterrows():
        lines.append(f"{r['dataset']} & {r['mean_calls']:.2f} & {int(r['max_calls'])} & "
                     f"{r['unresolved_pct']:.1f}\\% & {r['oracle_unresolved_pct']:.1f}\\%" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def plot(points, out_path, wide):
    if wide:
        fig, axes = plt.subplots(1, 4, figsize=(11, 2.0), sharey=True, sharex=True)
        sizes = dict(title=10, label=8, tick=8, legend=8, marker=55)
        legend_y = 1.12
    else:
        # Single WACV column is 3.25in wide: 2x2 panels, figure-level labels.
        fig, axes = plt.subplots(2, 2, figsize=(3.25, 3.0), sharey=True, sharex=True)
        sizes = dict(title=7, label=7, tick=6, legend=6, marker=18)
        legend_y = 1.06
    flat = axes.flat if hasattr(axes, "flat") else axes

    for ax, (dataset, methods) in zip(flat, points.items()):
        for method, (calls, acc) in methods.items():
            ax.scatter(calls, acc, s=sizes["marker"], label=method, **METHOD_STYLE[method])
        ax.set_title(dataset, fontsize=sizes["title"], pad=3)
        ax.set_xlim(-0.4, 7.2)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=sizes["tick"], length=2, pad=1.5)

    if wide:
        for ax in flat:
            ax.set_xlabel("mean API calls / image", fontsize=sizes["label"])
        axes[0].set_ylabel("accuracy (%)", fontsize=sizes["label"])
    else:
        fig.supxlabel("mean API calls / image", fontsize=sizes["label"], y=0.02)
        fig.supylabel("accuracy (%)", fontsize=sizes["label"], x=0.015)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=sizes["legend"],
               bbox_to_anchor=(0.5, legend_y), frameon=False, handletextpad=0.3,
               columnspacing=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wide", action="store_true",
                        help="full-width 1x4 layout (spans both columns)")
    parser.add_argument("-o", "--out", default=str(OUT_PATH))
    args = parser.parse_args()

    points, t6 = build_data()

    print("Figure 2 points (calls, accuracy):")
    for dataset, methods in points.items():
        print(f"  {dataset:9s} " + "  ".join(
            f"{m}=({c:.2f}, {a:.2f})" for m, (c, a) in methods.items()))
    print("\nTable 6:")
    print(t6.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    t6.to_csv(TABLE_DIR / "table6_inference_cost.csv", index=False)
    (TABLE_DIR / "tab_inference_cost.tex").write_text(tex_table6(t6) + "\n")
    plot(points, Path(args.out), args.wide)


if __name__ == "__main__":
    main()
