"""
Paired bootstrap confidence intervals for the Tree-vs-Zero-shot accuracy gap
(Plan Cycle 2, Phase A2). Zero-shot and tree predictions are stored on the
same image row, so this is a paired design: resample images with replacement,
recompute both accuracies each time, and look at the distribution of
Delta = Tree_acc - ZS_acc.

Usage:
    python evaluation/bootstrap_ci.py
"""
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_utils import ANSWERS

OUT_PATH = "results/bootstrap_ci_holm.csv"

DBS = {
    "GTSRB": "results/gtsrb_merged.db",
    "MNIST": "results/mnist.db",
    "CIFAR-10": "results/cifar10.db",
    "MSCOCO": "results/mscoco.db",
}
MODELS = ["gpt-4o", "meta-llama/llama-3.2-11b-vision-instruct", "qwen-vl-max"]
MODEL_LABEL = {
    "gpt-4o": "GPT-4o",
    "meta-llama/llama-3.2-11b-vision-instruct": "LLaMA",
    "qwen-vl-max": "Qwen-VL",
}
N_BOOTSTRAP = 10000
SEED = 42


def norm(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    try:
        return str(int(s))
    except ValueError:
        return s


def report_holm(results, alpha=0.05):
    """
    Holm-Bonferroni correction over the full family of dataset x model
    comparisons, so the paper can cite adjusted p-values rather than assert
    that an uncorrected p clears a threshold.

    Holm sorts the p-values ascending and compares the i-th (0-indexed) against
    alpha / (m - i); the adjusted p-value is the running maximum of
    (m - i) * p, capped at 1, which keeps the sequence monotone. Because the
    two-sided bootstrap p is floored at 2 / N_BOOTSTRAP, adjusted values at
    that floor are upper bounds and are flagged as such.
    """
    if not results:
        return
    m = len(results)
    ordered = sorted(results, key=lambda r: r["p_two_sided"])
    running = 0.0
    for i, r in enumerate(ordered):
        adjusted = min(1.0, (m - i) * r["p_two_sided"])
        running = max(running, adjusted)
        r["p_holm_adjusted"] = running
        r["significant_holm"] = running < alpha

    print(f"\nHolm-Bonferroni over {m} comparisons (family-wise alpha={alpha}):")
    print(f"{'Dataset':10s} {'Model':8s} {'p_two':>10s} {'p_holm':>10s}  sig")
    for r in ordered:
        flag = "<" if r["p_resolution_limited"] else " "
        print(f"{r['dataset']:10s} {r['model']:8s} {flag}{r['p_two_sided']:9.5f} "
              f"{flag}{r['p_holm_adjusted']:9.5f}  {'yes' if r['significant_holm'] else 'NO'}")

    n_sig = sum(r["significant_holm"] for r in ordered)
    largest = max(r["p_holm_adjusted"] for r in ordered)
    print(f"\n{n_sig}/{m} significant after correction; largest adjusted p = {largest:.4f}")
    print("'<' marks values floored by bootstrap resolution (upper bounds).")

    out = Path(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "model", "n", "zs_acc", "tree_acc", "delta", "ci_lo", "ci_hi",
            "p_zs_greater", "p_two_sided", "p_holm_adjusted", "significant_holm",
            "p_resolution_limited"]
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in ordered:
            writer.writerow({c: r[c] for c in cols})
    print(f"Wrote {out}")


def main():
    rng = np.random.default_rng(SEED)
    results = []
    print(f"{'Dataset':10s} {'Model':8s} {'n':>5s}  {'ZS%':>6s} {'Tree%':>6s} "
          f"{'Delta':>7s}  {'95% CI':>16s}  {'P(ZS>Tree)':>10s}")
    for dataset, dbpath in DBS.items():
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        for model in MODELS:
            cur.execute(f"""
                SELECT Class, LLM_output_without_tree_class, LLM_output_with_tree_class
                FROM {ANSWERS}
                WHERE Include_memory='0' AND Include_description='0' AND RunId=0
                  AND LLM_name=? AND temperature='0.7'
            """, (model,))
            rows = cur.fetchall()
            if not rows:
                continue
            zs_correct = np.array([norm(c) == norm(z) for c, z, t in rows])
            tree_correct = np.array([norm(c) == norm(t) for c, z, t in rows])
            n = len(rows)

            idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
            zs_boot = zs_correct[idx].mean(axis=1)
            tree_boot = tree_correct[idx].mean(axis=1)
            delta_boot = tree_boot - zs_boot

            zs_acc = zs_correct.mean() * 100
            tree_acc = tree_correct.mean() * 100
            delta = tree_acc - zs_acc
            ci_lo, ci_hi = np.percentile(delta_boot, [2.5, 97.5]) * 100
            p_zs_greater = (delta_boot < 0).mean()

            # Two-sided bootstrap p for "the two methods differ", derived from the
            # tail mass. It is resolution-limited at 1/N_BOOTSTRAP: a comparison
            # where every resample favours zero-shot reports p = 2/N_BOOTSTRAP,
            # which is an upper bound rather than an exact value.
            p_two = min(1.0, 2.0 * min(p_zs_greater, 1.0 - p_zs_greater))
            p_two = max(p_two, 2.0 / N_BOOTSTRAP)
            results.append({
                "dataset": dataset, "model": MODEL_LABEL[model], "n": n,
                "zs_acc": round(zs_acc, 2), "tree_acc": round(tree_acc, 2),
                "delta": round(delta, 2), "ci_lo": round(ci_lo, 2), "ci_hi": round(ci_hi, 2),
                "p_zs_greater": round(p_zs_greater, 4), "p_two_sided": p_two,
                "p_resolution_limited": p_two <= 2.0 / N_BOOTSTRAP,
            })

            print(f"{dataset:10s} {MODEL_LABEL[model]:8s} {n:5d}  {zs_acc:6.2f} {tree_acc:6.2f} "
                  f"{delta:7.2f}  [{ci_lo:6.2f}, {ci_hi:6.2f}]  {p_zs_greater:10.4f}")
        con.close()

    report_holm(results)


if __name__ == "__main__":
    main()
