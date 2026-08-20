"""
Parent-subsumption check for GTSRB's pedestrian pair.

The GTSRB tree separates two closely related warning signs:

    [L10] Q: Does it depict both child and pedestrian?
      -> yes:  Final: both child and pedestrian warning ([28])
      -> no:
        [L11] Q: Does it depict pedestrian?
          -> yes: Final: pedestrian warning ([27])

The specific class is tested first, so an image of class 28 whose child figure
goes unnoticed answers "no" at L10, meets the broader question at L11, and can
be labelled class 27 instead. That is the one direction in which a class can
be swallowed by its more general sibling, and it is worth measuring directly
rather than assuming greedy traversal produces it.

This script reports, for every stored traversal of a ground-truth class-28
image, how often the prediction lands on the parent class, on the correct
class, on an unresolved path, or on some unrelated class.

Rows are read through `db_utils.ANSWERS`, so the superseded duplicate paths
described there are excluded and each image is counted once per configuration.

Usage:
    python evaluation/parent_subsumption_analysis.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_utils import ANSWERS

DB_PATH = "results/gtsrb_merged.db"
OUT_PATH = Path("results/parent_subsumption_gtsrb.csv")

CHILD_CLASS = 28   # "both child and pedestrian warning"
PARENT_CLASS = 27  # "pedestrian warning"

UNRESOLVED_MARKERS = {"-1", "None", "nan", ""}


def main():
    if not Path(DB_PATH).exists():
        print(f"DB not found at {DB_PATH}")
        return

    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        f"SELECT LLM_output_with_tree_class FROM {ANSWERS} "
        "WHERE CAST(Class AS INTEGER) = ? AND LLM_output_with_tree IS NOT NULL",
        (CHILD_CLASS,),
    ).fetchall()
    con.close()

    preds = [str(r[0]).strip() for r in rows]
    n = len(preds)
    if n == 0:
        print("No traversals found for the child class.")
        return

    counts = {
        "correct_child_class": sum(p == str(CHILD_CLASS) for p in preds),
        "parent_class": sum(p == str(PARENT_CLASS) for p in preds),
        "unresolved": sum(p in UNRESOLVED_MARKERS for p in preds),
    }
    counts["other_class"] = n - sum(counts.values())

    out = pd.DataFrame(
        [{"outcome": k, "n": v, "percent": round(100 * v / n, 1)} for k, v in counts.items()]
    )
    out.loc[len(out)] = {
        "outcome": "unresolved_or_other",
        "n": counts["unresolved"] + counts["other_class"],
        "percent": round(100 * (counts["unresolved"] + counts["other_class"]) / n, 1),
    }

    print(f"Ground-truth class {CHILD_CLASS} traversals: n={n}")
    print(out.to_string(index=False))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.insert(0, "n_total", n)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
