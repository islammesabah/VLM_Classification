"""
Per-depth / per-question error analysis of tree-based classification runs.

Reconstructs, for every stored tree-traversal path in each dataset's SQLite DB,
whether each node-level decision kept the traversal on a path that could still
reach the image's ground-truth class, using the *current* tree definitions in
tree_generation/. A decision at depth d is scored only if every prior decision
on that path was also still consistent with the ground-truth class (i.e., the
model hadn't already left the correct branch) -- this operationalizes "error
propagation" directly from the data instead of asserting it qualitatively.

Historical rows store the raw model response text as the "answer" field
(pre-dating the regex-extraction step used to pick a branch), not the
extracted branch keyword. We therefore re-run the same word-boundary,
case-insensitive, longest-answer-first extraction main.py uses before falling
back to "unparseable."

Usage:
    python evaluation/tree_error_analysis.py
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.data_loader import CIFARLoader, GTSRBLoader, MNISTLoader, MSCOCOLoader
from db_utils import ANSWERS

DATASETS = {
    "GTSRB": {"db": "results/gtsrb_merged.db", "loader": GTSRBLoader},
    "MNIST": {"db": "results/mnist.db", "loader": MNISTLoader},
    "CIFAR-10": {"db": "results/cifar10.db", "loader": CIFARLoader},
    "MSCOCO": {"db": "results/mscoco.db", "loader": MSCOCOLoader},
}

OUT_DIR = Path("results/tree_error_analysis")


def extract_answer(node, text):
    """Reproduce main.py's regex-based branch extraction against free text."""
    if not text or not node.answers:
        return None
    sorted_answers = sorted(node.answers, key=len, reverse=True)
    pattern = "(?i)(" + "|".join(f"\\b{re.escape(a)}\\b" for a in sorted_answers) + ")"
    m = re.search(pattern.lower(), text.lower())
    if not m:
        return None
    return next((a for a in sorted_answers if a.lower() == m.group(1).lower()), None)


def walk_path(tree_root, target_class, path):
    """
    Yield one record per scored decision along `path`, stopping at the first
    decision that leaves the ground-truth-consistent subtree (or the first
    decision we can't resolve to a valid branch at all).
    """
    node = tree_root
    for depth, entry in enumerate(path):
        question = entry.get("question")
        raw_answer = entry.get("answer")
        if node.question != question:
            # Path no longer matches the current tree definition (tree edited
            # since these rows were generated) -- stop, don't fabricate a result.
            yield {"depth": depth, "question": question, "resolved": False,
                   "correct": None, "reason": "question_mismatch"}
            return
        answer = raw_answer if raw_answer in node.children else extract_answer(node, raw_answer)
        if answer is None or answer not in node.children:
            yield {"depth": depth, "question": question, "resolved": False,
                   "correct": None, "reason": "unparseable_answer"}
            return
        child = node.children[answer]
        correct = target_class in child.get_all_possible_classes()
        yield {"depth": depth, "question": question, "resolved": True,
               "correct": correct, "reason": None}
        if not correct:
            return
        node = child


def analyze_dataset(name, cfg):
    loader_cls = cfg["loader"]
    db_path = cfg["db"]
    if not Path(db_path).exists():
        print(f"[{name}] DB not found at {db_path}, skipping.")
        return pd.DataFrame()

    tree_root = loader_cls().build_tree()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT Class, LLM_name, temperature, Include_memory, Include_description, "
        f"RunId, LLM_output_with_tree FROM {ANSWERS} WHERE LLM_output_with_tree IS NOT NULL"
    )
    rows = cur.fetchall()
    conn.close()

    records = []
    n_badjson, n_badclass = 0, 0
    for class_str, model, temperature, include_memory, include_description, run_id, raw in rows:
        try:
            path = json.loads(raw)
            if not isinstance(path, list) or not path:
                n_badjson += 1
                continue
        except (json.JSONDecodeError, TypeError):
            n_badjson += 1
            continue
        try:
            target_class = int(class_str)
        except (TypeError, ValueError):
            n_badclass += 1
            continue

        for rec in walk_path(tree_root, target_class, path):
            rec.update(
                dataset=name, model=model, temperature=temperature,
                include_memory=include_memory, include_description=include_description,
                run_id=run_id,
            )
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    print(f"[{name}] rows={len(rows)} decisions_scored={len(df)} "
          f"dropped_bad_json={n_badjson} dropped_bad_class={n_badclass}")
    return df


def summarize(df: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "decisions_raw.csv", index=False)

    resolved = df[df["resolved"]].copy()
    resolved["correct"] = resolved["correct"].astype(bool)

    # Primary condition: temperature=0.7, no memory, no description, RunId=0
    # (matches the condition the paper's headline figures/tables are "centered on").
    primary = resolved[
        (resolved["temperature"] == 0.7)
        & (resolved["include_memory"] == "0")
        & (resolved["include_description"] == "0")
        & (resolved["run_id"] == 0)
    ]

    for label, subset in [("all_configs", resolved), ("primary_condition", primary)]:
        by_depth = (
            subset.groupby(["dataset", "depth"])["correct"]
            .agg(n_correct="sum", n_total="count")
            .reset_index()
        )
        by_depth["accuracy"] = by_depth["n_correct"] / by_depth["n_total"]
        by_depth.to_csv(OUT_DIR / f"per_depth_accuracy_{label}.csv", index=False)

        by_depth_model = (
            subset.groupby(["dataset", "model", "depth"])["correct"]
            .agg(n_correct="sum", n_total="count")
            .reset_index()
        )
        by_depth_model["accuracy"] = by_depth_model["n_correct"] / by_depth_model["n_total"]
        by_depth_model.to_csv(OUT_DIR / f"per_depth_accuracy_by_model_{label}.csv", index=False)

        by_question = (
            subset.groupby(["dataset", "question"])["correct"]
            .agg(n_correct="sum", n_total="count")
            .reset_index()
        )
        by_question["accuracy"] = by_question["n_correct"] / by_question["n_total"]
        by_question = by_question.sort_values(["dataset", "n_total"], ascending=[True, False])
        by_question.to_csv(OUT_DIR / f"per_question_accuracy_{label}.csv", index=False)

    print("Wrote per-depth and per-question CSVs to", OUT_DIR)
    return resolved, primary


def check_discussion_claims(df: pd.DataFrame):
    """Directly check the specific GTSRB claims made in the paper's Discussion section."""
    gtsrb = df[(df["dataset"] == "GTSRB") & df["resolved"]]
    print("\n=== Fact-check: Discussion section claims (GTSRB, all configs) ===")
    for keyword in ["number", "pedestrian"]:
        subset = gtsrb[gtsrb["question"].str.contains(keyword, case=False, na=False)]
        if subset.empty:
            print(f"  No decisions found for questions containing '{keyword}'.")
            continue
        by_q = (
            subset.groupby("question")["correct"]
            .agg(n_correct="sum", n_total="count")
            .assign(accuracy=lambda d: d["n_correct"] / d["n_total"])
            .sort_values("n_total", ascending=False)
        )
        print(f"  Questions containing '{keyword}':")
        print(by_q.to_string())


if __name__ == "__main__":
    all_dfs = []
    for name, cfg in DATASETS.items():
        df = analyze_dataset(name, cfg)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise SystemExit("No data found in any dataset DB.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    resolved, primary = summarize(full_df)
    check_discussion_claims(full_df)
