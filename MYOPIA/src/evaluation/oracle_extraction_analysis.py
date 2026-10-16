"""
Oracle re-parse of tree-traversal answers that main.py's strict regex marked
"unparseable," to separate genuine reasoning failure from answer-interface
failure (Plan Cycle 2, Phase A1).

Manual inspection of a sample of unparseable LLaMA responses (see session
notes) showed the failures cluster into a few concrete, recoverable patterns
rather than random noise:
  - number words instead of digits ("thirty" instead of "30", "one" instead of "1")
  - multi-word/hyphenated answers restated without the hyphen or reordered
    ("red and white" / "white-red" instead of "red-white")
  - a small set of adjectival synonyms for shape answers ("circular" for "circle")
  - yes/no questions answered as a full declarative sentence with an explicit
    negation marker instead of the literal word "no" ("is not a living animal")
  - genuine refusals/hedges ("I cannot confidently answer...", "too blurry to
    determine") -- deliberately left unresolved, not force-recovered

The oracle extractor below encodes only these observed, conservative rules; it
does not attempt open-ended fuzzy matching, and any case it can't confidently
resolve is left unresolved exactly like the strict extractor. This keeps the
"recovered" category auditable rather than a black box.

Usage:
    python evaluation/oracle_extraction_analysis.py
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
    "GTSRB": {"db": "results/gtsrb.db", "loader": GTSRBLoader},
    "MNIST": {"db": "results/mnist.db", "loader": MNISTLoader},
    "CIFAR-10": {"db": "results/cifar10.db", "loader": CIFARLoader},
    "MSCOCO": {"db": "results/mscoco.db", "loader": MSCOCOLoader},
}

OUT_DIR = Path("results/tree_error_analysis")

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100", "hundred and twenty": "120",
}
SHAPE_SYNONYMS = {
    "circular": "circle", "triangular": "triangle", "octagonal": "octagon",
    "diamond-shaped": "diamond",
}
HEDGE_MARKERS = [
    "cannot confidently", "unable to", "cannot determine", "can't determine",
    "not enough evidence", "too blurry to", "difficult to determine",
    "cannot identify", "unclear", "ambiguous", "cannot answer",
]
NEGATION_MARKERS = [" not ", "n't ", "not a ", "not capable", "not typically", "not present",
                     "rather than", "instead of"]
STOPWORDS = {"the", "is", "of", "or", "and", "in", "on", "for", "with", "does", "have",
             "has", "are", "typically", "primarily", "object", "image", "digit", "animal"}


def strict_extract(node, text):
    """Exact reproduction of main.py's regex extraction."""
    if not text or not node.answers:
        return None
    sorted_answers = sorted(node.answers, key=len, reverse=True)
    pattern = "(?i)(" + "|".join(f"\\b{re.escape(a)}\\b" for a in sorted_answers) + ")"
    m = re.search(pattern.lower(), text.lower())
    if not m:
        return None
    return next((a for a in sorted_answers if a.lower() == m.group(1).lower()), None)


def oracle_extract(node, text):
    """
    Conservative fallback extraction, applied only when strict_extract fails.
    Returns None (stays unresolved) unless one of the specific observed
    recovery patterns applies with high confidence.
    """
    if not text or not node.answers:
        return None
    lower = text.lower()

    if any(h in lower for h in HEDGE_MARKERS):
        return None  # explicit refusal/hedge -- do not force a recovery

    # 1. "no closed loops" / "none" style zero-count phrasing for numeric-count answers.
    #    Checked BEFORE generic number-word substitution below, since longer
    #    chain-of-thought responses often mention an unrelated number ("two
    #    curved strokes") before stating this actual (possibly zero) conclusion.
    if "0" in node.answers and re.search(r"\bno\b.*\b(loop|line|segment|curve)s?\b", lower):
        return "0"

    # 2. Number-word normalization -- prefer the LAST matching occurrence, since
    #    reasoning-style responses state intermediate facts before their final
    #    conclusion (e.g. "...two curved strokes... so, no closed loops" should
    #    resolve to the final "no", not the earlier "two").
    normalized = lower
    for word, digit in NUMBER_WORDS.items():
        normalized = re.sub(rf"\b{word}\b", digit, normalized)
    sorted_answers = sorted(node.answers, key=len, reverse=True)
    pattern = "(?i)(" + "|".join(f"\\b{re.escape(a)}\\b" for a in sorted_answers) + ")"
    matches = list(re.finditer(pattern.lower(), normalized))
    if matches:
        last = matches[-1].group(1)
        m = next((a for a in sorted_answers if a.lower() == last.lower()), None)
        if m:
            return m

    # 3. Shape-adjective synonyms.
    for syn, canonical in SHAPE_SYNONYMS.items():
        if syn in lower and canonical in node.answers:
            return canonical

    # 4. Multi-word/hyphenated compound answers restated without the hyphen or reordered
    #    (e.g. "red and white" / "white-red" for the valid answer "red-white").
    for ans in node.answers:
        if "-" in ans:
            parts = ans.split("-")
            if len(parts) >= 2 and all(re.search(rf"\b{re.escape(p)}\b", lower) for p in parts):
                return ans

    # 5. Yes/no questions restated as a full sentence instead of the literal word.
    if set(a.lower() for a in node.answers) == {"yes", "no"}:
        has_negation = any(marker in f" {lower} " for marker in NEGATION_MARKERS)
        # Only recover "yes" for a clean declarative sentence with no negation
        # and no hedge (already filtered above); "no" needs an explicit marker.
        if has_negation:
            return "no" if "no" in [a.lower() for a in node.answers] else None
        # A confident, unhedged, non-negated declarative sentence: treat as "yes"
        # -- but only if it actually engages with the question's content (shares
        # a substantive word with it), not a short/garbled non-answer like
        # "choice2" that happens to contain no negation marker either.
        question_words = {w for w in re.findall(r"[a-z]+", (node.question or "").lower())
                            if w not in STOPWORDS and len(w) > 2}
        response_words = {w for w in re.findall(r"[a-z]+", lower) if len(w) > 2}
        has_overlap = bool(question_words & response_words)
        is_sentence_like = len(text.split()) >= 4
        if (has_overlap and is_sentence_like
                and not lower.strip().startswith(("i ", "it depends", "maybe"))):
            return "yes" if "yes" in [a.lower() for a in node.answers] else None

    return None


def classify_path(tree_root, target_class, path):
    """
    Walk the path once with strict extraction (matches main.py/tree_error_analysis.py
    exactly) and, for the first point of failure, additionally try oracle extraction
    to label WHY it failed: wrong-but-parseable / recoverable-by-oracle / unparseable.
    """
    node = tree_root
    for depth, entry in enumerate(path):
        question, raw_answer = entry.get("question"), entry.get("answer")
        if node.question != question:
            return {"depth": depth, "category": "question_mismatch"}

        strict = raw_answer if raw_answer in node.children else strict_extract(node, raw_answer)
        if strict is not None and strict in node.children:
            child = node.children[strict]
            correct = target_class in child.get_all_possible_classes()
            if not correct:
                return {"depth": depth, "category": "wrong_but_parseable"}
            node = child
            continue

        # Strict extraction failed -- try the oracle fallback.
        oracle = oracle_extract(node, raw_answer)
        if oracle is not None and oracle in node.children:
            child = node.children[oracle]
            correct = target_class in child.get_all_possible_classes()
            return {"depth": depth,
                     "category": "recovered_correct" if correct else "recovered_wrong"}

        return {"depth": depth, "category": "genuinely_unparseable"}

    return {"depth": len(path), "category": "fully_resolved_correct"}


def classify_final_class_only(target_class, final_class_str):
    """
    Fallback for rows where LLM_output_with_tree is not the expected JSON list
    of {question, answer} traversal steps but a plain final-answer string (a
    logging-format inconsistency isolated to GPT-4o/GTSRB primary-condition
    rows: 667 of 916 store text like "The number is '20'." instead of a path).
    No per-node history survives for these, so depth-level re-parsing is not
    possible -- the only thing recoverable is whether main.py's own original
    extraction (LLM_output_with_tree_class) reached a valid leaf, and whether
    it was correct. This can only ever yield "fully_resolved_correct",
    "wrong_but_parseable", or "genuinely_unparseable" -- never
    "recovered_correct"/"recovered_wrong"/"question_mismatch", since oracle
    re-parsing requires the raw per-node response text this format lacks.
    """
    try:
        final_class = int(final_class_str)
    except (TypeError, ValueError):
        return {"depth": None, "category": "genuinely_unparseable"}
    if final_class < 0:
        return {"depth": None, "category": "genuinely_unparseable"}
    if final_class == target_class:
        return {"depth": None, "category": "fully_resolved_correct"}
    return {"depth": None, "category": "wrong_but_parseable"}


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
        f"RunId, LLM_output_with_tree, LLM_output_with_tree_class FROM {ANSWERS} "
        "WHERE LLM_output_with_tree IS NOT NULL"
    )
    rows = cur.fetchall()
    conn.close()

    records = []
    n_fallback = 0
    for (class_str, model, temperature, include_memory, include_description, run_id,
         raw, final_class_str) in rows:
        try:
            target_class = int(class_str)
        except (TypeError, ValueError):
            continue
        try:
            path = json.loads(raw)
            has_path = isinstance(path, list) and bool(path)
        except (json.JSONDecodeError, TypeError):
            has_path = False
        if has_path:
            result = classify_path(tree_root, target_class, path)
        else:
            result = classify_final_class_only(target_class, final_class_str)
            n_fallback += 1
        result.update(dataset=name, model=model, temperature=temperature,
                       include_memory=include_memory, include_description=include_description,
                       run_id=run_id)
        records.append(result)

    df = pd.DataFrame.from_records(records)
    print(f"[{name}] paths_classified={len(df)} (final-class-only fallback: {n_fallback})")
    return df


def summarize(df: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "oracle_classification_raw.csv", index=False)

    primary = df[
        (df["temperature"] == 0.7) & (df["include_memory"] == "0")
        & (df["include_description"] == "0") & (df["run_id"] == 0)
    ]

    for label, subset in [("all_configs", df), ("primary_condition", primary)]:
        counts = subset.groupby(["dataset", "category"]).size().unstack(fill_value=0)
        for col in ["fully_resolved_correct", "wrong_but_parseable", "recovered_correct",
                     "recovered_wrong", "genuinely_unparseable", "question_mismatch"]:
            if col not in counts.columns:
                counts[col] = 0
        total = counts.sum(axis=1)
        counts["n_total"] = total
        # "Unresolved" matches tab:inference_cost's existing definition: any path
        # that didn't cleanly resolve to a leaf via strict extraction, including
        # question_mismatch (tree definition edited since these rows were generated).
        counts["orig_unresolved_rate"] = (
            counts["question_mismatch"] + counts["genuinely_unparseable"]
            + counts["recovered_correct"] + counts["recovered_wrong"]
        ) / total
        counts["oracle_unresolved_rate"] = (
            counts["question_mismatch"] + counts["genuinely_unparseable"]
        ) / total
        counts["orig_tree_accuracy"] = counts["fully_resolved_correct"] / total
        counts["oracle_tree_accuracy"] = (
            counts["fully_resolved_correct"] + counts["recovered_correct"]
        ) / total
        counts.to_csv(OUT_DIR / f"oracle_failure_breakdown_{label}.csv")
        print(f"\n=== {label} ===")
        print(counts[["n_total", "wrong_but_parseable", "recovered_correct", "recovered_wrong",
                       "genuinely_unparseable", "orig_unresolved_rate", "oracle_unresolved_rate",
                       "orig_tree_accuracy", "oracle_tree_accuracy"]].round(4).to_string())

    print("\nWrote", OUT_DIR / "oracle_failure_breakdown_*.csv")


if __name__ == "__main__":
    all_dfs = []
    for name, cfg in DATASETS.items():
        d = analyze_dataset(name, cfg)
        if not d.empty:
            all_dfs.append(d)
    full_df = pd.concat(all_dfs, ignore_index=True)
    summarize(full_df)
