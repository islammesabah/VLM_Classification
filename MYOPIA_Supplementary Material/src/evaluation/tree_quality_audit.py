"""
Per-node tree-quality audit: joins static tree structure with in-traversal
VLM accuracy, the label-conditioned/image-grounded knowledge-verification
probes, and cross-model disagreement into one master table per dataset.

Motivation (WACV E&D self-review, "tree quality -> node difficulty ->
traversal accuracy" ask): the paper's Discussion section already argues
qualitatively that the hardest tree-traversal nodes are the finest-grained
perceptual discriminations (e.g. GTSRB's "What arrow is shown?" at 42.9%).
This script makes that argument quantitative and dataset-wide instead of a
few hand-picked examples, using only data already collected (existing
per-node CSVs) plus one static walk of each tree -- no new API calls.

Each row is keyed by `node_path`: the tuple of answers taken from the root to
reach that node, encoded as a "/"-joined string ("" for the root). This
identifies a node by its position, and stays stable even if two nodes in the
same tree were to ask an identical question string -- joining on question
text alone would silently collapse such nodes together.

No such duplicates exist in the current four trees: every internal node has a
distinct question (GTSRB 22, MNIST 8, CIFAR-10 9, MSCOCO 89, each with as many
unique question strings as nodes), so `duplicate_question` is False
throughout. The flag is kept as a guard because the label-conditioned and
image-grounded probes stored only question text, not node_path, so their join
would become ambiguous if a tree revision ever introduced a duplicate.

Only the three models the paper describes as its primary benchmark are
included (gpt-4o, llama3.2-vision:11b-instruct-fp16 as tree/ZS source,
qwen-vl-max). Any other model tag present in raw DBs is excluded here as an
exploratory run, not part of the reported three-model comparison.

Usage:
    python evaluation/tree_quality_audit.py
"""
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.data_loader import CIFARLoader, GTSRBLoader, MNISTLoader, MSCOCOLoader
from db_utils import ANSWERS
from tree_error_analysis import extract_answer

DATASETS = {
    "GTSRB": {"db": "results/gtsrb.db", "loader": GTSRBLoader},
    "MNIST": {"db": "results/mnist.db", "loader": MNISTLoader},
    "CIFAR-10": {"db": "results/cifar10.db", "loader": CIFARLoader},
    "MSCOCO": {"db": "results/mscoco.db", "loader": MSCOCOLoader},
}

PRIMARY_MODELS = {"gpt-4o", "llama3.2-vision:11b-instruct-fp16", "qwen-vl-max"}

OUT_DIR = Path("results/tree_quality_audit")

LABEL_COND_FILES = {
    "GTSRB": "results/label_conditioned_verification/gtsrb_gpt_4o_label_conditioned.csv",
    "MNIST": "results/label_conditioned_verification/mnist_gpt_4o_label_conditioned.csv",
    "CIFAR-10": "results/label_conditioned_verification/cifar10_gpt_4o_label_conditioned.csv",
    "MSCOCO": "results/label_conditioned_verification/mscoco_gpt_4o_label_conditioned.csv",
}
IMAGE_GROUND_FILES = {
    "GTSRB": "results/image_grounded_verification/gtsrb_gpt_4o_image_grounded.csv",
    "MNIST": "results/image_grounded_verification/mnist_gpt_4o_image_grounded.csv",
    "CIFAR-10": "results/image_grounded_verification/cifar10_gpt_4o_image_grounded.csv",
    "MSCOCO": "results/image_grounded_verification/mscoco_gpt_4o_image_grounded.csv",
}

# The wide per-node table above carries GPT-4o's grounding gap only, while its
# in-traversal accuracy pools all three primary models. That mismatch is fine
# for ranking/inspection but not for regression, so the model-matched analysis
# below pairs each model's own grounding gap with its own traversal accuracy.
# The traversal DBs and the verification CSVs tag the same LLaMA deployment
# differently, hence the explicit mapping.
DATASET_SLUGS = {"GTSRB": "gtsrb", "MNIST": "mnist", "CIFAR-10": "cifar10", "MSCOCO": "mscoco"}

MODEL_VERIFICATION_SLUGS = {
    # traversal-DB model tag -> (verification filename slug, verification `model` column value)
    "gpt-4o": ("gpt_4o", "gpt-4o"),
    "llama3.2-vision:11b-instruct-fp16": (
        "llama3.2_vision:11b_instruct_fp16",
        "llama3.2-vision:11b-instruct-fp16",
    ),  # probe CSVs use the underscore/colon filename stem
    "qwen-vl-max": ("qwen_vl_max", "qwen-vl-max"),
}


def entropy_bits(counts):
    """Shannon entropy (bits) of a discrete distribution given raw counts."""
    total = sum(counts)
    if total == 0 or len(counts) < 2:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def walk_static(node, path_answers=()):
    """
    Yield one static-structure record per node in the tree (internal nodes
    only; leaves are skipped since they have no question/split to audit).
    `path_answers` is the tuple of answers taken to reach `node`.
    """
    if node.question is None:
        return
    node_path = "/".join(path_answers)
    child_sizes = {ans: len(child.get_all_possible_classes()) for ans, child in node.children.items()}
    total_reachable = len(node.get_all_possible_classes())
    max_share = max(child_sizes.values()) / total_reachable if total_reachable else 0.0
    yield {
        "node_path": node_path,
        "depth": len(path_answers),
        "question": node.question,
        "branching_factor": len(node.answers),
        "reachable_classes": total_reachable,
        "split_entropy_bits": entropy_bits(list(child_sizes.values())),
        "max_child_share": max_share,
    }
    for ans, child in node.children.items():
        yield from walk_static(child, path_answers + (ans,))


def walk_traversal(tree_root, target_class, path):
    """
    Like tree_error_analysis.walk_path, but also tracks node_path (the
    "/"-joined answer sequence from root) and the extracted answer, so
    decisions can be grouped by node identity rather than question text.
    """
    node = tree_root
    path_answers = ()
    for depth, entry in enumerate(path):
        question = entry.get("question")
        raw_answer = entry.get("answer")
        node_path = "/".join(path_answers)
        if node.question != question:
            yield {"node_path": node_path, "depth": depth, "question": question,
                   "resolved": False, "correct": None, "reason": "question_mismatch"}
            return
        answer = raw_answer if raw_answer in node.children else extract_answer(node, raw_answer)
        if answer is None or answer not in node.children:
            yield {"node_path": node_path, "depth": depth, "question": question,
                   "resolved": False, "correct": None, "reason": "unparseable_answer"}
            return
        child = node.children[answer]
        correct = target_class in child.get_all_possible_classes()
        yield {"node_path": node_path, "depth": depth, "question": question,
               "resolved": True, "correct": correct, "reason": None}
        if not correct:
            return
        node = child
        path_answers = path_answers + (answer,)


def load_traversal_decisions(name, cfg, tree_root):
    db_path = cfg["db"]
    if not Path(db_path).exists():
        print(f"[{name}] DB not found at {db_path}, skipping traversal decisions.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT Class, LLM_name, temperature, Include_memory, Include_description, "
        f"RunId, LLM_output_with_tree FROM {ANSWERS} WHERE LLM_output_with_tree IS NOT NULL"
    )
    rows = cur.fetchall()
    conn.close()

    records = []
    for class_str, model, temperature, include_memory, include_description, run_id, raw in rows:
        if model not in PRIMARY_MODELS:
            continue
        try:
            path = json.loads(raw)
            if not isinstance(path, list) or not path:
                continue
        except (json.JSONDecodeError, TypeError):
            continue
        try:
            target_class = int(class_str)
        except (TypeError, ValueError):
            continue

        for rec in walk_traversal(tree_root, target_class, path):
            rec.update(dataset=name, model=model, temperature=temperature,
                       include_memory=include_memory, include_description=include_description,
                       run_id=run_id)
            records.append(rec)

    return pd.DataFrame.from_records(records)


def _aggregate_node_stats(subset, suffix):
    """Aggregate one config slice of decisions into per-node stats with a column suffix."""
    if subset.empty:
        return pd.DataFrame(columns=["node_path"])

    total = subset.groupby("node_path").size().rename(f"n_decisions_{suffix}")
    unresolved = (~subset["resolved"]).groupby(subset["node_path"]).sum().rename(f"n_unresolved_{suffix}")

    resolved = subset[subset["resolved"]].copy()
    resolved["correct"] = resolved["correct"].astype(bool)
    node_acc = (
        resolved.groupby("node_path")["correct"]
        .agg(**{f"n_resolved_{suffix}": "count", f"n_correct_{suffix}": "sum"})
    )
    node_acc[f"in_traversal_accuracy_{suffix}"] = node_acc[f"n_correct_{suffix}"] / node_acc[f"n_resolved_{suffix}"]

    per_model = resolved.groupby(["node_path", "model"])["correct"].mean().reset_index()
    model_spread = (
        per_model.groupby("node_path")["correct"]
        .agg(**{
            f"n_models_seen_{suffix}": "count",
            f"model_accuracy_spread_{suffix}": lambda s: s.max() - s.min() if len(s) > 1 else float("nan"),
        })
    )

    out = pd.concat([total, unresolved], axis=1).fillna(0)
    out[f"unresolved_rate_{suffix}"] = out[f"n_unresolved_{suffix}"] / out[f"n_decisions_{suffix}"]
    out = out.join(node_acc[[f"n_resolved_{suffix}", f"in_traversal_accuracy_{suffix}"]], how="left")
    out = out.join(model_spread, how="left")
    return out.reset_index()


def per_node_traversal_stats(decisions_df, dataset):
    """
    Computes two config slices per node, mirroring tree_error_analysis.py's
    convention so numbers here reproduce already-published per-question
    figures (e.g. the Discussion section's GTSRB "What arrow is shown?" at
    42.9%, n=1513, comes from the all-configs pool, not primary condition):

      - "primary": T=0.7, no memory, no description, RunId=0 only.
      - "allcfg": every stored config for the three primary models pooled
        (matches tree_error_analysis.py's "all_configs" output).
    """
    d = decisions_df[decisions_df["dataset"] == dataset]
    primary = d[
        (d["temperature"] == 0.7) & (d["include_memory"] == "0")
        & (d["include_description"] == "0") & (d["run_id"] == 0)
    ]

    primary_stats = _aggregate_node_stats(primary, "primary")
    allcfg_stats = _aggregate_node_stats(d, "allcfg")

    if primary_stats.empty and allcfg_stats.empty:
        return pd.DataFrame(columns=["node_path"])
    if primary_stats.empty:
        return allcfg_stats
    if allcfg_stats.empty:
        return primary_stats
    return primary_stats.merge(allcfg_stats, on="node_path", how="outer")


def audit_dataset(name, cfg):
    loader_cls = cfg["loader"]
    tree_root = loader_cls().build_tree()

    static_df = pd.DataFrame(list(walk_static(tree_root)))

    dup_questions = (
        static_df.groupby("question")["node_path"].nunique().pipe(lambda s: s[s > 1])
    )
    if not dup_questions.empty:
        print(f"[{name}] WARNING: {len(dup_questions)} question string(s) reused at "
              f"multiple tree positions; label-conditioned/image-grounded columns "
              f"(joined on question text) are ambiguous for these nodes: "
              f"{list(dup_questions.index)[:5]}{'...' if len(dup_questions) > 5 else ''}")
    static_df["duplicate_question"] = static_df["question"].isin(dup_questions.index)

    decisions_df = load_traversal_decisions(name, cfg, tree_root)
    traversal_stats = per_node_traversal_stats(decisions_df, name) if not decisions_df.empty else pd.DataFrame()

    merged = static_df.merge(traversal_stats, on="node_path", how="left") if not traversal_stats.empty else static_df

    label_path = LABEL_COND_FILES.get(name)
    if label_path and Path(label_path).exists():
        lc = pd.read_csv(label_path)
        lc_by_q = lc.groupby("question")["correct"].mean().rename("label_conditioned_accuracy")
        merged = merged.merge(lc_by_q, on="question", how="left")

    image_path = IMAGE_GROUND_FILES.get(name)
    if image_path and Path(image_path).exists():
        ig = pd.read_csv(image_path)
        ig_by_q = ig.groupby("question")["correct"].mean().rename("image_grounded_accuracy")
        merged = merged.merge(ig_by_q, on="question", how="left")

    if "label_conditioned_accuracy" in merged.columns and "image_grounded_accuracy" in merged.columns:
        merged["grounding_gap"] = merged["label_conditioned_accuracy"] - merged["image_grounded_accuracy"]

    merged.insert(0, "dataset", name)

    matched = build_matched_table(name, cfg, static_df, decisions_df)
    return merged, matched


def load_grounding_by_model(name):
    """
    Per-(question, model) label-conditioned and image-grounded accuracy for one
    dataset, so each model's grounding gap can be paired with that same model's
    traversal accuracy instead of borrowing GPT-4o's.
    """
    ds_slug = DATASET_SLUGS[name]
    frames = []
    for db_tag, (file_slug, csv_tag) in MODEL_VERIFICATION_SLUGS.items():
        lc_path = Path(f"results/label_conditioned_verification/{ds_slug}_{file_slug}_label_conditioned.csv")
        ig_path = Path(f"results/image_grounded_verification/{ds_slug}_{file_slug}_image_grounded.csv")
        if not lc_path.exists() or not ig_path.exists():
            print(f"[{name}] no matched verification pair for {db_tag}, skipping that model.")
            continue

        lc = pd.read_csv(lc_path)
        ig = pd.read_csv(ig_path)
        # Guard against a mislabelled file silently contributing another model's runs.
        lc = lc[lc["model"] == csv_tag]
        ig = ig[ig["model"] == csv_tag]

        merged = pd.merge(
            lc.groupby("question")["correct"].mean().rename("label_conditioned_accuracy"),
            ig.groupby("question")["correct"].mean().rename("image_grounded_accuracy"),
            on="question", how="inner",
        ).reset_index()
        merged["model"] = db_tag
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=["question", "model", "label_conditioned_accuracy",
                                     "image_grounded_accuracy", "grounding_gap"])
    out = pd.concat(frames, ignore_index=True)
    out["grounding_gap"] = out["label_conditioned_accuracy"] - out["image_grounded_accuracy"]
    return out


def per_node_model_traversal_stats(decisions_df, dataset):
    """Per-(node_path, model) in-traversal accuracy, all stored configs pooled."""
    d = decisions_df[(decisions_df["dataset"] == dataset) & decisions_df["resolved"]]
    if d.empty:
        return pd.DataFrame(columns=["node_path", "model"])
    d = d.copy()
    d["correct"] = d["correct"].astype(bool)
    stats = (
        d.groupby(["node_path", "model"])["correct"]
        .agg(n_resolved_model="count", in_traversal_accuracy_model="mean")
        .reset_index()
    )
    return stats


def build_matched_table(name, cfg, static_df, decisions_df):
    """
    One row per (node, model): that node's static structure joined to that
    model's own traversal accuracy and its own grounding gap. This is the
    table the model-matched regression runs on.
    """
    if decisions_df.empty:
        return pd.DataFrame()

    per_model = per_node_model_traversal_stats(decisions_df, name)
    if per_model.empty:
        return pd.DataFrame()

    matched = static_df.merge(per_model, on="node_path", how="inner")

    grounding = load_grounding_by_model(name)
    if grounding.empty:
        return pd.DataFrame()

    matched = matched.merge(grounding, on=["question", "model"], how="left")
    matched.insert(0, "dataset", name)
    return matched


def standardized_regression(df, predictors, target):
    """
    Standardized OLS via the normal equations (no statsmodels/sklearn
    dependency): z-scores each predictor first so coefficients are directly
    comparable across factors measured in different units (bits, counts,
    fractions), then reports coefficient, standard error, t-statistic, and
    two-sided p-value per predictor, plus R^2 and adjusted R^2.

    This is the multivariate complement to the bivariate correlations
    computed elsewhere in this script: it asks whether each factor's
    association with in-traversal accuracy survives controlling for the
    other three, rather than looking at each factor in isolation.
    """
    from scipy import stats

    sub = df[predictors + [target]].dropna().copy()
    n = len(sub)
    X_raw = sub[predictors].values.astype(float)
    X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=1)
    X = np.column_stack([np.ones(n), X_std])
    y = sub[target].values.astype(float)

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    k = X.shape[1]
    dof = n - k
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tvals = beta / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), dof))

    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)

    names = ["Intercept"] + predictors
    print(f"\nStandardized multivariate regression: {target} ~ {' + '.join(predictors)}")
    print(f"n={n}, R2={r2:.4f}, adj-R2={r2_adj:.4f}")
    for nm, b, s, t, p in zip(names, beta, se, tvals, pvals):
        print(f"  {nm:20s} beta={b:+.4f}  se={s:.4f}  t={t:+.3f}  p={p:.4f}")
    return pd.DataFrame({"predictor": names, "beta": beta, "se": se, "t": tvals, "p": pvals})


def clustered_and_mixed_regression(df, predictors, target, group_col="dataset"):
    """
    Robustness check for standardized_regression() above: the node-level
    analysis sample is nested within only a handful of trees (one per
    dataset), so treating the naive OLS's classical, i.i.d.-residual
    standard errors as if nodes were independent observations overstates
    precision. This refits the same standardized specification two ways
    that account for that clustering -- OLS with cluster-robust (sandwich)
    standard errors grouped by `group_col`, and a random-intercept
    mixed-effects model with `group_col` as the grouping factor -- and
    reports both alongside the naive fit for comparison.

    With only a few clusters (one per dataset here), cluster-robust/MixedLM
    standard errors are themselves imprecise, so this is reported as a
    cross-check on whether the naive OLS's conclusions survive accounting
    for dataset-level grouping, not as a strictly superior replacement.
    """
    import statsmodels.formula.api as smf

    sub = df[predictors + [target, group_col, "n_resolved_allcfg"]].dropna().copy()
    n = len(sub)
    n_clusters = sub[group_col].nunique()

    X_raw = sub[predictors].values.astype(float)
    X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=1)
    z_predictors = [f"{p}_z" for p in predictors]
    for i, zp in enumerate(z_predictors):
        sub[zp] = X_std[:, i]

    formula = f"{target} ~ " + " + ".join(z_predictors)

    ols_cluster = smf.ols(formula, data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub[group_col]}
    )
    mixed = smf.mixedlm(formula, data=sub, groups=sub[group_col]).fit(reml=False)
    # Sensitivity: weight nodes by their numbers of scored decisions, so a
    # node measured on 60 traversals counts more than one measured on 3.
    wls = smf.wls(formula, data=sub, weights=sub["n_resolved_allcfg"]).fit()

    print(f"\nCluster-robust OLS (clustered by {group_col}, n={n}, n_clusters={n_clusters}): {formula}")
    print(ols_cluster.summary())
    print(f"\nRandom-intercept mixed-effects model (groups={group_col}, n={n}, n_clusters={n_clusters}): {formula}")
    print(mixed.summary())

    names = ["Intercept"] + predictors
    fe_names = ["Intercept"] + z_predictors
    cluster_out = pd.DataFrame({
        "predictor": names,
        "beta": ols_cluster.params[fe_names].values,
        "se": ols_cluster.bse[fe_names].values,
        "t": ols_cluster.tvalues[fe_names].values,
        "p": ols_cluster.pvalues[fe_names].values,
        "method": f"OLS_cluster_robust_by_{group_col}",
        "n": n,
        "n_clusters": n_clusters,
    })
    mixed_out = pd.DataFrame({
        "predictor": names,
        "beta": mixed.fe_params[fe_names].values,
        "se": mixed.bse_fe[fe_names].values,
        "t": (mixed.fe_params[fe_names] / mixed.bse_fe[fe_names]).values,
        "p": mixed.pvalues[fe_names].values,
        "method": f"MixedLM_random_intercept_{group_col}",
        "n": n,
        "n_clusters": n_clusters,
    })
    wls_out = pd.DataFrame({
        "predictor": names,
        "beta": wls.params[fe_names].values,
        "se": wls.bse[fe_names].values,
        "t": wls.tvalues[fe_names].values,
        "p": wls.pvalues[fe_names].values,
        "method": "WLS_by_n_decisions",
        "n": n,
        "n_clusters": n_clusters,
    })
    print(f"\nWLS weighted by n_resolved_allcfg (n={n}): {formula}")
    for term in fe_names:
        print(f"  {term:28s} beta={wls.params[term]:+.4f}  se={wls.bse[term]:.4f}  p={wls.pvalues[term]:.4g}")
    return pd.concat([cluster_out, mixed_out, wls_out], ignore_index=True)


def model_matched_regression(df, predictors, target):
    """
    The model-matched counterpart to the two functions above. Each row here is
    one (node, model) pair carrying that model's own grounding gap and its own
    traversal accuracy, which removes the mismatch in the pooled analysis
    (GPT-4o's grounding gap regressed on three-model-pooled accuracy).

    Going long introduces a second, stronger source of non-independence: each
    node now contributes up to three rows. We therefore report the naive fit
    alongside three specifications that account for it -- standard errors
    clustered by node, standard errors clustered by dataset, and a
    random-intercept mixed model grouped by node -- and carry model as a fixed
    effect, since the three VLMs are the full set under evaluation rather than
    a sample from a population of models.
    """
    import statsmodels.formula.api as smf

    sub = df[predictors + [target, "dataset", "model", "node_path",
                           "n_resolved_model"]].dropna().copy()
    n = len(sub)
    sub["node_id"] = sub["dataset"] + "::" + sub["node_path"]
    n_nodes = sub["node_id"].nunique()
    n_datasets = sub["dataset"].nunique()

    X_raw = sub[predictors].values.astype(float)
    X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=1)
    z_predictors = [f"{p}_z" for p in predictors]
    for i, zp in enumerate(z_predictors):
        sub[zp] = X_std[:, i]

    formula = f"{target} ~ " + " + ".join(z_predictors) + " + C(model)"
    print(f"\n=== Model-matched analysis: n={n} (node,model) rows, "
          f"{n_nodes} distinct nodes, {n_datasets} datasets ===")
    print(f"Formula: {formula}")

    fits = {
        "OLS_naive": smf.ols(formula, data=sub).fit(),
        "OLS_cluster_by_node": smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["node_id"]}),
        "OLS_cluster_by_dataset": smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["dataset"]}),
        # Sensitivity: weight observations by their numbers of scored decisions.
        "WLS_by_n_resolved": smf.wls(
            formula, data=sub, weights=sub["n_resolved_model"]).fit(),
        "WLS_by_n_resolved_cluster_node": smf.wls(
            formula, data=sub, weights=sub["n_resolved_model"]).fit(
            cov_type="cluster", cov_kwds={"groups": sub["node_id"]}),
    }

    rows = []
    for method, fit in fits.items():
        print(f"\n--- {method} (R2={fit.rsquared:.4f}) ---")
        for term in fit.params.index:
            print(f"  {term:28s} beta={fit.params[term]:+.4f}  "
                  f"se={fit.bse[term]:.4f}  p={fit.pvalues[term]:.4g}")
            rows.append({"predictor": term, "beta": fit.params[term], "se": fit.bse[term],
                         "t": fit.tvalues[term], "p": fit.pvalues[term], "method": method,
                         "r2": fit.rsquared, "n": n, "n_nodes": n_nodes})

    mixed = smf.mixedlm(formula, data=sub, groups=sub["node_id"]).fit(reml=False)
    print(f"\n--- MixedLM_random_intercept_node ---")
    for term in mixed.fe_params.index:
        print(f"  {term:28s} beta={mixed.fe_params[term]:+.4f}  "
              f"se={mixed.bse_fe[term]:.4f}  p={mixed.pvalues[term]:.4g}")
        rows.append({"predictor": term, "beta": mixed.fe_params[term], "se": mixed.bse_fe[term],
                     "t": mixed.fe_params[term] / mixed.bse_fe[term], "p": mixed.pvalues[term],
                     "method": "MixedLM_random_intercept_node", "r2": float("nan"),
                     "n": n, "n_nodes": n_nodes})

    print("\nBivariate correlations (model-matched rows):")
    for p in predictors:
        print(f"  {p:22s} r={sub[p].corr(sub[target]):+.3f}")

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_tables = []
    all_matched = []
    for name, cfg in DATASETS.items():
        print(f"[{name}] auditing tree...")
        table, matched = audit_dataset(name, cfg)
        out_path = OUT_DIR / f"{name.lower().replace('-', '')}_node_table.csv"
        table.to_csv(out_path, index=False)
        print(f"[{name}] wrote {len(table)} nodes to {out_path}")
        all_tables.append(table)
        if not matched.empty:
            all_matched.append(matched)

    summary = pd.concat(all_tables, ignore_index=True)
    summary.to_csv(OUT_DIR / "all_datasets_node_table.csv", index=False)

    scored = summary.dropna(subset=["in_traversal_accuracy_allcfg"])
    if not scored.empty:
        worst = scored.sort_values("in_traversal_accuracy_allcfg").head(15)
        cols = ["dataset", "question", "depth", "branching_factor", "split_entropy_bits",
                "max_child_share", "n_resolved_allcfg", "in_traversal_accuracy_allcfg",
                "n_resolved_primary", "in_traversal_accuracy_primary",
                "label_conditioned_accuracy", "image_grounded_accuracy", "grounding_gap"]
        cols = [c for c in cols if c in worst.columns]
        worst[cols].to_csv(OUT_DIR / "worst_nodes_summary.csv", index=False)
        print(f"Wrote worst_nodes_summary.csv ({len(worst)} nodes)")

        reg = standardized_regression(
            scored,
            predictors=["split_entropy_bits", "branching_factor", "depth", "grounding_gap"],
            target="in_traversal_accuracy_allcfg",
        )
        reg.to_csv(OUT_DIR / "multivariate_regression.csv", index=False)
        print(f"Wrote multivariate_regression.csv")

        robust_reg = clustered_and_mixed_regression(
            scored,
            predictors=["split_entropy_bits", "branching_factor", "depth", "grounding_gap"],
            target="in_traversal_accuracy_allcfg",
            group_col="dataset",
        )
        robust_reg.to_csv(OUT_DIR / "multivariate_regression_clustered_mixed.csv", index=False)
        print(f"Wrote multivariate_regression_clustered_mixed.csv")

    if all_matched:
        matched_all = pd.concat(all_matched, ignore_index=True)
        matched_all.to_csv(OUT_DIR / "all_datasets_node_model_table.csv", index=False)
        print(f"Wrote all_datasets_node_model_table.csv ({len(matched_all)} node-model rows)")

        matched_reg = model_matched_regression(
            matched_all,
            predictors=["split_entropy_bits", "branching_factor", "depth", "grounding_gap"],
            target="in_traversal_accuracy_model",
        )
        matched_reg.to_csv(OUT_DIR / "multivariate_regression_model_matched.csv", index=False)
        print(f"Wrote multivariate_regression_model_matched.csv")

    print("Wrote", OUT_DIR / "all_datasets_node_table.csv")


if __name__ == "__main__":
    main()
