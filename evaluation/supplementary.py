"""
Build the complete MYOPIA supplementary document (results/paper/supplementary.tex)
and its figure, then report every number it contains.

The supplementary fulfils the Reproducibility paragraph of the main paper, which
promises "complete T=0, tree-with-history, 10-prompt zero-shot, global-label,
numbered-choice, and run-to-run results, along with full tree-audit
specifications and node counts."  Each of those is one section here:

  S1  scope and conventions (denominators, GTSRB dedup, coverage rule)
  S2  complete accuracy at T=0 and T=0.7, including tree-with-history
  S3  paired-bootstrap CIs and Holm-corrected tests for the Tree-vs-ZS gap
  S4  the ten-prompt zero-shot sweep (full scale) + matched small-scale checks
  S5  repeated-measurement scope (what the RunId axis does and does not cover)
  S6  global-label probe
  S7  numbered-choice probe
  S8  tree specifications and node counts
  S9  tree-quality audit coverage
  S10 reproducibility details (artifacts, sampling, decoding, known pitfalls)

Every table is computed from the databases and probe CSVs at build time, through
db_utils.ANSWERS, so the deduplicated GTSRB row set matches the main paper. S3
additionally reads results/bootstrap_ci_holm.csv; run evaluation/bootstrap_ci.py
first, or that section is omitted.

Usage (from the repository root):

    python evaluation/supplementary.py                 # write .tex + figure
    python evaluation/supplementary.py --report        # print the numbers only
"""

import argparse
import csv
import json
import sqlite3
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_utils import ANSWERS
import supp_table

OUT_TEX = Path('results/paper/supplementary.tex')
OUT_FIG = Path('results/paper/zs_prompt_sweep.pdf')
MODELS = supp_table.MODELS
DATASETS = supp_table.DATASETS


# --------------------------------------------------------------------------- #
# S2/S3: zero-shot prompt sweep
# --------------------------------------------------------------------------- #
def prompt_sweep(config, min_coverage=0.9):
    """Per-prompt zero-shot accuracy: full-scale sweeps and matched small checks."""
    full, matched = {}, {}
    for dataset in DATASETS:
        conn = sqlite3.connect(config[dataset]['database_name'])
        expected = config[dataset]['num_samples']
        for model, label in MODELS:
            rows = conn.execute(f"""
                SELECT CAST(RunId AS INTEGER), COUNT(*),
                       SUM(CAST(Class AS INTEGER) =
                           CAST(LLM_output_without_tree_class AS INTEGER))
                FROM {ANSWERS}
                WHERE LLM_name = ? AND temperature = 0.7
                  AND CAST(Include_memory AS INTEGER) = 0
                  AND CAST(Include_description AS INTEGER) = 0
                  AND RunId IS NOT NULL
                GROUP BY 1 ORDER BY 1""", (model,)).fetchall()
            complete = {r: (n, 100.0 * (k or 0) / n)
                        for r, n, k in rows if n >= min_coverage * expected}
            if len(complete) >= 2:
                acc = [a for _, a in complete.values()]
                full[(dataset, label)] = {
                    'runs': {r: a for r, (_, a) in complete.items()},
                    'n': next(iter(complete.values()))[0],
                    'mean': statistics.mean(acc), 'sd': statistics.stdev(acc),
                    'min': min(acc), 'max': max(acc)}
                continue
            # No full-scale sweep: fall back to the images every RunId shares,
            # so the small checks compare like with like.
            run_ids = [r for r, n, _ in rows if n]
            if len(run_ids) < 2:
                continue
            seen = {}
            for r in run_ids:
                seen[r] = {p for (p,) in conn.execute(
                    f"""SELECT Image_path FROM {ANSWERS}
                        WHERE LLM_name = ? AND temperature = 0.7
                          AND CAST(Include_memory AS INTEGER) = 0
                          AND CAST(Include_description AS INTEGER) = 0
                          AND CAST(RunId AS INTEGER) = ?""", (model, r))}
            common = set.intersection(*seen.values())
            if len(common) < 10:
                continue
            placeholders = ','.join('?' * len(common))
            accs = {}
            for r in run_ids:
                n, k = conn.execute(f"""
                    SELECT COUNT(*), SUM(CAST(Class AS INTEGER) =
                           CAST(LLM_output_without_tree_class AS INTEGER))
                    FROM {ANSWERS}
                    WHERE LLM_name = ? AND temperature = 0.7
                      AND CAST(Include_memory AS INTEGER) = 0
                      AND CAST(Include_description AS INTEGER) = 0
                      AND CAST(RunId AS INTEGER) = ?
                      AND Image_path IN ({placeholders})""",
                    (model, r, *common)).fetchone()
                accs[r] = 100.0 * (k or 0) / n
            acc = list(accs.values())
            matched[(dataset, label)] = {
                'runs': accs, 'n': len(common), 'mean': statistics.mean(acc),
                'sd': statistics.stdev(acc), 'min': min(acc), 'max': max(acc)}
        conn.close()
    return full, matched


def tree_run_coverage(config):
    """How many RunIds carry a tree traversal, per dataset x model."""
    coverage = {}
    for dataset in DATASETS:
        conn = sqlite3.connect(config[dataset]['database_name'])
        for model, label in MODELS:
            rows = conn.execute(f"""
                SELECT CAST(RunId AS INTEGER), COUNT(*),
                       SUM(LLM_output_with_tree IS NOT NULL)
                FROM {ANSWERS}
                WHERE LLM_name = ? AND temperature = 0.7
                  AND CAST(Include_memory AS INTEGER) = 0
                  AND CAST(Include_description AS INTEGER) = 0
                  AND RunId IS NOT NULL
                GROUP BY 1 ORDER BY 1""", (model,)).fetchall()
            coverage[(dataset, label)] = [(r, n, wp or 0) for r, n, wp in rows]
        conn.close()
    return coverage


# --------------------------------------------------------------------------- #
# S4/S5: probes
# --------------------------------------------------------------------------- #
def _load_probe(path):
    df = pd.read_csv(path)
    for col in ('correct', 'resolved'):
        df[col] = df[col].astype(str).str.lower().isin(('true', '1'))
    return df


def probe_summary(path, conditions):
    """Per-dataset node accuracy / unresolved rate / image accuracy, image-matched."""
    df = _load_probe(path)
    out = {}
    for dataset, group in df.groupby('dataset'):
        images = {c: set(s.image_index) for c, s in group.groupby('condition')}
        if set(images) != set(conditions):
            continue
        common = set.intersection(*images.values())
        matched = group[group.image_index.isin(common)]
        row = {'images': len(common)}
        for cond, sub in matched.groupby('condition'):
            final = sub.sort_values('depth').groupby('image_index').tail(1)
            row[cond] = {
                'nodes': len(sub),
                'node_acc': 100.0 * sub.correct.mean(),
                'unresolved': 100.0 * (1 - sub.resolved.mean()),
                'image_acc': 100.0 * final.correct.mean()}
        # Node-level pairing: same image AND same depth in both conditions.
        a = matched[matched.condition == conditions[0]].set_index(['image_index', 'depth'])
        b = matched[matched.condition == conditions[1]].set_index(['image_index', 'depth'])
        keys = a.index.intersection(b.index)
        row['paired'] = {
            'nodes': len(keys),
            conditions[0]: 100.0 * a.loc[keys].correct.mean() if len(keys) else float('nan'),
            conditions[1]: 100.0 * b.loc[keys].correct.mean() if len(keys) else float('nan')}
        out[dataset] = row
    return out


# --------------------------------------------------------------------------- #
# S6/S7: tree specifications and audit coverage
# --------------------------------------------------------------------------- #
def tree_specs():
    nodes = pd.read_csv('results/tree_quality_audit/all_datasets_node_table.csv')
    specs = {}
    for dataset, group in nodes.groupby('dataset'):
        counts = group.branching_factor.value_counts().sort_index()
        reached = group.n_decisions_primary.fillna(0) > 0
        specs[dataset] = {
            'internal': len(group),
            'leaves': int(group.reachable_classes.max()),
            'max_depth': int(group.depth.max()),
            'branching': {int(k): int(v) for k, v in counts.items()},
            'entropy_min': group.split_entropy_bits.min(),
            'entropy_max': group.split_entropy_bits.max(),
            'entropy_mean': group.split_entropy_bits.mean(),
            'unreached': int((~reached).sum()),
            'duplicate_questions': int(group.duplicate_question.sum())}
    return specs


def audit_coverage():
    pairs = pd.read_csv('results/tree_quality_audit/all_datasets_node_model_table.csv')
    complete = pairs.dropna(subset=['in_traversal_accuracy_model', 'grounding_gap'])
    per_dataset = complete.groupby('dataset').agg(
        pairs=('model', 'size'), nodes=('node_path', 'nunique'))
    per_model = complete.groupby('model').size()
    return {
        'rows': len(pairs), 'retained': len(complete),
        'nodes': complete.groupby(['dataset', 'node_path'], dropna=False).ngroups,
        'per_dataset': per_dataset, 'per_model': per_model}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(full, path=OUT_FIG):
    """Per-prompt zero-shot accuracy, one strip per dataset-model, sorted by mean."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                         'axes.linewidth': 0.6})
    keys = sorted(full, key=lambda k: (DATASETS.index(k[0]), k[1]))
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(keys) + 1.0))
    colours = {'GTSRB': '#c0392b', 'CIFAR-10': '#2471a3'}
    for i, key in enumerate(keys):
        stats, y = full[key], len(keys) - 1 - i
        values = list(stats['runs'].values())
        colour = colours.get(key[0], '#555555')
        ax.plot([stats['min'], stats['max']], [y, y], color=colour, lw=1.0,
                alpha=0.45, zorder=1)
        ax.scatter(values, [y] * len(values), s=16, color=colour, alpha=0.75,
                   zorder=2, edgecolors='none')
        ax.scatter([stats['runs'][0]], [y], s=52, marker='D', facecolors='none',
                   edgecolors='black', lw=0.9, zorder=3)
        ax.scatter([stats['mean']], [y], s=46, marker='|', color='black',
                   lw=1.4, zorder=3)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f'{d}  {m}' for d, m in keys][::-1])
    ax.set_xlabel('zero-shot accuracy (\\%)')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', ls=':', lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], marker='o', ls='', color='#555555', ms=4,
                   label='individual prompt'),
        plt.Line2D([], [], marker='D', ls='', mfc='none', mec='black', ms=6,
                   label='prompt 0 (reported in Table 1)'),
        plt.Line2D([], [], marker='|', ls='', color='black', ms=8,
                   label='mean over prompts'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.30),
              ncol=3, frameon=False, fontsize=7, handletextpad=0.4,
              columnspacing=1.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def tex_prompt_sweep(full):
    prompts = sorted({r for s in full.values() for r in s['runs']})
    head = ' & '.join(rf'\textbf{{P{p}}}' for p in prompts)
    lines = [
        r'\begin{table*}[t]', r'\centering',
        r'\caption{Zero-shot accuracy (\%) for each of the ten prompt '
        r'paraphrases at $T{=}0.7$, on the datasets where every prompt was run '
        r'at full scale. P0 is the prompt reported in Table~1 of the main '
        r'paper. Empty cells are prompts whose run did not reach full coverage '
        r'and are excluded from the summary statistics. $\Delta$ is '
        r'$\max-\min$ across prompts.}',
        r'\label{tab:supp_prompt_sweep}', r'\footnotesize',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}ll' + 'r' * len(prompts) + r'@{\hspace{6pt}}rrr@{}}',
        r'\toprule',
        rf'\textbf{{Dataset}} & \textbf{{Model}} & {head} & '
        r'\textbf{Mean} & \textbf{SD} & $\boldsymbol{\Delta}$ \\',
        r'\midrule',
    ]
    last = None
    for dataset, label in sorted(full, key=lambda k: (DATASETS.index(k[0]), k[1])):
        stats = full[(dataset, label)]
        if last is not None and dataset != last:
            lines.append(r'\midrule')
        head_cell = dataset if dataset != last else ''
        last = dataset
        cells = []
        for p in prompts:
            if p not in stats['runs']:
                cells.append('---')
            elif p == 0:
                cells.append(rf"\underline{{{stats['runs'][p]:.2f}}}")
            else:
                cells.append(f"{stats['runs'][p]:.2f}")
        lines.append(
            f'{head_cell:<9} & {label:<8} & ' + ' & '.join(f'{c:>16}' for c in cells)
            + f" & {stats['mean']:.2f} & {stats['sd']:.2f} & "
              f"{stats['max'] - stats['min']:.2f}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


def tex_matched_checks(matched):
    lines = [
        r'\begin{table}[t]', r'\centering',
        r'\caption{Small-scale prompt checks on the datasets without a '
        r'full-scale sweep. Each row compares the prompts that were run on the '
        r'same matched image subset, so the spread is not confounded by which '
        r'images each prompt saw.}',
        r'\label{tab:supp_matched_prompts}', r'\footnotesize',
        r'\begin{tabular}{@{}llrrrrr@{}}', r'\toprule',
        r'\textbf{Dataset} & \textbf{Model} & $n$ & \textbf{Prompts} & '
        r'\textbf{Mean} & \textbf{SD} & $\boldsymbol{\Delta}$ \\',
        r'\midrule',
    ]
    last = None
    for dataset, label in sorted(matched, key=lambda k: (DATASETS.index(k[0]), k[1])):
        s = matched[(dataset, label)]
        head_cell = dataset if dataset != last else ''
        last = dataset
        lines.append(
            f'{head_cell:<9} & {label:<8} & {s["n"]} & {len(s["runs"])} & '
            f'{s["mean"]:.2f} & {s["sd"]:.2f} & {s["max"] - s["min"]:.2f}' + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def tex_probe(summary, conditions, names, label, caption):
    lines = [
        r'\begin{table}[t]', r'\centering',
        rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\footnotesize',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule',
        r'& & \multicolumn{2}{c}{\textbf{Node acc.}} & '
        r'\multicolumn{2}{c}{\textbf{Unresolved}} & '
        r'\multicolumn{2}{c}{\textbf{Image acc.}} \\',
        r'\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(l){7-8}',
        rf'\textbf{{Dataset}} & $n_{{\text{{img}}}}$ & \textbf{{{names[0]}}} & '
        rf'\textbf{{{names[1]}}} & \textbf{{{names[0]}}} & \textbf{{{names[1]}}} & '
        rf'\textbf{{{names[0]}}} & \textbf{{{names[1]}}} \\',
        r'\midrule',
    ]
    for dataset in DATASETS:
        if dataset not in summary:
            continue
        row = summary[dataset]
        a, b = row[conditions[0]], row[conditions[1]]
        lines.append(
            f'{dataset:<9} & {row["images"]} & '
            f'{a["node_acc"]:.1f} & {b["node_acc"]:.1f} & '
            f'{a["unresolved"]:.1f} & {b["unresolved"]:.1f} & '
            f'{a["image_acc"]:.1f} & {b["image_acc"]:.1f}' + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def tex_paired(summary, conditions, names, label, caption):
    lines = [
        r'\begin{table}[t]', r'\centering',
        rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\footnotesize',
        r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'\textbf{Dataset} & \textbf{Paired nodes} & '
        rf'\textbf{{{names[0]}}} & \textbf{{{names[1]}}} & '
        r'$\boldsymbol{\Delta}$ \\',
        r'\midrule',
    ]
    for dataset in DATASETS:
        if dataset not in summary:
            continue
        p = summary[dataset]['paired']
        delta = p[conditions[1]] - p[conditions[0]]
        lines.append(
            f'{dataset:<9} & {p["nodes"]} & {p[conditions[0]]:.1f} & '
            f'{p[conditions[1]]:.1f} & {delta:+.1f}' + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def tex_tree_specs(specs):
    # Full width: the branching column is too wide for a single column.
    lines = [
        r'\begin{table*}[t]', r'\centering',
        r'\caption{Specifications of the four evaluated trees. Branching gives '
        r'the multiset of child counts over internal nodes. Split entropy $H_v$ '
        r'is in bits, unnormalised. "Unreached" counts internal nodes that no '
        r'primary-condition traversal ever visited.}',
        r'\label{tab:supp_tree_specs}', r'\footnotesize',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}lrrrlrrr@{}}', r'\toprule',
        r'\textbf{Dataset} & \textbf{Int.} & \textbf{Leaves} & '
        r'\textbf{Depth} & \textbf{Branching} & '
        r'$\boldsymbol{\bar{H}}$ & $\boldsymbol{H}$ \textbf{range} & '
        r'\textbf{Unreached} \\',
        r'\midrule',
    ]
    total_internal = total_leaves = total_unreached = 0
    for dataset in DATASETS:
        s = specs[dataset]
        branching = ', '.join(f'{v}$\\times${k}' for k, v in s['branching'].items())
        lines.append(
            f'{dataset:<9} & {s["internal"]} & {s["leaves"]} & {s["max_depth"]} & '
            f'{branching} & {s["entropy_mean"]:.2f} & '
            f'{s["entropy_min"]:.2f}--{s["entropy_max"]:.2f} & '
            f'{s["unreached"]}' + r' \\')
        total_internal += s['internal']
        total_leaves += s['leaves']
        total_unreached += s['unreached']
    lines += [
        r'\midrule',
        rf'\textbf{{Total}} & \textbf{{{total_internal}}} & '
        rf'\textbf{{{total_leaves}}} & & & & & \textbf{{{total_unreached}}} \\',
        r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


def tex_audit_coverage(cov):
    per_dataset = cov['per_dataset']
    lines = [
        r'\begin{table}[t]', r'\centering',
        r'\caption{Tree-quality audit coverage. A node--model observation is '
        r'retained when it has both an in-traversal accuracy and a grounding '
        r'gap.}',
        r'\label{tab:supp_audit_coverage}', r'\footnotesize',
        r'\begin{tabular}{@{}lrr@{}}', r'\toprule',
        r'\textbf{Dataset} & \textbf{Nodes} & \textbf{Node--model pairs} \\',
        r'\midrule',
    ]
    for dataset in DATASETS:
        if dataset not in per_dataset.index:
            continue
        row = per_dataset.loc[dataset]
        lines.append(f'{dataset:<9} & {int(row["nodes"])} & {int(row["pairs"])}' + r' \\')
    lines.append(r'\midrule')
    for model, label in MODELS:
        if model in cov['per_model'].index:
            lines.append(rf'\textit{{{label}}} & & {int(cov["per_model"][model])} \\')
    lines += [
        r'\midrule',
        rf'\textbf{{Total}} & \textbf{{{cov["nodes"]}}} & '
        rf'\textbf{{{cov["retained"]}}} \\',
        r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


BOOTSTRAP_CSV = Path('results/bootstrap_ci_holm.csv')


def bootstrap_stats(path=BOOTSTRAP_CSV):
    """Load the paired-bootstrap table written by evaluation/bootstrap_ci.py.

    Returns rows keyed by (dataset, model label) so the renderer can order them
    the same way as every other table in this document. Returns None when the
    CSV has not been generated, so the supplement still builds without it.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline='') as fh:
        return {(r['dataset'], r['model']): r for r in csv.DictReader(fh)}


def tex_bootstrap(rows):
    if not rows:
        return ''
    floor = 2.0 / 10000
    lines = [
        r'\begin{table*}[t]', r'\centering',
        r'\caption{Paired bootstrap over images for the tree-minus-zero-shot '
        r'accuracy gap at $T{=}0.7$ (no descriptions, no history, run~0). '
        r'Zero-shot and tree predictions share an image row, so images are '
        r'resampled with replacement ($10{,}000$ resamples, seed~42) and both '
        r'accuracies are recomputed on each resample. $\Delta$ is '
        r'Tree$-$ZS in points; the interval is the $2.5$--$97.5$ percentile '
        r'range of the resampled $\Delta$. $p_{\text{Holm}}$ is the two-sided '
        r'bootstrap $p$ after Holm--Bonferroni correction over all twelve '
        r'dataset--model comparisons. Entries marked $<$ are floored by '
        r'bootstrap resolution and are upper bounds. Every comparison is '
        r'negative and every corrected $p$ clears $\alpha=0.05$.}',
        r'\label{tab:supp_bootstrap}', r'\footnotesize',
        r'\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{@{}llrrrrcr@{}}', r'\toprule',
        r'\textbf{Dataset} & \textbf{Model} & $n$ & \textbf{ZS} & '
        r'\textbf{Tree} & $\Delta$ & \textbf{95\% CI} & '
        r'$p_{\text{Holm}}$ \\',
        r'\midrule',
    ]
    for i, dataset in enumerate(DATASETS):
        if i:
            lines.append(r'\midrule')
        for _, label in MODELS:
            r = rows.get((dataset, label))
            if r is None:
                continue
            lim = float(r['p_two_sided']) <= floor
            p = ('$<$' if lim else '') + f"{float(r['p_holm_adjusted']):.4f}"
            lines.append(
                f"{dataset} & {label} & {int(r['n'])} & "
                f"{float(r['zs_acc']):.2f} & {float(r['tree_acc']):.2f} & "
                f"${float(r['delta']):.2f}$ & "
                f"$[{float(r['ci_lo']):.2f}, {float(r['ci_hi']):.2f}]$ & "
                f"{p} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


STATS_TEXT = r"""
\section{Significance of the Tree--Zero-Shot Gap}
\label{sec:supp_stats}

Table~\ref{tab:supp_full} shows the tree condition below zero-shot almost
everywhere. Table~\ref{tab:supp_bootstrap} attaches an interval and a
corrected $p$-value to that gap, so the claim rests on more than a point
estimate.

The design is paired: for a given dataset, model, and temperature, the
zero-shot and tree predictions for one image are stored on the same database
row, so the two conditions see identical images. The bootstrap therefore
resamples \emph{images}, not predictions, and recomputes both accuracies on
each resample; the reported interval is over the difference, which absorbs the
correlation between the two conditions rather than treating them as
independent samples.

Two limits are worth stating. The two-sided bootstrap $p$ cannot fall below
$2/10{,}000$, so eleven of the twelve comparisons report that floor and the
corrected values for them are upper bounds rather than exact
quantities. And the interval covers sampling variation over images at a fixed
decoding temperature; it does not cover run-to-run decoding variation, which is
the separate quantity measured in Section~\ref{sec:supp_repeats}.

All twelve gaps are negative, and all twelve remain significant after
Holm--Bonferroni correction over the family of twelve comparisons (largest
adjusted $p=0.0278$, GTSRB with LLaMA). The single smallest gap,
$-3.88$ points for LLaMA on GTSRB, is also the only one whose interval
approaches zero ($[-7.33, -0.44]$); every other interval excludes zero by a
wide margin.
"""


REPRO_TEXT = r"""
\section{Reproducibility Details}
\label{sec:supp_repro}

This section records what a reader needs in order to regenerate the numbers
above, and what cannot be regenerated exactly.

\paragraph{Artifacts in this supplement.} The accompanying archive contains the
four evaluated decision trees in full (both the human-readable and the
machine-readable form), the complete prompt set (zero-shot paraphrases,
per-class descriptions, and the node-question templates), the
generator--evaluator tree-construction pipeline, the evaluation harness, and
the analysis scripts that emit every table in this document. It excludes API
credentials, raw provider traces, and the datasets themselves.

\paragraph{Sampling.} Each dataset is evaluated on a fixed image subset, listed
by identifier in the archive: 901 GTSRB, 1000 MNIST, 1000 CIFAR-10, and 850
MSCOCO images. The MSCOCO subset is restricted to images carrying exactly one
category label, so that a single ground-truth class is well defined. These
identifier lists are the sampling record; the subsets are not redrawn at run
time.

\paragraph{Decoding.} All primary conditions use temperature $0$ or $0.7$ with
provider defaults otherwise, one image per request, and no few-shot examples.
Tree traversal issues one request per visited node; the tree-with-history
ablation additionally carries prior node questions and answers forward in the
conversation.

\paragraph{Determinism and its limits.} The analysis layer is deterministic:
the bootstrap fixes seed~42 with $10{,}000$ resamples, and every table is
recomputed from the stored databases rather than from cached summaries. The
generation layer is not. All three models are served through hosted APIs, so
even $T{=}0$ is not bit-reproducible, and the providers may retire or silently
update a checkpoint. The LLaMA endpoint used for the main runs was in fact
retired during the study and replaced by a self-hosted deployment of the same
official weights at fp16. Re-running the pipeline should therefore reproduce
the qualitative ordering and the sign of every gap in
Table~\ref{tab:supp_bootstrap}, but not the exact percentages.

\paragraph{Recomputation path.} The stored databases are the ground truth for
every table here. Given them, the analysis scripts in the archive regenerate
this entire document, including the figure, without any API access or API key.
Reproducing the databases themselves requires API credentials and paid calls.

\paragraph{Known pitfalls.} Two details silently change results if missed. The
GTSRB deduplication described in Section~\ref{sec:supp_scope} must be applied
or GTSRB denominators inflate from 901 to as many as 915. And accuracy counts
unparseable answers and unlogged traversals as wrong; scoring only the
parseable subset raises every tree number and is not comparable to the values
reported here.
"""


PREAMBLE = r"""% MYOPIA -- supplementary material.
% Generated by evaluation/supplementary.py; edit that script, not this file.
%
% Compiles standalone with pdflatex. To drop into the WACV template instead,
% replace the documentclass/geometry lines below with the wacv.sty preamble
% and delete \title/\author/\maketitle.
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{times}
\usepackage[hidelinks]{hyperref}
\setcounter{table}{0}
\renewcommand{\thetable}{S\arabic{table}}
\setcounter{figure}{0}
\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{section}{0}
\renewcommand{\thesection}{S\arabic{section}}

\title{\vspace{-2em}MYOPIA: Understanding the Limits of Tree-Based Reasoning\\
in Vision-Language Models\\[0.4em]\large Supplementary Material}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-3em}
"""

INTRO = r"""
\section{Scope and Conventions}
\label{sec:supp_scope}

This supplement reports the complete results promised in the Reproducibility
paragraph of the main paper. Section~\ref{sec:supp_full} gives every condition
at both temperatures, including the tree-with-history ablation;
Section~\ref{sec:supp_stats} the paired-bootstrap intervals and corrected
significance tests for the tree--zero-shot gap;
Section~\ref{sec:supp_prompts} the ten-prompt zero-shot sweep;
Section~\ref{sec:supp_repeats} the scope of repeated measurement;
Section~\ref{sec:supp_global} the global-label probe;
Section~\ref{sec:supp_numbered} the numbered-choice probe; and
Sections~\ref{sec:supp_trees}--\ref{sec:supp_audit} the tree specifications,
node counts, and audit coverage. Section~\ref{sec:supp_repro} records the
reproducibility details: the artifacts shipped with this supplement, the fixed
image subsets, the decoding settings, and the limits on exact
re-execution.

Three conventions apply throughout, and are stated here because they determine
how every number below should be read.

\paragraph{Accuracy denominator.} A prediction is correct when the integer
ground-truth class equals the integer predicted class. The denominator is
\emph{every row of the condition}, so an unparseable answer and an image whose
traversal was never logged both count as wrong. Accuracy therefore blends
reasoning failure with answer-interface failure by construction; the two are
separated only in the answer-interface analysis of the main paper and in
Section~\ref{sec:supp_numbered} here.

\paragraph{GTSRB row deduplication.} The GTSRB database contains legacy rows
under \texttt{all\_sequences\_training/} that duplicate images stored under
\texttt{datasets/GTSRB/Training/}. All tables in the main paper and in this
supplement read through a view that drops the legacy copy whenever a matching
modern row exists for the same model, temperature, ablation, and run. Ignoring
this deduplication changes GTSRB denominators from 901 to as many as 915 and
shifts accuracies by up to 0.3 points, so any recomputation that does not
deduplicate will not reproduce these numbers exactly.

\paragraph{Coverage.} A full pass is 901 images for GTSRB, 1000 for MNIST and
CIFAR-10, and 850 for MSCOCO. The main paper prints a cell only when its
condition covers at least 90\% of a full pass. This supplement instead prints
every cell that has data and annotates the incomplete ones with their $n$, so
that what was and was not run to completion is visible rather than hidden.

\section{Complete Results at Both Temperatures}
\label{sec:supp_full}

Table~\ref{tab:supp_full} reports all five prompting conditions at $T{=}0$ and
$T{=}0.7$, extending Table~1 of the main paper with the $T{=}0$ block and the
tree-with-history (Tree+H) ablation.

Two features of the table deserve comment. First, the $T{=}0$ block is complete
only for GTSRB and CIFAR-10. On MNIST and MSCOCO the $T{=}0$ conditions were
not run to completion, and the surviving cells cover between 2 and 860 images;
they are printed with their $n$ but should not be read as dataset-level
estimates. This is the reason the main paper restricts its $T{=}0$ discussion to
GTSRB and CIFAR-10.

Second, Tree+H is a full pass only on GTSRB and CIFAR-10. On MNIST and MSCOCO
it covers 2--33 images, so the apparently strong values there (for example
75.00 for GPT-4o on MNIST, from 16 images) carry no weight. The
history ablation should be read from the GTSRB and CIFAR-10 rows alone, where
it reduces accuracy in five of six comparisons, the single exception being
Qwen-VL on GTSRB.

Best and second-best marking considers only cells at full coverage, so a
partial cell can never be marked as a winner.
"""

PROMPTS_TEXT = r"""
\section{Ten-Prompt Zero-Shot Sweep}
\label{sec:supp_prompts}

The main paper notes that VLMs are sensitive to prompt phrasing and reports
zero-shot results from one prompt. This section gives the full sweep behind
that choice. Ten paraphrases of the zero-shot instruction were written per
dataset; they differ in register and verbosity but request the same output, a
single class ID. The prompt index is stored as \texttt{RunId}, and prompt~0 is
the one reported in Table~1 of the main paper.

Table~\ref{tab:supp_prompt_sweep} and Figure~\ref{fig:supp_prompt_sweep} give
per-prompt accuracy for the six dataset--model pairs where all ten prompts were
run at full scale. Two observations follow, and both qualify the main paper's
zero-shot numbers.

\paragraph{Prompt sensitivity is large, not small.} Across-prompt standard
deviations range from 2.81 points (CIFAR-10, GPT-4o) to 16.97 points (GTSRB,
Qwen-VL), and the max--min spread reaches 41.5 points on GTSRB for Qwen-VL and
31.1 points for LLaMA. Prompt choice is therefore a first-order factor in the
zero-shot number, comparable in magnitude to the differences between models.
The spread is driven partly by catastrophic prompts: Qwen-VL falls to 5.55\% on
one GTSRB paraphrase and 50.1\% on one CIFAR-10 paraphrase, patterns consistent
with format non-compliance rather than a graded loss of recognition ability.

\paragraph{Prompt~0 is the strongest GTSRB prompt, not a typical one.} On
GTSRB, prompt~0 gives the highest accuracy of all sweep prompts for all three
models: 71.14 against a mean of 59.67 for GPT-4o, 37.07 against 16.57 for
LLaMA, and 47.06 against 35.84 for Qwen-VL. On CIFAR-10 it is close to the mean
for GPT-4o (87.90 against 87.08) but above it for Qwen-VL (77.60 against 71.01)
and below it for LLaMA (36.70 against 42.13). The GTSRB zero-shot column of
Table~1 is thus a best-prompt figure rather than a typical-prompt figure.

This matters for the central Tree--zero-shot comparison, because tree traversal
was run only under prompt~0 (Section~\ref{sec:supp_repeats}) and the tree
prompt does not vary with the prompt index. The comparison therefore places a
best-case zero-shot configuration against a single tree configuration on GTSRB.
We report this explicitly rather than reselecting a prompt: the direction of
the gap is unchanged, since even the weakest GTSRB zero-shot prompt for GPT-4o
(45.84) is within 12 points of the tree result and the mean across prompts
(59.67) still exceeds it, but the magnitude of the GTSRB gap should be read as
an upper bound.
"""

REPEATS_TEXT = r"""
\section{Scope of Repeated Measurement}
\label{sec:supp_repeats}

We state precisely what the repeated-measurement axis in our released databases
does and does not cover, since this bounds the stability claims that can be
made from them.

The \texttt{RunId} field selects the zero-shot prompt paraphrase. All ten
paraphrases are distinct in every dataset. Consequently the variation
quantified in Section~\ref{sec:supp_prompts} is \emph{prompt} variation, and it
applies to the zero-shot condition only.

Tree traversal was executed under \texttt{RunId}~0 alone; the tree prompt is
built from the node's own question and answer set and does not depend on the
prompt index. The database therefore contains one tree traversal per
dataset--model--temperature--ablation cell, and no repeated-run replicate of a
tree result exists at any scale. Reported tree accuracies are single-run
measurements, and we make no claim about their run-to-run stability. Readers
should treat the tree column of every table in the main paper and this
supplement accordingly, and we regard replicated tree runs as the most
important missing robustness check in this release.

Where a full-scale sweep is unavailable, Table~\ref{tab:supp_matched_prompts}
reports the small prompt checks that were run on MNIST and MSCOCO, restricted
to the images all prompts share so the spread reflects prompt choice rather
than differing image subsets. Three of the four checks are stable
(SD $\leq 2.9$ points), but MSCOCO with GPT-4o varies more (SD 4.79 points,
range 65.0--75.0 on 20 images). These checks use 20 images each; their standard
errors are wide and they do not substitute for a full-scale sweep.
"""

GLOBAL_TEXT = r"""
\section{Global-Label Probe}
\label{sec:supp_global}

Tree traversal shows the model only the current node's answer options, whereas
zero-shot prompting shows it the entire label set. If that difference in label
access were the primary cause of the Tree--zero-shot gap, restoring the global
label list to each node prompt should recover much of it. This probe tests
that, with GPT-4o, on class-stratified subsamples: each node prompt is issued
twice, once in the standard local-list form and once with the dataset's full
class list appended.

Table~\ref{tab:supp_global} reports both conditions on the images the two
conditions share. The global list produces small and inconsistent changes in
node accuracy: $+1.0$ point on GTSRB, $+0.1$ on MSCOCO, $-1.4$ on CIFAR-10 and
$-5.3$ on MNIST. Image-level accuracy moves by $+4$, $+4$, $-5$, and $0$ points
respectively. No dataset approaches its zero-shot accuracy from Table~1. The
global list also slightly \emph{raises} the unresolved rate on three of four
datasets, consistent with the longer prompt making format compliance harder
rather than easier.

We therefore find no support for label access as the primary cause of the gap.
The probe uses 20--50 images per dataset, so it excludes only a large effect;
differences of a few points are not resolvable at this scale.
"""

NUMBERED_TEXT = r"""
\section{Numbered-Choice Probe}
\label{sec:supp_numbered}

Because the strict extractor marks a response unresolved whenever it fails to
match a predefined answer, part of the tree deficit could be an
answer-interface artefact rather than a reasoning failure. This probe
prospectively controls for that: on the same images, each node is asked twice,
once with the free-form prompt and strict regex extraction, and once with an
explicit numbered-option format that requests only an option number.

Table~\ref{tab:supp_numbered} reports both conditions. The numbered format
eliminates unresolved responses entirely (0.0\% on all four datasets), while
free-form is already close to fully parseable for GPT-4o (0.0--2.4\%
unresolved). Node accuracy moves by at most a few points and not consistently
in one direction.

\paragraph{A caveat on matching.} Each traversal terminates at its first
incorrect or unresolved decision, so the two conditions do not visit the same
number of nodes even on the same image: a condition that fails earlier
accumulates fewer, shallower, and therefore easier nodes. The node accuracies
in Table~\ref{tab:supp_numbered} are computed over each condition's own node
set, which is what the main paper reports.
Table~\ref{tab:supp_numbered_paired} instead restricts to nodes reached at the
same depth on the same image in both conditions. Matching in this stricter
sense increases the MNIST gain from $+3.1$ to $+8.4$ points and deepens the
MSCOCO loss from $-1.5$ to $-3.8$ points, while GTSRB and CIFAR-10 are
essentially unchanged.

\paragraph{A caveat on power.} The probe uses 20 images per dataset, yielding
36--110 node decisions per condition. None of the four differences is
distinguishable from zero (Fisher exact, all $p > 0.75$), and at the image
level the direction reverses on two datasets. The correct reading is that this
probe excludes a \emph{large} interface effect for GPT-4o, not that it measures
a small one. Notably, free-form unresolved rates here are far below GPT-4o's
rate in the full run (9.9\% on GTSRB), which is what supports the main paper's
claim that GPT-4o's interface failures are concentrated in particular
conditions rather than inherent to free-form prompting.
"""

TREES_TEXT = r"""
\section{Tree Specifications and Node Counts}
\label{sec:supp_trees}

Table~\ref{tab:supp_tree_specs} gives the structure of the four evaluated
trees. Every internal node has at least two children, every leaf maps to
exactly one class, and no question repeats along any root-to-leaf path.

The trees differ substantially in shape. CIFAR-10 and MNIST are shallow
(maximum depth 3) with 8--9 internal nodes; MSCOCO is an 89-node binary tree of
depth up to 10; and GTSRB is the most irregular, mixing 16 binary nodes with
two ternary, two five-way, and two eight-way nodes, and reaching depth 14. The
eight-way arrow-direction question on GTSRB is the node an independent LLM
judge flagged for question design, and it is also the highest-entropy split in
any of the trees (3.00 bits).

@MSCOCO_UNREACHED@ MSCOCO internal nodes were never reached by any
primary-condition traversal. They are excluded from the audit because they
have no measured accuracy, which is why the audit covers @AUDIT_NODES@ rather
than @TOTAL_INTERNAL@ nodes.
"""

AUDIT_TEXT = r"""
\section{Tree-Quality Audit Coverage}
\label{sec:supp_audit}

Table~\ref{tab:supp_audit_coverage} shows how the @RETAINED@ node--model
observations of the audit distribute over datasets and models. A node--model
pair enters the audit only when it has both an in-traversal accuracy and a
grounding gap; @DROPPED@ of the @CANDIDATES@ candidate rows lack one of the
two and are dropped.

Coverage is strongly uneven. MSCOCO contributes @MSCOCO_PAIRS@ of the
@RETAINED@ observations (@MSCOCO_PCT@\%) from @MSCOCO_NODES@ nodes, while
MNIST contributes @MNIST_PAIRS@ from @MNIST_NODES@ nodes. Any pooled
association reported in the audit is therefore weighted towards the MSCOCO
tree, which is also the only deep binary tree in the set. This is the main
reason the audit's conclusions are limited to the four evaluated trees and are
not presented as general properties of hierarchical classifiers.

Per-model coverage is @GPT_NODES@ nodes for GPT-4o, @QWEN_NODES@ for Qwen-VL
Max, and @LLAMA_NODES@ for LLaMA-3.2; LLaMA reaches fewer nodes because its
traversals terminate earlier.
Node accuracy in the audit is thus conditional on reach, and nodes deep in a
tree are scored on the subset of traversals that arrived there correctly.
"""


def build_document(config, min_coverage=0.9, make_figure=True):
    table, expected = supp_table.collect(config)
    full, matched = prompt_sweep(config, min_coverage)
    global_probe = probe_summary('results/global_list_probe/results.csv',
                                 ['local_list', 'global_list'])
    numbered = probe_summary('results/constrained_choice_probe/results.csv',
                             ['free_form', 'constrained'])
    specs = tree_specs()
    coverage = audit_coverage()
    boot = bootstrap_stats()

    per_model = coverage['per_model']
    per_dataset = coverage['per_dataset']
    trees_text = (TREES_TEXT
        .replace('@MSCOCO_UNREACHED@',
                 ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six'][specs['MSCOCO']['unreached']])
        .replace('@AUDIT_NODES@', str(coverage['nodes']))
        .replace('@TOTAL_INTERNAL@', str(sum(s['internal'] for s in specs.values()))))
    audit_text = (AUDIT_TEXT
        .replace('@RETAINED@', str(coverage['retained']))
        .replace('@CANDIDATES@', str(coverage['rows']))
        .replace('@DROPPED@', str(coverage['rows'] - coverage['retained']))
        .replace('@MSCOCO_PAIRS@', str(int(per_dataset.loc['MSCOCO', 'pairs'])))
        .replace('@MSCOCO_PCT@', str(round(100 * per_dataset.loc['MSCOCO', 'pairs'] / coverage['retained'])))
        .replace('@MSCOCO_NODES@', str(int(per_dataset.loc['MSCOCO', 'nodes'])))
        .replace('@MNIST_PAIRS@', str(int(per_dataset.loc['MNIST', 'pairs'])))
        .replace('@MNIST_NODES@', str(int(per_dataset.loc['MNIST', 'nodes'])))
        .replace('@GPT_NODES@', str(int(per_model['gpt-4o'])))
        .replace('@QWEN_NODES@', str(int(per_model['qwen-vl-max'])))
        .replace('@LLAMA_NODES@', str(int(per_model['meta-llama/llama-3.2-11b-vision-instruct']))))

    figure_block = ''
    if make_figure:
        build_figure(full)
        figure_block = '\n'.join([
            r'\begin{figure}[t]', r'\centering',
            rf'\includegraphics[width=\linewidth]{{{OUT_FIG.name}}}',
            r'\caption{Zero-shot accuracy for each of the ten prompt '
            r'paraphrases, on the dataset--model pairs swept at full scale. '
            r'Prompt~0, reported in Table~1 of the main paper, is the best of '
            r'the ten on GTSRB for all three models.}',
            r'\label{fig:supp_prompt_sweep}', r'\end{figure}'])

    parts = [
        PREAMBLE, INTRO,
        supp_table.render(table, expected, min_coverage, mark_best=True),
        STATS_TEXT, tex_bootstrap(boot),
        PROMPTS_TEXT, tex_prompt_sweep(full), figure_block,
        REPEATS_TEXT, tex_matched_checks(matched),
        GLOBAL_TEXT,
        tex_probe(global_probe, ['local_list', 'global_list'],
                  ['Local', 'Global'], 'tab:supp_global',
                  r'Global-label probe with GPT-4o, on the images both '
                  r'conditions share. Local is the standard node prompt; '
                  r"Global appends the dataset's full class list to every "
                  r'node prompt. All values are percentages.'),
        NUMBERED_TEXT,
        tex_probe(numbered, ['free_form', 'constrained'],
                  ['Free', 'Num.'], 'tab:supp_numbered',
                  r'Numbered-choice probe with GPT-4o ($n{=}20$ images per '
                  r'dataset). Free is the free-form prompt with strict regex '
                  r'extraction; Num.\ requests only an option number. Node '
                  r"accuracy is computed over each condition's own visited "
                  r'nodes, as in the main paper. All values are percentages.'),
        tex_paired(numbered, ['free_form', 'constrained'], ['Free', 'Num.'],
                   'tab:supp_numbered_paired',
                   r'Numbered-choice probe restricted to nodes reached at the '
                   r'same depth on the same image in both conditions. '
                   r'Matching changes the MNIST and MSCOCO deltas relative to '
                   r'Table~\ref{tab:supp_numbered}.'),
        trees_text, tex_tree_specs(specs),
        audit_text, tex_audit_coverage(coverage),
        REPRO_TEXT,
        r'\end{document}',
    ]
    return '\n\n'.join(p for p in parts if p), {
        'full': full, 'matched': matched, 'global': global_probe,
        'numbered': numbered, 'specs': specs, 'coverage': coverage,
        'bootstrap': boot}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-path', default='config.json')
    parser.add_argument('--min-coverage', type=float, default=0.9)
    parser.add_argument('--report', action='store_true',
                        help='print the computed numbers, write nothing')
    parser.add_argument('-o', '--out', default=str(OUT_TEX))
    args = parser.parse_args()

    config = json.load(open(args.config_path))
    latex, data = build_document(config, args.min_coverage,
                                 make_figure=not args.report)

    print('=== full-scale prompt sweep ===', file=sys.stderr)
    for (ds, m), s in sorted(data['full'].items()):
        print(f'  {ds:9} {m:8} n={s["n"]:5} runs={len(s["runs"]):2} '
              f'mean={s["mean"]:6.2f} sd={s["sd"]:5.2f} '
              f'range=[{s["min"]:.2f},{s["max"]:.2f}] '
              f'prompt0={s["runs"][0]:.2f}', file=sys.stderr)
    print('=== matched small-scale checks ===', file=sys.stderr)
    for (ds, m), s in sorted(data['matched'].items()):
        print(f'  {ds:9} {m:8} n={s["n"]:3} sd={s["sd"]:.2f} '
              f'range=[{s["min"]:.2f},{s["max"]:.2f}]', file=sys.stderr)
    print('=== tree specs ===', file=sys.stderr)
    for ds, s in data['specs'].items():
        print(f'  {ds:9} internal={s["internal"]:3} leaves={s["leaves"]:3} '
              f'depth={s["max_depth"]:2} unreached={s["unreached"]}', file=sys.stderr)
    print(f'=== audit: {data["coverage"]["retained"]} pairs over '
          f'{data["coverage"]["nodes"]} nodes ===', file=sys.stderr)

    if args.report:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex + '\n')
    print(f'\nwritten to {out}', file=sys.stderr)
    print(f'figure written to {OUT_FIG}', file=sys.stderr)


if __name__ == '__main__':
    main()
