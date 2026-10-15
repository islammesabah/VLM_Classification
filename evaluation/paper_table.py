"""
Emit the extended zero-shot / tree / CoT LaTeX table (tab:zs_vs_tree).

Widens the primary-condition table with the two ablation conditions already
in the databases:

  * history  (Include_memory = 1)      -- tree traversal in chat mode, each
                                          question sees the previous Q&A
  * descriptions (Include_description) -- 1 injects image + class descriptions
                                          into tree inference, 2 injects them
                                          into zero-shot inference

Accuracy uses the same convention as `visualization.py`: correct when
`int(Class) == int(prediction_class)`, denominator = every row of the
condition, so unparsable answers and images with no logged tree traversal
count as wrong. CoT numbers are read from the `results/cot_baseline/` CSVs
(100 images per dataset x model, temperature 0.7 only).

A cell is printed only when its condition covers at least `--min-coverage`
of a full dataset pass; partial conditions print "---" and are listed in the
coverage report on stderr, so the table never mixes a 1000-image accuracy
with a 15-image one.

Usage (from the repository root):

    python evaluation/paper_table.py                 # LaTeX to stdout
    python evaluation/paper_table.py --report        # coverage/back-check only
    python evaluation/paper_table.py -o results/tab_zs_vs_tree_extended.tex
"""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_utils import ANSWERS

DATASETS = ['GTSRB', 'MNIST', 'CIFAR-10', 'MSCOCO']
MODELS = [
    ('gpt-4o', 'GPT-4o'),
    ('meta-llama/llama-3.2-11b-vision-instruct', 'LLaMA'),
    ('qwen-vl-max', 'Qwen-VL'),
]
# label -> (Include_memory, Include_description, which prediction column)
CONDITIONS = [
    ('ZS',     (0, 0, 'without')),
    ('ZS+D',   (0, 2, 'without')),
    ('Tree',   (0, 0, 'with')),
    ('Tree+H', (1, 0, 'with')),
    ('Tree+D', (0, 1, 'with')),
]
COT_DIR = Path('results/cot_baseline')
# The CoT baseline was run against the self-hosted fp16 Ollama endpoint for
# LLaMA (the OpenRouter endpoint that produced the database rows is gone), so
# its CSV slug does not match the LLM_name used everywhere else.
COT_SLUGS = {
    'gpt-4o': 'gpt_4o',
    'qwen-vl-max': 'qwen_vl_max',
    'meta-llama/llama-3.2-11b-vision-instruct': 'llama3.2_vision_11b_instruct_fp16',
}


def cell(conn, model, memory, description, temperature, column):
    """(rows, accuracy, rows_with_a_logged_traversal) for one condition."""
    sql = f"""
        SELECT COUNT(*),
               SUM(CAST(Class AS INTEGER) = CAST(LLM_output_{column}_tree_class AS INTEGER)),
               SUM(LLM_output_with_tree IS NOT NULL)
        FROM {ANSWERS}
        WHERE LLM_name = ?
          AND CAST(Include_memory AS INTEGER) = ?
          AND CAST(Include_description AS INTEGER) = ?
          AND temperature = ?
          AND CAST(RunId AS INTEGER) = 0
    """
    n, correct, with_path = conn.execute(
        sql, (model, memory, description, temperature)).fetchone()
    return n, (None if not n else 100.0 * correct / n), (with_path or 0)


def cot_accuracy(dataset, model):
    """Accuracy from the CoT baseline CSV, or None when that run is absent."""
    if model not in COT_SLUGS:
        return None, 0
    path = COT_DIR / f"{dataset.replace('-', '').lower()}_{COT_SLUGS[model]}_cot.csv"
    matches = [path] if path.exists() else []
    if not matches:
        return None, 0
    with open(matches[0]) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, 0
    correct = sum(r['correct'].strip().lower() == 'true' for r in rows)
    return 100.0 * correct / len(rows), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-path', default='config.json')
    parser.add_argument('--min-coverage', type=float, default=0.9,
                        help='fraction of a full dataset pass a condition must cover to be '
                             'printed (default: 0.9)')
    parser.add_argument('--report', action='store_true',
                        help='print the coverage report only, no LaTeX')
    parser.add_argument('-o', '--out', help='also write the LaTeX to this file')
    args = parser.parse_args()

    config = json.load(open(args.config_path))
    table = {}
    suppressed, incomplete_paths = [], []

    for dataset in DATASETS:
        conn = sqlite3.connect(config[dataset]['database_name'])
        expected = config[dataset]['num_samples']
        for model, label in MODELS:
            for temperature in (0.0, 0.7):
                for name, (memory, description, column) in CONDITIONS:
                    n, acc, with_path = cell(conn, model, memory, description,
                                             temperature, column)
                    if acc is None or n < args.min_coverage * expected:
                        table[(dataset, label, temperature, name)] = None
                        if n:
                            suppressed.append((dataset, label, temperature, name, n, expected, acc))
                        continue
                    table[(dataset, label, temperature, name)] = acc
                    if column == 'with' and with_path < n:
                        incomplete_paths.append((dataset, label, temperature, name, n,
                                                 with_path, acc, 100.0 * acc * n / with_path / 100))
            acc, n = cot_accuracy(dataset, model)
            table[(dataset, label, 0.7, 'CoT')] = acc
            table[(dataset, label, 0.0, 'CoT')] = None
        conn.close()

    if suppressed:
        print('Conditions suppressed for coverage below '
              f'{args.min_coverage:.0%} of a full pass:', file=sys.stderr)
        for ds, m, t, name, n, exp, acc in suppressed:
            print(f'  {ds:9} {m:8} T={t:<4} {name:7} n={n:4}/{exp} '
                  f'(would read {acc:.2f})', file=sys.stderr)
    if incomplete_paths:
        print('\nPrinted tree cells with images that logged no traversal '
              '(counted as wrong):', file=sys.stderr)
        for ds, m, t, name, n, wp, acc, acc_paths in incomplete_paths:
            print(f'  {ds:9} {m:8} T={t:<4} {name:7} {wp}/{n} images have a path; '
                  f'printed {acc:.2f}, paths-only {acc_paths:.2f}', file=sys.stderr)
    if args.report:
        return

    names = [n for n, _ in CONDITIONS]
    fmt = lambda v: '---' if v is None else f'{v:.2f}'

    lines = [
        r'\begin{table*}[ht]',
        r'\centering',
        r'\caption{Mean zero-shot (ZS), tree-based (Tree), and chain-of-thought (CoT) '
        r'accuracy (\%) by dataset, model, and temperature, with the conversation-history '
        r'(+H, \texttt{Include\_memory}) and description (+D, \texttt{Include\_description}) '
        r'ablations. +D injects image and class descriptions into tree inference (Tree+D) or '
        r'into zero-shot inference (ZS+D). A cell is reported only where the condition covers '
        r'at least 90\% of a full dataset pass; "---" marks conditions that were not run to '
        r'completion (all MNIST/MSCOCO $T{=}0$ conditions, and the history ablation on '
        r'MNIST/MSCOCO, where $n \leq 33$). CoT was run at $T{=}0.7$ only '
        r'($n$=100/dataset$\times$model, full class coverage).}',
        r'\footnotesize',
        r'\begin{tabular}{ll' + 'r' * (len(names) * 2 + 1) + '}',
        r'\hline',
        r'& & \multicolumn{%d}{c}{\textbf{T=0}} & \multicolumn{%d}{c}{\textbf{T=0.7}} \\'
        % (len(names), len(names) + 1),
        r'\textbf{Dataset} & \textbf{Model} & '
        + ' & '.join(rf'\textbf{{{n}}}' for n in names) + ' & '
        + ' & '.join(rf'\textbf{{{n}}}' for n in names + ['CoT']) + r' \\ \hline',
    ]
    for dataset in DATASETS:
        for i, (_, label) in enumerate(MODELS):
            head = rf'{dataset:<9}' if i == 0 else ' ' * 9
            cells = [fmt(table[(dataset, label, 0.0, n)]) for n in names]
            cells += [fmt(table[(dataset, label, 0.7, n)]) for n in names + ['CoT']]
            lines.append(f'{head} & {label:<8} & ' + ' & '.join(f'{c:>6}' for c in cells) + r' \\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}', r'\label{tab:zs_vs_tree}', r'\end{table*}']

    latex = '\n'.join(lines)
    print(latex)
    if args.out:
        Path(args.out).write_text(latex + '\n')
        print(f'\nwritten to {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
