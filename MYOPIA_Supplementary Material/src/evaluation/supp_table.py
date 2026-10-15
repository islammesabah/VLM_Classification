"""
Emit the full supplementary accuracy table (tab:supp_full) in WACV format.

Unlike the main-paper table (evaluation/paper_table.py), which suppresses any
condition covering less than 90% of a full dataset pass, this table reports
EVERY condition that has data at both temperatures, annotating each incomplete
cell with its sample size so the reader can see exactly what was and was not
run to completion.

Accuracy convention is identical to paper_table.py / visualization.py:
correct when `int(Class) == int(prediction_class)`, denominator = every row of
the condition, so unparseable answers and images with no logged tree traversal
count as wrong. Rows are read through db_utils.ANSWERS, which de-duplicates the
legacy GTSRB `all_sequences_training/` paths.

CoT numbers come from results/cot_baseline/ (n=100 per dataset x model,
temperature 0.7 only), so the T=0 CoT cells are structurally empty.

Usage (from the repository root):

    python evaluation/supp_table.py                        # LaTeX to stdout
    python evaluation/supp_table.py --report               # coverage report only
    python evaluation/supp_table.py -o results/paper/tab_supp_full.tex
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
    ('llama3.2-vision:11b-instruct-fp16', 'LLaMA'),
    ('qwen-vl-max', 'Qwen-VL'),
]
# Column order matches the main-paper table: zero-shot block, then tree block.
CONDITIONS = [
    ('ZS',     (0, 0, 'without')),
    ('ZS+D',   (0, 2, 'without')),
    ('Tree',   (0, 0, 'with')),
    ('Tree+H', (1, 0, 'with')),
    ('Tree+D', (0, 1, 'with')),
]
COLUMNS = ['ZS', 'ZS+D', 'CoT', 'Tree', 'Tree+H', 'Tree+D']
TEMPERATURES = [0.0, 0.7]
COT_DIR = Path('results/cot_baseline')
COT_SLUGS = {
    'gpt-4o': 'gpt_4o',
    'qwen-vl-max': 'qwen_vl_max',
    'llama3.2-vision:11b-instruct-fp16': 'llama3.2_vision_11b_instruct_fp16',
}


def cell(conn, model, memory, description, temperature, column):
    """(n, accuracy) for one condition; accuracy is None when the cell is empty."""
    sql = f"""
        SELECT COUNT(*),
               SUM(CAST(Class AS INTEGER) = CAST(LLM_output_{column}_tree_class AS INTEGER))
        FROM {ANSWERS}
        WHERE LLM_name = ?
          AND CAST(Include_memory AS INTEGER) = ?
          AND CAST(Include_description AS INTEGER) = ?
          AND temperature = ?
          AND CAST(RunId AS INTEGER) = 0
    """
    n, correct = conn.execute(
        sql, (model, memory, description, temperature)).fetchone()
    return n, (None if not n else 100.0 * (correct or 0) / n)


def cot_cell(dataset, model):
    """(n, accuracy) from the CoT baseline CSV, or (0, None) when absent."""
    if model not in COT_SLUGS:
        return 0, None
    path = COT_DIR / f"{dataset.replace('-', '').lower()}_{COT_SLUGS[model]}_cot.csv"
    if not path.exists():
        return 0, None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, None
    correct = sum(r['correct'].strip().lower() == 'true' for r in rows)
    return len(rows), 100.0 * correct / len(rows)


def collect(config):
    """{(dataset, label, temperature, condition): (n, accuracy)} plus expected n."""
    table, expected = {}, {}
    for dataset in DATASETS:
        conn = sqlite3.connect(config[dataset]['database_name'])
        expected[dataset] = config[dataset]['num_samples']
        for model, label in MODELS:
            for temperature in TEMPERATURES:
                for name, (memory, description, column) in CONDITIONS:
                    table[(dataset, label, temperature, name)] = cell(
                        conn, model, memory, description, temperature, column)
            table[(dataset, label, 0.7, 'CoT')] = cot_cell(dataset, model)
            table[(dataset, label, 0.0, 'CoT')] = (0, None)
        conn.close()
    return table, expected


def render(table, expected, min_coverage, mark_best):
    def fmt(dataset, label, temperature, name, best, second):
        n, acc = table[(dataset, label, temperature, name)]
        if acc is None:
            return '---'
        body = f'{acc:.2f}'
        if mark_best and name == best:
            body = rf'\textbf{{{body}}}'
        elif mark_best and name == second:
            body = rf'\underline{{{body}}}'
        # CoT is a deliberate 100-image probe, not an incomplete pass, so its
        # sample size is stated once in the caption rather than on every cell.
        if name != 'CoT' and n < min_coverage * expected[dataset]:
            body += rf'$^{{\,{n}}}$'
        return body

    def rank(dataset, label, temperature):
        """Best / second-best among cells that cover a full pass (CoT included)."""
        reliable = []
        for name in COLUMNS:
            n, acc = table[(dataset, label, temperature, name)]
            if acc is None:
                continue
            if name != 'CoT' and n < min_coverage * expected[dataset]:
                continue
            reliable.append((acc, name))
        reliable.sort(reverse=True)
        best = reliable[0][1] if len(reliable) >= 2 else None
        second = reliable[1][1] if len(reliable) >= 2 else None
        return best, second

    ncol = len(COLUMNS)
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Complete accuracy (\%) for every condition at both temperatures. '
        r'ZS: zero-shot; +D: class and image descriptions injected into the prompt; '
        r'+H: conversation history carried across tree nodes; CoT: chain-of-thought. '
        r'Accuracy counts unparseable answers and images with no logged traversal as '
        r'wrong, so every cell shares the denominator of its condition. Superscripts '
        r'give $n$ for cells that did not complete a full dataset pass '
        rf'($<{min_coverage * 100:.0f}\%$ coverage); an em dash marks conditions with '
        r'no logged rows. '
        r'CoT was run at $T{=}0.7$ only, on a $n{=}100$ probe per dataset--model pair '
        r'with full class coverage. Bold and underlined values denote the best and '
        r'second-best result within each dataset--model--temperature row, considering '
        r'only cells at full coverage. Full passes are $n=901$ (GTSRB), $1000$ (MNIST, '
        r'CIFAR-10), and $850$ (MSCOCO).}',
        r'\label{tab:supp_full}',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{4.6pt}',
        r'\begin{tabular}{@{}ll c ' + 'r' * ncol + r'@{}}',
        r'\toprule',
        r'\textbf{Dataset} & \textbf{Model} & $T$ & '
        + ' & '.join(rf'\textbf{{{c}}}' for c in COLUMNS) + r' \\',
        r'\midrule',
    ]

    for d_i, dataset in enumerate(DATASETS):
        if d_i:
            lines.append(r'\midrule')
        for m_i, (_, label) in enumerate(MODELS):
            if m_i:
                lines.append(rf'\cmidrule(l){{2-{ncol + 3}}}')
            for t_i, temperature in enumerate(TEMPERATURES):
                head = rf'{dataset}' if (m_i == 0 and t_i == 0) else ''
                model_head = label if t_i == 0 else ''
                best, second = rank(dataset, label, temperature)
                cells = [fmt(dataset, label, temperature, c, best, second)
                         for c in COLUMNS]
                lines.append(
                    f'{head:<9} & {model_head:<8} & ${temperature:g}$ & '
                    + ' & '.join(f'{c:>16}' for c in cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


def report(table, expected, min_coverage):
    print('Cell coverage (accuracy / n / fraction of a full pass):', file=sys.stderr)
    for dataset in DATASETS:
        for _, label in MODELS:
            for temperature in TEMPERATURES:
                for name in COLUMNS:
                    n, acc = table[(dataset, label, temperature, name)]
                    if acc is None:
                        flag = 'EMPTY'
                    elif name == 'CoT':
                        flag = 'probe'
                    elif n < min_coverage * expected[dataset]:
                        flag = 'PARTIAL'
                    else:
                        flag = 'full'
                    shown = '  ---' if acc is None else f'{acc:6.2f}'
                    print(f'  {dataset:9} {label:8} T={temperature:<4} {name:7} '
                          f'{shown}  n={n:5}/{expected[dataset]:<5} {flag}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-path', default='config.json')
    parser.add_argument('--min-coverage', type=float, default=0.9,
                        help='coverage below which a cell is annotated with its n '
                             '(default: 0.9)')
    parser.add_argument('--no-mark-best', action='store_true',
                        help='omit the bold/underline best and second-best marking')
    parser.add_argument('--report', action='store_true',
                        help='print the per-cell coverage report only, no LaTeX')
    parser.add_argument('-o', '--out', help='also write the LaTeX to this file')
    args = parser.parse_args()

    config = json.load(open(args.config_path))
    table, expected = collect(config)
    report(table, expected, args.min_coverage)
    if args.report:
        return

    latex = render(table, expected, args.min_coverage, not args.no_mark_best)
    print(latex)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(latex + '\n')
        print(f'\nwritten to {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
