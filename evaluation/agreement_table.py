"""
Table 2 of the main paper (instance-level Tree--ZS agreement) and the
class-level pooled comparison counts quoted in Sec. 5.1 (e.g. "72 of 129
class--model comparisons on GTSRB").

Primary condition only (T=0.7, no descriptions, no history, RunId 0), pooled
across the three models; per-model breakdowns are written to the CSVs. GTSRB
reads results/gtsrb_merged.db (see results/build_gtsrb_merged.py); all
queries go through db_utils.ANSWERS for the GTSRB legacy-row deduplication.

Usage (from the repository root):
    python evaluation/agreement_table.py [--gtsrb-db results/gtsrb.db]
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_utils import ANSWERS

OUT_DIR = Path('results/agreement')
DBS = {
    'GTSRB': 'results/gtsrb_merged.db',
    'MNIST': 'results/mnist.db',
    'CIFAR-10': 'results/cifar10.db',
    'MSCOCO': 'results/mscoco.db',
}
MODELS = ['gpt-4o', 'meta-llama/llama-3.2-11b-vision-instruct', 'qwen-vl-max']


def load(dbs):
    frames = []
    for dataset, db in dbs.items():
        conn = sqlite3.connect(db)
        sql = f"""
            SELECT LLM_name, Class,
                   LLM_output_without_tree_class AS zs_pred,
                   LLM_output_with_tree_class AS tree_pred
            FROM {ANSWERS}
            WHERE LLM_name IN ({','.join('?' * len(MODELS))})
              AND temperature = 0.7
              AND CAST(Include_memory AS INTEGER) = 0
              AND CAST(Include_description AS INTEGER) = 0
              AND CAST(RunId AS INTEGER) = 0
        """
        df = pd.read_sql_query(sql, conn, params=MODELS)
        conn.close()
        df['dataset'] = dataset
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    def as_int(v):
        try:
            return int(str(v).strip())
        except (TypeError, ValueError):
            return -1

    truth = df['Class'].map(as_int)
    df['zs_correct'] = truth == df['zs_pred'].map(as_int)
    df['tree_correct'] = truth == df['tree_pred'].map(as_int)
    return df


def instance_agreement(df):
    rows = []
    for keys, g in df.groupby(['dataset', 'LLM_name']):
        rows.append(dict(zip(['dataset', 'model'], keys)) | _counts(g))
    for dataset, g in df.groupby('dataset'):
        rows.append({'dataset': dataset, 'model': 'pooled'} | _counts(g))
    out = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(DBS)}
    return out.sort_values(['dataset', 'model'], key=lambda c: c.map(lambda v: order.get(v, v))
                           ).reset_index(drop=True)


def _counts(g):
    return {
        'n': len(g),
        'both_plus': int((g['tree_correct'] & g['zs_correct']).sum()),
        'both_minus': int((~g['tree_correct'] & ~g['zs_correct']).sum()),
        'tree_plus': int((g['tree_correct'] & ~g['zs_correct']).sum()),
        'zs_plus': int((~g['tree_correct'] & g['zs_correct']).sum()),
    }


def class_level_wins(df):
    per_class = (df.groupby(['dataset', 'LLM_name', 'Class'])
                   [['tree_correct', 'zs_correct']].mean().reset_index())
    per_class['winner'] = 'tie'
    per_class.loc[per_class['tree_correct'] > per_class['zs_correct'], 'winner'] = 'tree'
    per_class.loc[per_class['zs_correct'] > per_class['tree_correct'], 'winner'] = 'zs'
    pooled = (per_class.groupby('dataset')['winner'].value_counts().unstack(fill_value=0)
                .reindex(columns=['zs', 'tree', 'tie'], fill_value=0))
    pooled['total'] = pooled.sum(axis=1)
    return per_class, pooled


def tex_table(agg):
    lines = [
        r'\begin{table}[t]', r'\centering',
        r'\caption{Instance-level Tree--ZS agreement at $T=0.7$, pooled across models. '
        r'Both+/Both$-$ denote both correct/incorrect; Tree+/ZS+ denote only '
        r'Tree/ZS correct.}',
        r'\label{tab:agreement}', r'\footnotesize',
        r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'\textbf{Dataset} & \textbf{Both+} & \textbf{Both$-$} & \textbf{Tree+} & \textbf{ZS+} \\',
        r'\midrule',
    ]
    for dataset in DBS:
        row = agg[(agg['dataset'] == dataset) & (agg['model'] == 'pooled')].iloc[0]
        lines.append(f"{dataset} & {row['both_plus']} & {row['both_minus']} & "
                     f"{row['tree_plus']} & {row['zs_plus']}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gtsrb-db', default=DBS['GTSRB'],
                        help='GTSRB database (default: the merged one)')
    parser.add_argument('-o', '--out-dir', default=str(OUT_DIR))
    args = parser.parse_args()
    dbs = dict(DBS, GTSRB=args.gtsrb_db)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load(dbs)
    agg = instance_agreement(df)
    agg.to_csv(out / 'table2_instance_agreement.csv', index=False)
    (out / 'tab_agreement.tex').write_text(tex_table(agg) + '\n')
    print(f'GTSRB source: {dbs["GTSRB"]}')
    print('Table 2 (pooled rows):')
    print(agg[agg['model'] == 'pooled'].to_string(index=False))

    per_class, pooled = class_level_wins(df)
    per_class.to_csv(out / 'class_level_wins.csv', index=False)
    print('\nClass-level pooled comparisons (per dataset, over model x class cells):')
    for dataset, row in pooled.iterrows():
        print(f"  {dataset:9s} zero-shot better in {row['zs']}/{row['total']} "
              f"(tree better in {row['tree']}, ties {row['tie']})")
    print(f'\nWritten to {out}/')


if __name__ == '__main__':
    main()
