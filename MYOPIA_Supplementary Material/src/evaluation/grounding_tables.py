"""
Tables 4 and 5 of the main paper, from the stored knowledge-verification CSVs.

Table 4: label-conditioned (L) vs image-conditioned (I) node-question accuracy
per dataset and model, with a mean row. Table 5: GTSRB image-conditioned
accuracy by node depth. Both read the probe CSVs written by
evaluation/label_conditioned_knowledge_verification.py and
evaluation/image_grounded_knowledge_verification.py; no API access is needed.

Accuracy conventions: micro = mean over all node-question rows; macro = mean
of per-class means. Both are written to the CSVs; the LaTeX emits the paper's
stated convention -- macro for the label-conditioned column (Sec. 5.3) and
micro for the image-conditioned column (Table 5's Overall row equals Table
4's I column under micro).

Usage (from the repository root):
    python evaluation/grounding_tables.py
"""
import argparse
from pathlib import Path

import pandas as pd

OUT_DIR = Path('results/grounding')
DATASETS = ['GTSRB', 'MNIST', 'CIFAR-10', 'MSCOCO']
FILE_PREFIX = {'GTSRB': 'gtsrb', 'MNIST': 'mnist', 'CIFAR-10': 'cifar10', 'MSCOCO': 'mscoco'}
MODELS = [
    ('gpt-4o', 'gpt_4o', 'GPT-4o'),
    ('llama3.2-vision:11b-instruct-fp16', 'llama3.2_vision:11b_instruct_fp16', 'LLaMA'),
    ('qwen-vl-max', 'qwen_vl_max', 'Qwen-VL'),
]
DIRS = {
    'label': Path('results/label_conditioned_verification'),
    'image': Path('results/image_grounded_verification'),
}
SUFFIX = {'label': 'label_conditioned', 'image': 'image_grounded'}


def load(kind):
    """Concatenate the 12 paper-model probe CSVs for one condition."""
    frames = []
    for dataset in DATASETS:
        for _, fname, label in MODELS:
            path = DIRS[kind] / f'{FILE_PREFIX[dataset]}_{fname}_{SUFFIX[kind]}.csv'
            df = pd.read_csv(path)
            df['dataset'] = dataset
            df['model_label'] = label
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['correct'] = df['correct'].astype(str).str.strip().str.lower() == 'true'
    return df


def accuracies(df):
    """Micro and macro accuracy per (dataset, model)."""
    micro = df.groupby(['dataset', 'model_label'])['correct'].mean() * 100
    macro = (df.groupby(['dataset', 'model_label', 'true_class'])['correct'].mean()
               .groupby(['dataset', 'model_label']).mean() * 100)
    n = df.groupby(['dataset', 'model_label'])['correct'].size()
    return pd.DataFrame({'micro': micro, 'macro': macro, 'n': n}).reset_index()


def table4(label_df, image_df):
    l = accuracies(label_df).rename(columns={c: f'label_{c}' for c in ['micro', 'macro', 'n']})
    i = accuracies(image_df).rename(columns={c: f'image_{c}' for c in ['micro', 'macro', 'n']})
    out = l.merge(i, on=['dataset', 'model_label'])
    means = out.groupby('model_label', sort=False)[
        ['label_micro', 'label_macro', 'image_micro', 'image_macro']].mean().reset_index()
    means.insert(0, 'dataset', 'Mean')
    return pd.concat([out, means], ignore_index=True)


def table5(image_df):
    g = image_df[image_df['dataset'] == 'GTSRB']
    acc = g.pivot_table(index='depth', columns='model_label', values='correct',
                        aggfunc='mean') * 100
    n = g.pivot_table(index='depth', columns='model_label', values='correct', aggfunc='size')
    overall = g.groupby('model_label')['correct'].agg(['mean', 'size'])
    return acc, n, overall


def tex_table4(t4):
    order = [label for _, _, label in MODELS]
    lines = [
        r'\begin{table}[t]', r'\centering',
        r'\caption{Label-conditioned accuracy (L) and image-conditioned accuracy (I), '
        r'with all classes represented.}',
        r'\label{tab:grounding}', r'\footnotesize',
        r'\begin{tabular}{@{}l' + 'rr' * len(order) + r'@{}}', r'\toprule',
        ' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{\textbf{{{m}}}}}' for m in order) + r' \\',
        r'\textbf{Dataset}' + ' & L & I' * len(order) + r' \\', r'\midrule',
    ]
    for dataset in DATASETS + ['Mean']:
        cells = []
        for m in order:
            row = t4[(t4['dataset'] == dataset) & (t4['model_label'] == m)].iloc[0]
            cells += [f"{row['label_macro']:.2f}", f"{row['image_micro']:.2f}"]
        prefix = r'\textbf{Mean}' if dataset == 'Mean' else dataset
        if dataset == 'Mean':
            lines.append(r'\midrule')
            cells = [rf'\textbf{{{c}}}' for c in cells]
        lines.append(prefix + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def tex_table5(acc, overall):
    order = [label for _, _, label in MODELS]
    lines = [
        r'\begin{table}[t]', r'\centering',
        r'\caption{GTSRB image-conditioned accuracy by depth. Rows show depths 0--4; '
        r'Overall includes all depths.}',
        r'\label{tab:gtsrb_depth}', r'\footnotesize',
        r'\begin{tabular}{@{}l' + 'r' * len(order) + r'@{}}', r'\toprule',
        r'\textbf{Depth} & ' + ' & '.join(rf'\textbf{{{m}}}' for m in order) + r' \\',
        r'\midrule',
    ]
    for depth in sorted(acc.index):
        if depth > 4:
            break
        lines.append(f'{depth} & ' + ' & '.join(f'{acc.loc[depth, m]:.1f}' for m in order) + r' \\')
    lines.append(r'\midrule')
    lines.append(r'\textbf{Overall} & ' + ' & '.join(
        f"{overall.loc[m, 'mean'] * 100:.2f}" for m in order) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--out-dir', default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    label_df = load('label')
    image_df = load('image')

    t4 = table4(label_df, image_df)
    t4.to_csv(out / 'table4_label_vs_image.csv', index=False)
    (out / 'tab_grounding.tex').write_text(tex_table4(t4) + '\n')
    print('Table 4 (micro / macro):')
    print(t4.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    acc, n, overall = table5(image_df)
    t5 = acc.round(2)
    t5.columns = [f'{c}_acc' for c in t5.columns]
    for c in n.columns:
        t5[f'{c}_n'] = n[c]
    t5.to_csv(out / 'table5_gtsrb_by_depth.csv')
    (out / 'tab_gtsrb_depth.tex').write_text(tex_table5(acc, overall) + '\n')
    print('\nTable 5 (GTSRB image-conditioned by depth):')
    print(acc.round(1).to_string())
    print('\nPer-depth n:')
    print(n.to_string())
    print('\nOverall:')
    print((overall['mean'] * 100).round(2).to_string())
    print(f'\nWritten to {out}/')


if __name__ == '__main__':
    main()
