import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

with open("results/mean_accuracy_among_experiments.json", "r") as f:
    data = json.load(f)

# Convert data to DataFrame format suitable for plotting
plot_data = []
for dataset in data:
    for model in data[dataset]:
        for run_id, score in data[dataset][model].items():
            plot_data.append({
                'Dataset': dataset,
                'Model': model,
                'Accuracy': score
            })

df = pd.DataFrame(plot_data)

# Set up the plot with academic style
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0
})

# Define colors with opacity for datasets
colors = {
    'CIFAR-10': '#3498db',  # Blue
    'GTSRB': '#e74c3c'     # Red
}

# Alternative version with even more compact layout
plt.rcParams.update({'font.size': 12})
fig2, ax2 = plt.subplots(figsize=(6, 3))

box_plot2 = sns.boxplot(
    data=df,
    y='Model',
    x='Accuracy',
    hue='Dataset',
    palette=[colors['CIFAR-10'], colors['GTSRB']],
    linewidth=0.6,
    fliersize=1.5,
    width=0.5,
    saturation=0.6,
    ax=ax2
)

ax2.set_xlabel('Accuracy', fontsize=12)
ax2.set_ylabel('Model', fontsize=12)
ax2.grid(True, alpha=0.25, axis='x')

# Place legend inside the figure
ax2.legend(title='Dataset', fontsize=10, title_fontsize=8,
           loc='upper right', frameon=True, fancybox=True,
           shadow=True, framealpha=0.9)

plt.tight_layout()

# Save as PDF file
plt.savefig('results/zs_prompting.pdf',
            format='pdf',
            dpi=1200,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
