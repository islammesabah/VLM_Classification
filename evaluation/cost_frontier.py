"""
Accuracy-vs-inference-cost frontier for GPT-4o, the paper's primary reference
model: plots mean-accuracy against mean API calls per image for zero-shot
(ZS), tree-based (Tree), zero-shot chain-of-thought (CoT), and CLIP
zero-shot, one point per dataset per method.

All inputs below are numbers already reported elsewhere in the paper (Table
5: ZS/Tree/CoT accuracy at T=0.7, GPT-4o rows; Table 6: mean API calls per
image, pooled across the three primary models -- used here as the GPT-4o
tree-call estimate since a per-model breakdown of calls isn't separately
computed; Table 4: CLIP zero-shot accuracy). ZS and CoT are one call per
image by construction. CLIP zero-shot makes zero VLM API calls (a local,
open-weight model), plotted at calls=0 for reference. This script does not
call any API and reads no external files -- it is a plotting convenience,
not a new measurement.

Usage:
    python evaluation/cost_frontier.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUT_PATH = Path("VLMTreeProbe/Results_plots/cost_frontier.pdf")

# dataset -> {method: (mean_calls_per_image, accuracy_percent)}
DATA = {
    "GTSRB": {
        "CLIP ZS": (0.0, 18.9),
        "ZS": (1.0, 73.36),
        "CoT": (1.0, 63.0),
        "Tree": (3.70, 56.88),
    },
    "MNIST": {
        "CLIP ZS": (0.0, 21.6),
        "ZS": (1.0, 91.50),
        "CoT": (1.0, 94.0),
        "Tree": (2.45, 34.60),
    },
    "CIFAR-10": {
        "CLIP ZS": (0.0, 89.0),
        "ZS": (1.0, 87.90),
        "CoT": (1.0, 87.0),
        "Tree": (3.37, 76.00),
    },
    "MSCOCO": {
        "CLIP ZS": (0.0, 69.6),
        "ZS": (1.0, 72.82),
        "CoT": (1.0, 71.0),
        "Tree": (6.32, 46.71),
    },
}

METHOD_STYLE = {
    "CLIP ZS": dict(marker="D", color="tab:gray"),
    "ZS": dict(marker="o", color="tab:blue"),
    "CoT": dict(marker="^", color="tab:green"),
    "Tree": dict(marker="s", color="tab:red"),
}


def main():
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.0), sharey=True)
    for ax, (dataset, methods) in zip(axes, DATA.items()):
        for method, (calls, acc) in methods.items():
            style = METHOD_STYLE[method]
            ax.scatter(calls, acc, s=55, label=method, **style)
        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel("mean API calls / image", fontsize=8)
        ax.set_xlim(-0.4, 7.2)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("accuracy (%)", fontsize=8)
    axes[0].set_ylim(0, 100)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.12), frameon=False)
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
