"""
plot_subset_sensitivity.py — Plots test F1 vs entropy subset size.

Shows where increasing the subset size stops improving results (the stable plateau),
justifying the chosen subset size for the paper.

Usage:
    python plot_subset_sensitivity.py \\
        --base-dir experiments/subset_sensitivity \\
        --subsets 100 500 1000 2000 5000 10000 20000 full \\
        --output results/subset_sensitivity/subset_sensitivity.png
"""

import argparse
import glob
import json
import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_NAMES   = {"imdb": "IMDb", "sst2": "SST-2"}
STRATEGY_NAMES  = {"RetrainStrategy": "Retrain", "FineTuneStrategy": "Fine-tune"}
STRATEGY_COLORS = {"RetrainStrategy": "#2196F3", "FineTuneStrategy": "#FF5722"}


def load_results(base_dir, subset_label):
    """Load all result JSONs for a given subset size, grouped by dataset and strategy."""
    pattern = os.path.join(base_dir, f"subset_{subset_label}", "**", "*.json")
    results = {}  # (dataset, strategy) -> [f1 scores]
    for path in glob.glob(pattern, recursive=True):
        try:
            d = json.load(open(path))
            dataset  = d["cfg"]["data"]
            strategy = d["cfg"]["strategy_class"]
            f1 = d.get("final_test_stats", {}).get("f1_score")
            if f1 is not None:
                results.setdefault((dataset, strategy), []).append(f1)
        except Exception:
            continue
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="experiments/subset_sensitivity")
    parser.add_argument("--subsets", nargs="+",
                        default=["100","500","1000","2000","5000","10000","20000","full"])
    parser.add_argument("--output", default="results/subset_sensitivity/subset_sensitivity.png")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Collect all results
    data = {}  # (dataset, strategy) -> {subset_label -> [f1 scores]}
    for subset in args.subsets:
        results = load_results(args.base_dir, subset)
        for (dataset, strategy), f1s in results.items():
            data.setdefault((dataset, strategy), {})[subset] = f1s

    datasets   = sorted(set(d for d, _ in data.keys()))
    strategies = sorted(set(s for _, s in data.keys()))

    x_labels = args.subsets
    x_pos    = list(range(len(x_labels)))

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    stable_per_dataset = {}

    for ax, dataset in zip(axes, datasets):
        for strategy in strategies:
            key = (dataset, strategy)
            if key not in data:
                continue
            color = STRATEGY_COLORS.get(strategy, "gray")
            label = STRATEGY_NAMES.get(strategy, strategy)

            means, stds = [], []
            for subset in x_labels:
                f1s = data[key].get(subset, [])
                means.append(np.mean(f1s) * 100 if f1s else np.nan)
                stds.append(np.std(f1s) * 100 if f1s else 0)

            means = np.array(means)
            stds  = np.array(stds)

            ax.plot(x_pos, means, marker="o", linewidth=2, color=color, label=label)
            ax.fill_between(x_pos, means - stds, means + stds,
                            alpha=0.15, color=color)

            # Find stable point for this strategy (change < 0.1 F1)
            for i in range(1, len(means)):
                if not np.isnan(means[i]) and not np.isnan(means[i-1]):
                    if abs(means[i] - means[i-1]) < 0.1:
                        stable_per_dataset.setdefault(dataset, x_labels[i-1])
                        break

        # Draw vertical line at stable point (most conservative across strategies)
        stable = stable_per_dataset.get(dataset)
        if stable and stable in x_labels:
            ax.axvline(x=x_labels.index(stable), color="red", linestyle="--",
                       alpha=0.7, label=f"Plateau at {stable}")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.set_xlabel("Entropy Candidate Subset Size", fontsize=12)
        ax.set_ylabel("Test F1 (%)", fontsize=12)
        ax.set_title(DATASET_NAMES.get(dataset, dataset), fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle("Entropy Sampling Stability: F1 vs Candidate Subset Size\n"
                 "(Retrain & Fine-tune, 3 seeds, DistilBERT)", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # Choose stable subset: most conservative (largest) stable point across all datasets
    all_stable = [v for v in stable_per_dataset.values() if v in x_labels]
    if all_stable:
        # Pick the largest stable point seen across datasets
        stable_subset = sorted(all_stable, key=lambda s: int(s) if s != "full" else 999999)[-1]
    else:
        stable_subset = x_labels[-1]

    stable_size = 999999 if stable_subset == "full" else int(stable_subset)

    stable_path = Path(args.output).parent / "stable_subset.json"
    with open(stable_path, "w") as f:
        json.dump({"stable_subset_size": stable_size,
                   "stable_subset_label": stable_subset}, f, indent=2)
    print(f"Stable subset size chosen: {stable_subset} → saved to {stable_path}")


if __name__ == "__main__":
    main()
