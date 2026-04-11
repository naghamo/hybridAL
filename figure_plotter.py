"""
Visualization and analysis script for active learning experiment results.

This script provides utilities for loading experiment results from JSON files
and generating publication-quality visualizations and summary statistics for
comparing different active learning strategies across datasets.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import stats as scipy_stats

sns.set_theme(style="whitegrid")

dataset_names = {
    "agnews": "AG News",
    "imdb": "IMDb",
    "jigsaw": "Jigsaw",
    "sst2": "SST-2",
    "tweeteval": "TweetEval",
    "yahoo_answers": "Yahoo Answers",
}
dataset_labels = {
    "agnews": ["World", "Sports", "Business", "Sci/Tech"],
    "imdb": ["Negative", "Positive"],
    "jigsaw": ["Clean", "Toxic"],
    "sst2": ["Negative", "Positive"],
    "tweeteval": ["Negative", "Neutral", "Positive"],
    "yahoo_answers": [
        "Society & Culture", "Science & Math", "Health", "Education & Reference",
        "Computers & Internet", "Sports", "Business & Finance",
        "Entertainment & Music", "Family & Relationships", "Politics & Government"
    ],
}
# Legacy hardcoded switch rounds for old results that predate strategy_metadata tracking
_legacy_switch_rounds = {
    "agnews": {42: 10, 43: 10, 44: 10},
    "imdb": {42: 12, 43: 9, 44: 11},
    "jigsaw": {42: 19, 43: 11, 44: 6},
}
strategy_names = {
    "NewOnlyStrategy": "New only",
    "RetrainStrategy": "Retrain",
    "FineTuneStrategy": "Fine tuning",
    "DeltaF1Strategy": "HybridAL",
    "FixedSwitchStrategy": "Fixed Switch",
}
dpi = 250  # DPI for saving figures
figsize = (6, 4)  # Default figure size


def filter_experiments_df(df: pd.DataFrame, **filter_kwargs):
    """
    Filters the experiments DataFrame for a specific dataset and strategy. If the strategy is 'DeltaF1Strategy',
    it also filters based on provided hyperparameters.

    Args:
        df (pd.DataFrame): DataFrame containing experiment results.
        **filter_kwargs: Keyword arguments for filtering (e.g., dataset, strategy, seed).

    Returns:
        pd.DataFrame: Filtered DataFrame matching all specified criteria.
    """
    mask = pd.Series(True, index=df.index)
    if filter_kwargs:
        for key, value in filter_kwargs.items():
            if key in df.columns and value:
                mask &= (df[key] == value)

    return df.loc[mask]


def get_experiments_df(main_results_path: str):
    """
    Reads all experiment result files from the specified directory and compiles them into a single DataFrame.

    Args:
        main_results_path (str): Path to directory containing experiment JSON files.

    Returns:
        pd.DataFrame: DataFrame with columns including seed, strategy, dataset,
                     test_f1_score, test_accuracy, and round_val_stats.
    """
    all_data = []
    for root, dirs, files in os.walk(main_results_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    data['folder_name'] = os.path.basename(root)
                all_data.append(data)
                break

    df = pd.json_normalize(all_data, sep='_')

    # Core columns always present
    cols_to_keep = [
        'total_rounds', 'round_val_stats',
        'cfg_seed', 'cfg_strategy_class',
        'cfg_strategy_kwargs_epsilon', 'cfg_strategy_kwargs_k',
        'cfg_strategy_kwargs_validation_fraction',
        'cfg_strategy_kwargs_signal',
        'cfg_data', 'cfg_model_name_or_path', 'cfg_sampler_class',
        'final_pool_stats_labeled_count', 'final_pool_stats_unlabeled_count',
        'final_test_stats_loss', 'final_test_stats_f1_score', 'final_test_stats_accuracy',
        'confusion_matrix', 'folder_name',
    ]
    # strategy_metadata columns only present in newer result files
    for meta_col in ['strategy_metadata_switch_round', 'strategy_metadata_switched']:
        if meta_col in df.columns:
            cols_to_keep.append(meta_col)

    # Keep only columns that exist (older results may lack new fields)
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep]

    rename_map = {
        'cfg_seed': 'seed',
        'cfg_strategy_class': 'strategy',
        'cfg_strategy_kwargs_epsilon': 'epsilon',
        'cfg_strategy_kwargs_k': 'k',
        'cfg_strategy_kwargs_validation_fraction': 'validation_fraction',
        'cfg_strategy_kwargs_signal': 'signal',
        'cfg_data': 'dataset',
        'cfg_model_name_or_path': 'model',
        'cfg_sampler_class': 'sampler',
        'final_test_stats_f1_score': 'test_f1_score',
        'final_test_stats_accuracy': 'test_accuracy',
        'final_test_stats_loss': 'test_loss',
        'final_pool_stats_labeled_count': 'labeled_count',
        'final_pool_stats_unlabeled_count': 'unlabeled_count',
        'strategy_metadata_switch_round': 'switch_round',
        'strategy_metadata_switched': 'switched',
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Ensure switch_round column exists even for old results (NaN)
    if 'switch_round' not in df.columns:
        df['switch_round'] = float('nan')

    return df


def get_summary_table(experiments_df: pd.DataFrame, hybrid_hyper: dict):
    """
    Generate summary statistics table for all strategies and datasets.

    Computes mean and standard deviation across seeds for test F1 score,
    accuracy, training time, and total rounds for each strategy-dataset pair.
    For DeltaF1Strategy, uses the specified hyperparameters.

    Args:
        experiments_df (pd.DataFrame): DataFrame from get_experiments_df().
        hybrid_hyper (dict): Hyperparameters for DeltaF1Strategy
                            (keys: 'epsilon', 'k', 'validation_fraction').

    Returns:
        pd.DataFrame: Summary table with formatted statistics including
                     Dataset, Strategy, test scores, training time, and rounds.
    """
    summary_data = []

    for dataset in experiments_df['dataset'].unique():
        for strategy in experiments_df['strategy'].unique():
            is_delta_f1 = (strategy == 'DeltaF1Strategy')

            filtered_df = filter_experiments_df(experiments_df, dataset=dataset, strategy=strategy,
                                                epsilon=hybrid_hyper['epsilon'] if is_delta_f1 else None,
                                                k=hybrid_hyper['k'] if is_delta_f1 else None,
                                                validation_fraction=hybrid_hyper[
                                                    'validation_fraction'] if is_delta_f1 else None)

            training_times = []
            for _, row in filtered_df.iterrows():
                round_val_stats = row['round_val_stats']
                total_time = sum(round['training_time'] for round in round_val_stats)
                training_times.append(total_time)

            avg_f1 = filtered_df['test_f1_score'].mean()*100
            std_f1 = filtered_df['test_f1_score'].std()*100

            avg_accuracy = filtered_df['test_accuracy'].mean()*100
            std_accuracy = filtered_df['test_accuracy'].std()*100

            avg_time = np.mean(training_times)
            std_time = np.std(training_times)

            avg_total_rounds = filtered_df['total_rounds'].mean()
            std_total_rounds = filtered_df['total_rounds'].std()

            summary_data.append({
                'Dataset': dataset_names.get(dataset, dataset),
                'Strategy': strategy_names.get(strategy, strategy),
                'Avg Test Set Macro-F1 Score': f"{avg_f1:.2f}% ± {std_f1:.2f}%",
                'Avg Test Set Accuracy': f"{avg_accuracy:.2f}% ± {std_accuracy:.2f}%",
                'Avg Training Time (sec)': f"{avg_time:.2f} ± {std_time:.2f}",
                'Avg Total Rounds': f"{avg_total_rounds:.2f} ± {std_total_rounds:.2f}"
            })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def plot_f1_vs_time_avg(
        experiments_df: pd.DataFrame,
        hybrid_hyper: dict,
        save_dir_path: str = None,
        cmap: str = "Set2",
        n_grid: int = 250,
        show_individual: bool = False
):
    """
    Plots Macro-F1 vs cumulative training time, averaged across seeds per strategy, with interpolation onto a common
    time grid and a variability band.

    Args:
        experiments_df (pd.DataFrame): DataFrame from get_experiments_df().
        hybrid_hyper (dict): Hyperparameters for DeltaF1Strategy
                            (keys: 'epsilon', 'k', 'validation_fraction').
        save_dir_path (str, optional): Directory to save plots. If None, only displays.
        cmap (str): Matplotlib colormap name for strategy colors (default: "Set2").
        n_grid (int): Number of interpolation points for time grid (default: 250).
        show_individual (bool): Whether to show faint individual seed curves (default: False).
    """

    for dataset in experiments_df['dataset'].unique():
        plt.figure(figsize=figsize)
        color_map = plt.get_cmap(cmap)
        strategies = list(strategy_names.keys())

        for s_i, strategy in enumerate(strategies):
            # Gather the time points and F1 scores for each seed
            seed_curves = []
            for seed in sorted(experiments_df['seed'].unique()):
                is_delta_f1 = (strategy == 'DeltaF1Strategy')

                row = filter_experiments_df(
                    experiments_df, dataset=dataset,
                    strategy=strategy,
                    seed=seed,
                    epsilon=hybrid_hyper['epsilon'] if is_delta_f1 else None,
                    k=hybrid_hyper['k'] if is_delta_f1 else None,
                    validation_fraction=hybrid_hyper['validation_fraction'] if is_delta_f1 else None
                )

                # Build cumulative time curve
                round_val_stats = row['round_val_stats'].values[0]
                times, f1s = [], []
                t = 0.0
                for r in round_val_stats:
                    t += float(r['training_time'])
                    times.append(t)
                    f1s.append(float(r['f1_score']))

                if len(times) >= 2:
                    seed_curves.append((times, f1s))

            # Set the common time grid to be between 0 and the minimum max time across seeds
            max_times = [curve[0][-1] for curve in seed_curves]
            t_max_common = np.min(max_times)

            grid = np.linspace(0.0, t_max_common, n_grid)

            # Interpolate each seed's F1 scores onto the common time grid
            interp_f1s = []
            for times, f1s in seed_curves:
                f_on_grid = np.interp(grid, times, f1s)
                interp_f1s.append(f_on_grid)

                if show_individual:
                    plt.plot(times, f1s, color=color_map(s_i / len(color_map.colors)), alpha=0.25, lw=1)

            interp_f1s = np.vstack(interp_f1s)
            mean_f1 = interp_f1s.mean(axis=0)

            # Calculate standard deviation for the band
            std = interp_f1s.std(axis=0)
            lower, upper = mean_f1 - std, mean_f1 + std

            # Plot mean and band per strategy
            label = strategy_names[strategy]
            color = color_map(s_i / len(color_map.colors))
            plt.plot(grid, mean_f1, label=label, color=color, lw=2.0)
            plt.fill_between(grid, lower, upper, color=color, alpha=0.15, linewidth=0)

        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Validation Set Macro-F1 Score')
        plt.title(f'Validation Set Macro-F1 vs Training Time (mean across seeds) for {dataset_names[dataset]}')
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        if save_dir_path:
            plt.savefig(os.path.join(save_dir_path, f'f1_vs_time_{dataset}.png'), dpi=dpi)
        plt.show()


def plot_hybrid_hyper_variations(experiments_df: pd.DataFrame, best_hybrid_hyper: dict, save_dir_path: str = None,
                                 cmap: str = 'tab10'):
    """
    Plots 3 figures, where in each figure two of the three hybridAL hyperparameters are fixed to the best values,   and the third is varied.
    The figures show the mean test set F1 score vs the varied hyperparameter, with one line per dataset.

    Args:
        experiments_df (pd.DataFrame): DataFrame from get_experiments_df().
        best_hybrid_hyper (dict): Best hyperparameters for DeltaF1Strategy
                                 (keys: 'epsilon', 'k', 'validation_fraction').
        save_dir_path (str, optional): Directory to save plots. If None, only displays.
        cmap (str): Matplotlib colormap for dataset colors (default: 'tab10').
    """
    hyperparams_names = {"epsilon": r"$\varepsilon$",
                         "k": r"$k$",
                         "validation_fraction": r"$c$"}

    for param in best_hybrid_hyper.keys():
        plt.figure(figsize=figsize)
        color_map = plt.get_cmap(cmap)
        y_lim_lower = 0.0
        y_lim_upper = 1.0

        for i, dataset in enumerate(experiments_df['dataset'].unique()):
            subset = experiments_df[
                (experiments_df['strategy'] == 'DeltaF1Strategy') &
                (experiments_df['dataset'] == dataset)
                ]

            for other_param, value in best_hybrid_hyper.items():
                if other_param != param:
                    subset = subset[subset[other_param] == value]

            means = subset.groupby(param)['test_f1_score'].mean()
            plt.plot(means.index, means.values, marker='o', label=dataset_names.get(dataset, dataset),
                     color=color_map(i))

            # Std band (mean +- std)
            stds = subset.groupby(param)['test_f1_score'].std()
            lower, upper = means - stds, means + stds
            plt.fill_between(means.index, lower, upper, color=color_map(i), alpha=0.15, linewidth=0)

            # Buffering the y limits for legend aesthetics
            band_uppers = means + stds
            band_lowers = means - stds
            y_lim_upper = band_uppers.max() + 0.05
            y_lim_lower = band_lowers.min() - 0.05

        plt.xlabel(hyperparams_names[param])
        plt.ylabel('Mean Test Set Macro-F1 Score')
        plt.ylim(y_lim_lower, y_lim_upper)

        title_postfix = ', '.join([f"{hyperparams_names[k]}={v}" for k, v in best_hybrid_hyper.items() if k != param])
        plt.title(f'Mean Test Set Macro-F1 Score vs {hyperparams_names[param]} ({title_postfix})')

        plt.legend(frameon=False, fontsize='small')
        plt.grid(alpha=0.25)
        plt.tight_layout()

        if save_dir_path:
            plt.savefig(os.path.join(save_dir_path, f'hybrid_hyper_variation_{param}.png'), dpi=dpi)

        plt.show()


def plot_test_f1_bar_chart(experiments_df: pd.DataFrame, best_hybrid_hyper: dict, save_path: str = None):
    """
    Plots a bar chart comparing the mean test set macro-F1 scores of different strategies, averaged across all datasets and seeds.

    Args:
        experiments_df (pd.DataFrame): DataFrame from get_experiments_df().
        best_hybrid_hyper (dict): Best hyperparameters for DeltaF1Strategy
                                 (keys: 'epsilon', 'k', 'validation_fraction').
        save_path (str, optional): Full file path to save the plot. If None, only displays.
    """
    results = []

    plt.figure(figsize=figsize)

    # Fixed color mapping to match the f1 vs time plot
    cmap = plt.get_cmap("Set2")
    strategy_colors = {
        "New only": cmap(0),
        "Retrain": cmap(1),
        "Fine tuning": cmap(2),
        "HybridAL": cmap(3),
    }

    for strategy in experiments_df['strategy'].unique():
        info = filter_experiments_df(experiments_df, strategy=strategy,
                                     **best_hybrid_hyper) if strategy == 'DeltaF1Strategy' else filter_experiments_df(
            experiments_df, strategy=strategy)

        mean_f1 = info['test_f1_score'].mean()
        std_f1 = info['test_f1_score'].std()
        name = strategy_names[strategy]

        results.append((name, mean_f1, std_f1))

    # Sort by mean F1 score
    results.sort(key=lambda x: x[1])  # ascending order

    bars = []
    # Plot Sorted bars
    for name, mean_f1, std_f1 in results:
        color = strategy_colors[name]
        bar = plt.bar(name, mean_f1, yerr=std_f1, capsize=5, color=color)
        bars.append((bar, mean_f1, std_f1))

    # Print bar values atop bars
    for bar, mean, std in bars:
        for rect in bar:
            plt.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + std + 0.02,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9
            )

    plt.ylabel('Mean Test Set Macro-F1 Score')
    plt.title('Comparison of Strategies on Mean Test Set Macro-F1 Score')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


def plot_confusion_heatmaps_hybrid(experiments_df: pd.DataFrame, hybrid_hyper: dict, save_dir_path: str = None):
    """
    Plots 3 average normalized confusion matrix heatmaps for the hybrid strategy, one per dataset.

    Args:
        experiments_df (pd.DataFrame): DataFrame from get_experiments_df().
        hybrid_hyper (dict): Hyperparameters for DeltaF1Strategy
                            (keys: 'epsilon', 'k', 'validation_fraction').
        save_dir_path (str, optional): Directory to save plots. If None, only displays.
    """

    for dataset in experiments_df['dataset'].unique():
        relevant_hybrid = filter_experiments_df(
            experiments_df,
            strategy='DeltaF1Strategy',
            dataset=dataset,
            epsilon=hybrid_hyper['epsilon'],
            k=hybrid_hyper['k'],
            validation_fraction=hybrid_hyper['validation_fraction']
        )

        # Sum the confusion matrices
        num_labels = len(relevant_hybrid['confusion_matrix'].values[0])
        sum_cm = np.zeros((num_labels, num_labels))
        for cm in relevant_hybrid['confusion_matrix']:
            sum_cm += np.array(cm)

        # Normalize the summed confusion matrix
        avg_cm = sum_cm / sum_cm.sum(axis=1, keepdims=True)

        # Round and renormalize again to avoid floating point issues
        avg_cm = np.round(avg_cm, 2)
        avg_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            avg_cm, annot=True, fmt=".2f", cmap="Blues",
            cbar=True, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Proportion"}
        )

        # Rename the ticks to be the label meanings
        ax.set_xticklabels(dataset_labels[dataset], rotation=45, ha='right')
        ax.set_yticklabels(dataset_labels[dataset], rotation=0)

        plt.title(f'Normalized Global Confusion Matrix (HybridAL) - {dataset_names[dataset]}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        if save_dir_path:
            plt.savefig(os.path.join(save_dir_path, f'cm_hybrid_{dataset}.png'), dpi=dpi)
        plt.show()


def get_significance_table(experiments_df: pd.DataFrame, hybrid_hyper: dict) -> pd.DataFrame:
    """
    Compute Welch's t-test between DeltaF1Strategy and every other strategy, per dataset.

    Tests the null hypothesis that DeltaF1Strategy and the baseline have equal mean
    test F1 scores. Uses Welch's t-test (unequal variance assumed).

    Args:
        experiments_df (pd.DataFrame): DataFrame from get_experiments_df().
        hybrid_hyper (dict): Hyperparameters for DeltaF1Strategy
                            (keys: 'epsilon', 'k', 'validation_fraction').

    Returns:
        pd.DataFrame: Table with columns: Dataset, Baseline Strategy,
                      HybridAL mean F1, Baseline mean F1, t-statistic, p-value,
                      Significant (p<0.05).
    """
    rows = []

    for dataset in experiments_df['dataset'].unique():
        hybrid_df = filter_experiments_df(
            experiments_df,
            dataset=dataset,
            strategy='DeltaF1Strategy',
            epsilon=hybrid_hyper['epsilon'],
            k=hybrid_hyper['k'],
            validation_fraction=hybrid_hyper['validation_fraction'],
        )
        hybrid_f1 = hybrid_df['test_f1_score'].dropna().values

        for strategy in experiments_df['strategy'].unique():
            if strategy == 'DeltaF1Strategy':
                continue
            baseline_df = filter_experiments_df(experiments_df, dataset=dataset, strategy=strategy)
            baseline_f1 = baseline_df['test_f1_score'].dropna().values

            if len(hybrid_f1) < 2 or len(baseline_f1) < 2:
                continue

            t_stat, p_val = scipy_stats.ttest_ind(hybrid_f1, baseline_f1, equal_var=False)
            rows.append({
                'Dataset': dataset_names.get(dataset, dataset),
                'Baseline Strategy': strategy_names.get(strategy, strategy),
                'HybridAL mean F1': round(hybrid_f1.mean(), 4),
                'Baseline mean F1': round(baseline_f1.mean(), 4),
                'Mean ΔF1': round(hybrid_f1.mean() - baseline_f1.mean(), 4),
                't-statistic': round(t_stat, 3),
                'p-value': round(p_val, 4),
                'Significant (p<0.05)': p_val < 0.05,
            })

    return pd.DataFrame(rows)


def plot_switch_heatmap(experiments_df: pd.DataFrame, save_dir_path: str = None):
    """
    Plot a heatmap of average switch rounds across (epsilon, k) configurations.

    Shows at which round DeltaF1Strategy switched from retraining to fine-tuning,
    averaged across seeds and datasets, for each (epsilon, k) combination.
    Requires experiment results that include strategy_metadata_switch_round
    (i.e., results from updated code with switch_round tracking).

    Args:
        experiments_df (pd.DataFrame): DataFrame from get_experiments_df().
        save_dir_path (str, optional): Directory to save the plot. If None, only displays.
    """
    delta_df = experiments_df[experiments_df['strategy'] == 'DeltaF1Strategy'].copy()
    delta_df = delta_df.dropna(subset=['switch_round'])

    if delta_df.empty:
        print("No switch_round data available. Run new experiments with updated code first.")
        return

    pivot = delta_df.groupby(['epsilon', 'k'])['switch_round'].mean().unstack('k')

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        cbar_kws={"label": "Avg Switch Round"},
        linewidths=0.5
    )
    plt.title("Average Switch Round by Epsilon and k (DeltaF1Strategy)")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\varepsilon$")
    plt.tight_layout()
    if save_dir_path:
        plt.savefig(os.path.join(save_dir_path, "switch_round_heatmap.png"), dpi=dpi)
    plt.show()


def plot_f1_vs_round_switch(experiments_df: pd.DataFrame, best_hybrid_hyper: dict, save_dir_path: str = None):
    """
    Plots 3 plots of test set F1 score vs round number for HybridAL. There's a plot per dataset, and in each plot there is a line per seed. The round that the switch happened is marked.

    Args:
        experiments_df (pd.DataFrame): DataFrame from get_experiments_df().
        best_hybrid_hyper (dict): Best hyperparameters for DeltaF1Strategy
                                 (keys: 'epsilon', 'k', 'validation_fraction').
        save_dir_path (str, optional): Directory to save plots. If None, only displays.
    """

    for dataset in experiments_df['dataset'].unique():
        plt.figure(figsize=figsize)
        cmap = ListedColormap(['#9195F6', '#FB88B4', '#79d955'])

        color_i = 0
        total_rounds = []
        for j in experiments_df['seed'].unique():
            info = filter_experiments_df(
                experiments_df,
                dataset=dataset,
                strategy='DeltaF1Strategy',
                epsilon=best_hybrid_hyper['epsilon'],
                k=best_hybrid_hyper['k'],
                validation_fraction=best_hybrid_hyper['validation_fraction'],
                seed=j
            )

            f1_scores = [round['f1_score'] for round in info['round_val_stats'].values[0]]

            label = f"Seed {j}"
            color = cmap.colors[color_i]
            color_i += 1

            # Prefer switch_round from experiment data; fall back to legacy hardcoded dict
            switch_round_val = info['switch_round'].values[0] if 'switch_round' in info.columns else None
            if switch_round_val is None or (isinstance(switch_round_val, float) and np.isnan(switch_round_val)):
                switch_round_val = _legacy_switch_rounds.get(dataset, {}).get(j, None)

            total_rounds.append(info['total_rounds'].values[0])

            plt.plot(range(1, len(f1_scores) + 1), f1_scores, label=label, color=color)
            if switch_round_val is not None:
                switch_round_val = int(switch_round_val)
                plt.scatter(switch_round_val, f1_scores[switch_round_val - 1], color=color, s=50,
                            edgecolor='black', zorder=5, label=f"Switch Round (Seed {j})")

        plt.xlabel('Round Number')
        plt.ylabel('Validation Set Macro-F1 Score')
        plt.xticks(range(1, max(total_rounds) + 1), fontsize=10)
        plt.title(f'Validation Set Macro-F1 Score vs Round Number (HybridAL) - {dataset_names[dataset]}')
        plt.legend(fontsize='x-small')
        plt.grid(alpha=0.4)
        if save_dir_path is not None:
            plt.savefig(os.path.join(save_dir_path, f'f1_vs_round_hybrid_{dataset}.png'), dpi=dpi)
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table generation
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGY_ORDER   = ['RetrainStrategy', 'FineTuneStrategy', 'DeltaF1Strategy', 'NewOnlyStrategy']
_STRATEGY_DISPLAY = {
    'RetrainStrategy':     'Retrain',
    'FineTuneStrategy':    'Fine-tune',
    'DeltaF1Strategy':     'HybridAL',
    'NewOnlyStrategy':     'New Only',
    'FixedSwitchStrategy': 'Fixed Switch',
}


def _cell_stats(series_f1, series_acc, round_val_stats_series):
    """Return (f1_mean, f1_std, acc_mean, acc_std, time_mean, time_std)."""
    f1_mean  = series_f1.mean()  * 100
    f1_std   = series_f1.std()   * 100
    acc_mean = series_acc.mean() * 100
    acc_std  = series_acc.std()  * 100
    times = []
    for rvs in round_val_stats_series:
        if isinstance(rvs, list):
            times.append(sum(r['training_time'] for r in rvs))
    time_mean = np.mean(times) if times else 0.0
    time_std  = np.std(times)  if times else 0.0
    return f1_mean, f1_std, acc_mean, acc_std, time_mean, time_std


def generate_latex_table(experiments_df: pd.DataFrame,
                         hybrid_hyper: dict,
                         table_mode: str = 'main',
                         caption: str = None,
                         label: str = None,
                         output_path: str = None) -> str:
    """
    Generate a LaTeX results table from experiment data.

    Matches the paper style: thick borders, dataset-grouped rows, mean±std cells,
    and a dagger (†) on HybridAL rows where Welch's t-test gives p<0.05 vs all baselines.

    Required LaTeX packages:
        \\usepackage{booktabs, makecell, multirow, adjustbox, float}

    Args:
        experiments_df: Combined DataFrame from one or more get_experiments_df() calls.
        hybrid_hyper:   {'epsilon': ..., 'k': ..., 'validation_fraction': ...}
        table_mode:     'main' | 'backbones' | 'samplers' | 'fixed_switch' | 'signal'
        caption:        Custom caption (None = auto).
        label:          Custom \\label (None = auto).
        output_path:    If set, write .tex to this path.

    Returns:
        str: Full LaTeX table source.
    """
    from pathlib import Path as _Path

    group_col   = 'dataset' if table_mode in ('main', 'fixed_switch', 'signal') else \
                  'model'   if table_mode == 'backbones' else 'sampler'
    group_names = dataset_names if group_col == 'dataset' else {}

    # ── significance: HybridAL vs each baseline per group ───────────────── #
    sig = {}
    for gv in experiments_df[group_col].dropna().unique():
        gdf  = experiments_df[experiments_df[group_col] == gv]
        hyb  = filter_experiments_df(gdf, strategy='DeltaF1Strategy',
                                     epsilon=hybrid_hyper.get('epsilon'),
                                     k=hybrid_hyper.get('k'),
                                     validation_fraction=hybrid_hyper.get('validation_fraction'))
        hf1  = hyb['test_f1_score'].dropna().values
        for s in _STRATEGY_ORDER:
            if s == 'DeltaF1Strategy':
                continue
            bf1 = filter_experiments_df(gdf, strategy=s)['test_f1_score'].dropna().values
            if len(hf1) >= 2 and len(bf1) >= 2:
                _, p = scipy_stats.ttest_ind(hf1, bf1, equal_var=False)
                sig[(gv, s)] = (p < 0.05 and hf1.mean() > bf1.mean())

    # ── row builder ──────────────────────────────────────────────────────── #
    def make_row(strat_key, display, rdf, is_first, n, rounds_val, add_cline):
        if rdf.empty:
            return []
        f1m,f1s,am,as_,tm,ts = _cell_stats(
            rdf['test_f1_score'], rdf['test_accuracy'], rdf['round_val_stats'])
        marker = ''
        if strat_key == 'DeltaF1Strategy':
            gv = rdf[group_col].iloc[0]
            if any(sig.get((gv, s), False)
                   for s in _STRATEGY_ORDER if s != 'DeltaF1Strategy'):
                marker = r'$^\dagger$'
        f1c   = rf'{f1m:.2f}\% $\pm$ {f1s:.2f}\%{marker}'
        acc_c = rf'{am:.2f}\% $\pm$ {as_:.2f}\%'
        tc    = rf'{tm:.2f} $\pm$ {ts:.2f}'
        out   = []
        if is_first:
            rc = rf'\multirow{{{n}}}{{*}}{{\centering {rounds_val}}}'
            out.append(rf'\textbf{{{display}}} & {f1c} & {acc_c} & {tc} & {rc} \\')
        else:
            out.append(rf'\textbf{{{display}}} & {f1c} & {acc_c} & {tc} & \\')
        if add_cline:
            out.append(r'\cline{1-4}')
        return out

    # ── body ─────────────────────────────────────────────────────────────── #
    body  = []
    groups = sorted(experiments_df[group_col].dropna().unique())
    for g_idx, gv in enumerate(groups):
        gdf = experiments_df[experiments_df[group_col] == gv]
        gvd = group_names.get(gv, str(gv))
        body.append(r'\Xhline{2\arrayrulewidth}' if g_idx == 0 else r'\Xhline{3\arrayrulewidth}')
        body.append(rf'\multicolumn{{5}}{{!{{\vrule width 1.2pt}}c!{{\vrule width 1.2pt}}}}{{\emph{{{gvd}}}}} \\')
        body.append(r'\hline')

        rounds_val = int(round(gdf['total_rounds'].mean())) if 'total_rounds' in gdf.columns else '—'

        if table_mode == 'signal':
            signals = ['delta_f1','delta_loss','delta_accuracy','gradient_norm']
            sdisplay = {'delta_f1': r'HybridAL ($\Delta$F1)',
                        'delta_loss': r'HybridAL ($\Delta$Loss)',
                        'delta_accuracy': r'HybridAL ($\Delta$Acc)',
                        'gradient_norm': r'HybridAL (Grad Norm)'}
            present = [s for s in signals if 'signal' in gdf.columns and s in gdf['signal'].values]
            for i, sn in enumerate(present):
                sub = gdf[gdf['signal'] == sn]
                body += make_row('DeltaF1Strategy', sdisplay.get(sn, sn),
                                 sub, i == 0, len(present), rounds_val, i < len(present)-1)

        elif table_mode == 'fixed_switch':
            sr_col = 'cfg_strategy_kwargs_switch_round'
            srs = sorted(gdf[sr_col].dropna().unique()) if sr_col in gdf.columns else []
            for i, sr in enumerate(srs):
                sub = gdf[gdf[sr_col] == sr]
                body += make_row('FixedSwitchStrategy', f'Fixed-r{int(sr)}',
                                 sub, i == 0, len(srs), rounds_val, i < len(srs)-1)

        else:
            present = [s for s in _STRATEGY_ORDER if s in gdf['strategy'].values]
            for i, strat in enumerate(present):
                sub = gdf[gdf['strategy'] == strat]
                if strat == 'DeltaF1Strategy':
                    sub = filter_experiments_df(sub,
                                                epsilon=hybrid_hyper.get('epsilon'),
                                                k=hybrid_hyper.get('k'),
                                                validation_fraction=hybrid_hyper.get('validation_fraction'))
                body += make_row(strat, _STRATEGY_DISPLAY.get(strat, strat),
                                 sub, i == 0, len(present), rounds_val, i < len(present)-1)

    # ── captions & labels ────────────────────────────────────────────────── #
    _caps = {
        'main':
            r"Macro-F1, accuracy, training time, and rounds across strategies and datasets "
            r"(mean $\pm$ std). $^\dagger$~HybridAL is significantly better than all baselines "
            r"(Welch's $t$-test, $p<0.05$).",
        'backbones':    r'Backbone ablation. Same format as Table~\ref{tab:exp_summary}.',
        'samplers':     r'Sampler ablation. Same format as Table~\ref{tab:exp_summary}.',
        'fixed_switch': r'Fixed-switch baselines: switching at a predetermined round vs adaptive HybridAL.',
        'signal':       r'Switching-signal ablation: replacing $\Delta$F1 with other signals (fixed best hyperparameters).',
    }
    _labels = {'main':'tab:exp_summary','backbones':'tab:backbones',
               'samplers':'tab:samplers','fixed_switch':'tab:fixed_switch','signal':'tab:signal_ablation'}
    cap = caption or _caps.get(table_mode, '')
    lbl = label   or _labels.get(table_mode, 'tab:results')

    lines = [
        r'\begin{table}[H]', r'\centering', r'\footnotesize',
        r'\setlength{\tabcolsep}{6pt}', r'\renewcommand{\arraystretch}{1.32}',
        r'\begin{adjustbox}{width=\columnwidth}',
        r'\begin{tabular}{!{\vrule width 1.2pt}l|c|c|c|c!{\vrule width 1.2pt}}',
        r'\hline',
        r'\textbf{Strategy} & \textbf{Macro-F1} & \textbf{Accuracy} & '
        r'\textbf{Training time (s.)} & \textbf{N. rounds} \\',
        r'\hline', '',
    ] + body + [
        '', r'\Xhline{3\arrayrulewidth}',
        r'\end{tabular}', r'\end{adjustbox}',
        rf'\caption{{{cap}}}', rf'\label{{{lbl}}}',
        r'\end{table}',
    ]
    latex = '\n'.join(lines)

    if output_path:
        _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)

    return latex
