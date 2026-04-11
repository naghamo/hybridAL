"""
generate_results.py — Post-experiment plot + LaTeX table generator.

Called automatically by run_experiments.sh after each step, or manually:

  python generate_results.py \\
      --dirs experiments/ \\
      --table-mode main \\
      --hybrid-epsilon 0.01 --hybrid-k 4 --hybrid-vf 0.1 \\
      --output-dir results/

Table modes:
  main          — baselines + HybridAL, all datasets
  backbones     — backbone ablation (BERT, RoBERTa, DistilBERT)
  samplers      — sampler ablation (Entropy, Random, BADGE)
  fixed_switch  — fixed-switch baselines
  signal        — switching-signal ablation
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display required


def _load_best_hyper(hyper_file: str) -> dict:
    """Load best hyperparams saved by experimentation.py after Optuna."""
    p = Path(hyper_file)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate plots and LaTeX tables from experiment results.")
    parser.add_argument("--dirs", nargs="+", default=["experiments"],
                        help="Experiment output directories to load (space-separated).")
    parser.add_argument("--table-mode", default="main",
                        choices=["main", "backbones", "samplers", "fixed_switch", "signal"],
                        help="Which table layout to build.")
    parser.add_argument("--hybrid-epsilon", type=float, default=None,
                        help="Best epsilon for DeltaF1Strategy (from Optuna step).")
    parser.add_argument("--hybrid-k", type=int, default=None,
                        help="Best k for DeltaF1Strategy (from Optuna step).")
    parser.add_argument("--hybrid-vf", type=float, default=None,
                        help="Best validation_fraction for DeltaF1Strategy (from Optuna step).")
    parser.add_argument("--hyper-file", default="best_hyperparams.json",
                        help="JSON file written by experimentation.py with best hyperparams "
                             "(auto-loaded if --hybrid-epsilon/k/vf are not given).")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to write plots and .tex files into.")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Only generate LaTeX tables, skip plots.")
    parser.add_argument("--skip-table", action="store_true",
                        help="Only generate plots, skip LaTeX table.")
    args = parser.parse_args()

    # ── resolve best hyperparams ───────────────────────────────────────────── #
    hyper_from_file = _load_best_hyper(args.hyper_file)

    hybrid_hyper = {
        "epsilon":            args.hybrid_epsilon  or hyper_from_file.get("epsilon"),
        "k":                  args.hybrid_k        or hyper_from_file.get("k"),
        "validation_fraction": args.hybrid_vf      or hyper_from_file.get("validation_fraction"),
    }
    # Remove None values so filter_experiments_df doesn't filter on missing keys
    hybrid_hyper = {k: v for k, v in hybrid_hyper.items() if v is not None}

    # ── load data ─────────────────────────────────────────────────────────── #
    try:
        from figure_plotter import (
            get_experiments_df,
            get_significance_table,
            plot_switch_heatmap,
            generate_latex_table,
            plot_f1_vs_time_avg,
            plot_f1_vs_round_switch,
            plot_test_f1_bar_chart,
            plot_hybrid_hyper_variations,
            plot_confusion_heatmaps_hybrid,
        )
    except ImportError as e:
        print(f"[generate_results] ERROR: could not import figure_plotter: {e}", file=sys.stderr)
        sys.exit(1)

    # Load and concatenate experiments from all requested directories
    dfs = []
    for d in args.dirs:
        if not Path(d).exists():
            print(f"[generate_results] WARNING: directory not found: {d}")
            continue
        try:
            dfs.append(get_experiments_df(d))
        except Exception as e:
            print(f"[generate_results] WARNING: failed to load {d}: {e}")

    if not dfs:
        print("[generate_results] No experiments loaded — nothing to do.")
        sys.exit(0)

    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)
    print(f"[generate_results] Loaded {len(df)} experiment rows from {args.dirs}.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── plots ─────────────────────────────────────────────────────────────── #
    if not args.skip_plots:
        print("[generate_results] Generating plots...")

        # F1 vs time (averaged across seeds)
        try:
            plot_f1_vs_time_avg(df, hybrid_hyper,
                                save_dir_path=str(out / "f1_vs_time"))
            print("  [plot] f1_vs_time_avg: done")
        except Exception as e:
            print(f"  [plot] f1_vs_time_avg: SKIPPED ({e})")

        # Bar chart of final test F1
        try:
            plot_test_f1_bar_chart(df, hybrid_hyper,
                                   save_path=str(out / "test_f1_bar.png"))
            print("  [plot] test_f1_bar_chart: done")
        except Exception as e:
            print(f"  [plot] test_f1_bar_chart: SKIPPED ({e})")

        # F1 with switch point overlay
        try:
            plot_f1_vs_round_switch(df, hybrid_hyper,
                                    save_dir_path=str(out / "f1_switch"))
            print("  [plot] f1_vs_round_switch: done")
        except Exception as e:
            print(f"  [plot] f1_vs_round_switch: SKIPPED ({e})")

        # Hyperparameter variations (only if DeltaF1 experiments present)
        try:
            plot_hybrid_hyper_variations(df, hybrid_hyper,
                                         save_dir_path=str(out / "hyper_variations"))
            print("  [plot] hybrid_hyper_variations: done")
        except Exception as e:
            print(f"  [plot] hybrid_hyper_variations: SKIPPED ({e})")

        # Confusion heatmaps for HybridAL (one per dataset)
        if hybrid_hyper:
            try:
                plot_confusion_heatmaps_hybrid(df, hybrid_hyper,
                                               save_dir_path=str(out / "confusion_heatmaps"))
                print("  [plot] confusion_heatmaps_hybrid: done")
            except Exception as e:
                print(f"  [plot] confusion_heatmaps_hybrid: SKIPPED ({e})")

        # Switch heatmap (epsilon x k -> avg switch round)
        try:
            plot_switch_heatmap(df, save_dir_path=str(out / "switch_heatmap"))
            print("  [plot] switch_heatmap: done")
        except Exception as e:
            print(f"  [plot] switch_heatmap: SKIPPED ({e})")

        # Significance table (printed, not a plot)
        if hybrid_hyper:
            try:
                sig_df = get_significance_table(df, hybrid_hyper)
                print("\n[generate_results] Significance table (Welch's t-test):")
                print(sig_df.to_string(index=False))
                sig_path = out / "significance_table.csv"
                sig_df.to_csv(sig_path, index=False)
                print(f"  Saved to {sig_path}")
            except Exception as e:
                print(f"  [sig] SKIPPED ({e})")

    # ── LaTeX table ───────────────────────────────────────────────────────── #
    if not args.skip_table:
        tex_path = out / f"table_{args.table_mode}.tex"
        print(f"\n[generate_results] Generating LaTeX table ({args.table_mode}) → {tex_path}")
        try:
            latex = generate_latex_table(
                df,
                hybrid_hyper=hybrid_hyper,
                table_mode=args.table_mode,
                output_path=str(tex_path),
            )
            print(f"  Saved: {tex_path}")
            print("\n" + "=" * 70)
            print("LaTeX source:")
            print("=" * 70)
            print(latex)
            print("=" * 70)
        except Exception as e:
            print(f"  [latex] ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n[generate_results] Done.")


if __name__ == "__main__":
    main()
