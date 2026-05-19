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
matplotlib.use("Agg")


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
                        choices=["main", "backbones", "samplers", "fixed_switch", "signal", "all"],
                        help="Which table layout to build. 'all' generates every table.")
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


    hyper_from_file = _load_best_hyper(args.hyper_file)

    hybrid_hyper = {
        "epsilon": args.hybrid_epsilon or hyper_from_file.get("epsilon"),
        "k":       args.hybrid_k       or hyper_from_file.get("k"),
    }
    if args.hybrid_vf or hyper_from_file.get("validation_fraction"):
        hybrid_hyper["validation_fraction"] = args.hybrid_vf or hyper_from_file.get("validation_fraction")

    hybrid_hyper = {k: v for k, v in hybrid_hyper.items() if v is not None}


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
            plot_backbone_comparison,
            plot_signal_comparison,
            plot_sampler_comparison,
            plot_fixed_switch_comparison,
            plot_per_dataset_f1,
            plot_training_time_comparison,
        )
    except ImportError as e:
        print(f"[generate_results] ERROR: could not import figure_plotter: {e}", file=sys.stderr)
        sys.exit(1)


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


    if not args.skip_plots:
        print("[generate_results] Generating plots...")


        try:
            plot_f1_vs_time_avg(df, hybrid_hyper,
                                save_dir_path=str(out / "f1_vs_time"))
            print("  [plot] f1_vs_time_avg: done")
        except Exception as e:
            print(f"  [plot] f1_vs_time_avg: SKIPPED ({e})")


        try:
            plot_test_f1_bar_chart(df, hybrid_hyper,
                                   save_path=str(out / "test_f1_bar.png"))
            print("  [plot] test_f1_bar_chart: done")
        except Exception as e:
            print(f"  [plot] test_f1_bar_chart: SKIPPED ({e})")


        try:
            plot_f1_vs_round_switch(df, hybrid_hyper,
                                    save_dir_path=str(out / "f1_switch"))
            print("  [plot] f1_vs_round_switch: done")
        except Exception as e:
            print(f"  [plot] f1_vs_round_switch: SKIPPED ({e})")


        try:
            plot_hybrid_hyper_variations(df, hybrid_hyper,
                                         save_dir_path=str(out / "hyper_variations"))
            print("  [plot] hybrid_hyper_variations: done")
        except Exception as e:
            print(f"  [plot] hybrid_hyper_variations: SKIPPED ({e})")


        if hybrid_hyper:
            try:
                plot_confusion_heatmaps_hybrid(df, hybrid_hyper,
                                               save_dir_path=str(out / "confusion_heatmaps"))
                print("  [plot] confusion_heatmaps_hybrid: done")
            except Exception as e:
                print(f"  [plot] confusion_heatmaps_hybrid: SKIPPED ({e})")


        try:
            plot_switch_heatmap(df, save_dir_path=str(out / "switch_heatmap"))
            print("  [plot] switch_heatmap: done")
        except Exception as e:
            print(f"  [plot] switch_heatmap: SKIPPED ({e})")


        try:
            plot_backbone_comparison(df, hybrid_hyper, save_dir_path=str(out))
            print("  [plot] backbone_comparison: done")
        except Exception as e:
            print(f"  [plot] backbone_comparison: SKIPPED ({e})")

        try:
            plot_signal_comparison(df, hybrid_hyper, save_dir_path=str(out))
            print("  [plot] signal_comparison: done")
        except Exception as e:
            print(f"  [plot] signal_comparison: SKIPPED ({e})")

        try:
            plot_sampler_comparison(df, hybrid_hyper, save_dir_path=str(out))
            print("  [plot] sampler_comparison: done")
        except Exception as e:
            print(f"  [plot] sampler_comparison: SKIPPED ({e})")

        try:
            plot_fixed_switch_comparison(df, hybrid_hyper, save_dir_path=str(out))
            print("  [plot] fixed_switch_comparison: done")
        except Exception as e:
            print(f"  [plot] fixed_switch_comparison: SKIPPED ({e})")

        try:
            plot_per_dataset_f1(df, hybrid_hyper, save_dir_path=str(out))
            print("  [plot] per_dataset_f1: done")
        except Exception as e:
            print(f"  [plot] per_dataset_f1: SKIPPED ({e})")

        try:
            plot_training_time_comparison(df, hybrid_hyper, save_dir_path=str(out))
            print("  [plot] training_time_comparison: done")
        except Exception as e:
            print(f"  [plot] training_time_comparison: SKIPPED ({e})")


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


    if not args.skip_table:
        modes = ["main", "backbones", "samplers", "fixed_switch", "signal"]\
                if args.table_mode == "all" else [args.table_mode]
        for mode in modes:
            tex_path = out / f"table_{mode}.tex"
            print(f"\n[generate_results] Generating LaTeX table ({mode}) → {tex_path}")
            try:
                latex = generate_latex_table(
                    df,
                    hybrid_hyper=hybrid_hyper,
                    table_mode=mode,
                    output_path=str(tex_path),
                )
                print(f"  Saved: {tex_path}")
                print("\n" + "=" * 70)
                print(f"LaTeX source ({mode}):")
                print("=" * 70)
                print(latex)
                print("=" * 70)
            except Exception as e:
                print(f"  [latex] ERROR ({mode}): {e}")
                import traceback
                traceback.print_exc()

    print("\n[generate_results] Done.")


if __name__ == "__main__":
    main()
