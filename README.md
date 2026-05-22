# HybridAL: Switching Signal-Guided Training in Active Learning

Anonymous code release accompanying the EMNLP 2026 submission
*"Train Smarter, Not Harder: Switching Signal-Guided Training in Active Learning"*.

---

## Overview

**HybridAL** is a pool-based active-learning method that adaptively switches
its training strategy from full **retraining** (expensive, well-calibrated)
to **fine-tuning** (cheap, less calibrated) when a *stabilization signal*
indicates the learner has converged.

The two HybridAL variants in the paper instantiate this idea with
complementary signals:

| Variant | Signal | Trade-off |
|---|---|---|
| **HybridAL(Δα)** | spectral exponent of the weight matrices | weight-based, no validation pass, fastest |
| **HybridAL(ΔAcc)** | round-over-round validation accuracy | best calibration retention |

Across **3 transformer backbones** (DistilBERT, BERT, RoBERTa) and
**6 text-classification benchmarks** (IMDb, Jigsaw, SST-2, TweetEval,
AG News, Yahoo Answers) with **5 seeds** each, HybridAL preserves
test F1 (no statistically significant difference from retraining or
fine-tuning in the majority of cells, paired *t*-test, *p* > 0.05),
saves up to **49 %** of retraining time, and outperforms every
pre-committed `FixedSwitch@k` schedule on calibration.

---

## Pipeline

Standard pool-based AL loop, with one extra decision per round — *which
training strategy to use*:

1. **Initialize** — class-stratified labeled pool L₀ (|L₀|=200), unlabeled pool U₀.
2. **Train** — model `f_θ_t` on L_{t-1} using strategy `s_t`.
3. **Acquire** — top-`n` entropy samples from a random pre-filter of `N` candidates.
4. **Annotate** — move queried batch into the labeled pool.
5. **Monitor signal** — compute Δα or ΔAcc against the previous round.
6. **Switch decision** — if the signal stayed below ε for `k` consecutive rounds,
   permanently switch `s_t` from Retrain to FineTune.
7. **Repeat** — until budget `T=25` rounds is exhausted.
8. **Test** — evaluate macro-F1, NLL, and wall-clock time on held-out test set.

---

## Repository structure

```
.
├── adaptive_al/                  # core AL framework
│   ├── active_learning.py        # main AL loop
│   ├── config.py                 # run config dataclass
│   ├── evaluation.py             # metrics (F1, NLL, time)
│   ├── pool.py                   # labeled/unlabeled pool management
│   ├── samplers/                 # acquisition functions
│   │   ├── base_sampler.py
│   │   ├── entropy_sampler.py
│   │   ├── entropy_on_random_subset_sampler.py   # default (N=1000 pre-filter)
│   │   ├── random_sampler.py
│   │   └── badge_sampler.py
│   ├── strategies/               # per-round training strategies
│   │   ├── base_strategy.py
│   │   ├── retrain_strategy.py
│   │   ├── fine_tuning_strategy.py
│   │   ├── new_only_strategy.py
│   │   ├── fixed_switch_strategy.py   # FixedSwitch@k baselines
│   │   └── deltaf1_strategy.py        # HybridAL — supports any signal
│   └── utils/
│       ├── data_loader.py        # per-dataset loaders + splits
│       └── text_datasets.py      # tokenization / PyTorch wrappers
│
├── experiments_lib/              # experiment orchestration
│   ├── runners/                  # one module per experiment
│   │   ├── _base.py              # shared CLI + dispatcher
│   │   ├── main_results.py       # 9 methods × 3 backbones × 6 datasets × 5 seeds
│   │   ├── hyperparameter.py     # (ε, k) tuning for both signals
│   │   ├── pool_size.py          # |L₀| ∈ {50, 100, 200, 500}
│   │   ├── batch_size.py         # n ∈ {16, 32, 64, 128}
│   │   ├── samplers.py           # Entropy / Random / BADGE
│   │   ├── n_sensitivity.py      # entropy pre-filter N
│   │   ├── early_stopping.py     # max-10 + ES vs fixed-5
│   │   ├── calibration_sensitivity.py
│   │   ├── normalizer_sensitivity.py
│   │   └── calibration_eval.py   # re-run on DistilBERT, save logits for
│   │                             # ECE / temperature-scaling analysis
│   ├── aggregators/              # one module per experiment that builds _summary.csv
│   │   ├── _base.py
│   │   └── {main_results, hyperparameter, pool_size, batch_size,
│   │        samplers, n_sensitivity, early_stopping,
│   │        calibration_sensitivity, normalizer_sensitivity}.py
│   └── shared/                   # cross-experiment utilities
│       ├── config_factory.py     # build_cfg(...)
│       ├── constants.py          # backbones, datasets, seeds
│       ├── csv_io.py             # robust CSV read/write
│       ├── plan.py               # experiment grid / plan
│       ├── runner.py             # per-config execution
│       └── signals.py            # the 8 candidate switching signals
│
├── main.py                       # unified CLI (run / aggregate / list)
├── _calibration_normalizers.json # per-signal calibration normalizers
├── requirements.txt
└── README.md
```

Outputs from each runner land in `experiments/<experiment_name>/` (not
tracked); the aggregators read those and write per-experiment
`_summary.csv` files that back every number in the paper.

---

## Setup

Tested on Linux with Python 3.11 and CUDA 11.8.

```bash
git clone <anonymous repo URL>
cd hybridal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended. The full grid in the paper
was produced on two NVIDIA RTX 2080 Ti cards (≈280 GPU-hours total).

---

## Running experiments

`main.py` is a unified CLI with three subcommands:

```bash
python main.py list                       # show available experiments
python main.py run <experiment> [flags]   # launch an experiment grid
python main.py aggregate <experiment>     # build experiments/<exp>/_summary.csv
```

Each experiment has its own flags — see `python main.py run <experiment> --help`.

### Main results (810 runs)

```bash
python main.py run main_results \
    --signal-alpha delta_spectral_alpha --epsilon-alpha 1e-4 --k-alpha 3 \
    --signal-acc   delta_accuracy        --epsilon-acc   5e-3 --k-acc   2 \
    --gpu 0
python main.py aggregate main_results
```

Runs the 9 methods (Retrain, FineTune, NewOnly, HybridAL(Δα),
HybridAL(ΔAcc), FixedSwitch@{3,5,7,10}) across 3 backbones × 6 datasets
× 5 seeds. The aggregator writes `experiments/main_results/_summary.csv`.

### Hyperparameter tuning

```bash
python main.py run hyperparameter \
    --signal delta_spectral_alpha --confirm-grid \
    --epsilon-grid 0.00005 0.00010 0.00015 0.00020
python main.py aggregate hyperparameter
```

Sweeps (ε, k) on IMDb and AG News with 3 seeds. The selected
(ε⋆, k⋆) pair is `(1e-4, 3)` for Δα and `(5e-3, 2)` for ΔAcc.

### Robustness ablations

```bash
python main.py run pool_size        # |L₀| ∈ {50, 100, 200, 500}
python main.py run batch_size       # n   ∈ {16, 32, 64, 128}
python main.py run samplers         # Entropy / Random / BADGE
python main.py run n_sensitivity    # entropy pre-filter N
python main.py run early_stopping   # max-10 + ES vs fixed-5
```

Aggregate each with `python main.py aggregate <experiment>`.

### Calibration analysis (ECE / temperature scaling)

Re-runs the 5 main methods on DistilBERT (150 cells) and saves
per-run test/val logits + labels as `.npy` sidecars next to each
`results_*.json`, enabling offline ECE and post-temperature-scaling
NLL (Guo et al. 2017) without retraining.

```bash
python main.py run calibration_eval \
    --signal-alpha delta_spectral_alpha --epsilon-alpha 1e-4 --k-alpha 3 \
    --signal-acc   delta_accuracy        --epsilon-acc   5e-3 --k-acc   2
```

Outputs per run: `test_logits.npy`, `test_labels.npy`,
`val_logits.npy`, `val_labels.npy`. No aggregator; the
post-processing script is not shipped.

### Signal ablation

The signal ablation re-uses the main-results runner with one signal at a
time, all at the canonical comparison setting `(ε = 0.5, k = 3)` that
puts every candidate on equal footing. Pass any of the 8 names from
`experiments_lib/shared/signals.py` (`delta_spectral_alpha`,
`delta_accuracy`, `delta_f1`, `delta_loss`, `gradient_norm`,
`l2_weight_distance`, `cka`, `delta_nc`) as `--signal-alpha`, together
with `--epsilon-alpha 0.5 --k-alpha 3`. The main-results hyperparameters
(`1e-4, 3` for Δα; `5e-3, 2` for ΔAcc) are the *output* of the
hyperparameter sweep, not the signal-ablation setting.

---

## Output format

The per-experiment aggregator (`python main.py aggregate <experiment>`)
writes `experiments/<experiment>/_summary.csv` with one row per
`(method, backbone, dataset, seed)` cell. Columns include test F1,
test NLL, wall-clock training time, switch round, and per-round
metrics. These CSVs back every numerical claim in the paper. The
signal-ablation CSV omits test NLL; downstream analyses that need it
read `final_test_stats.loss` from each run's `results_*.json`. The
`calibration_eval` runner has no aggregator and instead writes the
four `.npy` sidecar arrays per run (see above) for offline
temperature-scaling and ECE computation.

---

## Datasets

| Dataset | Task | Classes | Train size |
|---|---|---|---|
| IMDb | Sentiment (binary) | 2 | 50,000 |
| Jigsaw | Toxicity (binary, imbalanced ≈9.6 %) | 2 | 159,571 |
| SST-2 | Sentiment (binary) | 2 | 68,221 |
| TweetEval | Tweet sentiment | 3 | 59,899 |
| AG News | Topic classification | 4 | 127,600 |
| Yahoo Answers | Topic Q&A (stratified subsample) | 10 | 50,000 |

All datasets are loaded via 🤗 `datasets`; see `adaptive_al/utils/data_loader.py`
for the exact splits used. A fixed stratified validation set is held
out per dataset and never enters the labeled pool.

---

## Training configuration

Defaults used across every experiment unless stated otherwise:

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr = 2e-5, weight_decay = 1e-3) |
| Batch size | 16 |
| Max epochs / round | 10 (early stopping, patience 2, min_delta = 1e-4) |
| Initial labeled pool `|L₀|` | 200 (class-stratified) |
| AL rounds `T` | 25 |
| Acquisition batch size `n` | 32 |
| Entropy pre-filter size `N` | 1000 |
| Seeds | {42, 43, 44, 45, 46} |

---

## Reproducibility

* No values are hardcoded — every knob is exposed via the runner CLI.
* All runs are seeded at the data-split, sampler, and model-init level.
* Per-run JSON configs are written alongside the metrics for each run.
* Aggregated CSVs (`experiments/<exp>/_summary.csv`) back every number
  in the paper.

---

## License

Released for academic use under the terms of the EMNLP 2026 submission
process. Authors and affiliation withheld for double-blind review.
