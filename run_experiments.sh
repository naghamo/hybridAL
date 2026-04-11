#!/bin/bash
# =============================================================================
# EMNLP Experiment Runner — full re-run from zero
#
# Activate conda env so the correct Python (3.11) is used:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybridAL
#
# HOW TO RUN IN THE BACKGROUND WITH A PROGRESS BAR:
#
#   1. Start everything in the background and save all output to run.log:
#        nohup bash run_experiments.sh > run.log 2>&1 &
#        echo "PID = $!"        # note the PID if you want to kill it later
#
#   2. In a second terminal, watch progress live:
#        python monitor.py --log run.log
#
#      Or with a known total (speeds up ETA):
#        python monitor.py --log run.log --total 1500
#
#   3. To stop the run:
#        kill <PID>             # kill just the bash process
#        # or
#        pkill -f experimentation.py
#
# WHAT EACH STEP ANSWERS:
#   Step 1 — How do the baselines perform? (all datasets, DistilBERT, Entropy)
#   Step 2 — How does HybridAL perform? (all datasets, DistilBERT, Entropy, Optuna)
#             Saves best hyperparams automatically to best_hyperparams.json
#   Step 3 — Does it hold across models? (BERT + RoBERTa, original datasets)
#   Step 4 — Does it hold across samplers? (Random + BADGE, original datasets)
#   Step 5 — Is adaptive switching better than fixed switching?
#   Step 6 — Is ΔF1 the right switching signal?
#             Uses best_hyperparams.json written by Step 2 — no manual edits needed.
# =============================================================================

set -e   # stop on first error

# ── Step 1: Baselines ─────────────────────────────────────────────────────── #
echo "============================================================"
echo " STEP 1 — Baselines (Retrain / FineTune / NewOnly)"
echo "============================================================"
python experimentation.py --mode baselines --force

echo "--- Generating plots and table for Step 1 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode main \
    --output-dir results/step1_baselines

# ── Step 2: HybridAL (Optuna) ────────────────────────────────────────────── #
echo "============================================================"
echo " STEP 2 — HybridAL with Optuna (saves best_hyperparams.json)"
echo "============================================================"
python experimentation.py --mode hybrid --force

echo "--- Generating plots and table for Step 2 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode main \
    --hyper-file best_hyperparams.json \
    --output-dir results/step2_hybrid

# ── Step 3: Backbone ablation ────────────────────────────────────────────── #
echo "============================================================"
echo " STEP 3 — Backbone ablation (BERT, RoBERTa)"
echo "============================================================"
python experimentation.py --mode backbones --force

echo "--- Generating plots and table for Step 3 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode backbones \
    --hyper-file best_hyperparams.json \
    --output-dir results/step3_backbones

# ── Step 4: Sampler ablation ─────────────────────────────────────────────── #
echo "============================================================"
echo " STEP 4 — Sampler ablation (Random, BADGE)"
echo "============================================================"
python experimentation.py --mode samplers --force

echo "--- Generating plots and table for Step 4 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode samplers \
    --hyper-file best_hyperparams.json \
    --output-dir results/step4_samplers

# ── Step 5: Fixed-switch baselines ───────────────────────────────────────── #
echo "============================================================"
echo " STEP 5 — Fixed-switch baselines (round 3 / 5 / 7 / 10)"
echo "============================================================"
python experimentation.py --mode fixed_switch --force

echo "--- Generating plots and table for Step 5 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode fixed_switch \
    --hyper-file best_hyperparams.json \
    --output-dir results/step5_fixed_switch

# ── Step 6: Signal ablation — reads best hyperparams automatically ────────── #
echo "============================================================"
echo " STEP 6 — Signal ablation (ΔLoss / ΔAcc / GradNorm vs ΔF1)"
echo "============================================================"

# Read best hyperparams from JSON written by Step 2
EPSILON=$(python -c "import json; d=json.load(open('best_hyperparams.json')); print(d['epsilon'])")
K=$(python -c "import json; d=json.load(open('best_hyperparams.json')); print(d['k'])")
VF=$(python -c "import json; d=json.load(open('best_hyperparams.json')); print(d['validation_fraction'])")

echo "Using best hyperparams: epsilon=${EPSILON}  k=${K}  validation_fraction=${VF}"

python experimentation.py --mode signal_ablation --force \
    --best-epsilon "${EPSILON}" \
    --best-k "${K}" \
    --best-validation-fraction "${VF}"

echo "--- Generating plots and table for Step 6 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode signal \
    --hyper-file best_hyperparams.json \
    --output-dir results/step6_signal_ablation

echo "============================================================"
echo " ALL STEPS COMPLETE"
echo " Results saved in:  results/"
echo "============================================================"
