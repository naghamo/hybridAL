#!/bin/bash
# GPU1: Step 2 (HybridAL Optuna) → Step 3 (backbones) → Step 4 (samplers) → Step 5 (fixed switch)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybridAL
export CUDA_VISIBLE_DEVICES=1

set -e

# Read stable subset size (already determined by Step 0)
BEST_SUBSET=$(python -c "
import json
d = json.load(open('results/subset_sensitivity/stable_subset.json'))
print(d['stable_subset_size'])
")
echo "Using stable subset size: ${BEST_SUBSET}"

# ── Step 2: HybridAL (Optuna) ────────────────────────────────────────────── #
echo "============================================================"
echo " [GPU1] STEP 2 — HybridAL with Optuna"
echo "============================================================"
python experimentation.py --mode hybrid --force \
    --random-subset-size "${BEST_SUBSET}"

echo "--- Generating plots and table for Step 2 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode main \
    --hyper-file best_hyperparams.json \
    --output-dir results/step2_hybrid

# Read best hyperparams written by Optuna
BEST_EPSILON=$(python -c "import json; d=json.load(open('best_hyperparams.json')); print(d['epsilon'])")
BEST_K=$(python -c "import json; d=json.load(open('best_hyperparams.json')); print(d['k'])")
echo "Using best hyperparams: epsilon=${BEST_EPSILON}  k=${BEST_K}"

# ── Step 3: Backbone ablation (BERT + RoBERTa only) ──────────────────────── #
echo "============================================================"
echo " [GPU1] STEP 3 — Backbone ablation (BERT, RoBERTa)"
echo "============================================================"
python experimentation.py --mode backbones --force \
    --models bert-base-uncased roberta-base \
    --best-epsilon "${BEST_EPSILON}" \
    --best-k "${BEST_K}" \
    --random-subset-size "${BEST_SUBSET}"

echo "--- Generating plots and table for Step 3 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode backbones \
    --hyper-file best_hyperparams.json \
    --output-dir results/step3_backbones

# ── Step 4: Sampler ablation ──────────────────────────────────────────────── #
echo "============================================================"
echo " [GPU1] STEP 4 — Sampler ablation (Random, BADGE)"
echo "============================================================"
python experimentation.py --mode samplers --force \
    --samplers RandomSampler BADGESampler \
    --best-epsilon "${BEST_EPSILON}" \
    --best-k "${BEST_K}" \
    --random-subset-size "${BEST_SUBSET}"

echo "--- Generating plots and table for Step 4 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode samplers \
    --hyper-file best_hyperparams.json \
    --output-dir results/step4_samplers

# ── Step 5: Fixed-switch baselines ───────────────────────────────────────── #
echo "============================================================"
echo " [GPU1] STEP 5 — Fixed-switch baselines"
echo "============================================================"
python experimentation.py --mode fixed_switch --force \
    --random-subset-size "${BEST_SUBSET}"

echo "--- Generating plots and table for Step 5 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode fixed_switch \
    --hyper-file best_hyperparams.json \
    --output-dir results/step5_fixed_switch

echo "============================================================"
echo " [GPU1] ALL DONE (Steps 2, 3, 4, 5)"
echo " Stable subset size used: ${BEST_SUBSET}"
echo "============================================================"
