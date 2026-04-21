#!/bin/bash
# GPU0: Step 1 (Baselines, continues from existing) → Step 6 (Signal ablation)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybridAL
export CUDA_VISIBLE_DEVICES=0

set -e

# Read stable subset size (already determined by GPU1 Step 0)
BEST_SUBSET=$(python -c "
import json
d = json.load(open('results/subset_sensitivity/stable_subset.json'))
print(d['stable_subset_size'])
")
echo "Using stable subset size: ${BEST_SUBSET}"

# ── Step 1: Baselines ─────────────────────────────────────────────────────── #
echo "============================================================"
echo " [GPU0] STEP 1 — Baselines (skips already-completed runs)"
echo "============================================================"
python experimentation.py --mode baselines \
    --random-subset-size "${BEST_SUBSET}"

echo "--- Generating plots and table for Step 1 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode main \
    --output-dir results/step1_baselines

# ── Wait for best_hyperparams.json from GPU1 Step 2 ─────────────────────── #
echo "Waiting for best_hyperparams.json from GPU1 Step 2 (Optuna)..."
while [ ! -f "best_hyperparams.json" ]; do
    sleep 60
done
echo "Found best_hyperparams.json."

EPSILON=$(python -c "import json; d=json.load(open('best_hyperparams.json')); print(d['epsilon'])")
K=$(python -c "import json; d=json.load(open('best_hyperparams.json')); print(d['k'])")
echo "Using best hyperparams: epsilon=${EPSILON}  k=${K}"

# ── Step 6: Signal ablation ───────────────────────────────────────────────── #
echo "============================================================"
echo " [GPU0] STEP 6 — Signal ablation"
echo "============================================================"
python experimentation.py --mode signal_ablation --force \
    --ablation-signals delta_loss delta_accuracy gradient_norm \
    --best-epsilon "${EPSILON}" \
    --best-k "${K}" \
    --random-subset-size "${BEST_SUBSET}"

echo "--- Generating plots and table for Step 6 ---"
python generate_results.py \
    --dirs experiments \
    --table-mode signal \
    --hyper-file best_hyperparams.json \
    --output-dir results/step6_signal_ablation

echo "============================================================"
echo " [GPU0] ALL DONE (Steps 1, 6)"
echo " Subset size used: ${BEST_SUBSET}"
echo "============================================================"
