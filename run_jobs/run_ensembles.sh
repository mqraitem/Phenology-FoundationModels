#!/bin/bash
# Create all ensembles for all month subsets.
# Run after eval_to_dataframe.py has exported per-seed CSVs for all models.
#
# Cross-model ensembles use ALL common seeds (per-seed paired ensembling, so
# variance comes for free across seeds).
# Intra-model "× 3" ensembles use 3 disjoint seed groups so we can report
# mean ± std across the three group-ensemble scores.

set -e

# 9 seeds total → 3 disjoint groups of 3. Edit here to change membership.
SEEDS_GA=("seed_42"  "seed_123" "seed_456")
SEEDS_GB=("seed_789" "seed_101" "seed_202")
SEEDS_GC=("seed_303" "seed_404" "seed_505")

INTRA_MODELS=(
    "transformer_1d_paper_1.0"
    "presto_1.0"
    "prithvi_final_100m_crop32_1.0"
)

for months in "3 6 9 12" "3 4 5 6 7 8 9 10" "1 2 3 4 5 6 7 8 9 10 11 12"; do
    echo ""
    echo "============================================================"
    echo "  Months: $months"
    echo "============================================================"

    # ---- Cross-model ensembles (per-seed paired) ----
    python misc_scripts/ensemble_from_csvs.py \
        --methods transformer_1d_paper_1.0 presto_1.0 \
        --selected_months $months \
        --name ensemble_transformer_presto

    python misc_scripts/ensemble_from_csvs.py \
        --methods transformer_1d_paper_1.0 prithvi_final_100m_crop32_1.0 \
        --selected_months $months \
        --name ensemble_transformer_prithvi

    python misc_scripts/ensemble_from_csvs.py \
        --methods transformer_1d_paper_1.0 presto_1.0 \
                  prithvi_final_100m_crop32_1.0 \
        --selected_months $months \
        --name ensemble_all

    # ---- Intra-model seed ensembles, three disjoint groups ----
    for model in "${INTRA_MODELS[@]}"; do
        python misc_scripts/ensemble_seeds.py \
            --model "$model" \
            --selected_months $months \
            --seeds "${SEEDS_GA[@]}" \
            --name_suffix _gA

        python misc_scripts/ensemble_seeds.py \
            --model "$model" \
            --selected_months $months \
            --seeds "${SEEDS_GB[@]}" \
            --name_suffix _gB

        python misc_scripts/ensemble_seeds.py \
            --model "$model" \
            --selected_months $months \
            --seeds "${SEEDS_GC[@]}" \
            --name_suffix _gC
    done
done

echo ""
echo "All ensembles created."
