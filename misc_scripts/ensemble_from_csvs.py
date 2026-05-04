"""Fit a global convex ensemble from per-pixel result CSVs and save weights + predictions.

Fits ensemble weights on training pixels, then applies to val and test.
This makes the global ensemble directly comparable to the regional ensemble
(ensemble_regional.py), which also fits on training.

Works with the per-model/per-seed results directory structure:
    results/{months_sub}/{model_name}/seed_42_{train,val,test}.csv

Produces an ensemble for each seed independently, then saves:
  1. Ensemble weights JSON to data/ensembles/{months_sub}/{ensemble_name}.json
  2. Per-seed blended CSVs to results/{months_sub}/{ensemble_name}/seed_N_{split}.csv

Usage:
    python misc_scripts/ensemble_from_csvs.py \
        --methods transformer_1d_paper_1.0 presto_1.0 \
        --selected_months 3 6 9 12 \
        --name ensemble_transformer_presto

    python misc_scripts/ensemble_from_csvs.py \
        --methods transformer_1d_paper_1.0 presto_1.0 \
                  prithvi_final_100m_crop32_1.0 \
        --selected_months 3 6 9 12 \
        --name ensemble_all
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from lib.utils import get_results_dir, months_to_str


PRED_COLS = ["G_pred_DOY", "M_pred_DOY", "S_pred_DOY", "D_pred_DOY"]
TRUTH_COLS = ["G_truth_DOY", "M_truth_DOY", "S_truth_DOY", "D_truth_DOY"]


def _fit_convex_weights(X, y):
    """Find w >= 0, sum(w) = 1 that minimizes MSE."""
    n = X.shape[1]
    w0 = np.full(n, 1.0 / n)
    res = minimize(
        fun=lambda w: np.mean((X @ w - y) ** 2),
        x0=w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )
    return res.x if res.success else w0


def sort_df(df):
    return df.sort_values(
        by=["years", "HLStile", "SiteID", "row", "col", "version"]
    ).reset_index(drop=True)


def load_seed_csv(results_base, method, seed, split):
    """Load a result CSV for a method/seed/split from the new directory structure."""
    path = os.path.join(results_base, method, f"{seed}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {split} CSV: {path}\n"
            f"Run eval_to_dataframe.py first to generate it."
        )
    return sort_df(pd.read_csv(path))


def get_common_seeds(results_base, methods):
    """Find seeds that exist for ALL methods."""
    seed_sets = []
    for method in methods:
        method_dir = os.path.join(results_base, method)
        if not os.path.isdir(method_dir):
            raise FileNotFoundError(f"Method directory not found: {method_dir}")
        seeds = set()
        for f in os.listdir(method_dir):
            if f.endswith("_test.csv") and f.startswith("seed_"):
                seeds.add(f.split("_test.csv")[0])  # e.g. "seed_42"
        seed_sets.append(seeds)
    common = seed_sets[0]
    for s in seed_sets[1:]:
        common = common & s
    return sorted(common)


def fit_ensemble_weights(val_dfs, methods, min_rows=20):
    """Learn per-date convex weights on validation predictions."""
    n_models = len(val_dfs)
    uniform = np.full(n_models, 1.0 / n_models)
    weights = {}

    for pred_col, truth_col in zip(PRED_COLS, TRUTH_COLS):
        X = np.column_stack([df[pred_col].values for df in val_dfs])
        y = val_dfs[0][truth_col].values
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)

        if valid.sum() < min_rows:
            weights[pred_col] = uniform.copy()
        else:
            weights[pred_col] = _fit_convex_weights(X[valid], y[valid])

    return weights


def apply_ensemble_weights(dfs, weights):
    """Apply learned weights to produce a blended DataFrame."""
    df_out = dfs[0].copy()
    for col in PRED_COLS:
        X = np.column_stack([df[col].values for df in dfs])
        df_out[col] = X @ weights[col]
    return df_out


def compute_mae(df):
    """Compute per-date and mean MAE from a results DataFrame."""
    maes = {}
    for pred_col, truth_col in zip(PRED_COLS, TRUTH_COLS):
        date = pred_col.split("_")[0]
        maes[date] = np.mean(np.abs(df[pred_col].values - df[truth_col].values))
    maes["Mean"] = np.mean(list(maes.values()))
    return maes


def main():
    parser = argparse.ArgumentParser(
        description="Fit convex ensemble from result CSVs (per-seed)"
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", required=True,
        help="Model group names (directory names under results/), e.g. "
             "transformer_1d_paper_1.0 presto_1.0",
    )
    parser.add_argument(
        "--selected_months", type=int, nargs="+", default=[3, 6, 9, 12],
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Ensemble name for output files.",
    )
    args = parser.parse_args()

    months_sub = f"m{months_to_str(args.selected_months)}"
    results_base = os.path.join("results", months_sub)

    if args.name is None:
        short_names = []
        for m in args.methods:
            short = m.replace("_1.0", "").replace("prithvi_final_100m_crop32", "prithvi")
            short = short.replace("transformer_1d_paper", "transformer")
            short_names.append(short)
        args.name = "ensemble_" + "_".join(short_names)

    print(f"Ensemble name: {args.name}")
    print(f"Methods: {args.methods}")
    print(f"Results dir: {results_base}")

    # Find common seeds
    common_seeds = get_common_seeds(results_base, args.methods)
    print(f"Common seeds: {common_seeds}")
    if not common_seeds:
        print("No common seeds found across all methods. Exiting.")
        return

    # Output directory for ensemble results
    ensemble_results_dir = get_results_dir(args.selected_months, group_name=args.name)

    all_weights = {}

    for seed in common_seeds:
        print(f"\n{'='*60}")
        print(f"  Seed: {seed}")
        print(f"{'='*60}")

        # Load train, val, and test CSVs for this seed
        train_dfs = [load_seed_csv(results_base, m, seed, "train") for m in args.methods]
        val_dfs = [load_seed_csv(results_base, m, seed, "val") for m in args.methods]
        test_dfs = [load_seed_csv(results_base, m, seed, "test") for m in args.methods]

        # Verify row alignment
        for split_name, dfs in [("train", train_dfs), ("val", val_dfs), ("test", test_dfs)]:
            for i, m in enumerate(args.methods):
                for col in ["years", "HLStile", "SiteID", "row", "col"]:
                    if not (dfs[0][col] == dfs[i][col]).all():
                        raise ValueError(f"{split_name} row mismatch: {args.methods[0]} vs {m} on '{col}'")

        # Fit weights on training data
        weights = fit_ensemble_weights(train_dfs, args.methods)
        all_weights[seed] = weights

        print("\nWeights (fitted on training):")
        for col, w in weights.items():
            date = col.split("_")[0]
            w_str = "  ".join(f"{args.methods[i]}: {w[i]:.3f}" for i in range(len(args.methods)))
            print(f"  {date}: {w_str}")

        # Apply to val and test
        val_blended = apply_ensemble_weights(val_dfs, weights)
        test_blended = apply_ensemble_weights(test_dfs, weights)

        # Report MAE
        val_maes = compute_mae(val_blended)
        print("\nVal MAE:")
        print(f"  ENSEMBLE: " + "  ".join(f"{k}: {v:.1f}" for k, v in val_maes.items()))

        print("\nTest MAE (individual):")
        for m, df in zip(args.methods, test_dfs):
            maes = compute_mae(df)
            print(f"  {m}: " + "  ".join(f"{k}: {v:.1f}" for k, v in maes.items()))

        test_maes = compute_mae(test_blended)
        print(f"  ENSEMBLE: " + "  ".join(f"{k}: {v:.1f}" for k, v in test_maes.items()))

        # Save per-seed CSVs
        val_blended.to_csv(os.path.join(ensemble_results_dir, f"{seed}_val.csv"), index=False)
        test_blended.to_csv(os.path.join(ensemble_results_dir, f"{seed}_test.csv"), index=False)
        print(f"Saved: {ensemble_results_dir}/{seed}_val.csv")
        print(f"Saved: {ensemble_results_dir}/{seed}_test.csv")

    # Save ensemble info JSON
    ensemble_dir = os.path.join("data", "ensembles", months_sub)
    os.makedirs(ensemble_dir, exist_ok=True)
    ensemble_path = os.path.join(ensemble_dir, f"{args.name}.json")

    ensemble_info = {
        "name": args.name,
        "methods": args.methods,
        "selected_months": args.selected_months,
        "weights_per_seed": {
            seed: {col: w.tolist() for col, w in weights.items()}
            for seed, weights in all_weights.items()
        },
    }
    with open(ensemble_path, "w") as f:
        json.dump(ensemble_info, f, indent=2)
    print(f"\nSaved ensemble info: {ensemble_path}")


if __name__ == "__main__":
    main()
