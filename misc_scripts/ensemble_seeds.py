"""Fit a global convex ensemble across seeds of the same model.

Fits per-date convex weights across seeds on training pixels,
applies them to val and test, and saves a single blended result.

Output structure:
    results/{months_sub}/{model}_seed_ensemble/seed_all_test.csv
    results/{months_sub}/{model}_seed_ensemble/seed_all_val.csv
    data/ensembles/{months_sub}/{model}_seed_ensemble.json

Usage:
    python misc_scripts/ensemble_seeds.py \
        --model transformer_1d_paper_1.0 \
        --selected_months 3 6 9 12

    python misc_scripts/ensemble_seeds.py \
        --model presto_1.0 \
        --selected_months 3 6 9 12
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


def compute_mae(df):
    maes = {}
    for pred_col, truth_col in zip(PRED_COLS, TRUTH_COLS):
        date = pred_col.split("_")[0]
        maes[date] = np.mean(np.abs(df[pred_col].values - df[truth_col].values))
    maes["Mean"] = np.mean(list(maes.values()))
    return maes


def main():
    parser = argparse.ArgumentParser(description="Fit convex ensemble across seeds of the same model")
    parser.add_argument("--model", type=str, required=True,
                        help="Model group name, e.g. transformer_1d_paper_1.0")
    parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--seeds", type=str, nargs="+", default=None,
                        help="Subset of seeds to ensemble (e.g. seed_42 seed_123 seed_456). "
                             "If omitted, uses all seeds found on disk.")
    parser.add_argument("--name_suffix", type=str, default="",
                        help="Suffix appended to the output ensemble name "
                             "(e.g. '_gA' produces <model>_seed_ensemble_gA).")
    args = parser.parse_args()

    months_sub = f"m{months_to_str(args.selected_months)}"
    model_dir = os.path.join("results", months_sub, args.model)

    if not os.path.isdir(model_dir):
        print(f"Model directory not found: {model_dir}")
        return

    # Find seeds available on disk
    available_seeds = sorted(set(
        f.split("_val.csv")[0].split("_test.csv")[0]
        for f in os.listdir(model_dir)
        if f.startswith("seed_") and f.endswith("_test.csv")
    ))

    # Filter to requested subset, if provided
    if args.seeds is not None:
        missing = [s for s in args.seeds if s not in available_seeds]
        if missing:
            print(f"Requested seeds not on disk: {missing}")
            return
        seeds = list(args.seeds)
    else:
        seeds = available_seeds

    print(f"Model: {args.model}")
    print(f"Available seeds: {available_seeds}")
    if args.seeds is not None:
        print(f"Using subset: {seeds}")
    if args.name_suffix:
        print(f"Name suffix: {args.name_suffix}")

    if len(seeds) < 2:
        print("Need at least 2 seeds for ensembling. Exiting.")
        return

    # Load per-seed CSVs
    train_dfs = []
    val_dfs = []
    test_dfs = []
    valid_seeds = []
    for seed in seeds:
        train_path = os.path.join(model_dir, f"{seed}_train.csv")
        val_path = os.path.join(model_dir, f"{seed}_val.csv")
        test_path = os.path.join(model_dir, f"{seed}_test.csv")
        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            print(f"Missing train/val/test CSV for {seed}, skipping.")
            continue
        train_dfs.append(sort_df(pd.read_csv(train_path)))
        val_dfs.append(sort_df(pd.read_csv(val_path)))
        test_dfs.append(sort_df(pd.read_csv(test_path)))
        valid_seeds.append(seed)

    seeds = valid_seeds
    if len(seeds) < 2:
        print("Not enough seeds with train/val/test CSVs. Exiting.")
        return

    # Verify row alignment
    for split_name, dfs in [("train", train_dfs), ("val", val_dfs), ("test", test_dfs)]:
        for i in range(1, len(dfs)):
            for col in ["years", "HLStile", "SiteID", "row", "col"]:
                if not (dfs[0][col] == dfs[i][col]).all():
                    raise ValueError(f"{split_name} row mismatch on '{col}' between seed 0 and {i}")

    # Fit per-date weights on training
    print("\nFitting convex weights on training...")
    weights = {}
    n_seeds = len(train_dfs)
    uniform = np.full(n_seeds, 1.0 / n_seeds)

    for pred_col, truth_col in zip(PRED_COLS, TRUTH_COLS):
        X = np.column_stack([df[pred_col].values for df in train_dfs])
        y = train_dfs[0][truth_col].values
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if valid.sum() < 20:
            weights[pred_col] = uniform.copy()
        else:
            weights[pred_col] = _fit_convex_weights(X[valid], y[valid])

    print("\nWeights per date:")
    for col, w in weights.items():
        date = col.split("_")[0]
        w_str = "  ".join(f"{seeds[i]}: {w[i]:.3f}" for i in range(len(seeds)))
        print(f"  {date}: {w_str}")

    # Apply weights
    val_blended = val_dfs[0].copy()
    test_blended = test_dfs[0].copy()
    for col in PRED_COLS:
        X_val = np.column_stack([df[col].values for df in val_dfs])
        X_test = np.column_stack([df[col].values for df in test_dfs])
        val_blended[col] = X_val @ weights[col]
        test_blended[col] = X_test @ weights[col]

    # Report
    print("\nTest MAE (individual seeds):")
    for seed, df in zip(seeds, test_dfs):
        maes = compute_mae(df)
        print(f"  {seed}: " + "  ".join(f"{k}: {v:.1f}" for k, v in maes.items()))

    maes = compute_mae(test_blended)
    print(f"  ENSEMBLE: " + "  ".join(f"{k}: {v:.1f}" for k, v in maes.items()))

    # Save
    ensemble_name = f"{args.model}_seed_ensemble{args.name_suffix}"
    out_dir = get_results_dir(args.selected_months, group_name=ensemble_name)
    val_blended.to_csv(os.path.join(out_dir, "seed_all_val.csv"), index=False)
    test_blended.to_csv(os.path.join(out_dir, "seed_all_test.csv"), index=False)
    print(f"\nSaved: {out_dir}/seed_all_val.csv")
    print(f"Saved: {out_dir}/seed_all_test.csv")

    # Save weights JSON
    ensemble_dir = os.path.join("data", "ensembles", months_sub)
    os.makedirs(ensemble_dir, exist_ok=True)
    info = {
        "name": ensemble_name,
        "model": args.model,
        "seeds": seeds,
        "selected_months": args.selected_months,
        "weights": {col: w.tolist() for col, w in weights.items()},
        "test_mae": maes,
    }
    info_path = os.path.join(ensemble_dir, f"{ensemble_name}.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved: {info_path}")


if __name__ == "__main__":
    main()
