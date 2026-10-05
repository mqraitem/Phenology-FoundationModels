"""Score reference spatial variation per test tile-year and summarize MAE by tertile.

For each phenophase, the score is the median absolute DOY difference between valid
horizontal and vertical pixel pairs, averaged over spatial lags 1, 2, 4, and 8. Each
phase score is standardized across the 48 test tile-years, and the four standardized
scores are averaged. Tile-years are split into three equal-count tertiles, and MAE
(tile-year mean over phenophases, averaged within seed, then across seeds) is
summarized per tertile and model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = ["seed_42", "seed_123", "seed_456"]
MODEL_GROUPS = {
    "temporal_transformer": "transformer_1d_paper_nl3_1.0",
    "presto": "presto_1.0",
    "prithvi": "prithvi_final_100m_crop32_1.0",
}
PHASE_CODES = ["G", "M", "S", "D"]
BIN_ORDER = ["Smoothest", "Intermediate", "Roughest"]


def robust_multiscale_score(doy: np.ndarray, valid: np.ndarray, lags=(1, 2, 4, 8), min_pairs=100) -> float:
    """Median right/down absolute DOY difference, averaged equally over spatial lags."""
    lag_scores = []
    for lag in lags:
        right_valid = valid[:, lag:] & valid[:, :-lag]
        down_valid = valid[lag:, :] & valid[:-lag, :]
        differences = np.concatenate([
            np.abs(doy[:, lag:] - doy[:, :-lag])[right_valid],
            np.abs(doy[lag:, :] - doy[:-lag, :])[down_valid],
        ])
        if differences.size < min_pairs:
            return np.nan
        lag_scores.append(float(np.median(differences)))
    return float(np.mean(lag_scores))


def score_tiles(tile_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(tile_dir.glob("*.npz")):
        tile = np.load(path, allow_pickle=True)
        gt, valid = tile["ground_truth_doy"], tile["ground_truth_valid"]
        row = {"tile_id": str(tile["tile_id"]), "site_id": str(tile["site_id"]),
               "year": int(str(tile["year"]))}
        for i, phase in enumerate(PHASE_CODES):
            row[f"score_{phase}"] = robust_multiscale_score(gt[i], valid[i])
        rows.append(row)
    scores = pd.DataFrame(rows)
    phase_cols = [f"score_{phase}" for phase in PHASE_CODES]
    standardized = (scores[phase_cols] - scores[phase_cols].mean()) / scores[phase_cols].std(ddof=0)
    scores["score"] = standardized.mean(axis=1)
    scores = scores.sort_values(["score", "tile_id"]).reset_index(drop=True)
    scores["bin"] = np.repeat(BIN_ORDER, len(scores) // len(BIN_ORDER))
    return scores


def load_tile_mae(results_dir: Path) -> pd.DataFrame:
    rows = []
    for model, group in MODEL_GROUPS.items():
        for seed in SEEDS:
            frame = pd.read_csv(results_dir / group / f"{seed}_test.csv")
            frame["tile_id"] = (frame["years"].astype(str) + "_" + frame["SiteID"].astype(str)
                                + "_" + frame["HLStile"].astype(str))
            for tile_id, tile in frame.groupby("tile_id"):
                mae = np.mean([np.abs(tile[f"{p}_pred_DOY"] - tile[f"{p}_truth_DOY"]).mean()
                               for p in PHASE_CODES])
                rows.append({"tile_id": tile_id, "model": model, "seed": seed, "mae_days": float(mae)})
    return pd.DataFrame(rows)


def summarize_by_tertile(scores: pd.DataFrame, tile_mae: pd.DataFrame) -> pd.DataFrame:
    merged = scores[["tile_id", "bin"]].merge(tile_mae, on="tile_id", validate="one_to_many")
    per_seed = (merged.groupby(["bin", "model", "seed"], as_index=False)
                .agg(seed_mae=("mae_days", "mean"), n_tile_years=("tile_id", "nunique")))
    return (per_seed.groupby(["bin", "model"], as_index=False)
            .agg(mean_mae=("seed_mae", "mean"),
                 seed_std=("seed_mae", lambda x: float(np.std(x, ddof=0))),
                 n_seeds=("seed", "nunique"), n_tile_years=("n_tile_years", "min")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-dir", type=Path, default=Path("data/stratified_analysis/tile_layers"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/m3-6-9-12"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/stratified_analysis/spatial_variation"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scores = score_tiles(args.tile_dir)
    assert len(scores) == 48, f"expected 48 test tile-years, found {len(scores)}"
    tile_mae = load_tile_mae(args.results_dir)
    assert tile_mae.groupby(["model", "seed"]).size().eq(48).all()
    summary = summarize_by_tertile(scores, tile_mae)

    scores.to_csv(args.out_dir / "tile_scores.csv", index=False)
    summary.to_csv(args.out_dir / "mae_by_tertile.csv", index=False)
    print(summary.pivot(index="bin", columns="model", values="mean_mae").reindex(BIN_ORDER).round(2))
    print(f"Saved tables to {args.out_dir}")


if __name__ == "__main__":
    main()
