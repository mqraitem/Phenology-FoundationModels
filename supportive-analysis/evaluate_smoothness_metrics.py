"""Select a ground-truth spatial-variation metric and form test-set tertiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SEEDS = ["seed_42", "seed_123", "seed_456"]
MODEL_GROUPS = {
    "temporal_transformer": "transformer_1d_paper_1.0",
    "presto": "presto_1.0",
    "prithvi": "prithvi_final_100m_crop32_1.0",
}
PHASE_CODES = ["G", "M", "S", "D"]
BIN_ORDER = ["Smoothest", "Intermediate", "Roughest"]
METRICS = ["global", "local", "combined", "robust_multiscale"]


def global_score(doy: np.ndarray, valid: np.ndarray, grid_sizes=range(8, 17)) -> float:
    """Mean std. dev. of valid block means over 8x8 through 16x16 grids."""
    height, width = doy.shape
    scale_scores = []
    for n in grid_sizes:
        row_edges = np.linspace(0, height, n + 1).astype(int)
        col_edges = np.linspace(0, width, n + 1).astype(int)
        block_means = []
        for row in range(n):
            for col in range(n):
                block_valid = valid[row_edges[row]:row_edges[row + 1], col_edges[col]:col_edges[col + 1]]
                if not block_valid.any():
                    continue
                block = doy[row_edges[row]:row_edges[row + 1], col_edges[col]:col_edges[col + 1]]
                block_means.append(float(block[block_valid].mean()))
        if len(block_means) >= 2:
            scale_scores.append(float(np.std(block_means, ddof=0)))
    return float(np.mean(scale_scores)) if scale_scores else np.nan


def local_score(doy: np.ndarray, valid: np.ndarray) -> float:
    """Student lag-1 directional madogram: raw right/down absolute DOY difference."""
    right_valid = valid[:, 1:] & valid[:, :-1]
    down_valid = valid[1:, :] & valid[:-1, :]
    right = np.abs(doy[:, 1:] - doy[:, :-1])[right_valid]
    down = np.abs(doy[1:, :] - doy[:-1, :])[down_valid]
    count = right.size + down.size
    return float((right.sum(dtype=np.float64) + down.sum(dtype=np.float64)) / count) if count else np.nan


def robust_multiscale_score(
    doy: np.ndarray, valid: np.ndarray, lags=(1, 2, 4, 8), min_pairs=100
) -> tuple[float, int]:
    """Median right/down DOY difference, averaged equally over spatial lags."""
    lag_scores = []
    minimum_count = np.inf
    for lag in lags:
        right_valid = valid[:, lag:] & valid[:, :-lag]
        down_valid = valid[lag:, :] & valid[:-lag, :]
        right = np.abs(doy[:, lag:] - doy[:, :-lag])[right_valid]
        down = np.abs(doy[lag:, :] - doy[:-lag, :])[down_valid]
        differences = np.concatenate([right, down])
        minimum_count = min(minimum_count, differences.size)
        if differences.size < min_pairs:
            return np.nan, int(minimum_count)
        lag_scores.append(float(np.median(differences)))
    return float(np.mean(lag_scores)), int(minimum_count)


def compute_gt_metrics(tile_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(tile_dir.glob("*.npz")):
        tile = np.load(path, allow_pickle=True)
        gt = tile["ground_truth_doy"]
        valid = tile["ground_truth_valid"]
        global_phases = [global_score(gt[i], valid[i]) for i in range(gt.shape[0])]
        local_phases = [local_score(gt[i], valid[i]) for i in range(gt.shape[0])]
        robust_results = [robust_multiscale_score(gt[i], valid[i]) for i in range(gt.shape[0])]
        row = {
            "tile_id": str(tile["tile_id"]),
            "site_id": str(tile["site_id"]),
            "year": int(str(tile["year"])),
            "global": float(np.mean(global_phases)),
            "local": float(np.mean(local_phases)),
            "robust_min_pairs": min(result[1] for result in robust_results),
        }
        for phase, (score, _) in zip(PHASE_CODES, robust_results):
            row[f"robust_{phase}"] = score
        rows.append(row)
    metrics = pd.DataFrame(rows)
    for column in ["global", "local"]:
        metrics[f"{column}_z"] = (metrics[column] - metrics[column].mean()) / metrics[column].std(ddof=0)
    metrics["combined"] = 0.5 * (metrics["global_z"] + metrics["local_z"])
    robust_z = []
    for phase in PHASE_CODES:
        column = metrics[f"robust_{phase}"]
        robust_z.append((column - column.mean()) / column.std(ddof=0))
    metrics["robust_multiscale"] = pd.concat(robust_z, axis=1).mean(axis=1)
    return metrics


def load_tile_mae(results_dir: Path) -> pd.DataFrame:
    rows = []
    for model, group in MODEL_GROUPS.items():
        for seed in SEEDS:
            path = results_dir / group / f"{seed}_test.csv"
            frame = pd.read_csv(path)
            frame["tile_id"] = (
                frame["years"].astype(int).astype(str) + "_" + frame["SiteID"].astype(str)
                + "_" + frame["HLStile"].astype(str)
            )
            for tile_id, tile in frame.groupby("tile_id"):
                phase_maes = [
                    np.abs(tile[f"{phase}_pred_DOY"] - tile[f"{phase}_truth_DOY"]).mean()
                    for phase in PHASE_CODES
                ]
                rows.append({
                    "tile_id": tile_id, "model": model, "seed": seed,
                    "mae_days": float(np.mean(phase_maes)),
                })
    return pd.DataFrame(rows)


def assign_tertiles(metrics: pd.DataFrame) -> pd.DataFrame:
    long_rows = []
    for metric in METRICS:
        ranked = metrics.sort_values([metric, "tile_id"]).reset_index(drop=True).copy()
        ranked["bin"] = pd.cut(
            np.arange(len(ranked)), bins=[-1, 15, 31, 47], labels=BIN_ORDER
        ).astype(str)
        ranked["metric"] = metric
        ranked["score"] = ranked[metric]
        long_rows.append(ranked[["tile_id", "site_id", "year", "metric", "score", "bin"]])
    return pd.concat(long_rows, ignore_index=True)


def summarize(metric_bins: pd.DataFrame, tile_mae: pd.DataFrame):
    merged = metric_bins.merge(tile_mae, on="tile_id", validate="many_to_many")
    per_seed_bin = (
        merged.groupby(["metric", "bin", "model", "seed"], as_index=False, observed=True)
        .agg(seed_mae=("mae_days", "mean"), n_tile_years=("tile_id", "nunique"))
    )
    bin_summary = (
        per_seed_bin.groupby(["metric", "bin", "model"], as_index=False, observed=True)
        .agg(
            mean_mae=("seed_mae", "mean"),
            seed_std=("seed_mae", lambda x: float(np.std(x, ddof=0))),
            n_seeds=("seed", "nunique"), n_tile_years=("n_tile_years", "min"),
        )
    )

    continuous_rows = []
    for (metric, model, seed), group in merged.groupby(["metric", "model", "seed"]):
        rho = float(spearmanr(group["score"], group["mae_days"]).statistic)
        coefficients = np.polyfit(group["score"], group["mae_days"], deg=1)
        fitted = np.polyval(coefficients, group["score"])
        residual = float(np.square(group["mae_days"] - fitted).sum())
        total = float(np.square(group["mae_days"] - group["mae_days"].mean()).sum())
        continuous_rows.append({
            "metric": metric, "model": model, "seed": seed,
            "spearman_rho": rho, "linear_r2": 1.0 - residual / total,
        })
    continuous = pd.DataFrame(continuous_rows)
    explanatory = (
        continuous.groupby(["metric", "model"], as_index=False)
        .agg(
            mean_spearman=("spearman_rho", "mean"),
            std_spearman=("spearman_rho", lambda x: float(np.std(x, ddof=0))),
            mean_r2=("linear_r2", "mean"),
            std_r2=("linear_r2", lambda x: float(np.std(x, ddof=0))),
        )
    )
    return merged, bin_summary, continuous, explanatory


def summarize_model_gaps(metric_bins: pd.DataFrame, tile_mae: pd.DataFrame):
    """Measure how smoothness explains paired tile-level differences between models."""
    wide = tile_mae.pivot(index=["tile_id", "seed"], columns="model", values="mae_days")
    comparisons = {
        "presto_vs_temporal": wide["temporal_transformer"] - wide["presto"],
        "prithvi_vs_temporal": wide["temporal_transformer"] - wide["prithvi"],
        "presto_vs_prithvi": wide["prithvi"] - wide["presto"],
    }
    gaps = pd.concat(comparisons, names=["comparison"]).rename("mae_improvement").reset_index()
    merged = metric_bins.merge(gaps, on="tile_id", validate="many_to_many")

    bin_per_seed = (
        merged.groupby(["metric", "bin", "comparison", "seed"], as_index=False, observed=True)
        .agg(mean_improvement=("mae_improvement", "mean"))
    )
    bin_summary = (
        bin_per_seed.groupby(["metric", "bin", "comparison"], as_index=False, observed=True)
        .agg(
            mean_improvement=("mean_improvement", "mean"),
            seed_std=("mean_improvement", lambda x: float(np.std(x, ddof=0))),
        )
    )

    continuous_rows = []
    for (metric, comparison, seed), group in merged.groupby(["metric", "comparison", "seed"]):
        rho = float(spearmanr(group["score"], group["mae_improvement"]).statistic)
        coefficients = np.polyfit(group["score"], group["mae_improvement"], deg=1)
        fitted = np.polyval(coefficients, group["score"])
        residual = float(np.square(group["mae_improvement"] - fitted).sum())
        total = float(np.square(group["mae_improvement"] - group["mae_improvement"].mean()).sum())
        continuous_rows.append({
            "metric": metric, "comparison": comparison, "seed": seed,
            "spearman_rho": rho, "linear_r2": 1.0 - residual / total,
        })
    continuous = pd.DataFrame(continuous_rows)
    explanatory = (
        continuous.groupby(["metric", "comparison"], as_index=False)
        .agg(mean_spearman=("spearman_rho", "mean"), mean_r2=("linear_r2", "mean"))
    )
    return bin_summary, continuous, explanatory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-dir", type=Path,
                        default=Path("student_test_tiles_m3-6-9-12/data/m3-6-9-12/test"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/m3-6-9-12"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/stratified_analysis/smoothness_metric_comparison"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_gt_metrics(args.tile_dir)
    assert len(metrics) == 48
    bins = assign_tertiles(metrics)
    tile_mae = load_tile_mae(args.results_dir)
    assert tile_mae.groupby(["model", "seed"]).size().eq(48).all()
    merged, bin_summary, continuous, explanatory = summarize(bins, tile_mae)
    gap_bins, gap_continuous, gap_explanatory = summarize_model_gaps(bins, tile_mae)

    metrics.to_csv(args.out_dir / "gt_smoothness_metrics.csv", index=False)
    bins.to_csv(args.out_dir / "gt_smoothness_tertiles.csv", index=False)
    bin_summary.to_csv(args.out_dir / "mae_by_smoothness_tertile.csv", index=False)
    continuous.to_csv(args.out_dir / "continuous_explanatory_power_per_seed.csv", index=False)
    explanatory.to_csv(args.out_dir / "continuous_explanatory_power_summary.csv", index=False)
    gap_bins.to_csv(args.out_dir / "model_gap_by_smoothness_tertile.csv", index=False)
    gap_continuous.to_csv(args.out_dir / "model_gap_explanatory_power_per_seed.csv", index=False)
    gap_explanatory.to_csv(args.out_dir / "model_gap_explanatory_power_summary.csv", index=False)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    print("\n=== GT metric correlations ===")
    print(metrics[["global", "local", "combined"]].corr(method="spearman").round(3).to_string())
    print("\n=== MAE by GT smoothness tertile ===")
    display_bins = bin_summary.copy()
    display_bins["bin"] = pd.Categorical(display_bins["bin"], BIN_ORDER, ordered=True)
    print(display_bins.sort_values(["metric", "bin", "model"]).round(3).to_string(index=False))
    print("\n=== Continuous explanatory power (mean across seeds) ===")
    print(explanatory.sort_values(["metric", "model"]).round(3).to_string(index=False))
    print("\n=== Model MAE improvement by GT smoothness tertile ===")
    display_gaps = gap_bins.copy()
    display_gaps["bin"] = pd.Categorical(display_gaps["bin"], BIN_ORDER, ordered=True)
    print(display_gaps.sort_values(["comparison", "metric", "bin"]).round(3).to_string(index=False))
    print("\n=== Smoothness explanation of paired model differences ===")
    print(gap_explanatory.sort_values(["comparison", "metric"]).round(3).to_string(index=False))

    spread = (
        bin_summary.pivot(index=["metric", "model"], columns="bin", values="mean_mae")
        .assign(rough_minus_smooth=lambda x: x["Roughest"] - x["Smoothest"])
        .reset_index()
    )
    print("\n=== Roughest minus smoothest MAE ===")
    print(spread[["metric", "model", "rough_minus_smooth"]].round(3).to_string(index=False))
    print(f"\nSaved tables to {args.out_dir}")


if __name__ == "__main__":
    main()
