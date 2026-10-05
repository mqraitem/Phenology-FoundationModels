"""Metrics and aggregation helpers for stratified phenology evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def masked_group_errors(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    group_ids: np.ndarray,
    excluded_ids: Iterable[int] = (0,),
) -> list[dict[str, float | int]]:
    """Pool linear absolute error across phases for each spatial group ID."""
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)
    valid = np.asarray(valid, dtype=bool)
    group_ids = np.asarray(group_ids)
    if prediction.shape != truth.shape or valid.shape != truth.shape:
        raise ValueError("prediction, truth, and valid must have matching shapes")
    if truth.ndim != 3 or group_ids.shape != truth.shape[1:]:
        raise ValueError("expected phase x height x width values and a height x width group map")

    excluded = set(excluded_ids)
    error = np.abs(prediction.astype(np.float64) - truth.astype(np.float64))
    rows = []
    for group_id in np.unique(group_ids[np.any(valid, axis=0)]):
        group_id = int(group_id)
        if group_id in excluded:
            continue
        mask = valid & (group_ids[None, :, :] == group_id)
        count = int(mask.sum())
        if not count:
            continue
        error_sum = float(error[mask].sum(dtype=np.float64))
        rows.append({
            "group_id": group_id,
            "abs_error_sum": error_sum,
            "n_valid": count,
            "mae_days": error_sum / count,
        })
    return rows


def summarize_seed_pooled(tile_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Pool absolute error over all valid observations within seed, then average seeds equally.

    Each tile-year contributes in proportion to its number of valid observations,
    so small patches or boundary slivers cannot carry the weight of a full tile-year.
    """
    required = {*group_cols, "seed", "abs_error_sum", "n_valid"}
    missing = required.difference(tile_df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    per_seed = (
        tile_df.groupby([*group_cols, "seed"], as_index=False)
        .agg(abs_error_sum=("abs_error_sum", "sum"), n_valid=("n_valid", "sum"))
    )
    per_seed["seed_mean"] = per_seed["abs_error_sum"] / per_seed["n_valid"]
    summary = (
        per_seed.groupby(group_cols, as_index=False)
        .agg(
            mean=("seed_mean", "mean"),
            seed_std=("seed_mean", lambda x: float(np.std(x, ddof=0))),
            n_seeds=("seed", "nunique"),
        )
    )
    return summary
