import numpy as np
import pandas as pd

from lib.stratified_analysis import masked_group_errors, summarize_seed_pooled


def test_masked_group_errors_pools_phases_linearly():
    truth = np.zeros((2, 2, 2), dtype=float)
    prediction = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=float)
    valid = np.ones_like(truth, dtype=bool)
    groups = np.array([[1, 1], [2, 0]])
    rows = {row["group_id"]: row for row in masked_group_errors(prediction, truth, valid, groups)}
    assert set(rows) == {1, 2}
    assert rows[1]["n_valid"] == 4
    assert rows[1]["abs_error_sum"] == 14.0
    assert rows[2]["mae_days"] == 5.0


def test_summary_pools_observations_within_seed():
    # Tile t2 is a 1-observation sliver: pooling weights it by size, not as a full tile-year.
    df = pd.DataFrame({
        "model": ["m"] * 4,
        "seed": ["a", "a", "b", "b"],
        "tile_id": ["t1", "t2", "t1", "t2"],
        "abs_error_sum": [90.0, 100.0, 180.0, 100.0],
        "n_valid": [9, 1, 9, 1],
    })
    result = summarize_seed_pooled(df, ["model"]).iloc[0]
    assert result["mean"] == 23.5  # seed a: 190/10 = 19; seed b: 280/10 = 28
    assert result["seed_std"] == 4.5
    assert result["n_seeds"] == 2
