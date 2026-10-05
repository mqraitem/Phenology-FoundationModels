"""Result loading helpers used by the paper notebook."""

import os

import pandas as pd

from lib.utils import get_results_dir


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["years", "HLStile", "SiteID", "row", "col", "version"]
    ).reset_index(drop=True)


def results_file(split="test", selected_months=(3, 6, 9, 12), seeds=None):
    """Load representative and per-seed result frames for one temporal setting."""
    base_dir = get_results_dir(selected_months=selected_months)
    results = {}
    results_per_seed = {}

    for model_dir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        seed_dfs = []
        for filename in sorted(os.listdir(model_path)):
            if not filename.endswith(f"_{split}.csv") or not filename.startswith("seed_"):
                continue
            seed_name = filename.removesuffix(f"_{split}.csv")
            if seeds is not None and seed_name not in seeds and seed_name != "seed_all":
                continue
            seed_dfs.append(pd.read_csv(os.path.join(model_path, filename)))

        if not seed_dfs:
            continue
        key = f"{model_dir}_{split}"
        results[key] = sort_df(seed_dfs[0])
        results_per_seed[key] = [sort_df(df) for df in seed_dfs]

    return results, results_per_seed
