"""Ablate sliding-window stride for Prithvi crop32 across month subsets.

Loads the best checkpoint per month subset, evaluates on val and test
at different strides, and reports MAE and time per tile.

Usage:
    python misc_scripts/stride_ablation.py [--device cuda]
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
import argparse
import torch
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader

import path_config
from lib.utils import (
    get_data_paths, get_masks_paper, eval_data_loader_crops,
    months_to_str, get_months_subdir, get_results_dir, build_model,
)
from lib.dataloaders.centroid_tile_dataset import CentroidTileDataset

MONTH_SUBSETS = {
    4: [3, 6, 9, 12],
    8: [3, 4, 5, 6, 7, 8, 9, 10],
    12: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}
GROUP = "prithvi_pretrained_multiscale_crops_conv3d_crop32_1.0"
CROP_SIZE = 32
STRIDES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def main():
    parser = argparse.ArgumentParser(description="Stride ablation for Prithvi crop32")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = args.device

    results = []

    for n_months, selected_months in MONTH_SUBSETS.items():
        months_sub = get_months_subdir(selected_months)
        n_timesteps = len(selected_months)
        m_str = months_to_str(selected_months)
        file_suffix = f"_m{m_str}"

        # Load best checkpoint (first seed)
        results_dir = get_results_dir(selected_months, group_name=GROUP)
        best_params_path = os.path.join(results_dir, "best_params.csv")
        if not os.path.exists(best_params_path):
            print(f"No best_params.csv for {n_months} months, skipping.")
            continue

        best_param_df = pd.read_csv(best_params_path)

        print(f"\n{'='*70}")
        print(f"  {n_months} months ({len(best_param_df)} seeds)")
        print(f"{'='*70}")

        # Load val and test dataloaders (shared across seeds)
        dataloaders = {}
        for split_name, data_split in [("val", "validation"), ("test", "testing")]:
            data_path = get_data_paths(data_split, 1.0, selected_months)
            dataset = CentroidTileDataset(data_path, split=data_split, data_percentage=1.0,
                                   n_timesteps=n_timesteps, file_suffix=file_suffix)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
            masks = get_masks_paper("train" if split_name == "val" else split_name)
            dataloaders[split_name] = (dataloader, masks)

        for split_name in ["val", "test"]:
            dataloader, masks = dataloaders[split_name]

            for stride in STRIDES:
                seed_maes = []
                seed_times = []

                for _, row_bp in best_param_df.iterrows():
                    seed = row_bp["Seed"]
                    best_param = row_bp["Best Param"]

                    config_dir = os.path.join(path_config.get_checkpoint_root(), months_sub, GROUP, seed)
                    ckpt_path = os.path.join(config_dir, best_param)

                    model, crop_size = build_model(GROUP, best_param, n_timesteps)
                    model = model.to(device)
                    ckpt = torch.load(ckpt_path)
                    model.load_state_dict(ckpt["model_state_dict"])

                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    acc, eval_loss, n_samples = eval_data_loader_crops(
                        dataloader, model, device, masks,
                        crop_size=CROP_SIZE, stride=stride, pixel_weighted=False,
                    )

                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    n_tiles = len(dataloader)
                    time_per_tile = elapsed / n_tiles

                    seed_maes.append(acc)
                    seed_times.append(time_per_tile)

                    del model
                    torch.cuda.empty_cache()

                # Average across seeds
                avg_acc = {}
                for date in seed_maes[0].keys():
                    avg_acc[date] = np.mean([s[date] for s in seed_maes])
                mean_mae = np.mean(list(avg_acc.values()))
                avg_time = np.mean(seed_times)

                print(f"  {split_name:>4s} | stride={stride:>2d} | MAE: ", end="")
                for date, mae in avg_acc.items():
                    print(f"{date}: {mae:.1f}  ", end="")
                print(f"Mean: {mean_mae:.1f} | {avg_time:.2f}s/tile")

                row = {
                    "Months": n_months,
                    "Split": split_name,
                    "Stride": stride,
                    "Mean MAE": mean_mae,
                    "Time/tile (s)": avg_time,
                }
                for date, mae in avg_acc.items():
                    row[date] = mae
                results.append(row)

    # Summary table
    df = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for n_months in MONTH_SUBSETS:
        for split in ["val", "test"]:
            sub = df[(df["Months"] == n_months) & (df["Split"] == split)]
            if len(sub) == 0:
                continue
            print(f"\n{n_months} months — {split}:")
            print(sub[["Stride", "Mean MAE", "Time/tile (s)"]].to_string(index=False))

    df.to_csv("results/stride_ablation.csv", index=False)
    print(f"\nSaved: results/stride_ablation.csv")


if __name__ == "__main__":
    main()
