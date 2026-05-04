"""Pre-generate all shared cached files (data paths, mean/stds, pixel caches, tile banks).

Run this BEFORE submitting batch jobs to avoid race conditions.

Usage:
    python regenerate_caches.py
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from lib.utils import get_data_paths, months_to_str
from lib.dataloaders.dataloaders_pixels import CycleDatasetPixels
from lib.dataloaders.dataloaders_crops import CycleDatasetCrops
from lib.dataloaders.dataloaders import CycleDataset
from lib.dataloaders.dataloaders_pixels_subsampled import CycleDatasetPixelsSubsampled
from lib.dataloaders.dataloaders_pixels_pixellatlon import CycleDatasetPixelsPixelLatLon

MONTH_SUBSETS = [
    [3, 6, 9, 12],
    # [3, 4, 5, 6, 7, 8, 9, 10],
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
]

DATA_PERCENTAGE = 1.0

for selected_months in MONTH_SUBSETS:
    months_str = months_to_str(selected_months)
    file_suffix = f"_m{months_str}"
    n_timesteps = len(selected_months)
    print(f"\n{'='*60}")
    print(f"  Months: {selected_months}")
    print(f"{'='*60}")

    # 1. Data paths (HLS, all splits)
    print("\n--- Data paths ---")
    for split in ["training", "validation", "testing"]:
        path_hls = get_data_paths(split, DATA_PERCENTAGE, selected_months)
        print(f"  {split}: HLS={len(path_hls)} tiles")

    # 2. HLS mean/stds + pixel cache (training split)
    print("\n--- HLS pixel cache (triggers mean/std computation) ---")
    path_train_hls = get_data_paths("training", DATA_PERCENTAGE, selected_months)
    pixel_ds = CycleDatasetPixels(
        path_train_hls, split="training",
        data_percentage=DATA_PERCENTAGE,
        n_timesteps=n_timesteps,
        file_suffix=file_suffix,
    )
    print(f"  HLS pixels: {len(pixel_ds)} samples")
    del pixel_ds

    # 3. HLS crop tile bank (training split)
    print("\n--- HLS crop tile bank ---")
    crop_ds = CycleDatasetCrops(
        path_train_hls, split="training",
        crop_size=48,
        data_percentage=DATA_PERCENTAGE,
        n_timesteps=n_timesteps,
        file_suffix=file_suffix,
        epoch_length=1,
    )
    print(f"  Crop tile bank: {len(crop_ds.all_images)} tiles")
    del crop_ds

    # 4. HLS pixel cache with per-pixel lat/lon (for Presto)
    print("\n--- HLS pixel cache with per-pixel lat/lon ---")
    pixel_ds_ll = CycleDatasetPixelsPixelLatLon(
        path_train_hls, split="training",
        data_percentage=DATA_PERCENTAGE,
        n_timesteps=n_timesteps,
        file_suffix=file_suffix,
        skip_normalization=True,
    )
    print(f"  HLS pixels (pixellatlon): {len(pixel_ds_ll)} samples")
    del pixel_ds_ll

    # 5. HLS subsampled pixel cache (for paper-matched transformer)
    print(f"\n--- HLS subsampled pixel cache (stride={CycleDatasetPixelsSubsampled.PIXEL_STRIDE}) ---")
    pixel_ds_sub = CycleDatasetPixelsSubsampled(
        path_train_hls, split="training",
        data_percentage=DATA_PERCENTAGE,
        n_timesteps=n_timesteps,
        file_suffix=file_suffix,
    )
    print(f"  HLS subsampled pixels (stride={CycleDatasetPixelsSubsampled.PIXEL_STRIDE}): {len(pixel_ds_sub)} samples")
    del pixel_ds_sub

print(f"\n{'='*60}")
print("  All caches regenerated successfully!")
print(f"{'='*60}")
