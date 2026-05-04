"""
Pixel dataloader with per-pixel lat/lon from rasterio transforms.

Same as dataloaders_pixels.py except each pixel gets its actual geographic
coordinate instead of the tile centroid.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import rasterio
from rasterio.warp import transform as warp_transform
from lib.utils import compute_or_load_means_stds, normalize_doy
import path_config


def load_raster(path, crop=None):
    if os.path.exists(path):
        with rasterio.open(path) as src:
            img = src.read()
            if crop:
                img = img[:, -crop[0]:, -crop[1]:]
    else:
        img = np.zeros((6, 330, 330))
    return img


def load_raster_input(path, target_size=330):
    if os.path.exists(path):
        with rasterio.open(path) as src:
            img = src.read()
        return img.astype(np.float32)
    else:
        return np.zeros((6, target_size, target_size), dtype=np.float32)


def load_raster_output(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"GT raster not found: {path}")
    with rasterio.open(path) as src:
        return src.read()


def compute_pixel_latlons_from_raster(raster_path, H=330, W=330):
    """Compute per-pixel lat/lon from a rasterio file's transform and CRS.

    Returns:
        latlons: (H, W, 2) numpy array of [lat, lon] per pixel
    """
    with rasterio.open(raster_path) as src:
        tile_crs = src.crs
        tile_transform = src.transform

    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs = tile_transform.c + (cols + 0.5) * tile_transform.a
    ys = tile_transform.f + (rows + 0.5) * tile_transform.e

    xs_flat = xs.ravel().tolist()
    ys_flat = ys.ravel().tolist()
    lons, lats = warp_transform(tile_crs, 'EPSG:4326', xs_flat, ys_flat)

    latlons = np.stack([np.array(lats).reshape(H, W),
                        np.array(lons).reshape(H, W)], axis=-1)
    return latlons.astype(np.float32)


class CycleDatasetPixelsPixelLatLon(Dataset):
    def __init__(self, data_dir, split, cache_path=None, data_percentage=1.0, target_size=330,
                 regenerate=False, n_timesteps=12, file_suffix="", skip_normalization=False):
        """
        Same as CycleDatasetPixels but with per-pixel lat/lon from rasterio.
        Uses a separate cache file (suffix _v2) to avoid conflicts.
        """
        self.data_dir = data_dir
        self.split = split
        self.data_percentage = data_percentage
        self.target_size = target_size
        self.n_timesteps = n_timesteps
        self.file_suffix = file_suffix
        self.skip_normalization = skip_normalization
        self.cache_path = cache_path if cache_path is not None else self._get_cache_path()

        self.correct_indices = [2, 5, 8, 11]
        self.correct_indices = [i - 1 for i in self.correct_indices]

        if not skip_normalization:
            self.get_means_stds()
            self._means_tensor = torch.tensor(self.means, dtype=torch.float32).view(1, -1)
            self._stds_tensor = torch.tensor(self.stds, dtype=torch.float32).view(1, -1)

        if os.path.exists(self.cache_path) and not regenerate:
            print(f"[PixelDatasetPixelLatLon] Loading preprocessed dataset from {self.cache_path}")
            data = np.load(self.cache_path, allow_pickle=True)
            self.inputs = data['inputs']
            self.targets = data['targets']
            self.latlons = data['latlons']
            self.meta = data['meta']
        else:
            print(f"[PixelDatasetPixelLatLon] Preprocessing {split} split into pixels (per-pixel lat/lon)...")
            self._build_dataset()
            print(f"[PixelDatasetPixelLatLon] Saved to {self.cache_path}")

    def _get_cache_path(self):
        if self.file_suffix.startswith("_m"):
            months_sub = self.file_suffix[1:]
            cache_dir = os.path.join(path_config.get_pixels_cache_dir(), months_sub)
        else:
            cache_dir = path_config.get_pixels_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        return f"{cache_dir}/{self.data_percentage}_pixels{self.file_suffix}_pixellatlon.npz"

    def get_means_stds(self):
        self.means, self.stds = compute_or_load_means_stds(
            data_dir=self.data_dir,
            split=self.split,
            data_percentage=self.data_percentage,
            num_bands=6,
            load_raster_fn=load_raster,
            file_suffix=self.file_suffix,
        )

    def process_gt(self, gt):
        invalid = (gt == 32767) | (gt < 0)
        gt = normalize_doy(gt)
        gt[invalid] = -1
        return gt.astype(np.float32)

    def _build_dataset(self):
        pixel_inputs, pixel_targets, pixel_latlons, pixel_meta = [], [], [], []

        for idx in tqdm(range(len(self.data_dir))):
            image_paths, gt_path, hls_tile_name = self.data_dir[idx]

            imgs = [load_raster_input(p, target_size=self.target_size)[:, np.newaxis]
                    for p in image_paths]
            img = np.concatenate(imgs, axis=1)  # (C, T, H, W)

            C, T, H, W = img.shape
            img_reshaped = img.reshape(C, T, H*W).transpose(2, 1, 0)  # (H*W, T, C)

            gt_mask = load_raster_output(gt_path)[self.correct_indices, :, :]
            labels = gt_mask.reshape(4, H*W).transpose(1, 0)
            labels = self.process_gt(labels)

            valid_labels_idx = ~(labels == -1).all(axis=1)
            non_dead_idx = ~(img_reshaped == 0).all(axis=(1, 2))
            valid_idx = valid_labels_idx & non_dead_idx

            img_valid = img_reshaped[valid_idx]
            labels_valid = labels[valid_idx]

            # Per-pixel lat/lon from rasterio transform
            # Use first available image path to get the geotransform
            raster_path = None
            for p in image_paths:
                if os.path.exists(p):
                    raster_path = p
                    break

            if raster_path is not None:
                pixel_ll_grid = compute_pixel_latlons_from_raster(raster_path, H, W)  # (H, W, 2)
                pixel_ll_flat = pixel_ll_grid.reshape(H*W, 2)  # (H*W, 2)
                latlons_valid = pixel_ll_flat[valid_idx]  # (N_valid, 2)
            else:
                # Fallback to zeros if no raster available
                latlons_valid = np.zeros((img_valid.shape[0], 2), dtype=np.float32)

            # Build meta
            h_coords, w_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            coords = np.stack([h_coords.ravel(), w_coords.ravel()], axis=1)
            coords = coords[valid_idx]
            meta = [(idx, h, w, hls_tile_name) for (h, w) in coords]

            pixel_inputs.append(img_valid.astype(np.float32))
            pixel_targets.append(labels_valid.astype(np.float32))
            pixel_latlons.append(latlons_valid)
            pixel_meta.extend(meta)

        self.inputs = np.concatenate(pixel_inputs, axis=0)
        self.targets = np.concatenate(pixel_targets, axis=0)
        self.latlons = np.concatenate(pixel_latlons, axis=0)
        self.meta = np.array(pixel_meta, dtype=object)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez_compressed(self.cache_path,
                            inputs=self.inputs,
                            targets=self.targets,
                            latlons=self.latlons,
                            meta=self.meta)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx])
        y = torch.from_numpy(self.targets[idx])
        ll = torch.from_numpy(self.latlons[idx])

        if not self.skip_normalization:
            dead = (x == 0).all(dim=-1, keepdim=True)
            x = (x - self._means_tensor) / (self._stds_tensor + 1e-6)
            x = x.masked_fill(dead.expand_as(x), 0.0)

        return {"image": x, "gt_mask": y, "latlons": ll}
