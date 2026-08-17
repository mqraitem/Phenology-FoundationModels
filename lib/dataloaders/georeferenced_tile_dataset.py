"""
Tile dataloader with per-pixel lat/lon from rasterio transforms.

Same as tile_dataset.py except latlons is (H, W, 2) per-pixel instead of (2,) tile centroid.
"""

import rasterio
from rasterio.warp import transform as warp_transform
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import path_config
from lib.utils import compute_or_load_means_stds, normalize_doy
from datetime import datetime


def load_raster(path, crop=None):
    if os.path.exists(path):
        with rasterio.open(path) as src:
            img = src.read()
            if crop:
                img = img[:, -crop[0]:, -crop[1]:]
    else:
        img = np.zeros((6, 330, 330))
    return img


def load_raster_input(path, target_size=336):
    if os.path.exists(path):
        with rasterio.open(path) as src:
            img = src.read()
        _, h, w = img.shape
        pad_h = (target_size - h) if h < target_size else 0
        pad_w = (target_size - w) if w < target_size else 0
        padded_img = np.pad(img, pad_width=((0, 0), (0, pad_h), (0, pad_w)),
                            mode='constant', constant_values=0)
        padded_img = padded_img.astype(np.float32)
    else:
        padded_img = np.zeros((6, target_size, target_size)).astype(np.float32)
    return padded_img


def load_raster_output(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"GT raster not found: {path}")
    with rasterio.open(path) as src:
        img = src.read()
    return img


def compute_pixel_latlons_from_raster(raster_path, H=330, W=330):
    """Compute per-pixel lat/lon from a rasterio file's transform and CRS."""
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


class GeoreferencedTileDataset(Dataset):
    """Same as TileDataset but returns per-pixel latlons (H, W, 2)."""

    def __init__(self, path, split, data_percentage=1.0, means=None, stds=None,
                 n_timesteps=12, file_suffix="", skip_normalization=False):

        self.data_dir = path
        self.split = split
        self.total_below_0 = 0
        self.total_above_365 = 0
        self.total_nan = 0
        self.total = 0
        self.data_percentage = data_percentage
        self.n_timesteps = n_timesteps
        self.file_suffix = file_suffix
        self.skip_normalization = skip_normalization

        self.correct_indices = [2, 5, 8, 11]
        self.correct_indices = [i - 1 for i in self.correct_indices]

        if not skip_normalization:
            if means is None or stds is None:
                self.get_means_stds()
            else:
                self.means = np.array(means)
                self.stds = np.array(stds)

        self.assign_location_time_info()

    def assign_location_time_info(self):
        import geopandas as gpd
        geo_path = path_config.get_data_geojson()
        geo_gdf = gpd.read_file(geo_path)
        geo_gdf = geo_gdf.rename(columns={"Site_ID": "SiteID"})
        geo_gdf["HLStile"] = "T" + geo_gdf["name"]
        geo_gdf = geo_gdf.set_crs("EPSG:4326")
        geo_gdf["centroid"] = geo_gdf.geometry.representative_point()

        self.all_locations = {}
        self.all_times = {}
        for input in self.data_dir:
            full_id = input[2]
            hls_tile = full_id.split("_")[-1]
            site_id = full_id.split("_")[-2]
            centroid = geo_gdf[(geo_gdf["HLStile"] == hls_tile) & (geo_gdf["SiteID"] == site_id)]["centroid"].iloc[0]
            self.all_locations[full_id] = [centroid.y, centroid.x]

            all_input_images_times = [x.split("/")[-1].split("_")[2].split("-") for x in input[0]]
            temp_coords = [[int(x[0]), int(x[1])] for x in all_input_images_times]
            temp_coords = [[x[0], datetime(x[0], x[1], 15).timetuple().tm_yday] for x in temp_coords]
            self.all_times[full_id] = temp_coords

    def get_means_stds(self):
        self.means, self.stds = compute_or_load_means_stds(
            data_dir=self.data_dir,
            split=self.split,
            data_percentage=self.data_percentage,
            num_bands=6,
            load_raster_fn=load_raster,
            file_suffix=self.file_suffix,
        )

    def __len__(self):
        return len(self.data_dir)

    def normalize_image(self, image, means, stds):
        number_of_channels = image.shape[0]
        number_of_time_steps = image.shape[1]
        bands, time, H, W = image.shape
        vh, vw = (330, 330)

        means1 = means.reshape(bands, 1, 1, 1)
        stds1 = stds.reshape(bands, 1, 1, 1)

        normalized = np.zeros_like(image, dtype=np.float32)
        valid_region = image[:, :, :vh, :vw]
        dead_ts_mask = (valid_region == 0).all(axis=0)
        normalized_valid = (valid_region.astype(np.float32) - means1) / (stds1 + 1e-6)
        normalized[:, :, :vh, :vw] = np.where(
            dead_ts_mask[np.newaxis, :, :, :], 0.0, normalized_valid,
        )

        normalized_tensor = torch.from_numpy(
            normalized.reshape(number_of_channels, number_of_time_steps, *image.shape[-2:])
        ).to(torch.float32)

        return normalized_tensor

    def process_gt(self, gt):
        invalid = (gt == 32767) | (gt < 0)
        gt = normalize_doy(gt)
        gt[invalid] = -1
        return gt.astype(np.float32)

    def __getitem__(self, idx):

        image_path = self.data_dir[idx][0]
        output_path = self.data_dir[idx][1]
        hls_tile_name = self.data_dir[idx][2]

        images = []
        for path in image_path:
            images.append(load_raster_input(path)[:, np.newaxis])

        gt_mask = load_raster_output(output_path)
        gt_mask = gt_mask[self.correct_indices, :, :]

        image = np.concatenate(images, axis=1)
        if self.skip_normalization:
            final_image = torch.from_numpy(image).to(torch.float32)
        else:
            final_image = self.normalize_image(image, self.means, self.stds)
        gt_mask = self.process_gt(gt_mask)

        H, W = gt_mask.shape[1], gt_mask.shape[2]
        fully_dead = (image[:, :, :H, :W] == 0).all(axis=(0, 1))
        gt_mask[:, fully_dead] = -1

        temporal_coords = self.all_times[hls_tile_name]

        # Per-pixel lat/lon from rasterio
        raster_path = None
        for p in image_path:
            if os.path.exists(p):
                raster_path = p
                break

        if raster_path is not None:
            pixel_latlons = compute_pixel_latlons_from_raster(raster_path, H, W)  # (H, W, 2)
        else:
            pixel_latlons = np.zeros((H, W, 2), dtype=np.float32)

        latlons = torch.from_numpy(pixel_latlons)  # (H, W, 2)

        return {
            "image": {
                "chip": final_image,
                "temporal_coords": torch.tensor(temporal_coords, dtype=torch.float32),
            },
            "image_unprocessed": image,
            "gt_mask": gt_mask,
            "hls_tile_name": hls_tile_name,
            "latlons": latlons,  # (H, W, 2) per-pixel
        }
