"""
Pixel-level dataset with spatial subsampling support.

Same as dataloaders_pixels.py but adds a pixel_stride parameter to subsample
every Nth pixel spatially (matching Tran et al. 2025 which samples every 10th pixel).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import geopandas as gpd
from lib.utils import compute_or_load_means_stds, normalize_doy
import path_config


def load_raster(path, crop=None):
	import rasterio
	if os.path.exists(path):
		with rasterio.open(path) as src:
			img = src.read()
			if crop:
				img = img[:, -crop[0]:, -crop[1]:]
	else:
		img = np.zeros((6, 330, 330))
	return img

def load_raster_input(path, target_size=330):
	import rasterio
	if os.path.exists(path):
		with rasterio.open(path) as src:
			img = src.read()  # (C, H, W)
		_, h, w = img.shape
		return img.astype(np.float32)
	else:
		return np.zeros((6, target_size, target_size), dtype=np.float32)

def load_raster_output(path):
	import rasterio
	if not os.path.exists(path):
		raise FileNotFoundError(f"GT raster not found: {path}")
	with rasterio.open(path) as src:
		return src.read()


class CycleDatasetPixelsSubsampled(Dataset):
	PIXEL_STRIDE = 10

	def __init__(self, data_dir, split, cache_path=None, data_percentage=1.0, target_size=330,
				 regenerate=False, n_timesteps=12, file_suffix="", skip_normalization=False):
		"""
		Args:
			data_dir: list of tuples [(image_paths, gt_path, hls_tile_name), ...]
			split: "train" / "val" / "test"
			cache_path: path to npz file where pixel dataset is cached (auto-derived if None)
			data_percentage: used for naming stats file
			target_size: padded size
			regenerate: if True, rebuild dataset even if cache exists
			n_timesteps: number of monthly time steps to use
			file_suffix: suffix for mean/std cache file naming
			skip_normalization: if True, return raw pixel values
		"""
		self.data_dir = data_dir
		self.split = split
		self.data_percentage = data_percentage
		self.target_size = target_size
		self.n_timesteps = n_timesteps
		self.file_suffix = file_suffix
		self.skip_normalization = skip_normalization
		self.pixel_stride = self.PIXEL_STRIDE
		self.cache_path = cache_path if cache_path is not None else self._get_cache_path()

		# correct gt indices
		self.correct_indices = [2, 5, 8, 11]
		self.correct_indices = [i - 1 for i in self.correct_indices]

		# load/compute means and stds
		if not skip_normalization:
			self.get_means_stds()
			self._means_tensor = torch.tensor(self.means, dtype=torch.float32).view(1, -1)
			self._stds_tensor = torch.tensor(self.stds, dtype=torch.float32).view(1, -1)

		self._assign_locations()

		if os.path.exists(self.cache_path) and not regenerate:
			print(f"[PixelDatasetSubsampled] Loading preprocessed dataset from {self.cache_path}")
			data = np.load(self.cache_path, allow_pickle=True)
			self.inputs = data['inputs']   # (N, T, C)
			self.targets = data['targets'] # (N, 4)
			self.latlons = data['latlons'] # (N, 2)
			self.meta = data['meta']
		else:
			print(f"[PixelDatasetSubsampled] Preprocessing {split} split (stride={self.pixel_stride})...")
			self._build_dataset()
			print(f"[PixelDatasetSubsampled] Saved to {self.cache_path}")


	def _get_cache_path(self):
		if self.file_suffix.startswith("_m"):
			months_sub = self.file_suffix[1:]
			cache_dir = os.path.join(path_config.get_pixels_cache_dir(), months_sub)
		else:
			cache_dir = path_config.get_pixels_cache_dir()
		os.makedirs(cache_dir, exist_ok=True)
		stride_tag = f"_stride{self.pixel_stride}" if self.pixel_stride > 1 else ""
		return f"{cache_dir}/{self.data_percentage}_pixels{self.file_suffix}{stride_tag}.npz"

	def get_means_stds(self):
		self.means, self.stds = compute_or_load_means_stds(
			data_dir=self.data_dir,
			split=self.split,
			data_percentage=self.data_percentage,
			num_bands=6,
			load_raster_fn=load_raster,
			file_suffix=self.file_suffix,
		)

	def _assign_locations(self):
		geo_path = path_config.get_data_geojson()
		geo_gdf = gpd.read_file(geo_path)
		geo_gdf = geo_gdf.rename(columns={"Site_ID": "SiteID"})
		geo_gdf["HLStile"] = "T" + geo_gdf["name"]
		geo_gdf = geo_gdf.set_crs("EPSG:4326")
		geo_gdf["centroid"] = geo_gdf.geometry.representative_point()

		self.all_locations = {}
		for entry in self.data_dir:
			full_id = entry[2]
			hls_tile = full_id.split("_")[-1]
			site_id = full_id.split("_")[-2]
			centroid = geo_gdf[(geo_gdf["HLStile"] == hls_tile) & (geo_gdf["SiteID"] == site_id)]["centroid"].iloc[0]
			self.all_locations[full_id] = [centroid.y, centroid.x]

	def process_gt(self, gt):
		invalid = (gt == 32767) | (gt < 0)
		gt = normalize_doy(gt)
		gt[invalid] = -1
		return gt.astype(np.float32)

	def _build_dataset(self):
		pixel_inputs, pixel_targets, pixel_latlons, pixel_meta = [], [], [], []

		for idx in tqdm(range(len(self.data_dir))):
			image_paths, gt_path, hls_tile_name = self.data_dir[idx]

			# load images
			imgs = [load_raster_input(p, target_size=self.target_size)[:, np.newaxis]
					for p in image_paths]
			img = np.concatenate(imgs, axis=1)  # (C, T, H, W)

			C, T, H, W = img.shape

			# Apply spatial subsampling before flattening
			if self.pixel_stride > 1:
				h_indices = np.arange(0, H, self.pixel_stride)
				w_indices = np.arange(0, W, self.pixel_stride)
				img_sub = img[:, :, h_indices][:, :, :, w_indices]  # (C, T, H', W')
				H_sub, W_sub = len(h_indices), len(w_indices)
			else:
				img_sub = img
				h_indices = np.arange(H)
				w_indices = np.arange(W)
				H_sub, W_sub = H, W

			img_reshaped = img_sub.reshape(C, T, H_sub * W_sub).transpose(2, 1, 0)  # (H'*W', T, C)

			# load and subsample GT
			gt_mask = load_raster_output(gt_path)[self.correct_indices, :, :]  # (4, H, W)
			if self.pixel_stride > 1:
				gt_mask = gt_mask[:, h_indices][:, :, w_indices]  # (4, H', W')
			labels = gt_mask.reshape(4, H_sub * W_sub).transpose(1, 0)  # (H'*W', 4)

			labels = self.process_gt(labels)

			# identify valid pixels
			valid_labels_idx = ~(labels == -1).all(axis=1)
			non_dead_idx = ~(img_reshaped == 0).all(axis=(1, 2))
			valid_idx = valid_labels_idx & non_dead_idx

			img_valid = img_reshaped[valid_idx]
			labels_valid = labels[valid_idx]

			# lat/lon for all valid pixels (tile-center)
			loc = self.all_locations.get(hls_tile_name, [0.0, 0.0])
			n_valid = img_valid.shape[0]
			latlons_tile = np.tile(np.array(loc, dtype=np.float32), (n_valid, 1))

			# build meta with original coordinates
			h_grid, w_grid = np.meshgrid(h_indices, w_indices, indexing="ij")
			coords = np.stack([h_grid.ravel(), w_grid.ravel()], axis=1)  # (H'*W', 2)
			coords = coords[valid_idx]
			meta = [(idx, h, w, hls_tile_name) for (h, w) in coords]

			pixel_inputs.append(img_valid.astype(np.float32))
			pixel_targets.append(labels_valid.astype(np.float32))
			pixel_latlons.append(latlons_tile)
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
		x = torch.from_numpy(self.inputs[idx])   # (T, C)
		y = torch.from_numpy(self.targets[idx])   # (4,)
		ll = torch.from_numpy(self.latlons[idx])  # (2,)

		if not self.skip_normalization:
			dead = (x == 0).all(dim=-1, keepdim=True)  # (T, 1)
			x = (x - self._means_tensor) / (self._stds_tensor + 1e-6)
			x = x.masked_fill(dead.expand_as(x), 0.0)

		return {"image": x, "gt_mask": y, "latlons": ll}
