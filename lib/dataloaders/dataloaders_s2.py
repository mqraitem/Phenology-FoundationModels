import os
import numpy as np
import torch
from torch.utils.data import Dataset
import geopandas as gpd
from scipy.ndimage import zoom
import path_config
from lib.utils import compute_or_load_means_stds, normalize_doy


CLOUD_SCORE_BAND = 7   # median_cs
CLOUD_THRESHOLD = 3000  # pixels with median_cs < 3000 are masked (zeroed out)


def load_raster_s2(path, target_size=336):
	"""Load S2 composite (8 bands, 990x990), downsample 3x to 330x330,
	mask cloudy pixels (median_cs < CLOUD_THRESHOLD), pad to target_size, return first 6 bands.

	Spectral bands (0-5) are downsampled with bicubic interpolation.
	Cloud score band (7) is downsampled with block averaging."""
	import rasterio
	if os.path.exists(path):
		with rasterio.open(path) as src:
			img = src.read()  # (8, 990, 990)
		C, H, W = img.shape
		new_h, new_w = H // 3, W // 3

		# Bicubic downsample spectral bands (0-5)
		spectral = img[:6].astype(np.float32)
		spectral_ds = zoom(spectral, (1, 1/3, 1/3), order=3)  # (6, 330, 330)

		# Block average downsample cloud score band
		cs = img[CLOUD_SCORE_BAND:CLOUD_SCORE_BAND+1, :new_h*3, :new_w*3]
		cs_ds = cs.reshape(1, new_h, 3, new_w, 3).mean(axis=(2, 4))  # (1, 330, 330)

		# Cloud mask: zero out spectral bands where median_cs < threshold
		cloud_mask = cs_ds[0] < CLOUD_THRESHOLD  # (330, 330)
		spectral_ds[:, cloud_mask] = 0

		# Pad to target_size
		_, h, w = spectral_ds.shape
		pad_h = max(0, target_size - h)
		pad_w = max(0, target_size - w)
		if pad_h > 0 or pad_w > 0:
			spectral_ds = np.pad(spectral_ds, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
		return spectral_ds
	else:
		return np.zeros((6, target_size, target_size), dtype=np.float32)


def load_raster_output(path):
	import rasterio
	if not os.path.exists(path):
		raise FileNotFoundError(f"GT raster not found: {path}")
	with rasterio.open(path) as src:
		return src.read()


class CycleDatasetS2(Dataset):
	def __init__(self, path, split, data_percentage=1.0, n_timesteps=12, file_suffix="", skip_normalization=True):
		"""
		Full-tile dataset for S2 composites.

		Args:
			path: list of tuples [(image_paths, gt_path, tile_name), ...]
			split: "train" / "val" / "test"
			data_percentage: for compatibility
			n_timesteps: number of monthly timesteps
			file_suffix: for compatibility
			skip_normalization: if True, return raw data (Presto normalizes internally).
			                    if False, apply per-band mean/std normalization.
		"""
		self.data_dir = path
		self.split = split
		self.data_percentage = data_percentage
		self.n_timesteps = n_timesteps
		self.file_suffix = file_suffix
		self.skip_normalization = skip_normalization

		self.correct_indices = [2, 5, 8, 11]
		self.correct_indices = [i - 1 for i in self.correct_indices]

		if not skip_normalization:
			self.get_means_stds()

		self.assign_location_time_info()

	def assign_location_time_info(self):
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
			self.all_locations[full_id] = [centroid.y, centroid.x]  # [lat, lon]

	def get_means_stds(self):
		s2_suffix = f"_s2{self.file_suffix}"
		self.means, self.stds = compute_or_load_means_stds(
			data_dir=self.data_dir,
			split=self.split,
			data_percentage=self.data_percentage,
			num_bands=6,
			load_raster_fn=load_raster_s2,
			file_suffix=s2_suffix,
		)

	def normalize_image(self, image, means, stds):
		"""Normalize (bands, time, H, W) image. Dead timesteps stay zero."""
		bands, time, H, W = image.shape
		vh, vw = 330, 330

		means1 = means.reshape(bands, 1, 1, 1)
		stds1 = stds.reshape(bands, 1, 1, 1)

		normalized = np.zeros_like(image, dtype=np.float32)
		valid_region = image[:, :, :vh, :vw]

		dead_ts_mask = (valid_region == 0).all(axis=0)  # (time, vh, vw)
		normalized_valid = (valid_region.astype(np.float32) - means1) / (stds1 + 1e-6)

		normalized[:, :, :vh, :vw] = np.where(
			dead_ts_mask[np.newaxis, :, :, :],
			0.0,
			normalized_valid,
		)

		return torch.from_numpy(normalized).to(torch.float32)

	def process_gt(self, gt):
		invalid = (gt == 32767) | (gt < 0)
		gt = normalize_doy(gt)
		gt[invalid] = -1
		return gt.astype(np.float32)

	def __len__(self):
		return len(self.data_dir)

	def __getitem__(self, idx):
		image_paths = self.data_dir[idx][0]
		output_path = self.data_dir[idx][1]
		hls_tile_name = self.data_dir[idx][2]

		images = []
		for p in image_paths:
			images.append(load_raster_s2(p)[:, np.newaxis])  # (6, 1, 336, 336)

		image = np.concatenate(images, axis=1)  # (6, T, 336, 336)
		if self.skip_normalization:
			image_tensor = torch.from_numpy(image).to(torch.float32)
		else:
			image_tensor = self.normalize_image(image, self.means, self.stds)

		gt_mask = load_raster_output(output_path)
		gt_mask = gt_mask[self.correct_indices, :, :]  # (4, 330, 330)
		gt_mask = self.process_gt(gt_mask)

		# Mask GT for fully-dead pixels (all bands zero across all timesteps)
		H, W = gt_mask.shape[1], gt_mask.shape[2]
		fully_dead = (image[:, :, :H, :W] == 0).all(axis=(0, 1))  # (H, W)
		gt_mask[:, fully_dead] = -1

		location_coords = self.all_locations[hls_tile_name]
		latlons = torch.tensor(location_coords, dtype=torch.float32)  # (2,)

		return {
			"image": image_tensor,
			"image_unprocessed": image,
			"gt_mask": gt_mask,
			"hls_tile_name": hls_tile_name,
			"latlons": latlons,
		}
