"""
Presto Phenology model for crop phenology prediction.

Key features:
  - Pretrained Presto encoder with band-group tokenization
  - Sigmoid output
  - Per-pixel lat/lon support (computed from rasterio transform)
  - Optional NDVI as additional band group
"""

import torch
import torch.nn as nn
import numpy as np

from lib.models.presto.presto.presto import Presto

from collections import OrderedDict
BANDS_GROUPS_IDX = OrderedDict({
    "S1": [0, 1],
    "S2_RGB": [2, 3, 4],
    "S2_Red_Edge": [5, 6, 7],
    "S2_NIR_10m": [8],
    "S2_NIR_20m": [9],
    "S2_SWIR": [10, 11],
    "ERA5": [12, 13],
    "SRTM": [14, 15],
    "NDVI": [16],
})

NDVI_IDX = 16
NUM_NORMED_BANDS = 17
DW_MISSING = 9

S2_TO_NORMED = [2, 3, 4, 9, 10, 11]
S2_NIR_LOCAL = 3
S2_RED_LOCAL = 2

HLS_TO_NORMED = [2, 3, 4, 9, 10, 11]
HLS_NIR_LOCAL = 3
HLS_RED_LOCAL = 2


def compute_pixel_latlons(raster_path, H=330, W=330):
    """Compute per-pixel lat/lon from a rasterio file's transform and CRS.

    Args:
        raster_path: path to any GeoTIFF for this tile
        H, W: spatial dimensions

    Returns:
        latlons: (H, W, 2) numpy array of [lat, lon] per pixel
    """
    import rasterio
    from rasterio.warp import transform as warp_transform

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


class PrestoPhenologyModel(nn.Module):
    def __init__(self, num_classes=4, freeze_encoder=False, input_mode="hls", feed_timeloc=False,
                 timeloc_mode=None, dropout=0.0, p_loc_drop=0.0,
                 pretrained=True, use_ndvi=False):
        """
        Same args as PrestoPhenologyModel. Only differences:
          - Sigmoid on output
          - forward() accepts per-pixel latlons (B, H, W, 2)
          - use_ndvi: whether to compute and feed NDVI (default False)
        """
        super().__init__()

        assert input_mode in ("s2", "hls"), f"Unknown input_mode: {input_mode}"
        self.input_mode = input_mode
        self.p_loc_drop = p_loc_drop
        self.use_ndvi = use_ndvi

        if timeloc_mode is not None:
            assert timeloc_mode in ("none", "time", "location", "both")
            self.timeloc_mode = timeloc_mode
        else:
            self.timeloc_mode = "both" if feed_timeloc else "none"

        if input_mode == "s2":
            self.band_indices = S2_TO_NORMED
            self.nir_local = S2_NIR_LOCAL
            self.red_local = S2_RED_LOCAL
        else:
            self.band_indices = HLS_TO_NORMED
            self.nir_local = HLS_NIR_LOCAL
            self.red_local = HLS_RED_LOCAL

        if pretrained:
            base_model = Presto.load_pretrained()
        else:
            base_model = Presto.construct()
        self.model = base_model.construct_finetuning_model(
            num_outputs=num_classes, regression=True
        )

        if dropout > 0.0:
            for blk in self.model.encoder.blocks:
                blk.attn.attn_drop.p = dropout
                blk.attn.proj_drop.p = dropout
                blk.mlp.drop1.p = dropout
                blk.mlp.drop2.p = dropout

        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad_(False)
            for param in self.model.head.parameters():
                param.requires_grad_(True)

        self.sigmoid = nn.Sigmoid()
        self.register_buffer("_location_codebook", None)

    def set_location_codebook(self, centers):
        if not isinstance(centers, torch.Tensor):
            centers = torch.tensor(centers, dtype=torch.float32)
        self.register_buffer("_location_codebook", centers)

    def _snap_tile_latlons(self, latlons):
        """Shift each tile's per-pixel lat/lon grid so its centroid lands on the
        nearest training-tile centroid. Preserves within-tile spatial structure
        while ensuring the model only sees coordinate ranges it saw at training
        time. `latlons` is (B, H, W, 2)."""
        if self._location_codebook is None:
            return latlons
        # (B, 2) tile centroids from per-pixel latlons
        tile_centroids = latlons.float().reshape(latlons.shape[0], -1, 2).mean(dim=1)
        dists = torch.cdist(tile_centroids, self._location_codebook.float())
        nearest = self._location_codebook[dists.argmin(dim=1)]  # (B, 2)
        shift = (nearest - tile_centroids).view(latlons.shape[0], 1, 1, 2)
        return latlons + shift

    def _prepare_presto_inputs(self, data, latlons, month=None):
        """Prepare inputs for Presto encoder."""
        B, T, C = data.shape
        device = data.device

        x = torch.zeros(B, T, NUM_NORMED_BANDS, device=device, dtype=data.dtype)
        x[:, :, self.band_indices] = data / 10000.0

        if self.use_ndvi:
            red = x[:, :, self.band_indices[self.red_local]]
            nir = x[:, :, self.band_indices[self.nir_local]]
            ndvi = torch.where(
                (nir + red) > 0,
                (nir - red) / (nir + red),
                torch.zeros_like(red),
            )
            x[:, :, NDVI_IDX] = ndvi

        mask = torch.ones(B, T, NUM_NORMED_BANDS, device=device, dtype=data.dtype)
        mask[:, :, self.band_indices] = 0.0
        if self.use_ndvi:
            mask[:, :, NDVI_IDX] = 0.0

        dead_ts = (data == 0).all(dim=-1)
        mask[dead_ts] = 1.0

        dynamic_world = torch.full((B, T), DW_MISSING, dtype=torch.long, device=device)

        feed_loc = self.timeloc_mode in ("location", "both")
        feed_time = self.timeloc_mode in ("time", "both")

        if not feed_loc:
            latlons = torch.zeros(B, 2, device=device)
        elif self.training and self.p_loc_drop > 0.0:
            drop = (torch.rand(B, 1, device=device) < self.p_loc_drop).float()
            latlons = latlons * (1.0 - drop)

        if feed_time:
            if month is None:
                month = 0
            elif isinstance(month, torch.Tensor) and month.dim() == 1 and month.shape[0] == T:
                month = month.unsqueeze(0).expand(B, T).to(device)
        else:
            month = 0

        return x, dynamic_world, latlons, mask, month

    def forward(self, x, processing_images=True, latlons=None, month=None, chunk_size=2048):
        """Forward pass with dual mode.

        Args:
            x: if processing_images=True: (B, 6, T, H, W) raw values or dict with "chip" key
               if processing_images=False: (B, T, 6) raw values
            latlons: (B, H, W, 2) per-pixel lat/lon for image mode,
                     (B, 2) per-pixel lat/lon for pixel mode,
                     or None
            month: (T,) 0-indexed month tensor, int, or None
            chunk_size: pixels per forward pass
        """
        device = next(self.parameters()).device

        if isinstance(x, dict):
            x = x["chip"]

        if processing_images:
            x = x.to(device)
            B, C, T, H, W = x.shape

            x_pixels = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)

            # Per-pixel latlons: (B, H_ll, W_ll, 2) → pad to (B, H, W, 2) if needed
            if latlons is not None:
                latlons = latlons.to(device)
                assert latlons.dim() == 4, f"Expected 4D per-pixel latlons, got {latlons.shape}"
                H_ll, W_ll = latlons.shape[1], latlons.shape[2]

                # Tile-level snap on the *unpadded* region, otherwise zero-padding
                # corrupts the centroid and the snap picks the wrong training tile.
                feed_loc = self.timeloc_mode in ("location", "both")
                if feed_loc and self._location_codebook is not None:
                    latlons = self._snap_tile_latlons(latlons)

                if H_ll < H or W_ll < W:
                    padded = torch.zeros(B, H, W, 2, device=device)
                    padded[:, :H_ll, :W_ll, :] = latlons
                    latlons = padded

                latlons_flat = latlons.reshape(B * H * W, 2)
            else:
                latlons_flat = torch.zeros(B * H * W, 2, device=device)

            outputs = []
            N = x_pixels.shape[0]
            for i in range(0, N, chunk_size):
                chunk = x_pixels[i:i + chunk_size]
                ll_chunk = latlons_flat[i:i + chunk_size]
                px, dw, ll, mk, mo = self._prepare_presto_inputs(chunk, ll_chunk, month)
                out = self.model(x=px, dynamic_world=dw, latlons=ll, mask=mk, month=mo)
                outputs.append(out)

            out = torch.cat(outputs, dim=0)
            out = out.view(B, H, W, -1).permute(0, 3, 1, 2)
            return self.sigmoid(out)

        else:
            x = x.to(device)
            if latlons is not None:
                latlons = latlons.to(device)
            else:
                latlons = torch.zeros(x.shape[0], 2, device=device)
            px, dw, ll, mk, mo = self._prepare_presto_inputs(x, latlons, month)
            out = self.model(x=px, dynamic_world=dw, latlons=ll, mask=mk, month=mo)
            return self.sigmoid(out)
