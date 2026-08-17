"""
Presto Phenology model for crop phenology prediction.

Key features:
  - Pretrained Presto encoder with band-group tokenization
  - Sigmoid output
  - Per-pixel lat/lon support
  - Optional NDVI as additional band group
"""

import torch
import torch.nn as nn
from lib.models.presto.presto.presto import Presto

NDVI_IDX = 16
NUM_NORMED_BANDS = 17
DW_MISSING = 9

HLS_TO_NORMED = [2, 3, 4, 9, 10, 11]
HLS_NIR_LOCAL = 3
HLS_RED_LOCAL = 2


class PrestoPhenologyModel(nn.Module):
    def __init__(self, num_classes=4, freeze_encoder=False, feed_timeloc=False,
                 timeloc_mode=None, dropout=0.0, p_loc_drop=0.0,
                 pretrained=True, use_ndvi=False):
        """
        Same args as PrestoPhenologyModel. Only differences:
          - Sigmoid on output
          - forward() accepts per-pixel latlons (B, H, W, 2)
          - use_ndvi: whether to compute and feed NDVI (default False)
        """
        super().__init__()

        self.p_loc_drop = p_loc_drop
        self.use_ndvi = use_ndvi

        if timeloc_mode is not None:
            assert timeloc_mode in ("none", "time", "location", "both")
            self.timeloc_mode = timeloc_mode
        else:
            self.timeloc_mode = "both" if feed_timeloc else "none"

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
