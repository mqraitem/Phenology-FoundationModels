"""Prithvi Phenology model.

Uses the final ViT layer output, upscales to full resolution,
concatenates the raw spectral input, and applies a temporal fusion head.
"""

import torch
import torch.nn as nn

from lib.models.prithvi_mae import PrithviMAE


# ===== Building blocks =====

class ChannelLayerNorm3D(nn.Module):
    """LayerNorm over channels for 5D tensors (B, C, T, H, W)."""
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class PrithviReshape3D(nn.Module):
    """Reshape backbone output to (B, embed_dim, T, H_patches, W_patches)."""
    def __init__(self, patch_size, input_size, num_frames):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_frames = num_frames
        self.spatial_size = int(self.input_size / self.patch_size[-1])

    def forward(self, latent):
        latent = latent[:, 1:, :]  # remove CLS token
        B, N, D = latent.shape
        H = W = self.spatial_size
        T = self.num_frames
        latent = latent.reshape(B, T, H, W, D)
        latent = latent.permute(0, 4, 1, 2, 3)
        return latent


class Conv3DTemporalSpatialBlock(nn.Module):
    """ConvTranspose3D for 2x spatial upsampling + Conv3D refinement."""
    def __init__(self, in_ch, out_ch, dropout=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            ChannelLayerNorm3D(out_ch),
            nn.GELU(),
            nn.Dropout(0.1) if dropout else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ChannelLayerNorm3D(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LocDropProxy(nn.Module):
    """Wrap a LocationEncoder so its output is bernoulli-masked to zero per-sample
    with probability `p` during training — equivalent to disabling the location
    contribution for that sample (unlike zeroing the coords, which still produces
    the constant sincos(0,0) vector)."""
    def __init__(self, inner: nn.Module, p: float = 0.0):
        super().__init__()
        self.inner = inner
        self.p = p

    def forward(self, location_coords):
        emb = self.inner(location_coords)
        if self.training and self.p > 0:
            B = emb.shape[0]
            keep = (torch.rand(B, 1, 1, device=emb.device) >= self.p).float()
            emb = emb * keep
        return emb


class TemporalFusionHead(nn.Module):
    """3D conv fusion + learned temporal collapse. Input: (B, C, T, H, W) → (B, n_classes, H, W)."""
    def __init__(self, in_ch, n_classes=4, n_layers=2, num_frames=4):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Conv3d(in_ch, in_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                ChannelLayerNorm3D(in_ch),
                nn.ReLU(inplace=True),
            ])
        self.layers = nn.Sequential(*layers)
        self.temporal_proj = nn.Conv3d(in_ch, n_classes, kernel_size=(num_frames, 1, 1))

    def forward(self, x):
        x = self.layers(x)
        x = self.temporal_proj(x)
        x = x.squeeze(2)
        return x


class PrithviBackbone(nn.Module):
    """Prithvi backbone that returns features from the final transformer block only."""

    def __init__(self, prithvi_params: dict, prithvi_ckpt_path: str = None):
        super().__init__()
        self.prithvi_params = prithvi_params
        self.model = PrithviMAE(**prithvi_params)

        if prithvi_ckpt_path is not None:
            checkpoint = torch.load(prithvi_ckpt_path, weights_only=False)

            if "encoder.pos_embed" not in checkpoint.keys():
                key = "model" if "model" in checkpoint.keys() else "state_dict"
                keys = list(checkpoint[key].keys())
                checkpoint = checkpoint[key]
            else:
                keys = list(checkpoint.keys())

            for k in keys:
                if ((prithvi_params.get("encoder_only", True)) and ("decoder" in k)) or "pos_embed" in k:
                    del checkpoint[k]
                elif "prithvi" in k:
                    new_k = k.replace("prithvi.", "")
                    checkpoint[new_k] = checkpoint[k]
                elif k in self.model.state_dict() and checkpoint[k].shape != self.model.state_dict()[k].shape:
                    print(f"Warning: size mismatch for layer {k}, deleting: "
                          f"{checkpoint[k].shape} != {self.model.state_dict()[k].shape}")
                    del checkpoint[k]

            _ = self.model.load_state_dict(checkpoint, strict=False)

    def forward(self, data):
        if isinstance(data, dict):
            chip = data.get("chip")
            temporal = data.get("temporal_coords")
            location = data.get("location_coords")
        else:
            chip = data
            temporal = None
            location = None

        # forward_features returns final layer output: (B, 1+T*Hp*Wp, embed_dim)
        x = self.model.forward_features(chip, temporal, location)
        return x


class PrithviPhenology(nn.Module):
    """Prithvi with single-scale features + raw input concatenation.

    1. Extract features from the final ViT layer only
    2. Reshape to (B, embed_dim, T, Hp, Wp)
    3. Upscale to full resolution via Conv3D upscaler
    4. Concatenate raw spectral input (B, in_chans, T, H, W)
    5. Apply temporal fusion head on the combined features
    """

    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 n_classes: int = 4,
                 model_size: str = "300m",
                 feed_timeloc: bool = False,
                 n_layers: int = 2,
                 concat_input: bool = True,
                 dropout: float = 0.0,
                 p_loc_drop: float = 0.0):
        super().__init__()
        self.concat_input = concat_input
        self.p_loc_drop = p_loc_drop

        # Toggle temporal/location encoding
        prithvi_params["coords_encoding"] = ["time", "location"] if feed_timeloc else []

        self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path)

        # Apply dropout to every transformer block's attention + MLP
        if dropout > 0.0:
            for blk in self.backbone.model.encoder.blocks:
                blk.attn.attn_drop.p = dropout
                blk.attn.proj_drop.p = dropout
                blk.mlp.drop1.p = dropout
                blk.mlp.drop2.p = dropout

        # Wrap the location encoder for per-sample bernoulli drop
        if feed_timeloc and p_loc_drop > 0.0:
            enc = self.backbone.model.encoder
            if hasattr(enc, "location_embed_enc"):
                enc.location_embed_enc = LocDropProxy(enc.location_embed_enc, p_loc_drop)

        embed_dim = prithvi_params["embed_dim"]
        num_frames = prithvi_params["num_frames"]
        in_chans = prithvi_params.get("in_chans", 6)

        self.reshaper = PrithviReshape3D(
            prithvi_params["patch_size"],
            prithvi_params["img_size"],
            prithvi_params["num_frames"],
        )

        # Spatial upscaler: embed_dim → embed_dim//8 at full resolution (4× 2x upsampling)
        self.upscale_blocks = nn.Sequential(
            Conv3DTemporalSpatialBlock(embed_dim, embed_dim // 2),
            Conv3DTemporalSpatialBlock(embed_dim // 2, embed_dim // 4),
            Conv3DTemporalSpatialBlock(embed_dim // 4, embed_dim // 8),
            Conv3DTemporalSpatialBlock(embed_dim // 8, embed_dim // 8),
        )

        final_ch = embed_dim // 8
        head_ch = final_ch + in_chans if concat_input else final_ch

        self.head = TemporalFusionHead(
            head_ch, n_classes, n_layers, num_frames=num_frames
        )

    def forward(self, x):
        if isinstance(x, dict):
            x = {k: v.cuda() for k, v in x.items()}
            raw_input = x.get("chip")  # (B, C, T, H, W)
        else:
            x = x.cuda()
            raw_input = x

        # Single-scale features from final ViT layer
        features = self.backbone(x)              # (B, 1+T*Hp*Wp, embed_dim)

        # Reshape to 5D (reshaper handles CLS token removal internally)
        features = self.reshaper(features)       # (B, embed_dim, T, Hp, Wp)

        # Upscale to full resolution
        features = self.upscale_blocks(features) # (B, embed_dim//8, T, H, W)

        # Optionally concatenate raw spectral input
        if self.concat_input:
            features = torch.cat([features, raw_input], dim=1)

        # Temporal fusion
        out = self.head(features)                # (B, n_classes, H, W)
        return out
