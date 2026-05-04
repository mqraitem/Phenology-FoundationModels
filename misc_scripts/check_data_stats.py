"""Check per-band input and output statistics (mean/std/min/max) before and after normalization.

Covers both HLS and Sentinel-2 data sources using the existing dataloaders.

Usage:
    python check_data_stats.py --selected_months 3 6 9 12
    python check_data_stats.py --selected_months 3 6 9 12 --source s2
    python check_data_stats.py --selected_months 1 2 3 4 5 6 7 8 9 10 11 12 --split validation
    python check_data_stats.py --selected_months 3 6 9 12 --max_tiles 50
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import numpy as np
from tqdm import tqdm

from lib.utils import get_data_paths, get_data_paths_s2, normalize_doy, compute_or_load_means_stds


def load_raster_hls(path, target_size=330):
    import rasterio
    if os.path.exists(path):
        with rasterio.open(path) as src:
            img = src.read()
        return img[:, :target_size, :target_size].astype(np.float32)
    else:
        return np.zeros((6, target_size, target_size), dtype=np.float32)


def load_raster_hls_for_stats(path, crop=None):
    import rasterio
    if os.path.exists(path):
        with rasterio.open(path) as src:
            img = src.read()
            if crop:
                img = img[:, -crop[0]:, -crop[1]:]
    else:
        img = np.zeros((6, 330, 330))
    return img


CLOUD_SCORE_BAND = 7
CLOUD_THRESHOLD = 3000


def load_raster_s2(path, target_size=330):
    import rasterio
    if os.path.exists(path):
        with rasterio.open(path) as src:
            img = src.read()  # (8, 990, 990)
        C, H, W = img.shape
        new_h, new_w = H // 3, W // 3
        img = img[:, :new_h*3, :new_w*3].reshape(C, new_h, 3, new_w, 3).mean(axis=(2, 4))
        cloud_mask = img[CLOUD_SCORE_BAND] < CLOUD_THRESHOLD
        img[:6, cloud_mask] = 0
        return img[:6].astype(np.float32)
    else:
        return np.zeros((6, target_size, target_size), dtype=np.float32)


def load_raster_s2_raw(path, target_size=330):
    """Load S2 composite with all 8 bands (including cloud scores), downsampled 3x."""
    import rasterio
    if os.path.exists(path):
        with rasterio.open(path) as src:
            img = src.read()  # (8, 990, 990)
        C, H, W = img.shape
        new_h, new_w = H // 3, W // 3
        img = img[:, :new_h*3, :new_w*3].reshape(C, new_h, 3, new_w, 3).mean(axis=(2, 4))
        return img.astype(np.float32)  # (8, 330, 330)
    else:
        return np.zeros((8, target_size, target_size), dtype=np.float32)


CORRECT_INDICES = [i - 1 for i in [2, 5, 8, 11]]
DATE_NAMES = ["Greenup", "Maturity", "Senescence", "Dormancy"]


def print_stats_table(title, stats_dict):
    """Print a formatted table of per-band statistics."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Band':<12s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for band_name, s in stats_dict.items():
        print(f"  {band_name:<12s} {s['mean']:>10.4f} {s['std']:>10.4f} "
              f"{s['min']:>10.4f} {s['max']:>10.4f}")
    print()


class OnlineStats:
    """Welford's online algorithm for computing mean/std/min/max."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = np.inf
        self.max_val = -np.inf

    def update(self, values):
        for x in values.ravel():
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
            self.min_val = min(self.min_val, x)
            self.max_val = max(self.max_val, x)

    def update_batch(self, values):
        """Batch update for efficiency."""
        if values.size == 0:
            return
        batch_n = values.size
        batch_mean = values.mean()
        batch_var = values.var()
        batch_min = values.min()
        batch_max = values.max()

        if self.n == 0:
            self.n = batch_n
            self.mean = float(batch_mean)
            self.M2 = float(batch_var * batch_n)
            self.min_val = float(batch_min)
            self.max_val = float(batch_max)
        else:
            total_n = self.n + batch_n
            delta = float(batch_mean) - self.mean
            self.mean = (self.n * self.mean + batch_n * float(batch_mean)) / total_n
            self.M2 += float(batch_var * batch_n) + delta**2 * self.n * batch_n / total_n
            self.n = total_n
            self.min_val = min(self.min_val, float(batch_min))
            self.max_val = max(self.max_val, float(batch_max))

    def result(self):
        std = np.sqrt(self.M2 / self.n) if self.n > 1 else 0.0
        return {
            "mean": self.mean,
            "std": std,
            "min": self.min_val if self.min_val != np.inf else 0.0,
            "max": self.max_val if self.max_val != -np.inf else 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Check per-band data statistics before/after normalization")
    parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--split", type=str, default="training",
                        choices=["training", "validation", "testing"])
    parser.add_argument("--source", type=str, default="hls", choices=["hls", "s2"],
                        help="Data source: hls or s2")
    parser.add_argument("--max_tiles", type=int, default=0,
                        help="Max tiles to process (0 = all)")
    parser.add_argument("--exclude_dead", action="store_true",
                        help="Exclude dead pixels (all-zero) from statistics")
    args = parser.parse_args()

    months_str = "-".join(str(m) for m in args.selected_months)
    file_suffix = f"_m{months_str}"
    n_timesteps = len(args.selected_months)

    print(f"Source: {args.source.upper()}")
    print(f"Split: {args.split}")
    print(f"Months: {args.selected_months}")
    print(f"Exclude dead pixels: {args.exclude_dead}")

    # Get data paths
    if args.source == "hls":
        data_paths = get_data_paths(args.split, data_percentage=1.0,
                                    selected_months=args.selected_months)
        load_fn = load_raster_hls
        # Get normalization means/stds
        means, stds = compute_or_load_means_stds(
            data_dir=data_paths, split=args.split, data_percentage=1.0,
            num_bands=6, load_raster_fn=load_raster_hls_for_stats,
            file_suffix=file_suffix,
        )
    else:
        data_paths = get_data_paths_s2(args.split, data_percentage=1.0,
                                       selected_months=args.selected_months)
        load_fn = load_raster_s2
        means = None  # S2/Presto normalizes internally
        stds = None

    n_tiles = len(data_paths)
    if args.max_tiles > 0:
        n_tiles = min(n_tiles, args.max_tiles)
    print(f"Tiles: {n_tiles}")

    num_bands = 6
    band_names = [f"Band {i}" for i in range(num_bands)]

    # --- Input statistics ---
    # Pre-normalization (raw)
    input_raw_stats = [OnlineStats() for _ in range(num_bands)]
    # Post-normalization
    input_norm_stats = [OnlineStats() for _ in range(num_bands)] if means is not None else None

    # --- Output/GT statistics ---
    gt_raw_stats = [OnlineStats() for _ in range(4)]
    gt_norm_stats = [OnlineStats() for _ in range(4)]

    # Dead pixel counts
    total_pixels = 0
    total_dead = 0

    # Per-month cloud masking stats (S2 only)
    # Per-band count of pixels > 10000 (S2 only)
    if args.source == "s2":
        cloud_stats = {m: {"total": 0, "masked": 0, "nodata": 0} for m in args.selected_months}
        above_10k = {"total": [0] * num_bands, "count": [0] * num_bands}

    for i in tqdm(range(n_tiles), desc="Processing tiles"):
        image_paths, gt_path, tile_name = data_paths[i]

        # Per-month cloud masking and >10000 stats (S2 only): read raw 8-band images
        if args.source == "s2":
            for t_idx, path in enumerate(image_paths):
                month = args.selected_months[t_idx]
                raw = load_raster_s2_raw(path)  # (8, 330, 330)
                H, W = raw.shape[1], raw.shape[2]
                n_px = H * W
                nodata = (raw[:6] == 0).all(axis=0)  # (330, 330)
                cloud_score = raw[CLOUD_SCORE_BAND]
                cloudy = (cloud_score < CLOUD_THRESHOLD) & ~nodata
                cloud_stats[month]["total"] += n_px
                cloud_stats[month]["masked"] += int(cloudy.sum())
                cloud_stats[month]["nodata"] += int(nodata.sum())

                # Count per-band pixels > 10000 (excluding nodata)
                for b in range(num_bands):
                    band = raw[b]
                    valid = ~nodata
                    above_10k["total"][b] += int(valid.sum())
                    above_10k["count"][b] += int((band[valid] > 10000).sum())

        # Load images
        imgs = [load_fn(p)[:, np.newaxis] for p in image_paths]
        img = np.concatenate(imgs, axis=1)  # (6, T, H, W)

        H, W = img.shape[2], img.shape[3]
        total_pixels += H * W

        # Dead pixel mask: all bands × all timesteps are zero
        dead_mask = (img == 0).all(axis=(0, 1))  # (H, W)
        total_dead += int(dead_mask.sum())

        # --- Raw input stats per band ---
        for b in range(num_bands):
            band_data = img[b]  # (T, H, W)
            if args.exclude_dead:
                # Exclude pixels that are dead across all timesteps
                live_mask = ~dead_mask  # (H, W)
                band_data = band_data[:, live_mask]  # (T, N_live)
            input_raw_stats[b].update_batch(band_data.astype(np.float64))

        # --- Normalized input stats per band (HLS only) ---
        if means is not None:
            means_r = means.reshape(6, 1, 1, 1)
            stds_r = stds.reshape(6, 1, 1, 1)

            # Per-timestep dead mask
            dead_ts_mask = (img == 0).all(axis=0)  # (T, H, W)
            img_norm = (img.astype(np.float32) - means_r) / (stds_r + 1e-6)
            img_norm = np.where(dead_ts_mask[np.newaxis], 0.0, img_norm)

            for b in range(num_bands):
                band_data = img_norm[b]  # (T, H, W)
                if args.exclude_dead:
                    live_mask = ~dead_mask
                    band_data = band_data[:, live_mask]
                input_norm_stats[b].update_batch(band_data.astype(np.float64))

        # --- GT stats ---
        import rasterio
        with rasterio.open(gt_path) as src:
            gt_full = src.read()
        gt = gt_full[CORRECT_INDICES, :H, :W].astype(np.float64)

        # Raw GT stats (exclude invalid: 32767 and negative)
        for d in range(4):
            gt_band = gt[d]
            valid = (gt_band != 32767) & (gt_band >= 0)
            if args.exclude_dead:
                valid = valid & (~dead_mask)
            if valid.any():
                gt_raw_stats[d].update_batch(gt_band[valid])

        # Normalized GT stats
        gt_proc = gt.copy()
        invalid = (gt_proc == 32767) | (gt_proc < 0)
        gt_proc = gt_proc / 547.0  # normalize_doy
        gt_proc[invalid] = -1

        for d in range(4):
            gt_band = gt_proc[d]
            valid = gt_band != -1
            if args.exclude_dead:
                valid = valid & (~dead_mask)
            if valid.any():
                gt_norm_stats[d].update_batch(gt_band[valid])

    # --- Print results ---
    print(f"\n{'#' * 70}")
    print(f"  DATA STATISTICS REPORT — {args.source.upper()}")
    print(f"  Split: {args.split} | Months: {args.selected_months}")
    print(f"  Tiles: {n_tiles} | Total pixels: {total_pixels:,}")
    print(f"  Dead pixels: {total_dead:,} ({100*total_dead/max(total_pixels,1):.2f}%)")
    if args.exclude_dead:
        print(f"  (Dead pixels EXCLUDED from statistics)")
    print(f"{'#' * 70}")

    # Input raw
    raw_input_dict = {band_names[b]: input_raw_stats[b].result() for b in range(num_bands)}
    print_stats_table("INPUT — Raw (before normalization)", raw_input_dict)

    # Input normalized (HLS only)
    if input_norm_stats is not None:
        norm_input_dict = {band_names[b]: input_norm_stats[b].result() for b in range(num_bands)}
        print_stats_table("INPUT — Normalized (after mean/std normalization)", norm_input_dict)

        # Print the means/stds used
        print(f"  Normalization parameters used:")
        for b in range(num_bands):
            print(f"    Band {b}: mean={means[b]:.4f}, std={stds[b]:.4f}")
        print()
    else:
        print(f"\n  [S2 data — no dataset-level normalization applied (Presto normalizes internally)]\n")

    # GT raw
    raw_gt_dict = {DATE_NAMES[d]: gt_raw_stats[d].result() for d in range(4)}
    print_stats_table("OUTPUT/GT — Raw DOY values (before normalization)", raw_gt_dict)

    # GT normalized
    norm_gt_dict = {DATE_NAMES[d]: gt_norm_stats[d].result() for d in range(4)}
    print_stats_table("OUTPUT/GT — Normalized (DOY / 547)", norm_gt_dict)

    # Per-month cloud masking report (S2 only)
    if args.source == "s2":
        print(f"\n{'=' * 70}")
        print(f"  PER-MONTH CLOUD MASKING (median_cs < {CLOUD_THRESHOLD})")
        print(f"{'=' * 70}")
        print(f"  {'Month':>5s}  {'Total px':>12s}  {'Masked':>12s}  {'% Masked':>10s}  {'Nodata':>12s}  {'% Nodata':>10s}")
        print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*10}")
        for m in args.selected_months:
            s = cloud_stats[m]
            valid = s["total"] - s["nodata"]
            pct_masked = 100 * s["masked"] / valid if valid > 0 else 0
            pct_nodata = 100 * s["nodata"] / s["total"] if s["total"] > 0 else 0
            print(f"  {m:>5d}  {s['total']:>12,}  {s['masked']:>12,}  {pct_masked:>9.2f}%  {s['nodata']:>12,}  {pct_nodata:>9.2f}%")
        print()

        # Per-band percentage of pixels > 10000
        print(f"{'=' * 70}")
        print(f"  PER-BAND PIXELS > 10000 (raw S2 reflectance, excluding nodata)")
        print(f"{'=' * 70}")
        print(f"  {'Band':>6s}  {'Valid px':>12s}  {'> 10000':>12s}  {'%':>10s}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}")
        for b in range(num_bands):
            total = above_10k["total"][b]
            count = above_10k["count"][b]
            pct = 100 * count / total if total > 0 else 0
            print(f"  {band_names[b]:>6s}  {total:>12,}  {count:>12,}  {pct:>9.2f}%")
        print()


if __name__ == "__main__":
    main()
