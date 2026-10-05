"""Build model-independent per-tile layers for the supportive analyses.

For every 4-month test tile-year, writes one compressed ``.npz`` with the HP-LSP
reference dates, the valid-pixel mask, and CEC Level I ecoregion IDs rasterized
to the 330x330 tile grid, plus ``eco_region_l1_lookup.csv`` to decode the IDs.
No model is involved, so this runs on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from lib.utils import get_data_paths


SELECTED_MONTHS = [3, 6, 9, 12]
TILE_SIZE = 330
# Band indices of the four phenophase dates in the HP-LSP reference raster.
GT_INDICES = [1, 4, 7, 10]
PHASE_NAMES = np.array(["Greenup", "Maturity", "Senescence/Silking", "Dormancy/Dough"])


def _load_hls_sequence(image_paths: list[str]) -> tuple[np.ndarray, dict]:
    """Stack the monthly HLS composites (missing months as zeros) and return the grid profile."""
    imgs = []
    profile = None
    for path in image_paths:
        if Path(path).exists():
            with rasterio.open(path) as src:
                imgs.append(src.read().astype(np.float32)[:, :TILE_SIZE, :TILE_SIZE])
                if profile is None:
                    profile = {"crs": src.crs, "transform": src.transform,
                               "height": TILE_SIZE, "width": TILE_SIZE}
        else:
            imgs.append(np.zeros((6, TILE_SIZE, TILE_SIZE), dtype=np.float32))
    if profile is None:
        raise FileNotFoundError(f"No available HLS raster among: {image_paths}")
    return np.stack(imgs, axis=0), profile


def _load_gt_doy(gt_path: str, hls_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference DOY per phase; pixels with no HLS data at any timestep are invalid."""
    with rasterio.open(gt_path) as src:
        gt = src.read()[GT_INDICES, :TILE_SIZE, :TILE_SIZE].astype(np.float32)
    valid = ~((gt == 32767) | (gt < 0))
    valid[:, (hls_sequence == 0).all(axis=(0, 1))] = False
    gt[~valid] = np.nan
    return gt, valid


def _make_l1_lookup(eco: gpd.GeoDataFrame) -> pd.DataFrame:
    table = (
        eco[["NA_L1CODE", "NA_L1NAME"]].drop_duplicates()
        .sort_values(["NA_L1CODE", "NA_L1NAME"], key=lambda s: s.astype(str))
        .reset_index(drop=True)
    )
    table.insert(0, "id", np.arange(1, len(table) + 1, dtype=np.int16))
    return table


def _rasterize_l1(eco: gpd.GeoDataFrame, lookup: pd.DataFrame, profile: dict) -> np.ndarray:
    key_to_id = {(str(c), str(n)): int(i)
                 for i, c, n in lookup[["id", "NA_L1CODE", "NA_L1NAME"]].itertuples(index=False)}
    eco = eco.to_crs(profile["crs"])
    shapes = [(geom, key_to_id[(str(c), str(n))])
              for geom, c, n in zip(eco.geometry, eco["NA_L1CODE"], eco["NA_L1NAME"])]
    return rasterize(shapes, out_shape=(profile["height"], profile["width"]),
                     transform=profile["transform"], fill=0, dtype="int16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/stratified_analysis/tile_layers"))
    parser.add_argument("--eco-shapefile", default="useco2/NA_CEC_Eco_Level2.shp")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eco = gpd.read_file(args.eco_shapefile)
    lookup = _make_l1_lookup(eco)
    lookup.to_csv(args.out_dir / "eco_region_l1_lookup.csv", index=False)

    test_paths = get_data_paths("testing", 1.0, SELECTED_MONTHS)
    for idx, (image_paths, gt_path, tile_id) in enumerate(test_paths):
        out_path = args.out_dir / f"{tile_id}.npz"
        if out_path.exists() and not args.overwrite:
            continue
        print(f"[{idx + 1}/{len(test_paths)}] {tile_id}", flush=True)
        hls_sequence, profile = _load_hls_sequence(image_paths)
        gt_doy, gt_valid = _load_gt_doy(gt_path, hls_sequence)
        year, site_id, hls_tile = tile_id.split("_")
        np.savez_compressed(
            out_path,
            tile_id=np.array(tile_id),
            year=np.array(year),
            site_id=np.array(site_id),
            hls_tile=np.array(hls_tile),
            phase_names=PHASE_NAMES,
            ground_truth_doy=gt_doy,
            ground_truth_valid=gt_valid,
            eco_region_l1_id=_rasterize_l1(eco, lookup, profile),
        )
    print(f"Wrote {len(test_paths)} tile layers to {args.out_dir}")


if __name__ == "__main__":
    main()
