"""Build compact tile-level tables for ecoregion, land-cover, and smoothness analysis.

Dense inference is streamed over the 4-month test split. Predictions are reduced
immediately to tile-level metrics and are not saved to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
import torch
from torch.utils.data import DataLoader

from lib.dataloaders.centroid_tile_dataset import CentroidTileDataset
from lib.dataloaders.pixel_coordinate_tile_dataset import PixelCoordinateTileDataset
from lib.stratified_analysis import masked_group_errors, neighbor_smoothness
from lib.utils import get_data_paths, months_to_str
from misc_scripts.build_student_test_tiles import MODEL_GROUPS, SELECTED_MONTHS, _predict_tile, _prepare_models


CANONICAL_SEEDS = ["seed_42", "seed_123", "seed_456"]
NLCD_NAMES = {
    11: "Open water", 12: "Perennial ice/snow", 21: "Developed, open space",
    22: "Developed, low intensity", 23: "Developed, medium intensity",
    24: "Developed, high intensity", 31: "Barren land", 41: "Deciduous forest",
    42: "Evergreen forest", 43: "Mixed forest", 52: "Shrub/scrub",
    71: "Grassland/herbaceous", 81: "Pasture/hay", 82: "Cultivated crops",
    90: "Woody wetlands", 95: "Emergent herbaceous wetlands",
}


def _first_raster_profile(image_paths: list[str]) -> dict:
    for path in image_paths:
        if Path(path).exists():
            with rasterio.open(path) as src:
                return {
                    "crs": src.crs,
                    "transform": src.transform,
                    "height": 330,
                    "width": 330,
                }
    raise FileNotFoundError(f"No HLS raster exists among {image_paths}")


def _align_landcover(src: rasterio.DatasetReader, profile: dict) -> np.ndarray:
    result = np.full((profile["height"], profile["width"]), src.nodata, dtype=np.uint8)
    reproject(
        source=rasterio.band(src, 1), destination=result,
        src_transform=src.transform, src_crs=src.crs,
        dst_transform=profile["transform"], dst_crs=profile["crs"],
        dst_nodata=src.nodata, resampling=Resampling.nearest,
    )
    return result


def _write_map_geometry(
    eco_path: str,
    states_path: str,
    lookup: pd.DataFrame,
    eco_out_path: Path,
    states_out_path: Path,
) -> None:
    eco = gpd.read_file(eco_path)
    eco["NA_L1CODE"] = eco["NA_L1CODE"].astype(str).str.replace(r"\.0$", "", regex=True)
    lookup = lookup.copy()
    lookup["NA_L1CODE"] = lookup["NA_L1CODE"].astype(str)
    level1 = eco.dissolve(by=["NA_L1CODE", "NA_L1NAME"], as_index=False)
    level1 = level1.merge(lookup, on=["NA_L1CODE", "NA_L1NAME"], how="inner")

    states = gpd.read_file(states_path).to_crs(level1.crs)
    states = states[~states["name"].isin(["Hawaii", "Puerto Rico"])]
    us_geometry = states.geometry.union_all()
    level1["geometry"] = level1.geometry.intersection(us_geometry)
    level1 = level1[~level1.geometry.is_empty & (level1["NA_L1NAME"] != "WATER")]
    level1["geometry"] = level1.geometry.simplify(1_000, preserve_topology=True)
    level1[["id", "NA_L1CODE", "NA_L1NAME", "geometry"]].to_crs("EPSG:4326").to_file(
        eco_out_path, driver="GeoJSON"
    )
    states["geometry"] = states.geometry.simplify(500, preserve_topology=True)
    states[["name", "geometry"]].to_crs("EPSG:4326").to_file(states_out_path, driver="GeoJSON")


def _build_invariant_data(args, test_paths: list, package_files: dict) -> tuple[dict, dict]:
    nlcd_sources = {
        2019: rasterio.open(args.nlcd_2019),
        2020: rasterio.open(args.nlcd_2020),
    }
    base_tiles, landcover = {}, {}
    try:
        for image_paths, _gt_path, tile_id in test_paths:
            if tile_id not in package_files:
                raise FileNotFoundError(f"Student tile package is missing {tile_id}")
            tile = np.load(package_files[tile_id], allow_pickle=True)
            base_tiles[tile_id] = {
                "ground_truth_doy": tile["ground_truth_doy"],
                "ground_truth_valid": tile["ground_truth_valid"],
                "eco_region_l1_id": tile["eco_region_l1_id"],
                "phase_names": [str(x) for x in tile["phase_names"]],
                "site_id": str(tile["site_id"]),
                "year": int(str(tile["year"])),
            }
            profile = _first_raster_profile(image_paths)
            landcover[tile_id] = _align_landcover(nlcd_sources[base_tiles[tile_id]["year"]], profile)
    finally:
        for src in nlcd_sources.values():
            src.close()
    return base_tiles, landcover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", default=CANONICAL_SEEDS)
    parser.add_argument("--out-dir", default="data/stratified_analysis/m3-6-9-12")
    parser.add_argument("--tile-package", default="student_test_tiles_m3-6-9-12")
    parser.add_argument("--eco-shapefile", default="useco2/NA_CEC_Eco_Level2.shp")
    parser.add_argument("--states-geojson", default="/projectnb/rise-ivc/Badr/us_states.geojson")
    parser.add_argument("--nlcd-2019", default="/projectnb/rise-ivc/Badr/Annual_NLCD_LndCov_2019_CU_C1V2/Annual_NLCD_LndCov_2019_CU_C1V2.tif")
    parser.add_argument("--nlcd-2020", default="/projectnb/rise-ivc/Badr/Annual_NLCD_LndCov_2020_CU_C1V2/Annual_NLCD_LndCov_2020_CU_C1V2.tif")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense inference")
    device = "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_paths = get_data_paths("testing", 1.0, SELECTED_MONTHS)
    if args.limit is not None:
        test_paths = test_paths[:args.limit]
    package_dir = Path(args.tile_package) / "data" / "m3-6-9-12" / "test"
    package_files = {p.stem: p for p in package_dir.glob("*.npz")}
    base_tiles, landcover = _build_invariant_data(args, test_paths, package_files)

    l1_lookup = pd.read_csv(Path(args.tile_package) / "eco_region_l1_lookup.csv")
    l1_names = dict(zip(l1_lookup["id"], l1_lookup["NA_L1NAME"]))
    _write_map_geometry(
        args.eco_shapefile, args.states_geojson, l1_lookup,
        out_dir / "ecoregion_l1_us.geojson", out_dir / "us_states.geojson",
    )

    suffix = f"_m{months_to_str(SELECTED_MONTHS)}"
    hls_dataset = CentroidTileDataset(test_paths, split="testing", data_percentage=1.0,
                               n_timesteps=4, file_suffix=suffix)
    presto_dataset = PixelCoordinateTileDataset(
        test_paths, split="testing", data_percentage=1.0,
        n_timesteps=4, file_suffix=suffix, skip_normalization=True,
    )
    month_tensor = torch.tensor([m - 1 for m in SELECTED_MONTHS], dtype=torch.long)

    eco_rows, landcover_rows, smooth_rows = [], [], []

    # Analysis 3: deterministic ground-truth spatial variation by tile and phase.
    for tile_id, tile in base_tiles.items():
        for phase_idx, phase in enumerate(tile["phase_names"]):
            value, count, total = neighbor_smoothness(
                tile["ground_truth_doy"][phase_idx], tile["ground_truth_valid"][phase_idx]
            )
            smooth_rows.append({
                "seed": "deterministic", "source": "ground_truth", "tile_id": tile_id,
                "site_id": tile["site_id"], "year": tile["year"], "phase": phase,
                "smoothness_days": value, "n_pairs": count, "difference_sum": total,
            })

    checkpoint_manifest = {}
    for seed in args.seeds:
        print(f"Loading models for {seed}", flush=True)
        models, model_params = _prepare_models(device, seed)
        checkpoint_manifest[seed] = model_params
        hls_loader = DataLoader(hls_dataset, batch_size=1, shuffle=False, num_workers=0)
        presto_loader = DataLoader(presto_dataset, batch_size=1, shuffle=False, num_workers=0)

        for idx, (data_hls, data_presto, path_row) in enumerate(zip(hls_loader, presto_loader, test_paths)):
            _image_paths, _gt_path, tile_id = path_row
            tile = base_tiles[tile_id]
            print(f"[{seed} {idx + 1}/{len(test_paths)}] {tile_id}", flush=True)
            for model_key, (model, crop_size) in models.items():
                prediction = _predict_tile(
                    model_key, model, crop_size, data_hls, data_presto,
                    month_tensor, device,
                )
                common = {
                    "seed": seed, "model": model_key, "tile_id": tile_id,
                    "site_id": tile["site_id"], "year": tile["year"],
                }
                # Analysis 1: model error within each represented ecoregion.
                for row in masked_group_errors(
                    prediction, tile["ground_truth_doy"], tile["ground_truth_valid"],
                    tile["eco_region_l1_id"], excluded_ids=(0, 1),
                ):
                    region_id = row.pop("group_id")
                    eco_rows.append({
                        **common, "eco_region_l1_id": region_id,
                        "eco_region_name": l1_names[region_id], **row,
                    })

                # Analysis 2: model error within each aligned Annual NLCD class.
                for row in masked_group_errors(
                    prediction, tile["ground_truth_doy"], tile["ground_truth_valid"],
                    landcover[tile_id], excluded_ids=(250,),
                ):
                    class_id = row.pop("group_id")
                    if class_id not in NLCD_NAMES:
                        continue
                    landcover_rows.append({
                        **common, "landcover_id": class_id,
                        "landcover_name": NLCD_NAMES[class_id], **row,
                    })

                # Analysis 3: prediction spatial variation by model and phase.
                for phase_idx, phase in enumerate(tile["phase_names"]):
                    value, count, total = neighbor_smoothness(
                        prediction[phase_idx], tile["ground_truth_valid"][phase_idx]
                    )
                    smooth_rows.append({
                        "seed": seed, "source": model_key, "tile_id": tile_id,
                        "site_id": tile["site_id"], "year": tile["year"], "phase": phase,
                        "smoothness_days": value, "n_pairs": count, "difference_sum": total,
                    })
        del models
        torch.cuda.empty_cache()

    pd.DataFrame(eco_rows).to_csv(out_dir / "ecoregion_tile_mae.csv", index=False)
    pd.DataFrame(landcover_rows).to_csv(out_dir / "landcover_tile_mae.csv", index=False)
    pd.DataFrame(smooth_rows).to_csv(out_dir / "smoothness_tile_phase.csv", index=False)
    manifest = {
        "split": "test", "selected_months": SELECTED_MONTHS, "n_tile_years": len(test_paths),
        "seeds": args.seeds, "models": list(MODEL_GROUPS), "checkpoints": checkpoint_manifest,
        "metrics": {
            "mae": "linear absolute error in the extended 1-547 target domain",
            "smoothness": "mean modulo-365 difference over valid right/down neighbor pairs",
        },
        "sources": {
            "tile_package": args.tile_package, "ecoregions": args.eco_shapefile,
            "states": args.states_geojson, "nlcd_2019": args.nlcd_2019, "nlcd_2020": args.nlcd_2020,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote analysis cache to {out_dir}")


if __name__ == "__main__":
    main()
