"""Result loading and geographic annotation helpers used by the paper notebook."""

import os

import geopandas as gpd
import pandas as pd

import path_config
from lib.utils import get_results_dir


def add_region_to_results(
    results_df: pd.DataFrame,
    geo_path: str,
    eco_path: str,
    region_column: str = "NA_L1NAME",
    tile_name_col: str = "name",
    site_id_col_geo: str = "Site_ID",
    predicate: str = "within",
) -> pd.DataFrame:
    """Attach an ecoregion label to each tile/site result row."""
    geo_gdf = gpd.read_file(geo_path).copy()
    if site_id_col_geo in geo_gdf.columns and "SiteID" not in geo_gdf.columns:
        geo_gdf = geo_gdf.rename(columns={site_id_col_geo: "SiteID"})
    if "HLStile" not in geo_gdf.columns:
        if tile_name_col not in geo_gdf.columns:
            raise ValueError(
                f"'{tile_name_col}' not found in tile file; provide tile_name_col."
            )
        geo_gdf["HLStile"] = "T" + geo_gdf[tile_name_col].astype(str)

    geo_gdf = geo_gdf.set_crs("EPSG:4326", allow_override=True).to_crs(3857)
    centroids = gpd.GeoDataFrame(
        geo_gdf[["HLStile", "SiteID"]].copy(),
        geometry=geo_gdf.geometry.centroid,
        crs=geo_gdf.crs,
    ).drop_duplicates(subset=["HLStile", "SiteID"])

    eco_gdf = gpd.read_file(eco_path).to_crs(centroids.crs)
    if region_column not in eco_gdf.columns:
        raise ValueError(
            f"'{region_column}' not found in ecoregion data. "
            f"Available columns: {list(eco_gdf.columns)}"
        )
    joined = gpd.sjoin(
        centroids,
        eco_gdf[[region_column, "geometry"]],
        how="left",
        predicate=predicate,
    )
    joined = joined[["HLStile", "SiteID", region_column]].drop_duplicates()
    return results_df.merge(joined, on=["HLStile", "SiteID"], how="left")


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["years", "HLStile", "SiteID", "row", "col", "version"]
    ).reset_index(drop=True)


def results_file(split="test", selected_months=(3, 6, 9, 12), seeds=None):
    """Load representative and per-seed result frames for one temporal setting."""
    base_dir = get_results_dir(selected_months=selected_months)
    results = {}
    results_per_seed = {}

    for model_dir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        seed_dfs = []
        for filename in sorted(os.listdir(model_path)):
            if not filename.endswith(f"_{split}.csv") or not filename.startswith("seed_"):
                continue
            seed_name = filename.removesuffix(f"_{split}.csv")
            if seeds is not None and seed_name not in seeds and seed_name != "seed_all":
                continue
            seed_dfs.append(pd.read_csv(os.path.join(model_path, filename)))

        if not seed_dfs:
            continue
        key = f"{model_dir}_{split}"
        results[key] = sort_df(seed_dfs[0])
        results_per_seed[key] = [sort_df(df) for df in seed_dfs]

    results_w_regions = {
        key: add_region_to_results(
            results_df=frame,
            geo_path=path_config.get_data_geojson(),
            eco_path="useco1/NA_CEC_Eco_Level1.shp",
            region_column="NA_L1NAME",
        )
        for key, frame in results.items()
    }
    return results, results_w_regions, results_per_seed
