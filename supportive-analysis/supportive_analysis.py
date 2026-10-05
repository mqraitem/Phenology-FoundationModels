"""Paper-supporting ecoregion, NLCD, and spatial-variation analyses.

This is a plain-text notebook: open it in VS Code and run the ``# %%`` cells
interactively, or review it as ordinary Python. Run it from the repository root.
"""

# # Ecoregion, Land-Cover, and Spatial Smoothness Analysis
# 
# This notebook analyzes the **4-month test setting** (March, June, September, December) for Temporal Transformer, Presto, and Prithvi. It reads compact tables built by the other scripts in this directory and runs on CPU.
# 
# Ecoregion and land-cover summaries pool absolute error over all valid pixels of each stratum within a seed, then average the three seeds, so small patches and boundary slivers contribute in proportion to their size. Spatial-variation tertiles use tile-year MAE (averaged within seed, then across seeds), matching the headline results.
# 
# To rebuild the inputs (see `supportive-analysis/README.md`):
# 
# ```bash
# python supportive-analysis/build_tile_layers.py          # CPU: reference dates, masks, ecoregion IDs
# python supportive-analysis/prepare_stratified_data.py    # GPU: dense inference, per-stratum errors
# python supportive-analysis/spatial_variation.py          # CPU: spatial-variation scores and tertiles
# ```

# %%


from pathlib import Path
import json
import sys
import textwrap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from shapely import make_valid
from shapely.geometry import box

from lib.stratified_analysis import summarize_seed_pooled

# A tile-year counts toward a stratum's site/tile-year support only if it has at
# least this many valid phase-pixel observations there (~100 pixels x 4 phases).
MIN_SUPPORT_OBS = 400

CACHE_DIR = REPO_ROOT / "data/stratified_analysis/m3-6-9-12"
SPATIAL_DIR = REPO_ROOT / "data/stratified_analysis/spatial_variation"
TILE_DIR = REPO_ROOT / "data/stratified_analysis/tile_layers"
IMAGE_DIR = REPO_ROOT / "paper_latex/Images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["temporal_transformer", "presto", "prithvi"]
MODEL_LABELS = {
    "temporal_transformer": "Temporal Transformer",
    "presto": "Presto",
    "prithvi": "Prithvi",
}
MODEL_COLORS = dict(zip(MODEL_ORDER, sns.color_palette("Set2", 3)))

required = [
    "manifest.json", "ecoregion_tile_mae.csv", "landcover_tile_mae.csv",
    "ecoregion_l1_us.geojson", "us_states.geojson",
]
missing = [name for name in required if not (CACHE_DIR / name).exists()]
assert not missing, f"Missing cache files: {missing}. Run the GPU cache builder first."
spatial_required = ["tile_scores.csv", "mae_by_tertile.csv"]
spatial_missing = [name for name in spatial_required if not (SPATIAL_DIR / name).exists()]
assert not spatial_missing, (
    f"Missing spatial-variation files: {spatial_missing}. Run "
    "supportive-analysis/spatial_variation.py first."
)

manifest = json.loads((CACHE_DIR / "manifest.json").read_text())
assert manifest["n_tile_years"] == 48
assert manifest["seeds"] == ["seed_42", "seed_123", "seed_456"]
manifest


# ## Analysis 1: Mean Error by CEC Level I Ecoregion
# 
# For each model, seed, and ecoregion, absolute error is pooled over all valid pixels and all four phenophases in that region, then averaged equally across seeds. Pooling weights each tile-year by its number of valid pixels in the region, so boundary slivers do not count as full tile-years. Water is excluded. The map uses categorical region colors; exact model MAEs and support counts are reported in the adjacent key.
# 
# The test set covers eight terrestrial Level I regions, but geographic support is uneven. A tile-year counts toward a region's support only if it has at least 100 valid pixels there (400 phase-pixel observations). We report only regions represented by at least two independent sites; the table gives independent-site and tile-year counts.

# %%


eco_raw = pd.read_csv(CACHE_DIR / "ecoregion_tile_mae.csv")
assert set(eco_raw["seed"]) == set(manifest["seeds"])
assert set(eco_raw["model"]) == set(MODEL_ORDER)

eco_summary = summarize_seed_pooled(
    eco_raw, ["model", "eco_region_l1_id", "eco_region_name"]
)
support = (
    eco_raw[eco_raw["n_valid"] >= MIN_SUPPORT_OBS]
    .groupby(["eco_region_l1_id", "eco_region_name"], as_index=False)
    .agg(n_sites=("site_id", "nunique"), n_tile_years=("tile_id", "nunique"))
)
supported_regions = support.loc[support["n_sites"] >= 2, "eco_region_l1_id"]
eco_summary = eco_summary[eco_summary["eco_region_l1_id"].isin(supported_regions)].copy()
eco_summary = eco_summary.merge(
    support, on=["eco_region_l1_id", "eco_region_name"], how="left"
)

display(
    eco_summary.pivot(index=["eco_region_name", "n_sites", "n_tile_years"],
                      columns="model", values="mean")
    .rename(columns=MODEL_LABELS).round(2)
)


# %%


regions = gpd.read_file(CACHE_DIR / "ecoregion_l1_us.geojson")
states = gpd.read_file(CACHE_DIR / "us_states.geojson")

represented_names = sorted(eco_summary["eco_region_name"].unique())
region_numbers = {name: i + 1 for i, name in enumerate(represented_names)}
region_palette = mpl.colormaps["tab10"]
region_colors = {
    name: mpl.colors.to_hex(region_palette(i / max(len(represented_names) - 1, 1)))
    for i, name in enumerate(represented_names)
}
conus_box = box(-125.5, 24.0, -66.0, 50.5)

region_support = (
    support[support["eco_region_l1_id"].isin(supported_regions)]
    .set_index(["eco_region_l1_id", "eco_region_name"])
    .sort_index()
)
region_mae = eco_summary.pivot(
    index=["eco_region_l1_id", "eco_region_name"],
    columns="model", values="mean",
)
region_table = region_support.join(region_mae).reset_index()
region_table["number"] = region_table["eco_region_name"].map(region_numbers)
region_table = region_table.sort_values("number").reset_index(drop=True)

map_regions = regions.merge(
    region_table[["eco_region_l1_id", "eco_region_name", "number"]],
    left_on="id", right_on="eco_region_l1_id", how="left",
)
map_regions["map_color"] = map_regions["eco_region_name"].map(region_colors)

def representative_point_in(geometry, clip_box):
    valid_geometry = make_valid(geometry)
    clipped = valid_geometry if clip_box is None else valid_geometry.intersection(clip_box)
    if clipped.is_empty:
        return None
    parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
    polygon_parts = [part for part in parts if part.geom_type in {"Polygon", "MultiPolygon"}]
    if not polygon_parts:
        return None
    return max(polygon_parts, key=lambda part: part.area).representative_point()

def add_region_numbers(ax, panel, clip_box):
    for _, row in panel[panel["number"].notna()].iterrows():
        point = representative_point_in(row.geometry, clip_box)
        if point is None:
            continue
        ax.text(
            point.x, point.y, str(int(row["number"])),
            ha="center", va="center", fontsize=12, fontweight="bold", zorder=10,
            bbox={"boxstyle": "square,pad=0.22", "facecolor": "white",
                  "edgecolor": "#222222", "linewidth": 0.8, "alpha": 0.94},
        )

fig, (map_ax, key_ax) = plt.subplots(
    1, 2, figsize=(16.0, 5.8), gridspec_kw={"width_ratios": [1.35, 1.0]}
)

states.plot(ax=map_ax, facecolor="#f4f4f4", edgecolor="white", linewidth=0.55)
map_regions.plot(
    ax=map_ax,
    color=[region_colors.get(name, "#dedede")
           for name in map_regions["eco_region_name"]],
    edgecolor="#555555", linewidth=0.55,
)
map_ax.set_xlim(-125.5, -66.0)
map_ax.set_ylim(24.0, 50.5)
map_ax.set_aspect(1 / np.cos(np.deg2rad(37.25)))
map_ax.set_axis_off()
add_region_numbers(map_ax, map_regions, conus_box)

key_ax.set_xlim(0, 1)
key_ax.set_ylim(0, 1)
key_ax.axis("off")
key_ax.text(0.01, 0.96, "CEC Level I ecoregion", fontsize=13.5,
            fontweight="bold", va="top")
key_ax.text(0.57, 0.96, "Sites /\ntile-years", fontsize=11.5,
            fontweight="bold", ha="center", va="top")
model_columns = [0.72, 0.855, 0.98]
for model, x_pos in zip(MODEL_ORDER, model_columns):
    label = {"temporal_transformer": "TT", "presto": "Presto",
             "prithvi": "Prithvi"}[model]
    key_ax.text(x_pos, 0.96, label, fontsize=10.5, fontweight="bold",
                ha="center", va="top")
key_ax.plot([0.01, 0.995], [0.82, 0.82], color="#777777", linewidth=0.8)

for (_, row), y_pos in zip(region_table.iterrows(), np.linspace(0.73, 0.10, len(region_table))):
    name = row["eco_region_name"]
    number = int(row["number"])
    key_ax.text(
        0.03, y_pos, str(number), ha="center", va="center",
        fontsize=12.5, fontweight="bold",
        bbox={"boxstyle": "square,pad=0.27", "facecolor": region_colors[name],
              "edgecolor": "#333333", "linewidth": 0.7},
    )
    key_ax.text(0.08, y_pos, textwrap.fill(name.title(), 20),
                fontsize=12.5, ha="left", va="center")
    key_ax.text(0.57, y_pos, f"{int(row['n_sites'])} / {int(row['n_tile_years'])}",
                fontsize=12.5, ha="center", va="center")
    best_mae = min(row[model] for model in MODEL_ORDER)
    for model, x_pos in zip(MODEL_ORDER, model_columns):
        key_ax.text(
            x_pos, y_pos, f"{row[model]:.1f}", fontsize=12.5,
            fontweight="bold" if np.isclose(row[model], best_mae) else "normal",
            ha="center", va="center",
        )

key_ax.text(0.99, 0.02, "MAE (days); best in bold", fontsize=11.5,
            fontweight="bold", ha="right", va="bottom")
fig.subplots_adjust(left=0.015, right=0.995, top=0.99, bottom=0.02, wspace=0.025)
eco_path = IMAGE_DIR / "ecoregion_mae_maps.pdf"
fig.savefig(eco_path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Saved: {eco_path}")


# ## Analysis 2: Mean Error by NLCD Land-Cover Class
# 
# Annual NLCD 2019/2020 classes are aligned to each 30 m HLS grid using nearest-neighbor resampling. A tile-year counts toward a class's support only if it has at least **400 valid phase-pixel observations** of that class (approximately 100 spatial pixels across four phases). A displayed class must be supported by at least **three tile-years**. Alaska is absent because Annual NLCD used here covers CONUS only.
# 
# Bars show MAE pooled over all valid pixels of each class within seed, then averaged equally across seeds. Error bars are standard deviations across the three seeds.

# %%


land_raw = pd.read_csv(CACHE_DIR / "landcover_tile_mae.csv")
class_support = (
    land_raw[land_raw["n_valid"] >= MIN_SUPPORT_OBS]
    .groupby(["landcover_id", "landcover_name"], as_index=False)
    .agg(n_tile_years=("tile_id", "nunique"), n_sites=("site_id", "nunique"))
)
keep_ids = class_support.loc[class_support["n_tile_years"] >= 3, "landcover_id"]
land_eligible = land_raw[land_raw["landcover_id"].isin(keep_ids)]

land_summary = summarize_seed_pooled(
    land_eligible, ["model", "landcover_id", "landcover_name"]
).merge(
    class_support, on=["landcover_id", "landcover_name"], how="left"
)

display(
    land_summary.pivot(index=["landcover_name", "n_sites", "n_tile_years"],
                       columns="model", values="mean")
    .rename(columns=MODEL_LABELS).round(2)
)


# %%


class_order = (
    land_summary.groupby(["landcover_id", "landcover_name"])["mean"].mean()
    .sort_values(ascending=False).index.tolist()
)

def wrap_landcover_label(name, n_tile_years, width=12):
    readable = name.replace("/", "/ ")
    lines = textwrap.wrap(
        readable, width=width, break_long_words=False, break_on_hyphens=False
    )
    return "\n".join(lines + [f"(n={n_tile_years})"])

labels = [
    wrap_landcover_label(
        name,
        int(class_support.loc[class_support["landcover_id"].eq(class_id), "n_tile_years"].iloc[0]),
    )
    for class_id, name in class_order
]
x = np.arange(len(class_order))
width = 0.25

lower_candidates = land_summary["mean"] - land_summary["seed_std"]
upper_candidates = land_summary["mean"] + land_summary["seed_std"]
y_min = max(0.0, np.floor(lower_candidates.min() - 0.5))
y_max = 2.0 * np.ceil((upper_candidates.max() + 1.0) / 2.0)

fig, ax = plt.subplots(figsize=(22, 9.5))

for offset, model in zip([-width, 0, width], MODEL_ORDER):
    indexed = land_summary[land_summary["model"] == model].set_index(["landcover_id", "landcover_name"])
    means = np.array([indexed.loc[key, "mean"] for key in class_order])
    stds = np.array([indexed.loc[key, "seed_std"] for key in class_order])
    ax.bar(
        x + offset, means - y_min, bottom=y_min, width=width * 0.92, yerr=stds,
        color=MODEL_COLORS[model], edgecolor="black", linewidth=0.6,
        error_kw={"linewidth": 0.8, "capsize": 2}, label=MODEL_LABELS[model],
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, linespacing=1.15)
ax.tick_params(axis="x", pad=8)
ax.tick_params(axis="y", labelsize=12)
ax.set_ylabel("Mean absolute error (days)", fontsize=15)
ax.set_ylim(y_min, y_max)
ax.set_yticks(np.arange(y_min, y_max + 0.1, 2.0))
ax.grid(axis="y", color="#dddddd", linewidth=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(ncol=3, frameon=False, loc="upper right", fontsize=13)
fig.subplots_adjust(left=0.065, right=0.99, top=0.98, bottom=0.39)
land_path = IMAGE_DIR / "landcover_mae.pdf"
fig.savefig(land_path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Saved: {land_path}")


# ## Analysis 3: Error by Ground-Truth Spatial Variation
# 
# Ground-truth spatial variation is measured robustly at multiple scales. For each phenophase, the median absolute DOY difference is calculated between valid horizontal and vertical pixel pairs at lags of 1, 2, 4, and 8 pixels. The four lag values are averaged, each phenophase is standardized across the 48 test tile-years, and the four standardized phase scores are averaged equally.
# 
# Tile-years are divided into equal-count tertiles of **16 tile-years each**. The map examples are illustrative and do not determine the aggregate bars: for the smoothest and roughest tertiles they are the nearly complete (>= 90% valid) tile-year nearest the tertile's median score; for the intermediate tertile we show WY-3 (2019), a nearly complete tile-year chosen for legibility rather than proximity to the median. Bars report tile-year-balanced MAE averaged over all four phenophases and then equally over seeds 42, 123, and 456. Error bars show standard deviation across seeds.

# %%


BIN_ORDER = ["Smoothest", "Intermediate", "Roughest"]
smooth_bins = pd.read_csv(SPATIAL_DIR / "tile_scores.csv")
smooth_mae = pd.read_csv(SPATIAL_DIR / "mae_by_tertile.csv")

assert smooth_bins.groupby("bin").size().reindex(BIN_ORDER).eq(16).all()
assert smooth_mae["n_seeds"].eq(3).all()

smooth_bins["valid_fraction"] = [
    np.load(TILE_DIR / f"{tile_id}.npz")["ground_truth_valid"].mean()
    for tile_id in smooth_bins["tile_id"]
]
bin_medians = smooth_bins.groupby("bin")["score"].median()
representatives = (
    smooth_bins[smooth_bins["valid_fraction"] >= 0.90]
    .assign(distance_from_bin_median=lambda frame:
            (frame["score"] - frame["bin"].map(bin_medians)).abs())
    .sort_values(["bin", "distance_from_bin_median", "tile_id"])
    .drop_duplicates("bin")
    .set_index("bin").reindex(BIN_ORDER)
)
# Intermediate example chosen for legibility (documented in the markdown above).
intermediate_example = smooth_bins.set_index("tile_id").loc["2019_WY-3_T12TWP"]
for column in ["site_id", "year", "score", "valid_fraction"]:
    representatives.loc["Intermediate", column] = intermediate_example[column]
representatives.loc["Intermediate", "tile_id"] = "2019_WY-3_T12TWP"

display(
    smooth_mae.pivot(index="bin", columns="model", values="mean_mae")
    .reindex(BIN_ORDER).rename(columns=MODEL_LABELS).round(2)
)
display(
    representatives[["tile_id", "score", "valid_fraction"]]
    .rename(columns={"score": "robust score"}).round(2)
)


# %%


representative_tiles = {}
for bin_name in BIN_ORDER:
    tile_id = representatives.loc[bin_name, "tile_id"]
    tile = np.load(TILE_DIR / f"{tile_id}.npz")
    representative_tiles[bin_name] = {
        "tile_id": tile_id,
        "gt": tile["ground_truth_doy"],
        "valid": tile["ground_truth_valid"],
    }

# Center each map on its own median so color represents within-tile timing
# variation rather than differences in mean seasonal timing between sites.
cmap_phenology = plt.cm.RdBu_r.copy()
cmap_phenology.set_bad("0.85")
for item in representative_tiles.values():
    phase_stack = np.ma.array(item["gt"], mask=~item["valid"])
    item["mean_doy"] = phase_stack.mean(axis=0)
    item["doy_anomaly"] = item["mean_doy"] - np.ma.median(item["mean_doy"])

absolute_anomalies = np.concatenate([
    np.abs(item["doy_anomaly"].compressed()) for item in representative_tiles.values()
])
anomaly_limit = float(np.percentile(absolute_anomalies, 98))
vmin, vmax = -anomaly_limit, anomaly_limit

fig = plt.figure(figsize=(3.55, 3.85))
grid = fig.add_gridspec(
    4, 3, height_ratios=[1, 0.055, 0.035, 1.08],
    hspace=0.075, wspace=0.055,
)
image_axes = [fig.add_subplot(grid[0, col]) for col in range(3)]

for col, bin_name in enumerate(BIN_ORDER):
    item = representative_tiles[bin_name]
    ax = image_axes[col]
    image = ax.imshow(
        np.clip(item["doy_anomaly"], vmin, vmax), cmap=cmap_phenology,
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_xticks([]); ax.set_yticks([])
    year, site_id, _ = item["tile_id"].split("_", maxsplit=2)
    ax.set_title(f"{bin_name}\n{site_id} ({year})", fontsize=6.5, fontweight="bold", pad=3)
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.65)

cbar_ax = fig.add_subplot(grid[1, :])
cbar = fig.colorbar(image, cax=cbar_ax, orientation="horizontal", extend="both")
midpoint = (vmin + vmax) / 2
cbar.set_ticks([vmin, midpoint, vmax])
cbar.set_ticklabels([f"≤{vmin:.0f}", "0", f"≥{vmax:.0f}"])
cbar.ax.tick_params(labelsize=5.5, pad=1)
cbar.ax.set_title("Mean phenophase DOY anomaly (days)", fontsize=6, pad=2)

ax = fig.add_subplot(grid[3, :])
x = np.arange(len(BIN_ORDER))
width = 0.25
for offset, model in zip([-width, 0, width], MODEL_ORDER):
    indexed = smooth_mae[smooth_mae["model"] == model].set_index("bin").reindex(BIN_ORDER)
    ax.bar(
        x + offset, indexed["mean_mae"], width=width * 0.92,
        yerr=indexed["seed_std"], color=MODEL_COLORS[model],
        edgecolor="black", linewidth=0.65, capsize=3, label=MODEL_LABELS[model],
    )
ax.set_xticks(x)
ax.set_xticklabels([f"{name}\n(n=16)" for name in BIN_ORDER], fontsize=6)
ax.tick_params(axis="y", labelsize=6)
ax.set_ylabel("Mean absolute error (days)", fontsize=7)
ax.set_ylim(8, 24)
ax.set_yticks(np.arange(8, 25, 2))
ax.grid(axis="y", color="#dddddd", linewidth=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(ncol=3, frameon=False, loc="upper left", fontsize=5.5,
          handlelength=1.4, columnspacing=0.9)

fig.subplots_adjust(left=0.16, right=0.985, top=0.99, bottom=0.08)
# Center the qualitative row over the full visible bar-chart footprint, which
# extends left of the plotting axis because of its tick labels and y-axis label.
for image_ax in image_axes:
    position = image_ax.get_position()
    image_ax.set_position([
        position.x0 - 0.04, position.y0, position.width, position.height
    ])
cbar_position = cbar_ax.get_position()
cbar_ax.set_position([
    cbar_position.x0 - 0.04, cbar_position.y0,
    cbar_position.width, cbar_position.height,
])
smooth_path = IMAGE_DIR / "spatial_variation_tertiles.pdf"
fig.savefig(smooth_path, bbox_inches="tight", dpi=300)
plt.show()
print(f"Saved: {smooth_path}")


# ## Interpretation Boundaries
# 
# - These are descriptive stratifications of 24 sites observed in two years, not estimates of performance over every US ecoregion or land-cover class.
# - Seed variation captures training variability, while the number of independent geographic sites remains the main limitation for regional inference.
# - Land-cover associations are not causal: class, geography, climate, target quality, and sample support are correlated.
# - Spatial-variation tertiles are relative to this test set, and hard bin boundaries discard information. The continuous robust score should be used for formal association tests.
# - The representative maps illustrate median-score tile-years; they are not averages and should not be interpreted as typical of every tile in a tertile.

# %%


expected_outputs = [eco_path, land_path, smooth_path]
for path in expected_outputs:
    assert path.exists() and path.stat().st_size > 1_000, f"Missing or empty figure: {path}"
print("All three paper-ready PDF figures were generated successfully.")
