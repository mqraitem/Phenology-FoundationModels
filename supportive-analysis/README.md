# Supportive Analyses

This directory contains the reproducible code for three analyses that support
the paper's main model comparison. All analyses use the four-month test setting
(March, June, September, and December), the fixed test split, and model seeds
42, 123, and 456.

Run commands from the repository root with the `geo` Conda environment.

## Files

- `supportive_analysis.py`: canonical plain-text notebook with `# %%` cells,
  suitable for code review and interactive execution in VS Code.
- `supportive_analysis.ipynb`: executed Jupyter companion containing the same
  analysis sections and saved outputs.
- `prepare_stratified_data.py`: GPU inference and spatial data preparation for
  the ecoregion and NLCD analyses.
- `evaluate_smoothness_metrics.py`: ground-truth spatial-variation metric
  comparison and tertile assignment.

The two notebook forms contain the same analysis logic. The plain-text form is
the reviewable source for published code. There is no Python notebook generator
or source file containing quoted notebook cells.

## Analysis 1: Ecoregions

`prepare_stratified_data.py` intersects each test tile with CEC Level I
ecoregions, computes valid-pixel absolute error within each represented region,
and stores tile-level measurements. The notebook averages tile-years within
each seed and then averages seeds equally.

Input geometry: `useco2/NA_CEC_Eco_Level2.shp`

Cached table: `data/stratified_analysis/m3-6-9-12/ecoregion_tile_mae.csv`

Paper figure: `paper_latex/Images/ecoregion_mae_maps.pdf`

## Analysis 2: NLCD Land Cover

`prepare_stratified_data.py` selects the Annual NLCD raster matching each tile
year, reprojects it to the HLS tile grid with nearest-neighbor resampling, and
computes valid-pixel absolute error separately for each represented NLCD class.
The notebook uses tile-year-balanced, seed-balanced aggregation.

Source rasters are supplied with `--nlcd-2019` and `--nlcd-2020`.

Cached table: `data/stratified_analysis/m3-6-9-12/landcover_tile_mae.csv`

Paper figure: `paper_latex/Images/landcover_mae.pdf`

## Analysis 3: Spatial Variation

`evaluate_smoothness_metrics.py` computes candidate ground-truth spatial
variation metrics and compares their association with model MAE. The selected
robust multiscale metric averages median horizontal and vertical absolute DOY
differences over spatial lags 1, 2, 4, and 8. It ranks the 48 tile-years and
assigns 16 to each of the smoothest, intermediate, and roughest tertiles.

Cached tables: `data/stratified_analysis/smoothness_metric_comparison/`

Paper figure: `paper_latex/Images/spatial_variation_tertiles.pdf`

## Reproduction

Build the inference cache on a GPU node:

```bash
conda run -n geo python supportive-analysis/prepare_stratified_data.py \
  --seeds seed_42 seed_123 seed_456 \
  --out-dir data/stratified_analysis/m3-6-9-12
```

Compute the spatial-variation analysis:

```bash
conda run -n geo python supportive-analysis/evaluate_smoothness_metrics.py
```

Then open `supportive-analysis/supportive_analysis.ipynb` with the `geo` kernel
and run all cells from the repository root.
