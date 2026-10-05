# Supportive Analyses

Ecoregion, land-cover, and spatial-variation analyses for the 4-month test
setting. Run from the repository root with the `geo` environment, after the main
models are trained and evaluated. The NLCD rasters and US-states GeoJSON are read
from `data.nlcd_2019`, `data.nlcd_2020`, and `data.us_states` in `config.json`.

```bash
python supportive-analysis/build_tile_layers.py          # CPU: reference dates, masks, ecoregion IDs
python supportive-analysis/prepare_stratified_data.py    # GPU: dense inference, per-stratum errors
python supportive-analysis/spatial_variation.py          # CPU: spatial-variation scores and tertiles
python supportive-analysis/supportive_analysis.py        # CPU: writes the three figures
```

Intermediate tables go to `data/stratified_analysis/`; figures go to
`paper_latex/Images/` (`ecoregion_mae_maps.pdf`, `landcover_mae.pdf`,
`spatial_variation_tertiles.pdf`).
