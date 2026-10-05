# Land Surface Phenology with Geospatial Foundation Models

Code for predicting four land surface phenology dates (greenup, maturity,
senescence, dormancy) from sparse monthly HLS composites with Prithvi EO 2.0,
Presto, and a temporal transformer baseline.

## Setup

1. Create the `geo` Conda environment (Python 3.10+, PyTorch with CUDA, `timm`,
   `einops`, `rasterio`, `geopandas`, `scipy`, `pandas`, `seaborn`, `wandb`).
2. Copy the configuration and fill in the data paths:
   ```bash
   cp config.example.json config.json
   ```
   Set `PHENOLOGY_CONFIG=/path/to/config.json` to keep it elsewhere.
3. Download the external data listed under [Data sources](#data-sources) and set
   the corresponding paths in `config.json`.
4. Place the Prithvi EO 2.0 weights at the `model_weights` paths in `config.json`
   (`data/prithvi_checkpoints/`). Presto's code and pretrained weights are included
   (`lib/models/presto/`).
5. Build the shared data caches once, before submitting jobs:
   ```bash
   python misc_scripts/regenerate_caches.py
   ```

All commands run from the repository root. Month subsets are `3 6 9 12` (T=4),
`3 4 5 6 7 8 9 10` (T=8), and `1 2 3 4 5 6 7 8 9 10 11 12` (T=12); scripts under
`run_jobs/run_jobs_{4,8,12}/` cover each subset.

## Data sources

| Input | Source | Location / config key |
|---|---|---|
| HP-LSP reference phenology rasters (2019, 2020) | ORNL DAAC, [doi:10.3334/ORNLDAAC/2248](https://doi.org/10.3334/ORNLDAAC/2248) | `data.lsp_ancillary` |
| HP-LSP pixel samples (`LSP_train_samples.csv`, `LSP_test_samples.csv`; define the evaluation pixels) | [Zenodo record 11583856](https://zenodo.org/records/11583856) | `data/` |
| Monthly HLS composites | Google Earth Engine (export script to be added) | `data.hls_composites` |
| HP-LSP site footprints | Included | `metadata/hp_lsp_site_extents.geojson` (`data.geojson`) |
| Prithvi EO 2.0 weights (tiny, 100M, 300M; `-TL`) | Hugging Face, `ibm-nasa-geospatial/Prithvi-EO-2.0-{tiny,100M,300M}-TL` | `model_weights.*` |
| Presto weights | Included, from [nasaharvest/presto](https://github.com/nasaharvest/presto) (MIT) | `lib/models/presto/data/default_model.pt` |
| Dense baseline (Tran et al., 2025) checkpoint | [khuonghtran/LSP_Transformer](https://github.com/khuonghtran/LSP_Transformer); run it on the test samples | `results/transformer_1d_priorwork_data_test.csv` |
| CEC ecoregions, Level I and II | [US EPA](https://www.epa.gov/eco-research/ecoregions-north-america) | `useco1/`, `useco2/` |
| Annual NLCD land cover 2019/2020 (CONUS) | [USGS/MRLC Annual NLCD](https://www.mrlc.gov/data) | `data.nlcd_2019`, `data.nlcd_2020` |
| US state boundaries (simplified, Census-derived; ecoregion map only) | Included | `metadata/us_states.geojson` (`data.us_states`) |

## 1. Train

The launchers submit SGE jobs (`qsub`, scripts in `run_jobs/sge_scripts/`; set
`-P` to your cluster's project) for the hyperparameter grid over seeds 42, 123,
and 456.

```bash
python run_jobs/run_jobs_4/transformer_1d_paper.py   # Temporal Transformer
python run_jobs/run_jobs_4/presto.py                 # Presto
python run_jobs/run_jobs_4/prithvi_crop32.py         # Prithvi (100M, 32x32 crops)
```

Ablations are the other modules in `run_jobs/run_jobs_4/` (crop size, raw-input
concatenation, pretraining and time/location for Prithvi and Presto); the Presto
regularization ablation (`presto_nolocdrop_nodropout.py`) exists for every T.
`run_jobs/presto_100epoch_check.py` runs the 100-epoch Presto comparison
reported in the appendix.

## 2. Select hyperparameters

```bash
python select_best_params.py --selected_months 3 6 9 12
```

Picks, per model group, the configuration with the lowest validation MAE averaged
over the three selection seeds (final-epoch checkpoints).

## 3. Extra seeds for seed ensembles

```bash
python run_jobs/run_jobs_4/best_params_extra_seeds.py          # trains seeds 789 101 202 303 404 505
python select_best_params.py --selected_months 3 6 9 12 --force --include-extra-seeds
```

## 4. Evaluate

```bash
for split in test val train; do
  python eval_to_dataframe.py --split $split --selected_months 3 6 9 12
done
```

Writes per-pixel predictions to `results/<months>/<group>/seed_*_{split}.csv`.

## 5. Ensembles

```bash
bash run_jobs/run_ensembles.sh
```

## 6. Figures and tables

```bash
python misc_scripts/benchmark_inference.py       # GPU: efficiency panel timings
python misc_scripts/build_qualitative_cache.py   # GPU: qualitative tile predictions
jupyter nbconvert --to notebook --execute --inplace results_overview_notebook.ipynb
```

The notebook writes the paper tables to `paper_latex/Tables/` and figures to
`paper_latex/Images/`. The ecoregion, land-cover, and spatial-variation figures
come from `supportive-analysis/` (see its README).

The dense prior-work baseline (`results/transformer_1d_priorwork_data_test.csv`)
was produced by running the released checkpoint of Tran et al. (2025) on the same
test pixels.
