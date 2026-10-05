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
3. Place the Prithvi EO 2.0 weights at the `model_weights` paths in `config.json`
   (`data/prithvi_checkpoints/`). Presto ships with the repo (`lib/models/presto/`).
4. Place the CEC ecoregion shapefiles in `useco1/` (Level I) and `useco2/` (Level II).
5. Build the shared data caches once, before submitting jobs:
   ```bash
   python misc_scripts/regenerate_caches.py
   ```

All commands run from the repository root. Month subsets are `3 6 9 12` (T=4),
`3 4 5 6 7 8 9 10` (T=8), and `1 2 3 4 5 6 7 8 9 10 11 12` (T=12); scripts under
`run_jobs/run_jobs_{4,8,12}/` cover each subset.

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
