# Crop Phenology Prediction with Foundation Models

Predicting crop phenology dates (day-of-year) from multi-temporal satellite imagery, comparing three architectures: a [Prithvi EO V2](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) ViT backbone with a Conv3D temporal-fusion head, the [Presto](https://github.com/nasaharvest/presto) pretrained pixel-level transformer, and a lightweight 1D temporal transformer baseline.

## Task

Given monthly satellite composites — Harmonized Landsat-Sentinel (HLS) at 30m or Sentinel-2 (S2) at 10m downsampled to 30m — with 6 spectral bands, predict 4 phenological dates per pixel:

| Output | Description |
|--------|-------------|
| **G** | Greenup / Germination |
| **M** | Maturity |
| **S** | Senescence / Silking |
| **D** | Dormancy / Dough |

Ground truth is the High Plains Land Surface Phenology (HP-LSP) dataset. Predictions are evaluated on 330x330 pixel tiles using Mean Absolute Error (MAE) in days.

## Models

### Prithvi (`train_prithvi.py`)
Prithvi EO V2 backbone (`100m`, `300m`, or `tiny`) feeding a Conv3D upscaler + linear regression head. Crops are sliced from the full tile at training time (`--crop_size`). Training uses layer-wise LR decay over the backbone (`--layer_decay`, `--backbone_lr_scale`).

### Presto (`train_presto.py`)
Presto pixel-level transformer encoder with a linear head, on Sentinel-2 (B2, B3, B4, B8A, B11, B12). S2 is 3x3 block-averaged 10m → 30m to match the GT grid; pixels with median cloud score ≥ 3000 are masked. Encoder can be frozen (`--freeze_encoder`) or fine-tuned. Optional location dropout (`--p_loc_drop`) and NDVI augmentation (`--use_ndvi`).

> `lib/models/presto/` contains the lightly modified model-runtime subset of [nasaharvest/presto](https://github.com/nasaharvest/presto); see `lib/models/presto/VENDORED.md`.

### Temporal Transformer (`train_transformer_1d_paper.py`)
Per-pixel 1D transformer over the HLS spectral time series. Configurable via `--d_model`, `--num_layers`, `--nhead`, `--dropout`.

## Setup

### Prerequisites
- Python 3.10+, PyTorch with CUDA
- `timm`, `einops`, `rasterio`, `wandb`, `geopandas`, `scipy`, `pandas`, `seaborn`

### Vendored Presto
The model-runtime subset under `lib/models/presto/` is included directly, with
small local modifications that avoid the unused Earth Engine and OpenMapFlow
pipelines. It requires PyTorch, Einops, NumPy, and python-dateutil. See
`lib/models/presto/VENDORED.md` for provenance and modifications.

### Configuration

Create the machine-local configuration before training:

```bash
cp config.example.json config.json
```

Edit the external data paths in `config.json`. Repository-relative storage and
weight paths are resolved from the repository root. `config.json` is ignored by
Git; `config.example.json` documents the published configuration structure.

To keep the configuration elsewhere, set:

```bash
export PHENOLOGY_CONFIG=/path/to/config.json
```

Code reads nested settings using dotted JSON keys, for example
`get_path("data.hls_composites")` and `get_value("evaluation.stride")`.

### Eco-region shapefile (optional, notebook only)
The notebook's regional analysis reads `useco1/NA_CEC_Eco_Level1.shp`. Download from EPA / CEC and place at `useco1/`.

## Training

### Prithvi (single run)

```bash
python train_prithvi.py \
    --load_checkpoint True \
    --model_size 100m \
    --crop_size 32 \
    --n_layers 4 \
    --concat_input True \
    --feed_timeloc True \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --n_epochs 150 \
    --epoch_length 5000 \
    --layer_decay 0.75 \
    --backbone_lr_scale 1.0 \
    --warmup_epochs 5 \
    --selected_months 3 6 9 12 \
    --logging True \
    --group_name prithvi_final_100m_crop32 \
    --wandb_name seed_42
```

### Presto (single run)

```bash
python train_presto.py \
    --pretrained True \
    --freeze_encoder False \
    --feed_timeloc True \
    --learning_rate 5e-5 \
    --batch_size 1024 \
    --n_epochs 150 \
    --warmup_epochs 5 \
    --selected_months 3 6 9 12 \
    --logging True \
    --group_name presto \
    --wandb_name seed_42
```

### Temporal Transformer (single run)

```bash
python train_transformer_1d_paper.py \
    --d_model 64 --num_layers 4 --nhead 4 --dropout 0.1 \
    --learning_rate 1e-4 --batch_size 1024 --n_epochs 150 \
    --selected_months 3 6 9 12 \
    --logging True --group_name transformer_1d_paper --wandb_name seed_42
```

### Batch job submission (SGE)
SGE launch templates live in `run_jobs/sge_scripts/` (`train_presto.sh`, `train_prithvi.sh`). Per-subset Python sweep drivers are in `run_jobs/run_jobs_{4,8,12}/` — one module per (model, ablation) pair, e.g.:

```bash
python -m run_jobs.run_jobs_4.prithvi_crop32     # Prithvi at T=4
python -m run_jobs.run_jobs_8.presto             # Presto at T=8
python -m run_jobs.run_jobs_12.transformer_1d_paper
```

Cross-model ensembling once seeds are trained:

```bash
bash run_jobs/run_ensembles.sh
```

## Evaluation

```bash
python select_best_params.py            # pick best val checkpoint per (group, seed)
python eval_to_dataframe.py             # export per-pixel predictions to results/<subset>/<group>/seed_*.csv
python misc_scripts/ensemble_seeds.py   # seed-level ensemble per group
python misc_scripts/ensemble_from_csvs.py   # cross-model ensemble (transformer + presto + prithvi)
```

Other utilities under `misc_scripts/`:

- `benchmark_inference.py` — produces `results/benchmark_results.csv` for the efficiency panel.
- `build_qualitative_cache.py` — caches per-tile predictions for the qualitative figure (notebook section 4.1).
- `regenerate_caches.py` — rebuilds dataloader caches after data changes.
- `stride_ablation.py` — sliding-window stride ablation for Prithvi eval.
- `check_data_stats.py`, `check_dead_pixel_gt_mismatch.py` — dataset sanity checks.

## Analysis notebook

Open `results_overview_notebook.ipynb` for the full set of paper figures and tables: per-subset comparisons, efficiency, crop-size effect, per-tile tables, qualitative tile predictions, eco-region performance, and the ablations (pretraining × time/location, regularization, concat, model size, ensembles, missing-timestep robustness).

## Reproducing this repo from scratch

The history is intentionally trimmed. To re-init a clean repo with only reproducer code (no datasets, checkpoints, or paper sources):

```bash
bash misc_scripts/setup_git_repo.sh
git commit -m "Initial commit"
```

## Key training details

- **LR schedule (Prithvi/Presto):** linear warmup (5 epochs) → cosine annealing.
- **LR schedule (1D Transformer):** cosine annealing, no warmup.
- **Layer-wise LR decay (Prithvi only):** factor 0.75 per backbone layer.
- **Loss:** MSE on normalized DOY (divided by 547; -1 marks invalid).
- **Evaluation:** sliding-window crops over 330x330 tiles, averaging overlapping predictions (Prithvi); full-tile pixel-wise (Presto, 1D Transformer).
- **Cloud masking (S2):** zero out pixels with median cloud score ≥ 3000 after 3x3 block averaging.
- **Logging:** Weights & Biases, per-epoch train/val/test metrics.
