"""Run dense inference and cache selected qualitative test tiles.

This script is the GPU inference stage for ``results_overview_notebook.ipynb``.
It resolves selected test tiles, loads each model's selected checkpoint, runs
full-tile inference, optionally applies a per-seed model ensemble, and writes
one compressed ``.npz`` cache per site/tile. Plotting remains in the notebook.

Tile selectors accept either ``SITE_ID=HLS_TILE`` (latest matching test year)
or ``YEAR=SITE_ID=HLS_TILE``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import path_config
from lib.dataloaders.tile_dataset import TileDataset
from lib.dataloaders.georeferenced_tile_dataset import GeoreferencedTileDataset
from lib.utils import (
    batched_sliding_window,
    build_model,
    get_data_paths,
    months_to_str,
)


TILE_SIZE = 330
DOY_SCALE = 547.0
PRED_COLUMNS = ["G_pred_DOY", "M_pred_DOY", "S_pred_DOY", "D_pred_DOY"]


def _parse_tile_selector(value: str) -> tuple[str | None, str, str]:
    parts = value.split("=")
    if len(parts) == 2:
        return None, parts[0], parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise argparse.ArgumentTypeError(
        f"Invalid tile selector {value!r}; use SITE_ID=HLS_TILE or YEAR=SITE_ID=HLS_TILE."
    )


def _select_test_paths(all_paths: list, selectors: list[tuple[str | None, str, str]]) -> list:
    """Resolve selectors in their requested order, preferring the latest year."""
    selected = []
    for requested_year, site_id, hls_tile in selectors:
        matches = []
        for row in all_paths:
            year, row_site, row_tile = row[2].split("_", 2)
            if row_site == site_id and row_tile == hls_tile:
                if requested_year is None or year == requested_year:
                    matches.append(row)
        if not matches:
            qualifier = f"{requested_year}=" if requested_year else ""
            raise ValueError(f"No test tile matches {qualifier}{site_id}={hls_tile}.")
        selected.append(max(matches, key=lambda row: int(row[2].split("_", 1)[0])))
    return selected


def _best_checkpoint(group: str, seed: str, months_sub: str) -> tuple[str, Path]:
    best_path = REPO_ROOT / "results" / months_sub / group / "best_params.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing selected-parameter table: {best_path}")
    best = pd.read_csv(best_path)
    row = best[best["Seed"] == seed]
    if row.empty:
        available = ", ".join(best["Seed"].astype(str))
        raise ValueError(f"No {seed} checkpoint for {group}; available seeds: {available}")
    filename = str(row.iloc[0]["Best Param"])
    checkpoint = Path(path_config.get_checkpoint_root()) / months_sub / group / seed / filename
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    return filename, checkpoint


def _load_model(group: str, seed: str, selected_months: list[int], device: str):
    months_sub = f"m{months_to_str(selected_months)}"
    filename, checkpoint_path = _best_checkpoint(group, seed, months_sub)
    model, crop_size = build_model(group, filename, len(selected_months))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), crop_size, filename


def _predict_tile(
    group: str,
    model: torch.nn.Module,
    crop_size: int | None,
    hls_data: dict,
    presto_data: dict,
    month_tensor: torch.Tensor,
    device: str,
) -> np.ndarray:
    with torch.inference_mode():
        if "presto" in group:
            prediction = model(
                presto_data["image"],
                processing_images=True,
                latlons=presto_data["latlons"].to(device),
                month=month_tensor,
            )[0, :, :TILE_SIZE, :TILE_SIZE]
        elif crop_size is not None:
            image = hls_data["image"]
            prediction = batched_sliding_window(
                model,
                image["chip"],
                crop_size,
                device,
                tile_size=TILE_SIZE,
                stride=path_config.get_eval_stride(),
                temporal_coords=image["temporal_coords"][0],
                location_coords=image["location_coords"][0],
            )
        else:
            prediction = model(hls_data["image"], processing_images=True)[
                0, :, :TILE_SIZE, :TILE_SIZE
            ]
    return prediction.detach().float().cpu().numpy().astype(np.float32)


def _make_rgb(hls_data: dict, mid_index: int) -> np.ndarray:
    raw = hls_data["image_unprocessed"][0, :, mid_index, :TILE_SIZE, :TILE_SIZE]
    rgb = raw[[2, 1, 0]].permute(1, 2, 0).numpy().astype(np.float32)
    valid = rgb[rgb > 0]
    if valid.size:
        low, high = np.percentile(valid, [2, 98])
        rgb = (rgb - low) / max(float(high - low), 1e-6)
    else:
        rgb.fill(0)
    return np.clip(rgb, 0, 1).astype(np.float32)


def _load_ensemble(path: str | None, models: list[str], seed: str, months: list[int]):
    if not path:
        return None
    ensemble_path = Path(path)
    if not ensemble_path.is_absolute():
        ensemble_path = REPO_ROOT / ensemble_path
    with ensemble_path.open(encoding="utf-8") as file:
        info = json.load(file)
    if info.get("selected_months") != months:
        raise ValueError(
            f"Ensemble months {info.get('selected_months')} do not match requested months {months}."
        )
    if info.get("methods") != models:
        raise ValueError(
            "Ensemble methods must exactly match --models and their order.\n"
            f"Ensemble: {info.get('methods')}\nModels:   {models}"
        )
    weights_by_seed = info.get("weights_per_seed", {})
    if seed not in weights_by_seed:
        raise ValueError(f"Ensemble {ensemble_path} has no weights for {seed}.")
    return info, weights_by_seed[seed], ensemble_path


def _ensemble_prediction(model_predictions: dict[str, np.ndarray], weights: dict) -> np.ndarray:
    output = np.empty_like(next(iter(model_predictions.values())))
    model_stack = np.stack(list(model_predictions.values()), axis=0)
    for phase, column in enumerate(PRED_COLUMNS):
        phase_weights = np.asarray(weights[column], dtype=np.float32)
        if phase_weights.shape != (model_stack.shape[0],):
            raise ValueError(f"Invalid weights for {column}: {phase_weights}")
        output[phase] = np.tensordot(phase_weights, model_stack[:, phase], axes=(0, 0))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--tile_ids", type=_parse_tile_selector, nargs="+", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--ensemble_file", default=None)
    parser.add_argument("--seed", default="seed_42")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cache_only",
        action="store_true",
        help="Retained for compatibility; this script only writes caches.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense qualitative-tile inference.")
    device = "cuda"
    months_sub = f"m{months_to_str(args.selected_months)}"
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ensemble = _load_ensemble(
        args.ensemble_file, args.models, args.seed, args.selected_months
    )
    all_test_paths = get_data_paths("testing", 1.0, args.selected_months)
    selected_paths = _select_test_paths(all_test_paths, args.tile_ids)

    file_suffix = f"_{months_sub}"
    hls_dataset = TileDataset(
        selected_paths,
        split="testing",
        data_percentage=1.0,
        n_timesteps=len(args.selected_months),
        file_suffix=file_suffix,
    )
    presto_dataset = GeoreferencedTileDataset(
        selected_paths,
        split="testing",
        data_percentage=1.0,
        n_timesteps=len(args.selected_months),
        file_suffix=file_suffix,
        skip_normalization=True,
    )
    hls_loader = DataLoader(hls_dataset, batch_size=1, shuffle=False, num_workers=0)
    presto_loader = DataLoader(presto_dataset, batch_size=1, shuffle=False, num_workers=0)

    loaded_models = {}
    for group in args.models:
        print(f"Loading {group} ({args.seed})...")
        loaded_models[group] = _load_model(group, args.seed, args.selected_months, device)

    month_tensor = torch.tensor([month - 1 for month in args.selected_months], dtype=torch.long)
    mid_index = len(args.selected_months) // 2
    for hls_data, presto_data in zip(hls_loader, presto_loader):
        full_tile_id = hls_data["hls_tile_name"][0]
        _, site_id, hls_tile = full_tile_id.split("_", 2)
        output_path = cache_dir / f"{site_id}_{hls_tile}.npz"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing cache: {output_path}")
            continue

        predictions = {}
        for group, (model, crop_size, _) in loaded_models.items():
            print(f"Predicting {full_tile_id} with {group}...")
            predictions[group] = _predict_tile(
                group, model, crop_size, hls_data, presto_data, month_tensor, device
            )

        if ensemble is not None:
            _, ensemble_weights, ensemble_path = ensemble
            predictions["__ensemble__"] = _ensemble_prediction(predictions, ensemble_weights)
        else:
            ensemble_path = None

        gt = hls_data["gt_mask"][0, :, :TILE_SIZE, :TILE_SIZE].numpy().astype(np.float32)
        prediction_stack = np.stack(list(predictions.values()), axis=0).astype(np.float32)
        valid = gt != -1
        global_min = np.array(
            [np.nanmin(np.where(valid[i], gt[i] * DOY_SCALE, np.nan)) for i in range(4)],
            dtype=np.float32,
        )
        global_max = np.array(
            [np.nanmax(np.where(valid[i], gt[i] * DOY_SCALE, np.nan)) for i in range(4)],
            dtype=np.float32,
        )

        np.savez_compressed(
            output_path,
            hls_tile_name=np.array(full_tile_id),
            site_id=np.array(site_id),
            hls_tile=np.array(hls_tile),
            selected_months=np.asarray(args.selected_months, dtype=np.int32),
            mid_month=np.int32(args.selected_months[mid_index]),
            ground_truth=gt,
            hls_rgb=_make_rgb(hls_data, mid_index),
            location_coords=hls_data["latlons"][0].numpy().astype(np.float32),
            global_min_doy=global_min,
            global_max_doy=global_max,
            model_names=np.asarray(list(predictions)),
            predictions=prediction_stack,
            seed=np.array(args.seed),
            checkpoints=np.asarray([loaded_models[group][2] for group in args.models]),
            ensemble_file=np.array(str(ensemble_path) if ensemble_path else ""),
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
