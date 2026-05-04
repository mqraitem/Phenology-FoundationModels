"""Benchmark inference time per tile for each model at different timestep counts.

Usage:
    python misc_scripts/benchmark_inference.py [--n_tiles 5] [--device cuda]

Measures wall-clock time to process tiles (330x330) with random inputs.
No real data or checkpoints needed — models are randomly initialized.
Model names in the output CSV match those used in results_overview_notebook.ipynb.
"""

import argparse
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import yaml
import pandas as pd
import path_config

TILE_SIZE = 330
N_BANDS = 6
N_CLASSES = 4
CROP_SIZE = 32
# STRIDE = path_config.get_eval_stride()
STRIDE = 10


def find_best_chunk_size(model, T, device, model_type="transformer"):
    """Find the largest chunk_size that doesn't OOM, then benchmark a few to pick fastest."""
    x = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE, device=device)
    candidates = [108900, 54450, 32768, 16384, 8192, 4096, 2048, 1024, 512]

    best_cs = candidates[-1]  # fallback
    best_time = float('inf')

    for cs in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast(device):
                model(x, chunk_size=cs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if elapsed < best_time:
                best_time = elapsed
                best_cs = cs
            print(f"    chunk_size={cs:>6d}: {elapsed:.4f}s")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    chunk_size={cs:>6d}: OOM")
                torch.cuda.empty_cache()
                continue
            raise

    return best_cs


def benchmark_transformer_1d(T, n_tiles, device, chunk_size=None):
    """Benchmark the 1D Temporal Transformer (pixel-level, processes full tile)."""
    from lib.models.transformer_1d_paper import TemporalTransformerPaper

    model = TemporalTransformerPaper(
        input_channels=N_BANDS, seq_len=T, num_classes=N_CLASSES,
        d_model=64, nhead=4, num_layers=4, dropout=0.0,
    ).to(device).eval()

    # Find best chunk size BEFORE compiling (avoid recompilation per shape)
    if chunk_size is None:
        print(f"  Finding best chunk_size for Transformer (T={T})...")
        chunk_size = find_best_chunk_size(model, T, device)
        print(f"  Best chunk_size: {chunk_size}")

    # Compile with the chosen chunk size
    model = torch.compile(model)

    # Warm up
    dummy = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE, device=device)
    with torch.no_grad(), torch.amp.autocast(device):
        model(dummy, chunk_size=chunk_size)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_tiles):
        x = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast(device):
            model(x, chunk_size=chunk_size)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    del model, dummy
    torch.cuda.empty_cache()
    return times, chunk_size


def benchmark_prithvi(T, n_tiles, device, batch_size=None):
    """Benchmark Prithvi (100M, crop32) with sliding-window inference."""
    from lib.models.prithvi_phenology import PrithviPhenology
    from lib.utils import batched_sliding_window

    with open('lib/models/prithvi_configs/prithvi_100m.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config["pretrained_cfg"]["img_size"] = CROP_SIZE
    config["pretrained_cfg"]["num_frames"] = T

    model = PrithviPhenology(
        config["pretrained_cfg"], prithvi_ckpt_path=None,
        n_classes=N_CLASSES, model_size="100m",
    ).to(device).eval()

    temporal_coords = torch.zeros(T, 2)
    location_coords = torch.zeros(2)

    # Find best batch size if not specified
    if batch_size is None:
        print(f"  Finding best batch_size for Prithvi (T={T})...")
        dummy = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE)
        candidates = [256, 128, 64, 32, 16, 8]
        best_bs = candidates[-1]
        best_time = float('inf')
        for bs in candidates:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    batched_sliding_window(model, dummy, CROP_SIZE, device,
                                           tile_size=TILE_SIZE, stride=STRIDE,
                                           batch_size=bs,
                                           temporal_coords=temporal_coords,
                                           location_coords=location_coords)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                if elapsed < best_time:
                    best_time = elapsed
                    best_bs = bs
                print(f"    batch_size={bs:>4d}: {elapsed:.4f}s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    batch_size={bs:>4d}: OOM")
                    torch.cuda.empty_cache()
                    continue
                raise
        batch_size = best_bs
        print(f"  Best batch_size: {batch_size}")

    # Compile with the chosen batch size
    model = torch.compile(model)

    # Warm up
    dummy = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE)
    with torch.no_grad():
        batched_sliding_window(model, dummy, CROP_SIZE, device,
                               tile_size=TILE_SIZE, stride=STRIDE, batch_size=batch_size,
                               temporal_coords=temporal_coords,
                               location_coords=location_coords)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_tiles):
        x = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            batched_sliding_window(model, x, CROP_SIZE, device,
                                   tile_size=TILE_SIZE, stride=STRIDE, batch_size=batch_size,
                               temporal_coords=temporal_coords,
                               location_coords=location_coords)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    del model, dummy
    torch.cuda.empty_cache()
    return times, batch_size


def benchmark_presto(T, n_tiles, device, chunk_size=None):
    """Benchmark Presto with time/location encoding (pixel-level, full tile)."""
    from lib.models.presto_model import PrestoPhenologyModel

    model = PrestoPhenologyModel(
        num_classes=N_CLASSES, freeze_encoder=False,
        input_mode="hls", feed_timeloc=True,
    ).to(device).eval()

    # Dummy month and latlons
    months_0idx = list(range(T))
    month_tensor = torch.tensor(months_0idx, dtype=torch.long)
    latlons = torch.randn(1, TILE_SIZE, TILE_SIZE, 2)

    # Find best chunk size BEFORE compiling (avoid recompilation per shape)
    if chunk_size is None:
        print(f"  Finding best chunk_size for Presto (T={T})...")
        # Need a wrapper since presto forward has extra args
        x_test = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE, device=device)
        candidates = [108900, 54450, 32768, 16384, 8192, 4096, 2048, 1024, 512]
        best_cs = candidates[-1]
        best_time = float('inf')
        for cs in candidates:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad(), torch.amp.autocast(device):
                    model(x_test, latlons=latlons, month=month_tensor, chunk_size=cs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                if elapsed < best_time:
                    best_time = elapsed
                    best_cs = cs
                print(f"    chunk_size={cs:>6d}: {elapsed:.4f}s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    chunk_size={cs:>6d}: OOM")
                    torch.cuda.empty_cache()
                    continue
                raise
        chunk_size = best_cs
        print(f"  Best chunk_size: {chunk_size}")

    # Compile with the chosen chunk size
    model = torch.compile(model)

    # Warm up
    dummy = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE, device=device)
    with torch.no_grad(), torch.amp.autocast(device):
        model(dummy, latlons=latlons, month=month_tensor, chunk_size=chunk_size)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_tiles):
        x = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast(device):
            model(x, latlons=latlons, month=month_tensor, chunk_size=chunk_size)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    del model, dummy
    torch.cuda.empty_cache()
    return times, chunk_size


def benchmark_ensemble(T, n_tiles, device, transformer_cs=1000, presto_cs=2048, prithvi_bs=64):
    """Benchmark full ensemble: run all 3 models + combine predictions.

    This measures the real wall-clock cost of the ensemble at inference.
    Uses the best chunk sizes found for individual models.
    """
    from lib.models.transformer_1d_paper import TemporalTransformerPaper
    from lib.models.prithvi_phenology import PrithviPhenology
    from lib.models.presto_model import PrestoPhenologyModel
    from lib.utils import batched_sliding_window

    # Initialize all 3 models
    transformer = TemporalTransformerPaper(
        input_channels=N_BANDS, seq_len=T, num_classes=N_CLASSES,
        d_model=64, nhead=4, num_layers=4, dropout=0.0,
    ).to(device).eval()
    transformer = torch.compile(transformer)

    with open('lib/models/prithvi_configs/prithvi_100m.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config["pretrained_cfg"]["img_size"] = CROP_SIZE
    config["pretrained_cfg"]["num_frames"] = T
    prithvi = PrithviPhenology(
        config["pretrained_cfg"], prithvi_ckpt_path=None,
        n_classes=N_CLASSES, model_size="100m",
    ).to(device).eval()
    prithvi = torch.compile(prithvi)

    presto = PrestoPhenologyModel(
        num_classes=N_CLASSES, freeze_encoder=False,
        input_mode="hls", feed_timeloc=True,
    ).to(device).eval()
    presto = torch.compile(presto)

    temporal_coords = torch.zeros(T, 2)
    location_coords = torch.zeros(2)
    months_0idx = list(range(T))
    month_tensor = torch.tensor(months_0idx, dtype=torch.long)
    latlons = torch.randn(1, TILE_SIZE, TILE_SIZE, 2)

    # Warm up
    dummy = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE, device=device)
    dummy_cpu = dummy.cpu()
    with torch.no_grad():
        with torch.amp.autocast(device):
            transformer(dummy, chunk_size=transformer_cs)
        batched_sliding_window(prithvi, dummy_cpu, CROP_SIZE, device,
                               tile_size=TILE_SIZE, stride=STRIDE, batch_size=prithvi_bs,
                               temporal_coords=temporal_coords,
                               location_coords=location_coords)
        with torch.amp.autocast(device):
            presto(dummy, latlons=latlons, month=month_tensor, chunk_size=presto_cs)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_tiles):
        x = torch.randn(1, N_BANDS, T, TILE_SIZE, TILE_SIZE, device=device)
        x_cpu = x.cpu()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast(device):
                transformer(x, chunk_size=transformer_cs)
            batched_sliding_window(prithvi, x_cpu, CROP_SIZE, device,
                                   tile_size=TILE_SIZE, stride=STRIDE, batch_size=prithvi_bs,
                               temporal_coords=temporal_coords,
                               location_coords=location_coords)
            with torch.amp.autocast(device):
                presto(x, latlons=latlons, month=month_tensor, chunk_size=presto_cs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    del transformer, prithvi, presto, dummy
    torch.cuda.empty_cache()
    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference time per tile")
    parser.add_argument("--n_tiles", type=int, default=5, help="Number of tiles to benchmark")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    n_tiles = args.n_tiles

    timesteps_our = [4, 8, 12]
    timesteps_prior_work = [244]

    results = []

    # --- Transformer: find best chunk_size at T=4, reuse for all ---
    print("=" * 60)
    print("Transformer")
    print("=" * 60)
    _, best_cs_t = benchmark_transformer_1d(4, 2, device)  # quick run to find chunk_size
    for T in timesteps_our + timesteps_prior_work:
        print(f"Benchmarking Transformer (T={T}, chunk_size={best_cs_t})...")
        times, _ = benchmark_transformer_1d(T, n_tiles, device, chunk_size=best_cs_t)
        for t in times:
            results.append({"Model": "1D Transformer", "Timesteps": T, "Time (s)": t})
        print(f"  Mean: {np.mean(times):.3f}s per tile")

    # --- Prithvi (crop32): find best batch size ---
    torch.cuda.empty_cache()
    print("\n" + "=" * 60)
    print("Prithvi")
    print("=" * 60)
    _, best_bs_prithvi = benchmark_prithvi(4, 2, device)  # quick run to find batch_size
    for T in timesteps_our:
        print(f"Benchmarking Prithvi (T={T}, batch_size={best_bs_prithvi})...")
        times, _ = benchmark_prithvi(T, n_tiles, device, batch_size=best_bs_prithvi)
        for t in times:
            results.append({"Model": "Prithvi (100M, crop32)", "Timesteps": T, "Time (s)": t})
        print(f"  Mean: {np.mean(times):.3f}s per tile")

    # --- Presto (with time/loc): find best chunk_size at T=4, reuse ---
    torch.cuda.empty_cache()
    print("\n" + "=" * 60)
    print("Presto")
    print("=" * 60)
    _, best_cs_p = benchmark_presto(4, 2, device)  # quick run to find chunk_size
    for T in timesteps_our:
        print(f"Benchmarking Presto (T={T}, chunk_size={best_cs_p})...")
        times, _ = benchmark_presto(T, n_tiles, device, chunk_size=best_cs_p)
        for t in times:
            results.append({"Model": "Presto", "Timesteps": T, "Time (s)": t})
        print(f"  Mean: {np.mean(times):.3f}s per tile")

    # --- Ensemble (all 3 models) ---
    torch.cuda.empty_cache()
    print("\n" + "=" * 60)
    print("Ensemble (all)")
    print("=" * 60)
    for T in timesteps_our:
        print(f"Benchmarking Ensemble (T={T})...")
        times = benchmark_ensemble(T, n_tiles, device,
                                   transformer_cs=best_cs_t, presto_cs=best_cs_p,
                                   prithvi_bs=best_bs_prithvi)
        for t in times:
            results.append({"Model": "Ensemble (all)", "Timesteps": T, "Time (s)": t})
        print(f"  Mean: {np.mean(times):.3f}s per tile")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("results/benchmark_results.csv", index=False)
    print(f"\nResults saved to results/benchmark_results.csv")

    # Print summary table
    summary = df.groupby(["Model", "Timesteps"])["Time (s)"].agg(["mean", "std"]).round(4)
    print(f"\n{'='*60}")
    print("Inference time per tile (seconds)")
    print(f"{'='*60}")
    print(summary)


if __name__ == "__main__":
    main()
