"""Select best hyperparameters per model group based on average val MAE across seeds.

For each model group, evaluates all checkpoints across all seeds, groups them by
hyperparameter (filename minus seed suffix), averages val MAE across seeds, and
picks the hyperparameter with the best average. Saves one row per seed in
best_params.csv, all pointing to the same hyperparameter (with seed-specific filename).

By default, skips any group whose best_params.csv already exists. Pass --force to
recompute from scratch (e.g. after adding new seeds or new checkpoints).
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import re
import argparse
from tqdm import tqdm
import pandas as pd
import path_config
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from lib.utils import get_data_paths, eval_data_loader, eval_data_loader_crops, eval_data_loader_presto, get_masks_paper, str2bool, months_to_str, get_months_subdir, get_results_dir, build_model
from lib.dataloaders.tile_dataset import TileDataset
from lib.dataloaders.georeferenced_tile_dataset import GeoreferencedTileDataset

#######################################################################################


def strip_seed_from_filename(filename):
	"""Remove _seed-N suffix to get the hyperparameter key.
	e.g. 'model_lr-0.001_seed-42.pth' -> 'model_lr-0.001'
	"""
	return re.sub(r'_seed-\d+\.pth$', '', filename)


def eval_checkpoint(model, device, is_presto, crop_size,
                    val_dataloader, cycle_dataset_val, orig_means, orig_stds,
                    params, month_tensor=None):
	"""Evaluate a single checkpoint and return mean val MAE."""
	if is_presto:
		acc, _, _ = eval_data_loader_presto(val_dataloader, model, device,
		            get_masks_paper("train"), month=month_tensor, pixel_weighted=False)
	else:
		use_config_norm = "_confignorm-True" in params
		if use_config_norm:
			with open('lib/models/prithvi_configs/prithvi_300m.yaml', 'r') as f:
				norm_cfg = yaml.safe_load(f)
			cycle_dataset_val.means = np.array(norm_cfg["pretrained_cfg"]["mean"])
			cycle_dataset_val.stds  = np.array(norm_cfg["pretrained_cfg"]["std"])

		if crop_size is not None:
			acc, _, _ = eval_data_loader_crops(val_dataloader, model, device,
			            get_masks_paper("train"), crop_size=crop_size, stride=crop_size,
			            pixel_weighted=False)
		else:
			acc, _, _ = eval_data_loader(val_dataloader, model, device,
			            get_masks_paper("train"), pixel_weighted=False)

		if use_config_norm:
			cycle_dataset_val.means = orig_means
			cycle_dataset_val.stds  = orig_stds

	return np.mean(list(acc.values()))


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12],
						help="Which months to use (e.g., 3 6 9 12)")
	parser.add_argument("--force", action="store_true",
						help="Recompute even if best_params.csv already exists")
	args = parser.parse_args()

	selected_months = args.selected_months
	months_sub = get_months_subdir(selected_months)
	n_timesteps = len(selected_months)
	m_str = months_to_str(selected_months)
	file_suffix = f"_m{m_str}"

	device = "cuda"
	groups_dir = os.path.join(path_config.get_checkpoint_root(), months_sub)

	if not os.path.exists(groups_dir):
		print(f"No checkpoint directory found at {groups_dir}")
		return

	all_groups = os.listdir(groups_dir)

	# Filter to supported model groups only
	supported = ["prithvi", "transformer_1d", "presto"]

	all_groups = [g for g in all_groups if any(s in g for s in supported)]

	for group_idx, group_name in enumerate(all_groups):

		group = group_name
		data_percentage = group.split("_")[-1]
		is_presto = "presto" in group

		# Skip if best_params.csv already exists (unless --force)
		results_dir = get_results_dir(selected_months, group_name=group_name)
		best_params_path = os.path.join(results_dir, "best_params.csv")
		if os.path.exists(best_params_path) and not args.force:
			print(f"\n[{group_idx+1}/{len(all_groups)}] {group_name}: best_params.csv exists, skipping (use --force to recompute)")
			continue

		batch_size = 2

		# Find seed subdirectories
		group_path = os.path.join(groups_dir, group)
		seed_dirs = sorted([d for d in os.listdir(group_path)
		                    if d.startswith("seed_") and os.path.isdir(os.path.join(group_path, d))])

		if not seed_dirs:
			print(f"No seed directories found in {group_path}, skipping.")
			continue

		# Count total checkpoints
		total_ckpts = sum(len([p for p in os.listdir(os.path.join(group_path, sd)) if p.endswith(".pth")]) for sd in seed_dirs)
		n_hp = len(set(strip_seed_from_filename(p) for sd in seed_dirs for p in os.listdir(os.path.join(group_path, sd)) if p.endswith(".pth")))

		print(f"\n{'='*70}")
		print(f"  [{group_idx+1}/{len(all_groups)}] {group_name}")
		print(f"  Seeds: {seed_dirs}")
		print(f"  Hyperparameters: {n_hp}, Checkpoints: {total_ckpts} ({total_ckpts // len(seed_dirs)} per seed)")
		print(f"{'='*70}")

		month_tensor = None
		if is_presto:
			selected_months_0idx = [m - 1 for m in selected_months]
			month_tensor = torch.tensor(selected_months_0idx, dtype=torch.long)

		# Build dataloader once (shared across all seeds)
		cycle_dataset_val = None
		orig_means = None
		orig_stds = None

		if is_presto:
			path_val = get_data_paths("validation", data_percentage, selected_months)
			cycle_dataset_val = GeoreferencedTileDataset(path_val, split="validation", data_percentage=data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix, skip_normalization=True)
			val_dataloader = DataLoader(cycle_dataset_val, batch_size=1, shuffle=False, num_workers=2)
		else:
			path_val = get_data_paths("validation", data_percentage, selected_months)
			cycle_dataset_val = TileDataset(path_val, split="validation", data_percentage=data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix)
			orig_means = cycle_dataset_val.means.copy()
			orig_stds  = cycle_dataset_val.stds.copy()
			val_dataloader = DataLoader(cycle_dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)

		# Collect val MAE for every (hyperparameter, seed) pair
		# hp_seed_maes: {hp_key: {seed_dir: (mae, filename)}}
		hp_seed_maes = {}

		ckpt_counter = 0
		for seed_dir in seed_dirs:
			seed_path = os.path.join(group_path, seed_dir)
			pth_files = sorted([p for p in os.listdir(seed_path) if p.endswith(".pth")])
			if not pth_files:
				continue

			print(f"\n  --- {seed_dir} ({len(pth_files)} checkpoints) ---")

			for params in pth_files:
				ckpt_counter += 1

				hp_key = strip_seed_from_filename(params)
				checkpoint = os.path.join(seed_path, params)

				try:
					model, crop_size = build_model(group, params, n_timesteps)
				except ValueError as e:
					print(f"  [{ckpt_counter}/{total_ckpts}] SKIP: {e}")
					continue

				model = model.to(device)
				model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

				mae = eval_checkpoint(model, device, is_presto, crop_size,
				                      val_dataloader, cycle_dataset_val, orig_means, orig_stds,
				                      params, month_tensor)

				print(f"  [{ckpt_counter}/{total_ckpts}] {hp_key} | {seed_dir} | val MAE = {mae:.2f}")

				if hp_key not in hp_seed_maes:
					hp_seed_maes[hp_key] = {}
				hp_seed_maes[hp_key][seed_dir] = (mae, params)

				del model
				torch.cuda.empty_cache()

		if not hp_seed_maes:
			print(f"No eligible checkpoints for {group_name}, skipping.")
			continue

		# Pick hyperparameter with best average MAE across seeds
		best_hp = None
		best_avg_mae = float('inf')

		print(f"\n  {'─'*60}")
		print(f"  RANKING ({len(hp_seed_maes)} hyperparameters, {len(seed_dirs)} seeds)")
		print(f"  {'─'*60}")

		ranked = []
		for hp_key, seed_results in hp_seed_maes.items():
			maes = [v[0] for v in seed_results.values()]
			ranked.append((hp_key, np.mean(maes), np.std(maes) if len(maes) > 1 else 0.0, len(maes)))
		ranked.sort(key=lambda x: x[1])

		for rank, (hp_key, avg, std, ns) in enumerate(ranked):
			marker = " <-- BEST" if rank == 0 else ""
			std_str = f" +/- {std:.2f}" if ns > 1 else ""
			print(f"  #{rank+1:2d}  avg MAE = {avg:.2f}{std_str}  ({ns} seeds)  {hp_key}{marker}")

			if rank == 0:
				best_avg_mae = avg
				best_hp = hp_key

		print(f"\n  SELECTED: {best_hp}")

		# Print detailed table with per-seed MAEs
		print(f"\n  {'─'*60}")
		print(f"  DETAILED TABLE")
		print(f"  {'─'*60}")

		# Find common prefix to strip from hyperparameter names
		all_hp_keys = list(hp_seed_maes.keys())
		if len(all_hp_keys) > 1:
			common_prefix = os.path.commonprefix(all_hp_keys)
			# Trim to last underscore so we don't cut mid-word
			last_sep = common_prefix.rfind("_")
			common_prefix = common_prefix[:last_sep + 1] if last_sep >= 0 else ""
		else:
			common_prefix = ""

		table_rows = []
		for hp_key, seed_results in hp_seed_maes.items():
			maes = [v[0] for v in seed_results.values()]
			row = {"Hyperparameter": hp_key[len(common_prefix):]}
			for sd in seed_dirs:
				if sd in seed_results:
					row[sd] = f"{seed_results[sd][0]:.2f}"
				else:
					row[sd] = "---"
			row["Avg"] = f"{np.mean(maes):.2f}"
			row["Std"] = f"{np.std(maes):.2f}" if len(maes) > 1 else "---"
			table_rows.append(row)

		table_df = pd.DataFrame(table_rows)
		table_df = table_df.sort_values("Avg")
		print(table_df.to_string(index=False))

		# Save best_params.csv with one row per seed
		rows = []
		for seed_dir in seed_dirs:
			if seed_dir in hp_seed_maes.get(best_hp, {}):
				_, filename = hp_seed_maes[best_hp][seed_dir]
				rows.append({"Seed": seed_dir, "Best Param": filename})

		if rows:
			param_df = pd.DataFrame(rows)
			param_df.to_csv(best_params_path, index=False)
			print(f"  Saved: {best_params_path}")
			print(param_df.to_string(index=False))
		else:
			print(f"  WARNING: No seeds found for best hyperparameter {best_hp}")

if __name__ == "__main__":
	main()
