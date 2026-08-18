import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import argparse
import pandas as pd
import path_config

from lib.utils import get_masks_paper, eval_data_loader_df, eval_data_loader_crops_df, eval_data_loader_presto_df, get_data_paths, str2bool, months_to_str, get_months_subdir, get_results_dir, build_model
from lib.dataloaders.centroid_tile_dataset import CentroidTileDataset
from lib.dataloaders.pixel_coordinate_tile_dataset import PixelCoordinateTileDataset

#######################################################################################


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
						help="Dataset split to evaluate on (train/val/test)")
	parser.add_argument("--selected_months", type=int, nargs="+", default=[3, 6, 9, 12],
						help="Which months to use (e.g., 3 6 9 12)")
	parser.add_argument("--model-groups", nargs="+", default=None,
						help="Optional exact model-group names to evaluate")
	args = parser.parse_args()

	selected_months = args.selected_months
	months_sub = get_months_subdir(selected_months)

	device = "cuda"

	# Find all model group directories that have best_params.csv
	results_base = os.path.join("results", months_sub)
	if not os.path.exists(results_base):
		print(f"No results directory found at {results_base}")
		return

	model_dirs = [d for d in os.listdir(results_base)
	              if os.path.isdir(os.path.join(results_base, d))
	              and os.path.exists(os.path.join(results_base, d, "best_params.csv"))]
	if args.model_groups is not None:
		requested = set(args.model_groups)
		model_dirs = [d for d in model_dirs if d in requested]

	# For val/train splits, only evaluate the models used in ensembles
	ensemble_models = ["prithvi_final_100m_crop32", "transformer_1d_paper", "presto"]

	for model_name in model_dirs:

		if args.split in ("val", "train"):
			if not any(m in model_name for m in ensemble_models):
				print(f"Skipping {model_name} for {args.split} split (not needed for ensembles)")
				continue

		group = model_name
		data_percentage = float(group.split("_")[-1])

		results_dir = get_results_dir(selected_months, group_name=model_name)
		best_param_df = pd.read_csv(os.path.join(results_dir, "best_params.csv"))

		# Map split name for data loading
		if args.split == "test":
			data_split = "testing"
		elif args.split == "val":
			data_split = "validation"
		else:  # train
			data_split = "training"

		data_loader_name = args.split

		n_timesteps = len(selected_months)
		months_str = months_to_str(selected_months)
		file_suffix = f"_m{months_str}"

		is_presto = "presto" in group

		selected_months_0idx = [m - 1 for m in selected_months]
		month_tensor = torch.tensor(selected_months_0idx, dtype=torch.long)

		for _, row in best_param_df.iterrows():
			seed = row["Seed"]  # e.g. "seed_42"
			best_param = row["Best Param"]

			output_file = os.path.join(results_dir, f"{seed}_{data_loader_name}.csv")
			if os.path.exists(output_file):
				print(f"Results for {model_name}/{seed} on {data_loader_name} already exist, skipping...")
				continue

			# Load checkpoint from seed subdir
			config_dir = os.path.join(path_config.get_checkpoint_root(), months_sub, group, seed)
			ckpt = torch.load(os.path.join(config_dir, best_param))

			print(f"Model: {model_name}, {seed}")
			print(f"Best parameters: {best_param}")
			print(f"Selected months: {selected_months} (n_timesteps={n_timesteps})")

			try:
				model, crop_size = build_model(group, best_param, n_timesteps)
			except ValueError as e:
				print(f"Skipping: {e}")
				continue

			if is_presto:
				data_path = get_data_paths(data_split, data_percentage, selected_months)
				cycle_dataset = PixelCoordinateTileDataset(data_path, split=data_split, data_percentage=data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix, skip_normalization=True)
				data_loader = DataLoader(cycle_dataset, batch_size=1, shuffle=False, num_workers=2)

				model = model.to(device)
				model.load_state_dict(ckpt["model_state_dict"])

				out_df = eval_data_loader_presto_df(data_loader, model, device, get_masks_paper(data_loader_name), month=month_tensor)
			else:
				data_path = get_data_paths(data_split, data_percentage, selected_months)
				cycle_dataset = CentroidTileDataset(data_path, split=data_split, data_percentage=data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix)

				data_loader = DataLoader(cycle_dataset, batch_size=2, shuffle=False, num_workers=2)

				use_config_norm = "_confignorm-True" in best_param
				if use_config_norm:
					with open('lib/models/prithvi_configs/prithvi_300m.yaml', 'r') as f:
						norm_cfg = yaml.safe_load(f)
					cycle_dataset.means = np.array(norm_cfg["pretrained_cfg"]["mean"])
					cycle_dataset.stds  = np.array(norm_cfg["pretrained_cfg"]["std"])

				model = model.to(device)
				model.load_state_dict(ckpt["model_state_dict"])

				if crop_size is not None:
					out_df = eval_data_loader_crops_df(data_loader, model, device, get_masks_paper(data_loader_name), crop_size=crop_size, stride=path_config.get_eval_stride())
				else:
					out_df = eval_data_loader_df(data_loader, model, device, get_masks_paper(data_loader_name))

			out_df.to_csv(output_file, index=False)
			print(f"Saved: {output_file}")

			del model
			torch.cuda.empty_cache()


if __name__ == "__main__":
	main()
