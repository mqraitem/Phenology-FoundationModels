"""
Training script for the paper-matched 1D Temporal Transformer on S2 data.

Same architecture as train_transformer_1d_paper.py but uses Sentinel-2 dataloaders.
"""

import os

# Limit threading libraries BEFORE importing torch/numpy
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["GDAL_NUM_THREADS"] = "4"
os.environ["GDAL_CACHEMAX"] = "512"

import torch
from torch.utils.data import DataLoader
import numpy as np

import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from lib.models.transformer_1d_paper import TemporalTransformerPaper
from lib.utils import (
	segmentation_loss_pixels, segmentation_loss_pixels_mae,
	segmentation_loss, segmentation_loss_mae,
	eval_data_loader, get_masks_paper, print_trainable_parameters,
	save_checkpoint, str2bool, months_to_str, get_checkpoint_dir, get_data_paths_s2,
)
from lib.dataloaders.dataloaders_pixels_s2 import CycleDatasetPixelsS2
from lib.dataloaders.dataloaders_s2 import CycleDatasetS2
from arg_configs import get_core_parser, set_seed

#######################################################################################

def main():

	parser = get_core_parser()
	parser.add_argument("--dropout", type=float, default=0.1,
					   help="Dropout rate for the transformer")
	parser.add_argument("--d_model", type=int, default=64,
					   help="Transformer feature dimension")
	parser.add_argument("--num_layers", type=int, default=4,
					   help="Number of transformer encoder layers")
	parser.add_argument("--nhead", type=int, default=4,
					   help="Number of attention heads")
	args = parser.parse_args()

	set_seed(args.seed)

	months_str = months_to_str(args.selected_months)
	file_suffix = f"_m{months_str}"
	n_timesteps = len(args.selected_months)

	wandb_config = {
		"learningrate": args.learning_rate,
		"batch_size": args.batch_size,
		"data_percentage": args.data_percentage,
		"loss": args.loss,
		"optimizer": args.optimizer,
		"selected_months": args.selected_months,
		"dropout": args.dropout,
		"d_model": args.d_model,
		"num_layers": args.num_layers,
		"nhead": args.nhead,
		"warmup_epochs": args.warmup_epochs,
		"min_lr": args.min_lr,
	}

	wandb_name = f"{args.wandb_name}_seed-{args.seed}"
	group_name = args.group_name

	if args.logging:
		wandb.init(
			project=args.wandb_project or f"phenology_paper_crop_{args.data_percentage}",
			group=group_name,
			config=wandb_config,
			name=wandb_name,
		)
		# wandb.run.log_code(".")  # disabled: snapshots 58k+ files including submodules

	path_train = get_data_paths_s2("training", args.data_percentage, args.selected_months)
	path_val = get_data_paths_s2("validation", args.data_percentage, args.selected_months)
	path_test = get_data_paths_s2("testing", args.data_percentage, args.selected_months)

	# Create train dataset first so mean/std stats are computed from training data
	cycle_dataset_train = CycleDatasetPixelsS2(
		path_train, split="training",
		data_percentage=args.data_percentage, n_timesteps=n_timesteps,
		file_suffix=file_suffix, skip_normalization=False,
	)

	# Val/test datasets load cached stats computed above
	cycle_dataset_val = CycleDatasetS2(path_val, split="validation", data_percentage=args.data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix, skip_normalization=False)
	cycle_dataset_test = CycleDatasetS2(path_test, split="testing", data_percentage=args.data_percentage, n_timesteps=n_timesteps, file_suffix=file_suffix, skip_normalization=False)

	train_dataloader = DataLoader(cycle_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
	val_dataloader = DataLoader(cycle_dataset_val, batch_size=1, shuffle=False, num_workers=0)
	test_dataloader = DataLoader(cycle_dataset_test, batch_size=1, shuffle=False, num_workers=0)

	device = "cuda"

	model = TemporalTransformerPaper(
		input_channels=6,
		seq_len=n_timesteps,
		num_classes=4,
		d_model=args.d_model,
		nhead=args.nhead,
		num_layers=args.num_layers,
		dropout=args.dropout,
	)

	print_trainable_parameters(model)
	model = model.to(device)

	checkpoint_dir = get_checkpoint_dir(group_name, args.data_percentage, args.selected_months, seed=args.seed)
	checkpoint = f"{checkpoint_dir}/{wandb_name}.pth"

	optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

	warmup_epochs = min(args.warmup_epochs, args.n_epochs)
	cosine_epochs = max(1, args.n_epochs - warmup_epochs)
	warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
	cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.min_lr)
	scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
	print(f"LR schedule: {warmup_epochs} warmup epochs + {cosine_epochs} cosine epochs (min_lr={args.min_lr})")

	train_loss_fn = segmentation_loss_pixels_mae if args.loss == "mae" else segmentation_loss_pixels
	eval_loss_fn = segmentation_loss_mae if args.loss == "mae" else segmentation_loss
	print(f"Using loss function: {args.loss}")

	for epoch in range(args.n_epochs):

		loss_i = 0.0

		print("iteration started")
		model.train()

		for j, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

			input = batch_data["image"]
			mask = batch_data["gt_mask"]

			mask = mask.to(device)

			optimizer.zero_grad()
			out = model(input, processing_images=False)

			loss = train_loss_fn(mask, out, device=device)
			loss_i += loss.item() * input.size(0)

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			if j % 500 == 0:
				to_print = f"Epoch: {epoch}, iteration: {j}, loss: {loss.item()} \n "
				print(to_print)

		epoch_loss_train = loss_i / len(train_dataloader.dataset)

		# Validation Phase
		acc_dataset_val, val_tile_errors, epoch_loss_val = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"), loss_fn=eval_loss_fn, pixel_weighted=False)
		acc_dataset_test, _, epoch_loss_test = eval_data_loader(test_dataloader, model, device, get_masks_paper("test"), loss_fn=eval_loss_fn)

		if args.logging:
			to_log = {}
			to_log["epoch"] = epoch + 1
			to_log["val_loss"] = epoch_loss_val
			to_log["test_loss"] = epoch_loss_test
			to_log["train_loss"] = epoch_loss_train
			to_log["learning_rate"] = optimizer.param_groups[0]['lr']
			for idx in range(4):
				to_log[f"acc_val_{idx}"] = acc_dataset_val[idx]
				to_log[f"acc_test_{idx}"] = acc_dataset_test[idx]
			for tile_name, tile_errs in val_tile_errors.items():
				tile_mae = np.mean([tile_errs[i]["sum_ae"] / tile_errs[i]["n_pixels"] for i in range(4) if tile_errs[i]["n_pixels"] > 0])
				to_log[f"val_tile/{tile_name}"] = tile_mae
			wandb.log(to_log)

		print("=" * 100)
		to_print = f"Epoch: {epoch}, val_loss: {epoch_loss_val} \n "
		for idx in range(4):
			to_print += f"acc_val_{idx}: {acc_dataset_val[idx]} \n "
		for idx in range(4):
			to_print += f"acc_test_{idx}: {acc_dataset_test[idx]} \n "
		print(to_print)
		print("=" * 100)

		scheduler.step()

	save_checkpoint(model, optimizer, epoch, epoch_loss_train, epoch_loss_val, checkpoint, selected_months=args.selected_months)

	if args.logging:
		wandb.finish()


if __name__ == "__main__":
	main()
