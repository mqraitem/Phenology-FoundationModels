
import os

# Limit threading libraries BEFORE importing torch/numpy
os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Intel MKL threads
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # OpenBLAS threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # vecLib threads
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # NumExpr threads
os.environ["GDAL_NUM_THREADS"] = "4"
os.environ["GDAL_CACHEMAX"] = "512"  # Limit cache to 512MB

import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml

import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import path_config

from lib.models.prithvi_phenology import PrithviPhenology
from lib.utils import (
	segmentation_loss, segmentation_loss_mae,
	eval_data_loader_crops, get_masks_paper, save_checkpoint,
	str2bool, months_to_str, get_checkpoint_dir,
	get_data_paths, print_trainable_parameters, get_layer_lr_groups,
)
from lib.dataloaders.dataloaders_crops import CycleDatasetCrops
from lib.dataloaders.dataloaders import CycleDataset
from arg_configs import get_core_parser, set_seed

#######################################################################################

def get_layer_lr_groups_singlescale(model, head_lr, backbone_lr, layer_decay=0.75):
	"""Build per-layer param groups for single-scale model.

	Handles model.reshaper, model.upscale_blocks, model.head (all at head_lr),
	and model.backbone with layer-wise decay.
	"""
	encoder = model.backbone.model.encoder
	num_layers = len(encoder.blocks)

	param_groups = []
	seen_params = set()

	# 1) Head parameters — full learning rate
	head_params = list(model.head.parameters())
	if head_params:
		param_groups.append({
			'params': head_params,
			'lr': head_lr,
			'name': 'head',
		})
		seen_params.update(id(p) for p in head_params)

	# 2) Upscale blocks — full learning rate (newly initialized)
	upscale_params = [p for p in model.upscale_blocks.parameters() if id(p) not in seen_params]
	if upscale_params:
		param_groups.append({
			'params': upscale_params,
			'lr': head_lr,
			'name': 'upscale_blocks',
		})
		seen_params.update(id(p) for p in upscale_params)

	# 3) Reshaper has no parameters, but check anyway
	reshaper_params = [p for p in model.reshaper.parameters() if id(p) not in seen_params]
	if reshaper_params:
		param_groups.append({
			'params': reshaper_params,
			'lr': head_lr,
			'name': 'reshaper',
		})
		seen_params.update(id(p) for p in reshaper_params)

	# 3) Backbone blocks — layer-wise decay
	for i, block in enumerate(encoder.blocks):
		block_params = [p for p in block.parameters() if id(p) not in seen_params]
		if block_params:
			layer_lr = backbone_lr * (layer_decay ** (num_layers - 1 - i))
			param_groups.append({
				'params': block_params,
				'lr': layer_lr,
				'name': f'backbone.block.{i}',
			})
			seen_params.update(id(p) for p in block_params)

	# 4) Remaining backbone params (patch_embed, cls_token, norm) — lowest LR
	remaining = [p for p in model.backbone.parameters()
				 if id(p) not in seen_params and p.requires_grad]
	if remaining:
		lowest_lr = backbone_lr * (layer_decay ** num_layers)
		param_groups.append({
			'params': remaining,
			'lr': lowest_lr,
			'name': 'backbone.other',
		})

	return param_groups

#######################################################################################

def main():

	parser = get_core_parser()

	# Prithvi-specific arguments
	parser.add_argument("--model_size", type=str, default="300m",
	                   help="Model size to use (tiny, 100m, or 300m)")
	parser.add_argument("--load_checkpoint", type=str2bool, default=False,
	                   help="Whether to load pretrained checkpoint")
	parser.add_argument("--feed_timeloc", type=str2bool, default=True,
	                   help="Whether to feed time/loc coords")
	parser.add_argument("--n_layers", type=int, default=2,
	                   help="Number of Conv3d layers in temporal fusion head")
	parser.add_argument("--concat_input", type=str2bool, default=True,
	                   help="Concatenate raw spectral input with upscaled features")
	parser.add_argument("--backbone_lr_scale", type=float, default=0.1,
	                   help="Backbone peak LR as a fraction of head LR")
	parser.add_argument("--load_finetuned", type=str2bool, default=False,
	                   help="Load domain-adapted MAE checkpoint")
	parser.add_argument("--dropout", type=float, default=0.0,
	                   help="Dropout rate applied to encoder attention + MLP")
	parser.add_argument("--p_loc_drop", type=float, default=0.0,
	                   help="Prob. of per-sample zeroing the location encoding during training")

	# Crop-specific arguments
	parser.add_argument("--crop_size", type=int, default=48,
	                   help="Spatial size of random crops (must be multiple of 16)")
	parser.add_argument("--epoch_length", type=int, default=5000,
	                   help="Number of crops per epoch")
	# Gradient accumulation
	parser.add_argument("--grad_accum_steps", type=int, default=1,
	                   help="Gradient accumulation steps. Effective batch = batch_size * this")

	# Layer-wise LR decay
	parser.add_argument("--layer_decay", type=float, default=0.75,
	                   help="Layer-wise LR decay factor (0-1)")

	# Normalization
	parser.add_argument("--use_config_normalization", type=str2bool, default=False,
	                   help="Use mean/std from Prithvi config instead of computing from dataset")


	args = parser.parse_args()

	set_seed(args.seed)

	months_str = months_to_str(args.selected_months)
	file_suffix = f"_m{months_str}"
	n_timesteps = len(args.selected_months)

	# --- Config ---
	with open(f'lib/models/prithvi_configs/prithvi_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)

	config["pretrained_cfg"]["img_size"] = args.crop_size
	config["pretrained_cfg"]["num_frames"] = n_timesteps

	group_name = args.group_name
	wandb_name = f"{args.wandb_name}_seed-{args.seed}"

	wandb_config = {
		"learningrate": args.learning_rate,
		"model_size": args.model_size,
		"load_checkpoint": args.load_checkpoint,
		"batch_size": args.batch_size,
		"data_percentage": args.data_percentage,
		"n_layers": args.n_layers,
		"loss": args.loss,
		"backbone_lr_scale": args.backbone_lr_scale,
		"selected_months": args.selected_months,
		"crop_size": args.crop_size,
		"epoch_length": args.epoch_length,
		"optimizer": args.optimizer,
		"grad_accum_steps": args.grad_accum_steps,
		"effective_batch_size": args.batch_size * args.grad_accum_steps,
		"layer_decay": args.layer_decay,
		"warmup_epochs": args.warmup_epochs,
		"min_lr": args.min_lr,
	}

	if args.logging:
		wandb.init(
				project=args.wandb_project or f"phenology_paper_crop_{args.data_percentage}",
				group=group_name,
				config=wandb_config,
				name=wandb_name,
				)
		# wandb.run.log_code(".")  # disabled: snapshots 58k+ files including submodules

	# --- Data ---
	path_train = get_data_paths("training", args.data_percentage, args.selected_months)
	path_val = get_data_paths("validation", args.data_percentage, args.selected_months)
	path_test = get_data_paths("testing", args.data_percentage, args.selected_months)

	print(f"Train: {len(path_train)}, Val: {len(path_val)}, Test: {len(path_test)}")

	# Normalization: use Prithvi pretrained mean/std or compute from dataset
	if args.use_config_normalization:
		cfg_means = np.array(config["pretrained_cfg"]["mean"], dtype=np.float64)
		cfg_stds = np.array(config["pretrained_cfg"]["std"], dtype=np.float64)
		print(f"Using Prithvi config normalization: mean={cfg_means.tolist()}, std={cfg_stds.tolist()}")
	else:
		cfg_means = None
		cfg_stds = None

	# Crop dataset for training
	cycle_dataset_train = CycleDatasetCrops(
		path_train, split="training",
		crop_size=args.crop_size,
		data_percentage=args.data_percentage,
		n_timesteps=n_timesteps,
		file_suffix=file_suffix,
		epoch_length=args.epoch_length,
		means=cfg_means, stds=cfg_stds,
	)

	# Standard tile datasets for val/test (sliding window eval)
	cycle_dataset_val = CycleDataset(path_val, split="validation", data_percentage=args.data_percentage,
	                                  means=cfg_means, stds=cfg_stds,
	                                  n_timesteps=n_timesteps, file_suffix=file_suffix)
	cycle_dataset_test = CycleDataset(path_test, split="testing", data_percentage=args.data_percentage,
	                                   means=cfg_means, stds=cfg_stds,
	                                   n_timesteps=n_timesteps, file_suffix=file_suffix)

	train_dataloader = DataLoader(cycle_dataset_train, batch_size=args.batch_size,
	                              shuffle=True, num_workers=4)
	val_dataloader = DataLoader(cycle_dataset_val, batch_size=1, shuffle=False, num_workers=0)
	test_dataloader = DataLoader(cycle_dataset_test, batch_size=1, shuffle=False, num_workers=0)

	effective_batch = args.batch_size * args.grad_accum_steps
	print(f"Crop size: {args.crop_size}x{args.crop_size}")
	print(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum_steps}, Effective batch: {effective_batch}")
	print(f"Training iterations per epoch: {len(train_dataloader)}")

	# --- Model ---
	device = "cuda"
	if args.load_finetuned:
		weights_path = "data/checkpoints/pretrained_prithvi_1.0/default.pth"
		args.load_checkpoint = True
	elif args.load_checkpoint:
		if args.model_size == "300m" and not args.feed_timeloc:
			weights_path = path_config.get_path("model_weights.300m_nontl")
		else:
			weights_path = path_config.get_model_weights(args.model_size)
	else:
		weights_path = None

	model = PrithviPhenology(
		config["pretrained_cfg"], weights_path,
		n_classes=4, model_size=args.model_size,
		feed_timeloc=args.feed_timeloc, n_layers=args.n_layers,
		concat_input=args.concat_input,
		dropout=args.dropout, p_loc_drop=args.p_loc_drop,
	)
	model = model.to(device)

	n_epochs = args.n_epochs

	print_trainable_parameters(model, detailed=True)

	checkpoint_dir = get_checkpoint_dir(group_name, args.data_percentage, args.selected_months, seed=args.seed)
	checkpoint = f"{checkpoint_dir}/{wandb_name}.pth"

	# --- Optimizer with layer-wise LR decay ---
	head_lr = args.learning_rate
	backbone_lr = args.learning_rate * args.backbone_lr_scale

	if args.load_checkpoint:
		param_groups = get_layer_lr_groups_singlescale(
			model,
			head_lr=head_lr,
			backbone_lr=backbone_lr,
			layer_decay=args.layer_decay,
		)
	else:
		param_groups = [{'params': list(model.parameters()), 'lr': head_lr, 'name': 'all'}]

	# Print LR schedule summary
	print(f"\nLayer-wise LR schedule (decay={args.layer_decay}):")
	for pg in param_groups:
		n_params = sum(p.numel() for p in pg['params'])
		print(f"  {pg['name']:25s}  lr={pg['lr']:.2e}  params={n_params:,}")
	print()

	optimizer = AdamW(param_groups, weight_decay=1e-4)

	# --- Cosine schedule with linear warmup ---
	warmup_epochs = min(args.warmup_epochs, n_epochs)
	cosine_epochs = n_epochs - warmup_epochs

	warmup_scheduler = LinearLR(
		optimizer,
		start_factor=0.01,
		end_factor=1.0,
		total_iters=warmup_epochs,
	)
	cosine_scheduler = CosineAnnealingLR(
		optimizer,
		T_max=cosine_epochs,
		eta_min=args.min_lr,
	)
	scheduler = SequentialLR(
		optimizer,
		schedulers=[warmup_scheduler, cosine_scheduler],
		milestones=[warmup_epochs],
	)
	print(f"LR schedule: {warmup_epochs} warmup epochs + {cosine_epochs} cosine epochs (min_lr={args.min_lr})")

	loss_fn = segmentation_loss_mae if args.loss == "mae" else segmentation_loss
	print(f"Using loss function: {args.loss}")

	# --- Training Loop with Gradient Accumulation ---
	accum_steps = args.grad_accum_steps

	for epoch in range(n_epochs):

		loss_i = 0.0

		current_lr = optimizer.param_groups[0]['lr']
		print(f"Epoch {epoch} started (head_lr={current_lr:.2e})")
		model.train()
		optimizer.zero_grad()

		for j, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

			input = batch_data["image"]
			mask = batch_data["gt_mask"]

			mask = mask.to(device)

			out = model(input)
			cs = args.crop_size
			out = out[:, :, :cs, :cs]
			mask = mask[:, :, :cs, :cs]

			loss = loss_fn(mask=mask, pred=out, device=device)
			loss = loss / accum_steps  # normalize for accumulation
			loss.backward()

			loss_i += loss.item() * accum_steps * mask.size(0)

			# Step optimizer every accum_steps iterations
			if (j + 1) % accum_steps == 0 or (j + 1) == len(train_dataloader):
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
				optimizer.zero_grad()

			if j % 50 == 0:
				to_print = f"Epoch: {epoch}, iteration: {j}, loss: {loss.item() * accum_steps} \n "
				print(to_print)

		epoch_loss_train = loss_i / len(train_dataloader.dataset)

		# Step LR schedule (per epoch, after training)
		scheduler.step()

		# Validation Phase (sliding window, crop_size x crop_size crops over full tiles)
		acc_dataset_val, val_tile_errors, epoch_loss_val = eval_data_loader_crops(
			val_dataloader, model, device, get_masks_paper("train"),
			crop_size=args.crop_size, loss_fn=loss_fn, pixel_weighted=False)
		acc_dataset_test, _, epoch_loss_test = eval_data_loader_crops(
			test_dataloader, model, device, get_masks_paper("test"),
			crop_size=args.crop_size, loss_fn=loss_fn)

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

	save_checkpoint(model, optimizer, epoch, epoch_loss_train, epoch_loss_val, checkpoint, selected_months=args.selected_months)

	if args.logging:
		wandb.finish()


if __name__ == "__main__":
	main()
