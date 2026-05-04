"""
Training script for Presto phenology model — TEST VERSION.

Same as train_presto.py but uses manual batched iteration over in-RAM tensors
instead of a DataLoader for the train loop. Eval is unchanged.
"""

import os

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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from lib.models.presto_model import PrestoPhenologyModel
from lib.utils import (
    segmentation_loss_pixels, segmentation_loss_pixels_mae,
    segmentation_loss, segmentation_loss_mae,
    eval_data_loader_presto, get_masks_paper,
    print_trainable_parameters, save_checkpoint, str2bool,
    months_to_str, get_checkpoint_dir, get_data_paths,
)
from lib.dataloaders.dataloaders_pixels_pixellatlon import CycleDatasetPixelsPixelLatLon
from lib.dataloaders.dataloaders_pixellatlon import CycleDatasetPixelLatLon
from arg_configs import get_core_parser, set_seed

#######################################################################################

def get_layer_lr_groups_presto(model, head_lr, backbone_lr, layer_decay=0.75):
    """Build per-layer param groups for Presto fine-tuning model."""
    encoder = model.model.encoder
    num_layers = len(encoder.blocks)

    param_groups = []
    seen_params = set()

    head_params = list(model.model.head.parameters())
    if head_params:
        param_groups.append({
            'params': head_params, 'lr': head_lr, 'name': 'head',
        })
        seen_params.update(id(p) for p in head_params)

    for i, block in enumerate(encoder.blocks):
        block_params = [p for p in block.parameters() if id(p) not in seen_params]
        if block_params:
            layer_lr = backbone_lr * (layer_decay ** (num_layers - 1 - i))
            param_groups.append({
                'params': block_params, 'lr': layer_lr, 'name': f'encoder.block.{i}',
            })
            seen_params.update(id(p) for p in block_params)

    remaining = [p for p in encoder.parameters()
                 if id(p) not in seen_params and p.requires_grad]
    if remaining:
        lowest_lr = backbone_lr * (layer_decay ** num_layers)
        param_groups.append({
            'params': remaining, 'lr': lowest_lr, 'name': 'encoder.other',
        })

    return param_groups


def main():

    parser = get_core_parser()
    parser.add_argument("--freeze_encoder", type=str2bool, default=False)
    parser.add_argument("--backbone_lr_scale", type=float, default=1.0)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--feed_timeloc", type=str2bool, default=False)
    parser.add_argument("--timeloc_mode", type=str, default=None,
                       choices=["none", "time", "location", "both"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--p_loc_drop", type=float, default=0.0,
                       help="Probability of zeroing lat/lon per sample during training")
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--use_ndvi", type=str2bool, default=False)
    parser.add_argument("--all_pixels", type=str2bool, default=True)
    args = parser.parse_args()

    set_seed(args.seed)

    months_str = months_to_str(args.selected_months)
    file_suffix = f"_m{months_str}"
    n_timesteps = len(args.selected_months)

    selected_months_0idx = [m - 1 for m in args.selected_months]
    month_tensor = torch.tensor(selected_months_0idx, dtype=torch.long)

    wandb_config = {
        "learningrate": args.learning_rate,
        "batch_size": args.batch_size,
        "data_percentage": args.data_percentage,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "selected_months": args.selected_months,
        "freeze_encoder": args.freeze_encoder,
        "backbone_lr_scale": args.backbone_lr_scale,
        "layer_decay": args.layer_decay,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "input_mode": "hls",
        "feed_timeloc": args.feed_timeloc,
        "timeloc_mode": args.timeloc_mode,
        "dropout": args.dropout,
        "p_loc_drop": args.p_loc_drop,
        "pretrained": args.pretrained,
        "per_pixel_latlon": True,
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

    path_train = get_data_paths("training", args.data_percentage, args.selected_months)
    path_val = get_data_paths("validation", args.data_percentage, args.selected_months)
    path_test = get_data_paths("testing", args.data_percentage, args.selected_months)

    cycle_dataset_val = CycleDatasetPixelLatLon(path_val, split="validation", data_percentage=args.data_percentage,
                                       n_timesteps=n_timesteps, file_suffix=file_suffix, skip_normalization=True)
    cycle_dataset_test = CycleDatasetPixelLatLon(path_test, split="testing", data_percentage=args.data_percentage,
                                        n_timesteps=n_timesteps, file_suffix=file_suffix, skip_normalization=True)

    cycle_dataset_train = CycleDatasetPixelsPixelLatLon(
        path_train, split="training",
        data_percentage=args.data_percentage, n_timesteps=n_timesteps,
        file_suffix=file_suffix, skip_normalization=True,
    )

    val_dataloader = DataLoader(cycle_dataset_val, batch_size=1, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(cycle_dataset_test, batch_size=1, shuffle=False, num_workers=0)

    device = "cuda"

    # ====== Pre-load training tensors into RAM (skip DataLoader entirely) ======
    print("Pre-loading training tensors into RAM...")
    train_inputs  = torch.from_numpy(cycle_dataset_train.inputs)
    train_targets = torch.from_numpy(cycle_dataset_train.targets)
    train_latlons = torch.from_numpy(cycle_dataset_train.latlons)
    N_train = train_inputs.shape[0]
    print(f"  inputs:  {tuple(train_inputs.shape)}  dtype={train_inputs.dtype}")
    print(f"  targets: {tuple(train_targets.shape)} dtype={train_targets.dtype}")
    print(f"  latlons: {tuple(train_latlons.shape)} dtype={train_latlons.dtype}")
    print(f"  N_train = {N_train:,}")
    # Pin memory once for fast async H2D transfers
    train_inputs  = train_inputs.pin_memory()
    train_targets = train_targets.pin_memory()
    train_latlons = train_latlons.pin_memory()

    model = PrestoPhenologyModel(num_classes=4, freeze_encoder=args.freeze_encoder, input_mode="hls",
                              feed_timeloc=args.feed_timeloc, timeloc_mode=args.timeloc_mode,
                              dropout=args.dropout,
                              p_loc_drop=args.p_loc_drop,
                              pretrained=args.pretrained, use_ndvi=args.use_ndvi)

    print_trainable_parameters(model)
    model = model.to(device)

    checkpoint_dir = get_checkpoint_dir(group_name, args.data_percentage, args.selected_months, seed=args.seed)
    checkpoint = f"{checkpoint_dir}/{wandb_name}.pth"

    head_lr = args.learning_rate
    backbone_lr = args.learning_rate * args.backbone_lr_scale

    if not args.freeze_encoder:
        param_groups = get_layer_lr_groups_presto(
            model, head_lr=head_lr, backbone_lr=backbone_lr,
            layer_decay=args.layer_decay,
        )
    else:
        param_groups = [{'params': list(model.model.head.parameters()), 'lr': head_lr, 'name': 'head'}]

    print(f"\nLayer-wise LR schedule (decay={args.layer_decay}):")
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']:25s}  lr={pg['lr']:.2e}  params={n_params:,}")
    print()

    optimizer = AdamW(param_groups, weight_decay=1e-4)

    warmup_epochs = min(args.warmup_epochs, args.n_epochs)
    cosine_epochs = max(1, args.n_epochs - warmup_epochs)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    train_loss_fn = segmentation_loss_pixels_mae if args.loss == "mae" else segmentation_loss_pixels
    eval_loss_fn = segmentation_loss_mae if args.loss == "mae" else segmentation_loss
    print(f"Using loss function: {args.loss}")

    bs = args.batch_size
    n_batches_per_epoch = N_train // bs  # drop_last behavior

    for epoch in range(args.n_epochs):

        loss_i = 0.0
        n_seen = 0
        print("iteration started")
        model.train()

        # Shuffle indices once per epoch
        perm = torch.randperm(N_train)

        for j in tqdm(range(n_batches_per_epoch), total=n_batches_per_epoch):
            idx = perm[j * bs : (j + 1) * bs]

            input   = train_inputs[idx]                                # CPU tensor (pinned)
            mask    = train_targets[idx].to(device, non_blocking=True)
            latlons = train_latlons[idx].to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(input, processing_images=False, latlons=latlons, month=month_tensor)
            loss = train_loss_fn(mask, out, device=device)

            loss_i += loss.item() * input.size(0)
            n_seen += input.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if j % 500 == 0:
                print(f"Epoch: {epoch}, iteration: {j}, loss: {loss.item()} \n ")

        epoch_loss_train = loss_i / max(n_seen, 1)

        acc_dataset_val, val_tile_errors, epoch_loss_val = eval_data_loader_presto(
            val_dataloader, model, device, get_masks_paper("train"),
            month=month_tensor, loss_fn=eval_loss_fn, pixel_weighted=False,
        )
        acc_dataset_test, _, epoch_loss_test = eval_data_loader_presto(
            test_dataloader, model, device, get_masks_paper("test"),
            month=month_tensor, loss_fn=eval_loss_fn,
        )

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
