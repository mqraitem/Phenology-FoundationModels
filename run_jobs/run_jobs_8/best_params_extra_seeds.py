"""Submit extra-seed runs for ensemble models at their existing best HP (8 months).

Reads each model's best_params.csv, extracts the (single) winning hyperparameter
combo, and re-submits training jobs for `extra_seeds` only — leaving the chosen
HP untouched so that re-running select_best_params.py after these complete will
just enrich the seed-mean estimate rather than shift the HP choice.
"""
import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from run_jobs_8.common import is_done, setup_records_dir, get_wandb_project
from lib.utils import parse_param, str2bool

selected_months = [3, 4, 5, 6, 7, 8, 9, 10]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)

# 6 new seeds (combined with existing {42, 123, 456} → 9 total → 3 disjoint groups of 3)
extra_seeds = [789, 101, 202, 303, 404, 505]
data_percentage = 1.0
wandb_project = get_wandb_project(data_percentage, months_str)


def best_hp_name(group_with_pct):
    bp = f"results/m{months_str}/{group_with_pct}/best_params.csv"
    if not os.path.exists(bp):
        print(f"[skip] no best_params.csv for {group_with_pct}")
        return None
    df = pd.read_csv(bp)
    fname = df["Best Param"].iloc[0]
    return re.sub(r"_seed-\d+\.pth$", "", fname)


def submit(cmd, seed, name):
    if is_done(f"{records_dir}/seed_{seed}/{name}"):
        return
    os.makedirs(f"{records_dir}/seed_{seed}", exist_ok=True)
    os.system(cmd)


# ---- Temporal Transformer ----
def submit_transformer():
    group = "transformer_1d_paper"
    name = best_hp_name(f"{group}_{data_percentage}")
    if name is None:
        return
    lr     = parse_param(name, "lr",   cast=float)
    bs     = parse_param(name, "bs",   cast=int)
    dm     = parse_param(name, "dm",   cast=int)
    nl     = parse_param(name, "nl",   cast=int)
    epochs = parse_param(name, "e",    cast=int)
    loss   = parse_param(name, "loss", cast=str)
    nhead, dropout, warmup, all_px = 4, 0.1, 10, True

    for seed in extra_seeds:
        cmd = (f"qsub -v args='"
               f" --seed {seed}"
               f" --n_epochs {epochs}"
               f" --selected_months {months_args}"
               f" --loss {loss}"
               f" --wandb_name {name}"
               f" --wandb_project {wandb_project}"
               f" --data_percentage {data_percentage}"
               f" --batch_size {bs}"
               f" --dropout {dropout}"
               f" --d_model {dm}"
               f" --num_layers {nl}"
               f" --nhead {nhead}"
               f" --all_pixels {all_px}"
               f" --warmup_epochs {warmup}"
               f" --optimizer adamw"
               f" --group_name {group}"
               f" --logging True"
               f" --learning_rate {lr}'"
               f" -o {records_dir}/seed_{seed}/{name}"
               f" run_jobs/sge_scripts/train_transformer_1d_paper.sh")
        submit(cmd, seed, name)


# ---- Presto ----
def submit_presto():
    group = "presto"
    name = best_hp_name(f"{group}_{data_percentage}")
    if name is None:
        return
    lr           = parse_param(name, "lr",            cast=float)
    bs           = parse_param(name, "bs",            cast=int)
    dropout      = parse_param(name, "dropout",       cast=float)
    plocdrop     = parse_param(name, "plocdrop",      cast=float)
    epochs       = parse_param(name, "e",             cast=int)
    loss         = parse_param(name, "loss",          cast=str)
    feed_timeloc = parse_param(name, "feed_timeloc",  default=True, cast=str2bool)
    warmup = 10

    for seed in extra_seeds:
        cmd = (f"qsub -v args='"
               f" --seed {seed}"
               f" --n_epochs {epochs}"
               f" --selected_months {months_args}"
               f" --loss {loss}"
               f" --wandb_name {name}"
               f" --wandb_project {wandb_project}"
               f" --data_percentage {data_percentage}"
               f" --batch_size {bs}"
               f" --dropout {dropout}"
               f" --p_loc_drop {plocdrop}"
               f" --feed_timeloc {feed_timeloc} --use_ndvi True"
               f" --warmup_epochs {warmup}"
               f" --optimizer adamw"
               f" --group_name {group}"
               f" --logging True"
               f" --learning_rate {lr}'"
               f" -o {records_dir}/seed_{seed}/{name}"
               f" run_jobs/sge_scripts/train_presto.sh")
        submit(cmd, seed, name)


# ---- Prithvi (100M, crop32) ----
def submit_prithvi():
    group = "prithvi_final_100m_crop32"
    name = best_hp_name(f"{group}_{data_percentage}")
    if name is None:
        return
    lr           = parse_param(name, "lr",                 cast=float)
    bs           = parse_param(name, "batch_size",         cast=int)
    grad_accum   = parse_param(name, "gradaccum",          default=1,    cast=int)
    loss         = parse_param(name, "loss",               default="mae", cast=str)
    n_layers     = parse_param(name, "n_layers",           default=4,    cast=int)
    crop_size    = parse_param(name, "crop",               default=32,   cast=int)
    epoch_length = parse_param(name, "epochlen",           default=10209, cast=int)
    bb_lr_scale  = parse_param(name, "backbone_lr_scale",  default=1.0,  cast=float)
    feed_timeloc = parse_param(name, "feed_timeloc",       default=True, cast=str2bool)
    concat_input = parse_param(name, "concat_input",       default=True, cast=str2bool)
    epochs = 100
    model_size = "100m"
    load_checkpoint = True

    for seed in extra_seeds:
        cmd = (f"qsub -v args='"
               f" --seed {seed}"
               f" --model_size {model_size}"
               f" --backbone_lr_scale {bb_lr_scale}"
               f" --crop_size {crop_size}"
               f" --epoch_length {epoch_length}"
               f" --grad_accum_steps {grad_accum}"
               f" --optimizer adamw"
               f" --n_epochs {epochs}"
               f" --selected_months {months_args}"
               f" --n_layers {n_layers}"
               f" --loss {loss}"
               f" --wandb_name {name}"
               f" --wandb_project {wandb_project}"
               f" --feed_timeloc {feed_timeloc}"
               f" --concat_input {concat_input}"
               f" --data_percentage {data_percentage}"
               f" --batch_size {bs}"
               f" --group_name {group}"
               f" --load_checkpoint {load_checkpoint}"
               f" --logging True"
               f" --learning_rate {lr}'"
               f" -o {records_dir}/seed_{seed}/{name}"
               f" run_jobs/sge_scripts/train_prithvi.sh")
        submit(cmd, seed, name)


if __name__ == "__main__":
    submit_transformer()
    submit_presto()
    submit_prithvi()
