"""Submit six extra seeds for the selected 100-epoch Presto configurations."""

import argparse
import os
import re
import shlex
import subprocess

import pandas as pd

from run_jobs_4.common import get_wandb_project
import path_config


MONTH_SETTINGS = (
    (3, 6, 9, 12),
    (3, 4, 5, 6, 7, 8, 9, 10),
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
)
MONTH_SETTING_LABELS = {len(months): months for months in MONTH_SETTINGS}
SEEDS = (789, 101, 202, 303, 404, 505)
GROUP_NAME = "presto_full_lr_e100"
GROUP_WITH_PERCENTAGE = f"{GROUP_NAME}_1.0"
SGE_SCRIPT = "run_jobs/sge_scripts/train_presto.sh"


def selected_name(months_str):
    path = os.path.join("results", f"m{months_str}", GROUP_WITH_PERCENTAGE, "best_params.csv")
    frame = pd.read_csv(path)
    return re.sub(r"_seed-\d+\.pth$", "", frame["Best Param"].iloc[0])


def checkpoint_path(months_str, seed, name):
    return os.path.join(
        path_config.get_checkpoint_root(), f"m{months_str}", GROUP_WITH_PERCENTAGE,
        f"seed_{seed}", f"{name}_seed-{seed}.pth",
    )


def param(name, key):
    match = re.search(rf"(?:^|_){re.escape(key)}-([^_]+)", name)
    if match is None:
        raise ValueError(f"Missing {key} in selected parameter name: {name}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--month-settings", type=int, nargs="+", choices=sorted(MONTH_SETTING_LABELS),
        default=sorted(MONTH_SETTING_LABELS), metavar="N_MONTHS",
    )
    args = parser.parse_args()

    submitted = skipped = 0
    for n_months in args.month_settings:
        months = MONTH_SETTING_LABELS[n_months]
        months_str = "-".join(map(str, months))
        months_args = " ".join(map(str, months))
        name = selected_name(months_str)
        wandb_project = get_wandb_project(1.0, months_str)

        for seed in SEEDS:
            if os.path.exists(checkpoint_path(months_str, seed, name)):
                skipped += 1
                continue
            record_dir = os.path.join(
                "records", GROUP_NAME, f"m{months_str}", f"seed_{seed}"
            )
            os.makedirs(record_dir, exist_ok=True)
            training_args = (
                f" --seed {seed} --n_epochs 100 --selected_months {months_args} --loss mae"
                f" --wandb_name {name} --wandb_project {wandb_project} --data_percentage 1.0"
                f" --batch_size {param(name, 'bs')} --dropout {param(name, 'dropout')}"
                f" --p_loc_drop {param(name, 'plocdrop')} --feed_timeloc True --use_ndvi True"
                f" --warmup_epochs 10 --optimizer adamw --group_name {GROUP_NAME}"
                f" --logging True --learning_rate {param(name, 'lr')}"
            )
            command = [
                "qsub", "-v", f"args={training_args}", "-o",
                os.path.join(record_dir, name), SGE_SCRIPT,
            ]
            if args.submit:
                subprocess.run(command, check=True)
            else:
                print(shlex.join(command))
            submitted += 1

    action = "Submitted" if args.submit else "Would submit"
    print(f"{action} {submitted} jobs; skipped {skipped} completed jobs.")


if __name__ == "__main__":
    main()
