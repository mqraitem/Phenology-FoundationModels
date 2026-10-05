"""Submit the independent Presto full-LR, 100-epoch robustness sweep."""

import argparse
import os
import shlex
import subprocess

from run_jobs_4.common import get_wandb_project
import path_config


MONTH_SETTINGS = (
    (3, 6, 9, 12),
    (3, 4, 5, 6, 7, 8, 9, 10),
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
)
MONTH_SETTING_LABELS = {len(months): months for months in MONTH_SETTINGS}
LEARNING_RATES = (1e-5, 1e-4, 1e-3)
DROPOUTS = (0.05, 0.1)
LOCATION_DROPOUTS = (0.1, 0.2)
SEEDS = (42, 123, 456)

EPOCHS = 100
BATCH_SIZE = 1024
WARMUP_EPOCHS = 10
GROUP_NAME = "presto_full_lr_e100"
SGE_SCRIPT = "run_jobs/sge_scripts/train_presto.sh"


def checkpoint_path(months_str, seed, name):
    return os.path.join(
        path_config.get_checkpoint_root(),
        f"m{months_str}",
        f"{GROUP_NAME}_1.0",
        f"seed_{seed}",
        f"{name}_seed-{seed}.pth",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs with qsub; otherwise print the commands only.",
    )
    parser.add_argument(
        "--month-settings",
        type=int,
        nargs="+",
        choices=sorted(MONTH_SETTING_LABELS),
        default=sorted(MONTH_SETTING_LABELS),
        metavar="N_MONTHS",
        help="Temporal settings to submit (choices: 4, 8, 12).",
    )
    args = parser.parse_args()

    submitted = 0
    skipped = 0

    selected_settings = [MONTH_SETTING_LABELS[n] for n in args.month_settings]
    for months in selected_settings:
        months_str = "-".join(map(str, months))
        months_args = " ".join(map(str, months))
        records_dir = os.path.join("records", GROUP_NAME, f"m{months_str}")
        wandb_project = get_wandb_project(1.0, months_str)

        for seed in SEEDS:
            seed_records_dir = os.path.join(records_dir, f"seed_{seed}")
            os.makedirs(seed_records_dir, exist_ok=True)

            for learning_rate in LEARNING_RATES:
                for location_dropout in LOCATION_DROPOUTS:
                    for dropout in DROPOUTS:
                        name = (
                            f"{GROUP_NAME}_lr-{learning_rate}_bs-{BATCH_SIZE}"
                            f"_dropout-{dropout}_plocdrop-{location_dropout}"
                            f"_e-{EPOCHS}_loss-mae_feed_timeloc-True"
                        )
                        record_path = os.path.join(seed_records_dir, name)
                        if os.path.exists(checkpoint_path(months_str, seed, name)):
                            skipped += 1
                            continue

                        training_args = (
                            f" --seed {seed}"
                            f" --n_epochs {EPOCHS}"
                            f" --selected_months {months_args}"
                            " --loss mae"
                            f" --wandb_name {name}"
                            f" --wandb_project {wandb_project}"
                            " --data_percentage 1.0"
                            f" --batch_size {BATCH_SIZE}"
                            f" --dropout {dropout}"
                            f" --p_loc_drop {location_dropout}"
                            " --feed_timeloc True --use_ndvi True"
                            f" --warmup_epochs {WARMUP_EPOCHS}"
                            " --optimizer adamw"
                            f" --group_name {GROUP_NAME}"
                            " --logging True"
                            f" --learning_rate {learning_rate}"
                        )
                        command = [
                            "qsub",
                            "-v",
                            f"args={training_args}",
                            "-o",
                            record_path,
                            SGE_SCRIPT,
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
