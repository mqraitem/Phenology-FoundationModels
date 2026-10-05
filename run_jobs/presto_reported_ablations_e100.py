"""Submit the reported Presto ablations with the 100-epoch protocol.

The regularization ablation is evaluated for 4, 8, and 12 input months. The
pretraining/time-location factorial is reported only for the 4-month setting.
Results use new ``*_e100`` groups so the original 50-epoch runs are preserved.
"""

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import path_config
from run_jobs_4.common import get_wandb_project


MONTH_SETTINGS = {
    4: (3, 6, 9, 12),
    8: (3, 4, 5, 6, 7, 8, 9, 10),
    12: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
}
LEARNING_RATES = (1e-5, 1e-4, 1e-3)
SEEDS = (42, 123, 456)
DROPOUTS = (0.05, 0.1)
LOCATION_DROPOUTS = (0.1, 0.2)

EPOCHS = 100
BATCH_SIZE = 1024
WARMUP_EPOCHS = 10
SGE_SCRIPT = "run_jobs/sge_scripts/train_presto.sh"


@dataclass(frozen=True)
class Variant:
    group: str
    month_settings: tuple[int, ...]
    pretrained: bool
    feed_timeloc: bool
    dropouts: tuple[float, ...]
    location_dropouts: tuple[float, ...]


VARIANTS = (
    Variant(
        "presto_nolocdrop_nodropout_e100",
        (4, 8, 12),
        True,
        True,
        (0.0,),
        (0.0,),
    ),
    Variant(
        "presto_notimeloc_e100",
        (4,),
        True,
        False,
        DROPOUTS,
        (0.0,),
    ),
    Variant(
        "presto_nopretrain_e100",
        (4,),
        False,
        True,
        DROPOUTS,
        LOCATION_DROPOUTS,
    ),
    Variant(
        "presto_notimeloc_nopretrain_e100",
        (4,),
        False,
        False,
        DROPOUTS,
        (0.0,),
    ),
)


def checkpoint_path(group, months_str, seed, name):
    return os.path.join(
        path_config.get_checkpoint_root(),
        f"m{months_str}",
        f"{group}_1.0",
        f"seed_{seed}",
        f"{name}_seed-{seed}.pth",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs with qsub; otherwise print commands only.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant.group for variant in VARIANTS],
        help="Optional subset of variant groups to run.",
    )
    args = parser.parse_args()

    requested = set(args.variants or [variant.group for variant in VARIANTS])
    submitted = skipped = 0
    for variant in VARIANTS:
        if variant.group not in requested:
            continue
        for n_months in variant.month_settings:
            months = MONTH_SETTINGS[n_months]
            months_str = "-".join(map(str, months))
            months_args = " ".join(map(str, months))
            wandb_project = get_wandb_project(1.0, months_str)

            for seed in SEEDS:
                record_dir = os.path.join(
                    "records", variant.group, f"m{months_str}", f"seed_{seed}"
                )
                os.makedirs(record_dir, exist_ok=True)
                for learning_rate in LEARNING_RATES:
                    for dropout in variant.dropouts:
                        for location_dropout in variant.location_dropouts:
                            name = (
                                f"{variant.group}_lr-{learning_rate}_bs-{BATCH_SIZE}"
                                f"_dropout-{dropout}_plocdrop-{location_dropout}"
                                f"_e-{EPOCHS}_loss-mae_feed_timeloc-{variant.feed_timeloc}"
                            )
                            if os.path.exists(
                                checkpoint_path(variant.group, months_str, seed, name)
                            ):
                                skipped += 1
                                continue

                            training_args = (
                                f" --seed {seed} --n_epochs {EPOCHS}"
                                f" --selected_months {months_args} --loss mae"
                                f" --wandb_name {name} --wandb_project {wandb_project}"
                                " --data_percentage 1.0"
                                f" --batch_size {BATCH_SIZE} --dropout {dropout}"
                                f" --p_loc_drop {location_dropout}"
                                f" --feed_timeloc {variant.feed_timeloc} --use_ndvi True"
                                f" --pretrained {variant.pretrained}"
                                f" --warmup_epochs {WARMUP_EPOCHS} --optimizer adamw"
                                f" --group_name {variant.group} --logging True"
                                f" --learning_rate {learning_rate}"
                            )
                            command = [
                                "qsub",
                                "-v",
                                f"args={training_args}",
                                "-o",
                                os.path.join(record_dir, name),
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
