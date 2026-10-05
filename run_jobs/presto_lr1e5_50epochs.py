"""Submit missing Presto runs for the reported 50-epoch experiments."""

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import path_config
from run_jobs.run_jobs_4.common import get_wandb_project


MONTH_SETTINGS = {
    4: (3, 6, 9, 12),
    8: (3, 4, 5, 6, 7, 8, 9, 10),
    12: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
}
SEEDS = (42, 123, 456)
DEFAULT_LEARNING_RATES = (1e-5,)
EPOCHS = 50
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
    Variant("presto", (4, 8, 12), True, True, (0.05, 0.1), (0.1, 0.2)),
    Variant("presto_nolocdrop_nodropout", (4, 8, 12), True, True, (0.0,), (0.0,)),
    Variant("presto_notimeloc", (4,), True, False, (0.05, 0.1), (0.0,)),
    Variant("presto_nopretrain", (4,), False, True, (0.05, 0.1), (0.1, 0.2)),
    Variant("presto_notimeloc_nopretrain", (4,), False, False, (0.05, 0.1), (0.0,)),
)


def checkpoint_path(group: str, months_slug: str, seed: int, run_name: str) -> str:
    return os.path.join(
        path_config.get_checkpoint_root(),
        f"m{months_slug}",
        f"{group}_1.0",
        f"seed_{seed}",
        f"{run_name}_seed-{seed}.pth",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant.group for variant in VARIANTS],
        help="Optional subset of model groups to submit.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=DEFAULT_LEARNING_RATES,
        help="Learning rates to check and submit (default: 1e-5).",
    )
    args = parser.parse_args()

    requested = set(args.variants or [variant.group for variant in VARIANTS])
    submitted = skipped = 0
    for variant in VARIANTS:
        if variant.group not in requested:
            continue
        for setting in variant.month_settings:
            months = MONTH_SETTINGS[setting]
            months_slug = "-".join(map(str, months))
            months_args = " ".join(map(str, months))
            project = get_wandb_project(1.0, months_slug)
            for seed in SEEDS:
                record_dir = os.path.join("records", f"m{months_slug}", f"seed_{seed}")
                os.makedirs(record_dir, exist_ok=True)
                for learning_rate in args.learning_rates:
                    for dropout in variant.dropouts:
                        for location_dropout in variant.location_dropouts:
                            location_part = (
                                f"_plocdrop-{location_dropout}"
                                if variant.feed_timeloc
                                else ""
                            )
                            run_name = (
                                f"{variant.group}_lr-{learning_rate}_bs-{BATCH_SIZE}"
                                f"_dropout-{dropout}{location_part}_e-{EPOCHS}"
                                f"_loss-mae_feed_timeloc-{variant.feed_timeloc}"
                            )
                            if os.path.exists(
                                checkpoint_path(variant.group, months_slug, seed, run_name)
                            ):
                                skipped += 1
                                continue

                            training_args = (
                                f" --seed {seed} --n_epochs {EPOCHS}"
                                f" --selected_months {months_args} --loss mae"
                                f" --wandb_name {run_name} --wandb_project {project}"
                                f" --data_percentage 1.0 --batch_size {BATCH_SIZE}"
                                f" --dropout {dropout} --p_loc_drop {location_dropout}"
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
                                os.path.join(record_dir, run_name),
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
