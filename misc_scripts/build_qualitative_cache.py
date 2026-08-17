"""Build the qualitative-tile cache consumed by Section 4.1 of
results_overview_notebook.ipynb.

Reads tile selection and cache location from config.json. Runs
misc_scripts/visualize_tile_predictions.py in --cache_only mode.

Usage (on a GPU node):
    python misc_scripts/build_qualitative_cache.py \
        [--ensemble_file data/ensembles/m3-6-9-12/ensemble_all_e100.json] \
        [--overwrite]
"""
import argparse
import os
import subprocess
import sys

# Make repo root importable regardless of CWD
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import path_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--selected_months", type=int, nargs="+",
                   default=[3, 6, 9, 12])
    p.add_argument("--models", type=str, nargs="+",
                   default=["transformer_1d_paper_1.0",
                            "presto_full_lr_e100_1.0",
                            "prithvi_final_100m_crop32_1.0"])
    p.add_argument("--ensemble_file", type=str, default=None,
                   help="Ensemble JSON from ensemble_from_csvs.py. "
                        "Defaults to data/ensembles/m<months_slug>/ensemble_all_e100.json "
                        "if that file exists; pass '' to skip ensemble.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing tile caches.")
    args = p.parse_args()

    tile_pairs = path_config.get_qual_tile_ids()
    if not tile_pairs:
        sys.exit("evaluation.qualitative_tiles is empty in config.json.")

    months_slug = "-".join(str(m) for m in args.selected_months)
    cache_dir = os.path.join(path_config.get_qual_cache_dir(), f"m{months_slug}")

    # Resolve ensemble file default
    if args.ensemble_file is None:
        default_ens = os.path.join(_REPO_ROOT, "data", "ensembles",
                                   f"m{months_slug}", "ensemble_all_e100.json")
        if os.path.exists(default_ens):
            args.ensemble_file = default_ens
            print(f"Using default ensemble: {default_ens}")
        else:
            print(f"No ensemble file at {default_ens}; continuing without ensemble.")

    tile_args = [f"{sid}={htile}" for sid, htile in tile_pairs]

    cmd = [
        sys.executable,
        os.path.join("misc_scripts", "visualize_tile_predictions.py"),
        "--models", *args.models,
        "--selected_months", *map(str, args.selected_months),
        "--tile_ids", *tile_args,
        "--cache_dir", cache_dir,
        "--cache_only",
    ]
    if args.ensemble_file:
        cmd += ["--ensemble_file", args.ensemble_file]
    if args.overwrite:
        cmd.append("--overwrite")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    print(f"\nCache written to {cache_dir}")


if __name__ == "__main__":
    main()
