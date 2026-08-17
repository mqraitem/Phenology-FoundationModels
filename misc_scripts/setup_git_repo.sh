#!/usr/bin/env bash
# Initialize a fresh git repo and stage only the code needed to reproduce results
# (training scripts, library, configs, run jobs, notebook). Excludes datasets,
# checkpoints, results, paper sources, and editor/tooling cruft via .gitignore.
#
# Usage:
#   bash misc_scripts/setup_git_repo.sh
#   git commit -m "Initial commit"

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
    git init -b main
fi

# --- Top-level meta + configs + entry points ---
git add \
    .gitignore \
    README.md \
    arg_configs.py \
    path_config.py \
    dirs.txt \
    eval_to_dataframe.py \
    select_best_params.py \
    train_presto.py \
    train_prithvi.py \
    train_transformer_1d_paper.py \
    results_overview_notebook.ipynb

# --- Library code ---
git add \
    lib/__init__.py \
    lib/utils.py \
    lib/utils_plotting.py
git add lib/dataloaders/
git add lib/models/__init__.py \
        lib/models/presto_model.py \
        lib/models/prithvi_mae.py \
        lib/models/prithvi_phenology.py \
        lib/models/transformer_1d_paper.py \
        lib/models/prithvi_configs/

# Vendored Presto (modified copy of nasaharvest/presto; see lib/models/presto/VENDORED.md)
git add lib/models/presto/

# --- Misc scripts (data prep, benchmarking, ensembling) and SLURM jobs ---
git add misc_scripts/
git add run_jobs/

echo
echo "Staged files:"
git status --short
echo
echo "Next: git commit -m 'Initial commit'"
