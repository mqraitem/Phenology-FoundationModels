#!/bin/bash -l
# Activate your environment

#$ -P ivc-ml
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
#$ -l h_rt=48:00:00
#$ -l gpu_c=8.6

conda activate geo
wandb_tmp="${TMPDIR:-/tmp}/wandb-${JOB_ID:-$$}"
mkdir -p "$wandb_tmp/cache" "$wandb_tmp/config"
export WANDB_CACHE_DIR="$wandb_tmp/cache"
export WANDB_CONFIG_DIR="$wandb_tmp/config"
export WANDB_DIR="$wandb_tmp"
export WANDB_CONSOLE=off

# Increase wandb tolerance for slow nodes / long epochs
export WANDB__SERVICE_WAIT=300
export WANDB_HTTP_TIMEOUT=120
export WANDB_INIT_TIMEOUT=300

# Run your commands
python train_presto.py $args
