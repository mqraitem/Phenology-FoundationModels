#!/bin/bash -l
# Activate your environment

#$ -P ivc-ml
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
#$ -l h_rt=48:00:00
#$ -l gpu_c=8.6

conda activate geo
export WANDB_CACHE_DIR=/projectnb/ivc-ml/mqraitem/.cache/wandb

# Increase wandb tolerance for slow nodes / long epochs
export WANDB__SERVICE_WAIT=300
export WANDB_HTTP_TIMEOUT=120
export WANDB_INIT_TIMEOUT=300

# Run your commands
python train_presto.py $args
