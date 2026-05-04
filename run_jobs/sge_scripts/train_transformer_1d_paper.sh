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

# Run your commands
python train_transformer_1d_paper.py $args
