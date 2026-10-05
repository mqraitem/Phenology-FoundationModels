#!/bin/bash -l
# Generic GPU job: runs the shell command in $CMD from the repo root.
# SGE project: change -P for your cluster.
#$ -P ivc-ml
#$ -cwd
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
#$ -l h_rt=24:00:00
#$ -l gpu_c=8.6

set -e
conda activate geo
export MPLBACKEND=Agg
nvidia-smi --query-gpu=name --format=csv,noheader || true
eval "$CMD"
