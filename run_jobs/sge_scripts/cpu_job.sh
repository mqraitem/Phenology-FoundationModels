#!/bin/bash -l
# Generic CPU job: runs the shell command in $CMD from the repo root.
# SGE project: change -P for your cluster.
#$ -P ivc-ml
#$ -cwd
#$ -pe omp 4
#$ -j y
#$ -l h_rt=12:00:00

set -e
conda activate geo
export MPLBACKEND=Agg
eval "$CMD"
