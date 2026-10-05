#!/bin/bash -l
# Generic CPU job: runs the shell command in $CMD from the repo root.
#$ -P ivc-ml
#$ -pe omp 4
#$ -j y
#$ -l h_rt=12:00:00

set -e
conda activate geo
cd /projectnb/ivc-ml/mqraitem/geospatial/phenology
export MPLBACKEND=Agg
eval "$CMD"
