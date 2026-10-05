#!/bin/bash -l

#$ -P ivc-ml
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
#$ -l h_rt=24:00:00
#$ -l gpu_c=8.6

conda activate geo

cd /projectnb/ivc-ml/mqraitem/geospatial/phenology
python $args
