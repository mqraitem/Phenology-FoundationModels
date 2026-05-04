#!/bin/bash -l

#$ -P ivc-ml
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
#$ -l h_rt=48:00:00
#$ -l gpu_c=8.9
#$ -l h=!scc-213*

conda activate geo

python train_presto_test.py \
    --seed 42 \
    --n_epochs 50 \
    --selected_months 1 2 3 4 5 6 7 8 9 10 11 12 \
    --loss mae \
    --data_percentage 1.0 \
    --batch_size 1024 \
    --dropout 0.1 \
    --p_loc_drop 0.2 \
    --feed_timeloc True --use_ndvi True \
    --warmup_epochs 10 \
    --optimizer adamw \
    --learning_rate 0.005
