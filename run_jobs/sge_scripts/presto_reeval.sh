#!/bin/bash -l
# Re-select and re-evaluate one Presto group after the eval-time attention-dropout fix.
# Env (passed via qsub -v): MONTHS="3 6 9 12", GROUP=presto_1.0, EXTRA=1 (include extra seeds),
# SPLITS="test val train".

#$ -P ivc-ml
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
#$ -l h_rt=24:00:00
#$ -l gpu_c=8.6

set -e
conda activate geo
cd /projectnb/ivc-ml/mqraitem/geospatial/phenology

extra_flag=""
if [ "${EXTRA:-0}" = "1" ]; then extra_flag="--include-extra-seeds"; fi

python select_best_params.py --selected_months $MONTHS --group "$GROUP" --force $extra_flag

for split in $SPLITS; do
    python eval_to_dataframe.py --split "$split" --selected_months $MONTHS \
        --model-groups "$GROUP" --force
done
