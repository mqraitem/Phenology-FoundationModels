#!/bin/bash
# Submit Presto re-selection + re-evaluation jobs (eval-time attention-dropout fix).
# DRY_RUN=1 prints the qsub commands instead of submitting.
set -e
cd "$(dirname "$0")/.."

submit() {  # months_label months group extra splits
    local label="$1" months="$2" group="$3" extra="$4" splits="$5"
    mkdir -p "records/m${label}/presto_reeval"
    local cmd=(qsub -v "MONTHS=${months},GROUP=${group},EXTRA=${extra},SPLITS=${splits}"
               -N "reeval_${group}_m${label}"
               -o "records/m${label}/presto_reeval/${group}.log"
               run_jobs/sge_scripts/presto_reeval.sh)
    if [ -n "$DRY_RUN" ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
}

# 4 months: main model (9 seeds, all splits for ensembles), paper ablations, 100-epoch run (item 30)
submit 3-6-9-12 "3 6 9 12" presto_1.0                       1 "test val train"
submit 3-6-9-12 "3 6 9 12" presto_nopretrain_1.0            0 "test"
submit 3-6-9-12 "3 6 9 12" presto_notimeloc_1.0             0 "test"
submit 3-6-9-12 "3 6 9 12" presto_notimeloc_nopretrain_1.0  0 "test"
submit 3-6-9-12 "3 6 9 12" presto_full_lr_e100_1.0          0 "test val"

for spec in "3-4-5-6-7-8-9-10|3 4 5 6 7 8 9 10" "1-2-3-4-5-6-7-8-9-10-11-12|1 2 3 4 5 6 7 8 9 10 11 12"; do
    label="${spec%%|*}"; months="${spec##*|}"
    submit "$label" "$months" presto_1.0              1 "test val train"
    submit "$label" "$months" presto_full_lr_e100_1.0 0 "test val"
done
