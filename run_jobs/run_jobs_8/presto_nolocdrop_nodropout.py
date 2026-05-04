"""Submit Presto runs with no location drop and no dropout (8 months)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_jobs_8.common import is_done, setup_records_dir, get_wandb_project

selected_months = [3, 4, 5, 6, 7, 8, 9, 10]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)

loss = "mae"
data_percentage = 1.0
batch_size = 1024
epochs = 50
warmup_epochs = 10
learning_rates = [1e-04, 1e-03]
p_loc_drop = 0.0
dropout = 0.0
seeds = [42, 123, 456]

group_name = "presto_nolocdrop_nodropout"
wandb_project = get_wandb_project(data_percentage, months_str)

for seed in seeds:
    for learning_rate in learning_rates:
        name = (f"{group_name}_lr-{learning_rate}_bs-{batch_size}"
                f"_dropout-{dropout}_plocdrop-{p_loc_drop}"
                f"_e-{epochs}_loss-{loss}_feed_timeloc-True")

        if is_done(f"{records_dir}/seed_{seed}/{name}"):
            continue

        command = (f"qsub -v args='"
                   f" --seed {seed}"
                   f" --n_epochs {epochs}"
                   f" --selected_months {months_args}"
                   f" --loss {loss}"
                   f" --wandb_name {name}"
                   f" --wandb_project {wandb_project}"
                   f" --data_percentage {data_percentage}"
                   f" --batch_size {batch_size}"
                   f" --dropout {dropout}"
                   f" --p_loc_drop {p_loc_drop}"
                   f" --feed_timeloc True --use_ndvi True"
                   f" --warmup_epochs {warmup_epochs}"
                   f" --optimizer adamw"
                   f" --group_name {group_name}"
                   f" --logging True"
                   f" --learning_rate {learning_rate}'"
                   f" -o {records_dir}/seed_{seed}/{name}"
                   f" run_jobs/sge_scripts/train_presto.sh")
        os.makedirs(f"{records_dir}/seed_{seed}", exist_ok=True)
        os.system(command)
