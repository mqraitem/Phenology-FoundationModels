"""Submit Presto runs without time/location and without pretraining, keeping NDVI on (4 months)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_jobs_4.common import is_done, setup_records_dir, get_wandb_project

selected_months = [3, 6, 9, 12]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)

loss = "mae"
data_percentage = 1.0
batch_size = 1024
epochs = 50
warmup_epochs = 10
learning_rates = [1e-04, 1e-03]
dropouts = [0.05, 0.1]
seeds = [42, 123, 456]

group_name = "presto_notimeloc_nopretrain"
wandb_project = get_wandb_project(data_percentage, months_str)

for seed in seeds:
    for learning_rate in learning_rates:
        for dropout in dropouts:
            name = (f"{group_name}_lr-{learning_rate}_bs-{batch_size}"
                    f"_dropout-{dropout}"
                    f"_e-{epochs}_loss-{loss}_feed_timeloc-False")

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
                       f" --feed_timeloc False --use_ndvi True"
                       f" --pretrained False"
                       f" --warmup_epochs {warmup_epochs}"
                       f" --optimizer adamw"
                       f" --group_name {group_name}"
                       f" --logging True"
                       f" --learning_rate {learning_rate}'"
                       f" -o {records_dir}/seed_{seed}/{name}"
                       f" run_jobs/sge_scripts/train_presto.sh")
            os.makedirs(f"{records_dir}/seed_{seed}", exist_ok=True)
            os.system(command)
