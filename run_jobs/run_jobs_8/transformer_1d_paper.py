"""Submit jobs for paper-matched 1D Transformer (8 months)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_jobs_8.common import is_done, setup_records_dir, get_wandb_project

selected_months = [3, 4, 5, 6, 7, 8, 9, 10]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)

# ===== Fixed defaults =====
loss = "mae"
data_percentage = 1.0
dropout = 0.1
nhead = 4
batch_size = 1024
epochs = 100
warmup_epochs = 10
d_model = 64
num_layers = 4

# ===== Sweep grid =====
learning_rates = [1e-05, 0.0001, 0.001]
all_pixels_options = [True]
seeds = [42, 123, 456]

group_name = "transformer_1d_paper"
wandb_project = get_wandb_project(data_percentage, months_str)
for seed in seeds:
    for learning_rate in learning_rates:
        for all_pixels in all_pixels_options:
            px_tag = "_allpx" if all_pixels else ""
            name = (f"{group_name}_lr-{learning_rate}_bs-{batch_size}"
                    f"_dm-{d_model}_nl-{num_layers}_e-{epochs}{px_tag}_loss-{loss}")

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
                       f" --d_model {d_model}"
                       f" --num_layers {num_layers}"
                       f" --nhead {nhead}"
                       f" --all_pixels {all_pixels}"
                       f" --warmup_epochs {warmup_epochs}"
                       f" --optimizer adamw"
                       f" --group_name {group_name}"
                       f" --logging True"
                       f" --learning_rate {learning_rate}'"
                       f" -o {records_dir}/seed_{seed}/{name}"
                       f" run_jobs/sge_scripts/train_transformer_1d_paper.sh")
            # print(command)
            os.makedirs(f"{records_dir}/seed_{seed}", exist_ok=True)
            os.system(command)
