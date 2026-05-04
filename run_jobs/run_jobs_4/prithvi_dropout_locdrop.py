"""Submit Prithvi runs with dropout + location dropping (100M, crop32, 4 months)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_jobs_4.common import is_done, setup_records_dir, get_wandb_project

selected_months = [3, 6, 9, 12]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)

# ===== Fixed defaults =====
loss = "mae"
n_layers = 4
load_checkpoint = True
grad_accum_steps = 1
data_percentage = 1.0
feed_timeloc = True
concat_input = True
crop_size = 32
epoch_length = 10209
epochs = 100
model_size = "100m"
backbone_lr_scale = 1.0

# ===== Sweep grid =====
learning_rates = [1e-05, 0.0001, 0.001]
batch_size = 36  # best from prithvi_final_100m_crop32 best_params.csv (4 months)
p_loc_drops = [0.1, 0.2]
dropouts = [0.05, 0.1]
seeds = [42, 123, 456]

group_name = "prithvi_dropout_locdrop_100m_crop32"
wandb_project = get_wandb_project(data_percentage, months_str)

for seed in seeds:
    for learning_rate in learning_rates:
        for p_loc_drop in p_loc_drops:
            for dropout in dropouts:
                name = (f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
                        f"_gradaccum-{grad_accum_steps}_loss-{loss}_n_layers-{n_layers}"
                        f"_crop-{crop_size}_epochlen-{epoch_length}"
                        f"_backbone_lr_scale-{backbone_lr_scale}"
                        f"_feed_timeloc-{feed_timeloc}"
                        f"_concat_input-{concat_input}"
                        f"_dropout-{dropout}_plocdrop-{p_loc_drop}")

                if is_done(f"{records_dir}/seed_{seed}/{name}"):
                    continue

                command = (f"qsub -v args='"
                           f" --seed {seed}"
                           f" --model_size {model_size}"
                           f" --backbone_lr_scale {backbone_lr_scale}"
                           f" --crop_size {crop_size}"
                           f" --epoch_length {epoch_length}"
                           f" --grad_accum_steps {grad_accum_steps}"
                           f" --optimizer adamw"
                           f" --n_epochs {epochs}"
                           f" --selected_months {months_args}"
                           f" --n_layers {n_layers}"
                           f" --loss {loss}"
                           f" --wandb_name {name}"
                           f" --wandb_project {wandb_project}"
                           f" --feed_timeloc {feed_timeloc}"
                           f" --concat_input {concat_input}"
                           f" --dropout {dropout}"
                           f" --p_loc_drop {p_loc_drop}"
                           f" --data_percentage {data_percentage}"
                           f" --batch_size {batch_size}"
                           f" --group_name {group_name}"
                           f" --load_checkpoint {load_checkpoint}"
                           f" --logging True"
                           f" --learning_rate {learning_rate}'"
                           f" -o {records_dir}/seed_{seed}/{name}"
                           f" run_jobs/sge_scripts/train_prithvi.sh")
                os.makedirs(f"{records_dir}/seed_{seed}", exist_ok=True)
                os.system(command)
