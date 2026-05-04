"""Submit jobs for Prithvi 100M crop size ablation (4 months)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_jobs_4.common import is_done, setup_records_dir, get_wandb_project

selected_months = [3, 6, 9, 12]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)

# ===== Fixed defaults =====
model_size = "100m"
loss = "mae"
n_layers = 4
load_checkpoint = True
grad_accum_steps = 1
data_percentage = 1.0
feed_timeloc = True
concat_input = True
epochs = 100

# ===== Sweep grid =====
learning_rates = [1e-05, 0.0001, 0.001]
backbone_lr_scale = 1.0
seeds = [42, 123, 456]

# Crop sizes to ablate (all must be multiples of 16).
# Batch sizes are per-crop: larger crops need smaller batches to fit in 44 GiB.
crop_to_batch_sizes = {
    16:  [36, 72],
    48:  [36, 72],
    96:  [36, 72],
    144: [18, 36],
    224: [9, 18],
}
crop_sizes = list(crop_to_batch_sizes.keys())

# Epoch length scales with crop area to maintain comparable pixel throughput
# Base: crop32 → 10209 crops/epoch (96 tiles × 330²/32² ≈ 10209)
base_crop = 32
base_epoch_length = 10209

wandb_project = get_wandb_project(data_percentage, months_str)

for crop_size in crop_sizes:
    # Scale epoch length so total pixels per epoch stays ~constant
    epoch_length = int(base_epoch_length * (base_crop ** 2) / (crop_size ** 2))
    epoch_length = max(epoch_length, 100)  # floor

    group_name = f"prithvi_final_100m_crop{crop_size}"
    batch_sizes = crop_to_batch_sizes[crop_size]

    for seed in seeds:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                name = (f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
                        f"_gradaccum-{grad_accum_steps}_loss-{loss}_n_layers-{n_layers}"
                        f"_crop-{crop_size}_epochlen-{epoch_length}"
                        f"_backbone_lr_scale-{backbone_lr_scale}"
                        f"_feed_timeloc-{feed_timeloc}"
                        f"_concat_input-{concat_input}")

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
