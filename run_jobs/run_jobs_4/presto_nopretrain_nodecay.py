"""Submit from-scratch Presto runs without layer-wise LR decay (4 months).

Same grids as the reported from-scratch ablations (presto_nopretrain,
presto_notimeloc_nopretrain), but with --layer_decay 1.0 so every parameter group
trains at the full learning rate, as from-scratch Prithvi does.
Set DRY_RUN=1 to print the qsub commands without submitting.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_jobs_4.common import is_done, setup_records_dir, get_wandb_project

selected_months = [3, 6, 9, 12]
months_str = "-".join(str(m) for m in selected_months)
months_args = " ".join(str(m) for m in selected_months)
records_dir = setup_records_dir(selected_months)
dry_run = os.environ.get("DRY_RUN") == "1"

loss = "mae"
data_percentage = 1.0
batch_size = 1024
epochs = 50
warmup_epochs = 10
learning_rates = [1e-05, 1e-04, 1e-03]
dropouts = [0.05, 0.1]
seeds = [42, 123, 456]

# (group name, feed time/loc, location-dropout grid)
variants = [
    ("presto_nopretrain_nodecay", True, [0.1, 0.2]),
    ("presto_notimeloc_nopretrain_nodecay", False, [None]),
]

wandb_project = get_wandb_project(data_percentage, months_str)

for group_name, feed_timeloc, p_loc_drops in variants:
    for seed in seeds:
        for learning_rate in learning_rates:
            for dropout in dropouts:
                for p_loc_drop in p_loc_drops:
                    plocdrop_tag = f"_plocdrop-{p_loc_drop}" if p_loc_drop is not None else ""
                    name = (f"{group_name}_lr-{learning_rate}_bs-{batch_size}"
                            f"_dropout-{dropout}{plocdrop_tag}"
                            f"_e-{epochs}_loss-{loss}_feed_timeloc-{feed_timeloc}")

                    if is_done(f"{records_dir}/seed_{seed}/{name}"):
                        continue

                    plocdrop_arg = f" --p_loc_drop {p_loc_drop}" if p_loc_drop is not None else ""
                    command = (f"qsub -N presto_nodecay -v args='"
                               f" --seed {seed}"
                               f" --n_epochs {epochs}"
                               f" --selected_months {months_args}"
                               f" --loss {loss}"
                               f" --wandb_name {name}"
                               f" --wandb_project {wandb_project}"
                               f" --data_percentage {data_percentage}"
                               f" --batch_size {batch_size}"
                               f" --dropout {dropout}"
                               f"{plocdrop_arg}"
                               f" --feed_timeloc {feed_timeloc} --use_ndvi True"
                               f" --pretrained False --layer_decay 1.0"
                               f" --warmup_epochs {warmup_epochs}"
                               f" --optimizer adamw"
                               f" --group_name {group_name}"
                               f" --logging True"
                               f" --learning_rate {learning_rate}'"
                               f" -o {records_dir}/seed_{seed}/{name}"
                               f" run_jobs/sge_scripts/train_presto.sh")
                    if dry_run:
                        print(command)
                        continue
                    os.makedirs(f"{records_dir}/seed_{seed}", exist_ok=True)
                    os.system(command)
