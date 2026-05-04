"""Shared utilities for run_jobs_8 scripts (8-month experiments)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import path_config

def is_done(record_path):
    """Check if a job already ran to completion."""
    if not os.path.exists(record_path):
        return False
    file_content = open(record_path, "r", encoding='latin-1').read()
    if not file_content:
        return False
    return "wandb: Find logs" in file_content


def setup_records_dir(selected_months):
    """Create and return the records directory for the given months."""
    months_str = "-".join(str(m) for m in selected_months)
    records_dir = f"records/m{months_str}"
    os.makedirs(records_dir, exist_ok=True)
    return records_dir


def get_wandb_project(data_percentage, months_str):
    """Get wandb project name from dirs.txt config."""
    base = path_config.get_wandb_project()
    return f"{base}_{data_percentage}_m{months_str}"
