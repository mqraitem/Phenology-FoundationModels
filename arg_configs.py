"""
Core argument parsing configuration for phenology training scripts.

This module provides core arguments shared across all training scripts.
Each training script should add its own model-specific arguments.
"""

import argparse
from lib.utils import str2bool


def get_core_parser():
    """
    Create an argument parser with core arguments shared by all training scripts.

    Returns:
        argparse.ArgumentParser: Parser with core arguments added

    Example:
        >>> parser = get_core_parser()
        >>> # Add model-specific arguments
        >>> parser.add_argument("--freeze", type=str2bool, default=False)
        >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for the model")
    parser.add_argument("--logging", type=str2bool, default=False,
                       help="Whether to log the results to wandb")
    parser.add_argument("--group_name", type=str, default="default",
                       help="Group name for wandb logging")
    parser.add_argument("--wandb_name", type=str, default="default",
                       help="Run name for wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="Project name for wandb logging (default: phenology_crop_{data_percentage})")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--data_percentage", type=float, default=1.0,
                       help="Fraction of data to use (0.0-1.0)")
    parser.add_argument("--n_epochs", type=int, default=120,
                       help="Number of training epochs")
    parser.add_argument("--selected_months", type=int, nargs='+',
                       default=[3, 4, 5, 6, 7, 8, 9, 10],
                       help="Which months to include (e.g., --selected_months 3 6 9 12)")
    parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"],
                       help="Loss function: mse or mae")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"],
                       help="Optimizer")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                       help="Number of linear warmup epochs")
    parser.add_argument("--min_lr", type=float, default=1e-7,
                       help="Minimum LR for cosine schedule")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    return parser


def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
