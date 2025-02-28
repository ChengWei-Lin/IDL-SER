import os
import json
import yaml
import wandb
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_wandb(args):
    """
    Initializes Weights & Biases (wandb) and returns the configuration.
    """
    run = wandb.init(
        project="SER",
        config={
            "learning_rate": args.learning_rate,
            "scheduler": args.scheduler,
            "gamma": args.gamma,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_filters": args.num_filters,
            "output_size": args.output_size,
            "kernel_sizes": [5, 7],
            "pool_kernel": 30,
            "pool_stride": 3,
            "dropout": args.dropout,
            "output_dir": args.output_dir,
            "model_name": args.model_name,
            "run_id": None,
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "hidden_dim": args.hidden_dim,
        },
        )
    # Update the run_id in the config
    wandb.config.update({"run_id": wandb.run.id}, allow_val_change=True)
    print("Run ID from config:", wandb.config.run_id)
    wandb.config.update({"model_name": args.model_name}, allow_val_change=True)
    print("Model name from config:", wandb.config.model_name)
    return wandb.config

def load_config(config_file):
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def output_json(predictions, output_dir, run_id):
    """
    Save predictions to a JSON file.

    Args:
        predictions (dict): Dictionary with keys as sample indices and values as prediction dicts.
        output_dir (str): Base output directory.
        run_id (str): Run ID.
    """
    run_dir = os.path.join(output_dir, run_id)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    output_file = os.path.join(run_dir, "ser_test_2.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f)