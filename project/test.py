import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from data_utils import load_data, collate_fn
from models import CNN, ACNN
from utils import load_config, output_json, set_seed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="wandb/run-20250210_230754-iqi39pfp/files/config.yaml", help="Path to configuration file.")
args = parser.parse_args()
set_seed(42)

def perform_inference(model, test_loader):
    """
    Performs inference on test data and returns predictions.

    Args:
        model (nn.Module): The CNN model.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        tuple: (all_activation_preds, all_valence_preds) as numpy arrays.
    """
    all_activation_preds = []
    all_valence_preds = []
    with torch.no_grad():
        for X_batch, _, _, _ in test_loader:
            a_pred, v_pred = model(X_batch)
            # Convert probabilities to binary labels using threshold 0.5.
            a_pred_labels = (a_pred.squeeze() >= 0.5).long()
            v_pred_labels = (v_pred.squeeze() >= 0.5).long()
            all_activation_preds.append(a_pred_labels.cpu())
            all_valence_preds.append(v_pred_labels.cpu())
    all_activation_preds = torch.cat(all_activation_preds).numpy()
    all_valence_preds = torch.cat(all_valence_preds).numpy()
    return all_activation_preds, all_valence_preds

def main(args):
    # Load test data.
    test_file = "data/ser_test_2.json"
    with open(test_file, "r") as f:
        test_data = json.load(f)
    
    # Load configuration.
    config_file = args.config_file
    config = load_config(config_file)
    output_dir = config["output_dir"]["value"]
    run_id = config["run_id"]["value"]
    model_name = config["model_name"]["value"]
    max_duration = config["max_duration"]["value"]

    # max_duration = 376  # Set maximum sequence length.
    test_features, _, _, _ = load_data(test_data, max_duration=max_duration)
    # Create test dataset with dummy labels.
    test_dataset = list(zip(test_features, [0] * len(test_features), [0] * len(test_features)))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


    # Recreate model with the same hyperparameters used during training.
    input_dim = 26     # e.g., 26 logMel features.
    seq_len = max_duration      # Must match training sequence length. Set 376.
    num_filters = config['num_filters']['value']  # Number of filters per branch.
    output_size = config['output_size']['value']  # Fully connected layer size.
    if config["model_name"]["value"] == "CNN":
        model = CNN(input_dim, seq_len, num_filters, output_size)
    elif config["model_name"]["value"] == "ACNN":
        model = ACNN(input_dim, seq_len, num_filters, output_size)

    # Load the saved model weights (mapping to CPU for inference).
    state_dict_path = os.path.join(f"{output_dir}", f"{run_id}", "best_model.pt")
    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    # Perform inference.
    all_activation_preds, all_valence_preds = perform_inference(model, test_loader)

    # Prepare output dictionary.
    predictions = {}
    for i, (a, v) in enumerate(zip(all_activation_preds, all_valence_preds)):
        predictions[str(i)] = {"valence": int(v), "activation": int(a)}

    # Save predictions to JSON.
    output_json(predictions, output_dir, run_id)
    print("Inference Complete.")
    print("Predictions saved to:", os.path.join(output_dir, run_id, "ser_test_2.json"))

if __name__ == "__main__":
    main(args)