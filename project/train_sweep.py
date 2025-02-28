import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from data_utils import load_data, collate_fn, calculate_duration
from models import CNN, ACNN, TransformerDualClassifier, BiLSTMModel
from utils import init_wandb, set_seed
import os

########################################
# Device Setup
########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

set_seed(42)

########################################
# Training Function
########################################

def train_model(model, train_loader, dev_loader, criterion, optimizer, scheduler, epochs, patience, run_id, output_dir):
    """
    Trains the model and evaluates on the dev set. Implements early stopping.
    
    Args:
        model (nn.Module): The CNN model.
        train_loader (DataLoader): Training data loader.
        dev_loader (DataLoader): Development (validation) data loader.
        criterion (Loss): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (Scheduler): Learning rate scheduler.
        epochs (int): Maximum number of epochs.
        patience (int): Number of epochs to wait for improvement before stopping.
    """
    best_dev_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_v, y_a, lengths in train_loader:
            X_batch = X_batch.to(device)
            y_v = y_v.to(device)
            y_a = y_a.to(device)
            optimizer.zero_grad()
            a_pred, v_pred = model(X_batch)
            loss_a = criterion(a_pred.squeeze(), y_a)
            loss_v = criterion(v_pred.squeeze(), y_v)
            loss = loss_a + loss_v
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Evaluate on dev set
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for X_batch, y_v, y_a, lengths in dev_loader:
                X_batch = X_batch.to(device)
                y_v = y_v.to(device)
                y_a = y_a.to(device)
                a_pred, v_pred = model(X_batch)
                loss_a = criterion(a_pred.squeeze(), y_a)
                loss_v = criterion(v_pred.squeeze(), y_v)
                loss = loss_a + loss_v
                dev_loss += loss.item()
        avg_dev_loss = dev_loss / len(dev_loader)

        # Log metrics to wandb
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "dev_loss": avg_dev_loss})
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}")

        # Early stopping check
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{output_dir}/{run_id}/best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

def main():
    # Initialize wandb and load hyperparameters
    wandb.init()
    config = wandb.config
    print("Config:", config)

    # Load JSON data
    with open("data/train.json") as f:
        train_data = json.load(f)
    with open("data/dev.json") as f:
        dev_data = json.load(f)

    # Create output directory if it does not exist
    if not os.path.exists(f"{config.output_dir}/{config.run_id}"):
        os.makedirs(f"{config.output_dir}/{config.run_id}")

    # # Calculate duration and set max_duration
    # if args.max_length:
    #     print(f"Using max_length: {args.max_length}")
    #     max_duration = args.max_length
    # else:
    std_lengths, mean_lengths, max_length = calculate_duration(train_data)
    print(f"Mean length: {mean_lengths}, std length: {std_lengths}, max length: {max_length}")
    length = int(mean_lengths + 1 * std_lengths)
    max_duration = length
    wandb.config.update({"max_duration": max_duration}, allow_val_change=True)
    print(f"Using calculated length: {max_duration}")

    # add to config
    config.max_duration = max_duration

    # Load and preprocess datasets
    train_features, train_valence_labels, train_activation_labels, train_max_duration= load_data(train_data, max_duration=max_duration)
    dev_features, dev_valence_labels, dev_activation_labels, dev_max_duration= load_data(dev_data, max_duration=max_duration)

    # Prepare datasets and DataLoaders
    train_dataset = list(zip(train_features, train_valence_labels, train_activation_labels))
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    dev_dataset = list(zip(dev_features, dev_valence_labels, dev_activation_labels))
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model instantiation: input_dim is the number of features (e.g., 26)
    input_dim = 26
    seq_len = train_max_duration  # Maximum sequence length from training data
    if config.model_name == "CNN":
        model = CNN(
            input_dim,
            seq_len,
            num_filters=config.num_filters,
            output_size=config.output_size,
            dropout=config.dropout,
            kernel_sizes=tuple(config.kernel_sizes),
            pool_kernel=config.pool_kernel,
            pool_stride=config.pool_stride,
        )
    elif config.model_name == "ACNN":
        model = ACNN(
        input_dim,
        seq_len,
        num_filters=config.num_filters,
        output_size=config.output_size,
        dropout=config.dropout,
        kernel_sizes=tuple(config.kernel_sizes),
        pool_kernel=config.pool_kernel,
        pool_stride=config.pool_stride,
        )
    elif config.model_name == "TransformerDualClassifier":
        model = TransformerDualClassifier(
            input_dim=26, 
            model_dim=config.model_dim, 
            num_heads=config.num_heads, 
            num_layers=config.num_layers, 
            dropout=config.dropout,
        )
    elif config.model_name == "BiLSTMModel":
        model = BiLSTMModel(
            input_dim=26, 
            hidden_dim=config.hidden_dim, 
            num_layers=config.num_layers, 
            dropout=config.dropout,
        )

    model = model.to(device)

    # Set up loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.scheduler == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=config.gamma)


    # Train the model with early stopping (patience = 5 epochs)
    train_model(model, train_loader, dev_loader, criterion, optimizer, scheduler, epochs=config.epochs, patience=5, run_id=config.run_id, output_dir=config.output_dir)

if __name__ == "__main__":
    main()