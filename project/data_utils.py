import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def calculate_duration(data):
    features = []
    for sample_id, sample_data in data.items():
        if "features" in sample_data:
            features.append(np.array(sample_data['features']))

    lengths = np.array([len(f) for f in features])
    max_length = np.max(lengths)
    mean_lengths = np.mean(lengths)
    std_lengths = np.std(lengths)
    return std_lengths, mean_lengths, max_length

def load_data(data, max_duration=None):
    """
    Loads data from a dictionary and returns padded features,
    valence labels, activation labels, and max_duration.
    
    Args:
        data (dict): Dictionary containing samples.
        max_duration (int, optional): Maximum length to pad/trim sequences.
    
    Returns:
        features (Tensor): Padded tensor of features.
        valence_labels (Tensor): Tensor of valence labels.
        activation_labels (Tensor): Tensor of activation labels.
        max_duration (int): The maximum sequence length used. 
    """
    features = []
    valence_labels = []
    activation_labels = []

    for sample_id, sample_data in data.items():
        if "features" in sample_data:
            features.append(np.array(sample_data["features"]))
        if "valence" in sample_data:
            valence_labels.append(sample_data["valence"])
        if "activation" in sample_data:
            activation_labels.append(sample_data["activation"])

    # Determine maximum duration if not provided
    if max_duration is None:
        max_duration = np.max([len(f) for f in features])

    # Pad (or trim) each feature sequence to max_duration
    features = pad_sequence([torch.tensor(f[:max_duration]) for f in features], batch_first=True)
    valence_labels = torch.tensor(valence_labels)
    activation_labels = torch.tensor(activation_labels)

    return features, valence_labels, activation_labels, max_duration


def collate_fn(batch):
    """
    Custom collate function to pad sequences and rearrange dimensions.
    
    Expected original shape of each sample's features:
        (seq_length, feature_dim)
    
    After processing, each sample is reshaped to:
        (1, feature_dim, seq_length)
    
    Returns:
        padded_seqs (Tensor): Padded tensor of shape (batch, 1, feature_dim, seq_length).
        valence_labels (Tensor): Float tensor of valence labels.
        activation_labels (Tensor): Float tensor of activation labels.
        lengths (list): List of original sequence lengths.
    """
    sequences, valence_labels, activation_labels = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]

    # Pad sequences: (batch, max_seq_length, feature_dim)
    padded_seqs = pad_sequence(sequences, batch_first=True)

    return padded_seqs, torch.tensor(valence_labels).float(), torch.tensor(activation_labels).float(), lengths
