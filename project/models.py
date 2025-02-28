import torch
from torch import nn
import torch.nn.functional as F
import math

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=128, num_layers=2, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        # Define a bidirectional LSTM.
        # Set batch_first=True so that inputs are of shape (batch, seq_len, feature_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )
        
        # After the BiLSTM, the hidden dimension is doubled (for forward and backward).
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        
        # Two separate classifiers for valence and activation.
        # For binary classification, you can output one logit per classifier.
        self.valence_classifier = nn.Linear(hidden_dim, 1)
        self.activation_classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # Get the LSTM outputs; we ignore the hidden states here.
        x = x.float()
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size*2)
        
        # Pool the outputs over the time dimension.
        # You can also try other pooling strategies (max pooling, last timestep, etc.)
        pooled = torch.mean(lstm_out, dim=1)  # shape: (batch_size, hidden_size*2)
        
        # Pass through a fully connected layer and ReLU.
        fc_out = self.relu(self.fc(pooled))  # shape: (batch_size, hidden_size)
        
        # Compute logits for both classifiers.
        valence_logits = F.sigmoid(self.valence_classifier(fc_out))      # shape: (batch_size, 1)
        activation_logits = F.sigmoid(self.activation_classifier(fc_out))  # shape: (batch_size, 1)
        
        return valence_logits, activation_logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on 
        # position and dimension
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # apply sin to even indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cos to odd indices in the array
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerDualClassifier(nn.Module):
    def __init__(self, input_dim=26, model_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: Dimension of input features (26 for log-mel features)
            model_dim: Dimension to project inputs into (d_model)
            num_heads: Number of attention heads in transformer layers
            num_layers: Number of transformer encoder layers
            dropout: Dropout probability
        """
        super(TransformerDualClassifier, self).__init__()
        # Project input features to model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        # Add positional encoding to capture sequence order
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        # Define transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Two separate heads for valence and activation classification (binary classification)
        self.valence_classifier = nn.Linear(model_dim, 1)    # output one logit for valence
        self.activation_classifier = nn.Linear(model_dim, 1)  # output one logit for activation

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            src_key_padding_mask: Optional mask to ignore padded positions (batch_size, seq_len)
        Returns:
            Tuple of logits:
                valence_logits: Tensor of shape (batch_size, 1)
                activation_logits: Tensor of shape (batch_size, 1)
        """
        # x = x.permute(0, 2, 1)
        x = x.float()
        # Project inputs: (batch_size, seq_len, input_dim) -> (batch_size, seq_len, model_dim)
        x = self.input_linear(x)
        # Add positional encodings
        x = self.positional_encoding(x)
        # Transformer expects input as (seq_len, batch_size, model_dim)
        x = x.transpose(0, 1)
        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Pool over the sequence dimension (e.g., mean pooling)
        pooled = x.mean(dim=0)  # shape becomes (batch_size, model_dim)
        # Compute logits for each task, add sigmoid activation
        valence_logits = torch.sigmoid(self.valence_classifier(pooled))
        activation_logits = torch.sigmoid(self.activation_classifier(pooled))
        return valence_logits, activation_logits

class CNN(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        num_filters=100,
        output_size=128,
        dropout=0.8,
        kernel_sizes=(5, 7),
        pool_kernel=30,
        pool_stride=3,
    ):
        """
        CNN for SER using two convolution branches with different kernel widths.
        
        Args:
            input_dim (int): Feature dimension (e.g., 26 for logMel features).
            seq_len (int): Maximum sequence length.
            num_filters (int): Number of filters per convolution branch.
            output_size (int): Number of units in the fully connected layer.
            dropout (float): Dropout probability.
            kernel_sizes (tuple): Tuple containing two kernel widths (for conv branches).
            pool_kernel (int): Kernel size for max pooling.
            pool_stride (int): Stride for max pooling.
        """
        super(CNN, self).__init__()
        k1, k2 = kernel_sizes

        # Each convolution branch spans the entire feature dimension
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(input_dim, k1),
            stride=1,
            padding=(0, k1 // 2),
        )
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(input_dim, k2),
            stride=1,
            padding=(0, k2 // 2),
        )

        # Max pooling over the sequence (width) dimension
        self.pool = nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride))

        # Use a dummy input to compute the flattened feature size after conv and pooling
        dummy_input = torch.zeros(1, 1, input_dim, seq_len)
        out1 = F.relu(self.conv1(dummy_input))
        out2 = F.relu(self.conv2(dummy_input))
        combined = torch.cat([out1, out2], dim=1)
        pooled = self.pool(combined)
        fc_input_dim = pooled.view(1, -1).size(1)

        self.fc1 = nn.Linear(fc_input_dim, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.clf_activation = nn.Linear(output_size, 1)
        self.clf_valence = nn.Linear(output_size, 1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input of shape (batch, 1, feature_dim, seq_len).
            
        Returns:
            activation_output (Tensor): Activation predictions (after sigmoid).
            valence_output (Tensor): Valence predictions (after sigmoid).
        """
        # Add a channel dimension: (batch, 1, max_seq_length, feature_dim)
        x = x.unsqueeze(1)
        # Permute to (batch, 1, feature_dim, max_seq_length)
        x = x.permute(0, 1, 3, 2)
        x = x.float()
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        # Concatenate feature maps along the channel dimension
        x = torch.cat([x1, x2], dim=1)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        activation_output = torch.sigmoid(self.clf_activation(x))
        valence_output = torch.sigmoid(self.clf_valence(x))
        return activation_output, valence_output

class ACNN(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        num_filters=100,
        output_size=128,
        dropout=0.8,
        kernel_sizes=(5, 7),
        pool_kernel=30,
        pool_stride=3,
    ):
        """
        ACNN for SER with an attention mechanism.

        This model applies two convolution branches (with different kernel widths)
        to extract features from the input spectrogram (e.g., logMel features). The outputs 
        from both branches are concatenated and then processed in two parallel ways:
        
          1. Global max pooling over the time dimension.
          2. An attention mechanism that computes weights over time using a linear scoring function
             f(x) = W^T x and computes a weighted sum.
        
        The two resulting vectors are concatenated and passed to a fully connected layer (with dropout),
        and then branched into separate outputs for activation and valence predictions.

        Args:
            input_dim (int): Feature dimension (e.g., 26 for logMel features).
            seq_len (int): Maximum sequence length.
            num_filters (int): Number of filters per convolution branch.
            output_size (int): Number of units in the fully connected layer.
            dropout (float): Dropout probability.
            kernel_sizes (tuple): Tuple of two kernel widths for the convolution branches.
            pool_kernel (int): Pooling kernel size (applied over the time dimension).
            pool_stride (int): Pooling stride (applied over the time dimension).
        """
        super(ACNN, self).__init__()
        k1, k2 = kernel_sizes

        # Convolution branches spanning the entire frequency (input) dimension.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(input_dim, k1),
            stride=1,
            padding=(0, k1 // 2),
        )
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(input_dim, k2),
            stride=1,
            padding=(0, k2 // 2),
        )

        # Max pooling over the sequence (time) dimension using pool_kernel and pool_stride.
        self.pool = nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride))

        # Attention layer: scoring function f(x) = W^T x (implemented as a linear layer without bias).
        # The input dimension to the attention layer is 2 * num_filters.
        self.attention = nn.Linear(2 * num_filters, 1, bias=False)
        
        # The final feature vector is a concatenation of:
        #   - Global max pooling vector: (2 * num_filters)
        #   - Attention vector: (2 * num_filters)
        # Thus, the concatenated feature vector has dimension: 4 * num_filters.
        self.fc1 = nn.Linear(4 * num_filters, output_size)
        self.dropout = nn.Dropout(p=dropout)
        
        # Output classifiers for multi-view learning.
        self.clf_activation = nn.Linear(output_size, 1)
        self.clf_valence = nn.Linear(output_size, 1)

        # (Optional) Compute dummy flattened size for debugging purposes.
        # Create a dummy input to simulate the network flow.
        dummy_input = torch.zeros(1, 1, input_dim, seq_len)
        out1 = F.relu(self.conv1(dummy_input))
        out2 = F.relu(self.conv2(dummy_input))
        x_cat = torch.cat([out1, out2], dim=1)  # Shape: (1, 2*num_filters, 1, T)
        x_cat = x_cat.squeeze(2)                # Shape: (1, 2*num_filters, T)
        # Apply pooling.
        pooled = self.pool(x_cat.unsqueeze(1))  # Here we temporarily add a channel for pooling compatibility.
        # Note: This dummy computation is only to check dimensions.
        # Remove or comment out if not needed.
        fc_input_dim = pooled.view(1, -1).size(1)
        # print("Dummy fc_input_dim:", fc_input_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input of shape (batch, 1, input_dim, seq_len).

        Returns:
            activation_output (Tensor): Activation predictions (after sigmoid).
            valence_output (Tensor): Valence predictions (after sigmoid).
        """
        # Add a channel dimension: (batch, 1, max_seq_length, feature_dim)
        x = x.unsqueeze(1)
        # Permute to (batch, 1, feature_dim, max_seq_length)
        x = x.permute(0, 1, 3, 2)
        x = x.float()
        # Convolution branches with ReLU activation.
        x1 = F.relu(self.conv1(x))  # Shape: (batch, num_filters, 1, T)
        x2 = F.relu(self.conv2(x))  # Shape: (batch, num_filters, 1, T)
        
        # Concatenate the outputs along the channel dimension.
        x_cat = torch.cat([x1, x2], dim=1)  # Shape: (batch, 2*num_filters, 1, T)
        
        # Remove the singleton dimension (height) to form a sequence.
        # x_seq shape: (batch, 2*num_filters, T)
        x_seq = x_cat.squeeze(2)
        
        # ---------------------------
        # Global Max Pooling Branch
        # ---------------------------
        # Compute max pooling over the time dimension.
        max_vector = torch.max(x_seq, dim=2)[0]  # Shape: (batch, 2*num_filters)
        
        # ---------------------------
        # Attention Mechanism Branch
        # ---------------------------
        # Transpose to shape (batch, T, 2*num_filters) for the attention layer.
        scores = self.attention(x_seq.permute(0, 2, 1))  # Shape: (batch, T, 1)
        alpha = torch.softmax(scores, dim=1)             # Shape: (batch, T, 1)
        # Compute the weighted sum over time.
        attention_vector = torch.bmm(x_seq, alpha)         # Shape: (batch, 2*num_filters, 1)
        attention_vector = attention_vector.squeeze(2)     # Shape: (batch, 2*num_filters)
        
        # ---------------------------
        # Combine and Classify
        # ---------------------------
        # Concatenate the max pooling and attention vectors.
        combined = torch.cat([max_vector, attention_vector], dim=1)  # Shape: (batch, 4*num_filters)
        fc_out = F.relu(self.fc1(combined))
        fc_out = self.dropout(fc_out)
        
        activation_output = torch.sigmoid(self.clf_activation(fc_out))
        valence_output = torch.sigmoid(self.clf_valence(fc_out))
        return activation_output, valence_output