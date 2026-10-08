"""
bilstm.py
=========
Bidirectional LSTM (BiLSTM) for Temporal Sequential Dynamics Modeling.
Captures forward and backward temporal dependencies across EEG time steps.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn

class BiLSTMTemporalEncoder(nn.Module):
    """
    Bidirectional LSTM module for temporal EEG sequence modeling.

    Parameters:
        input_dim: Number of input features per time step (default 19 for 19 channels).
        hidden_dim: Number of hidden units per LSTM direction (default 32).
        num_layers: Number of stacked LSTM layers (default 2).
        latent_dim: Output projection dimension (default 32).
        dropout: Dropout rate between LSTM layers (default 0.2).
    """
    def __init__(
        self,
        input_dim: int = 19,
        hidden_dim: int = 32,
        num_layers: int = 2,
        latent_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 2 * hidden_dim because of bidirectionality
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU()
        )

    def forward(
        self,
        x: torch.Tensor,
        return_sequences: bool = False
    ) -> torch.Tensor:
        """
        Input: (batch, seq_len, input_dim)
        Output:
            if return_sequences: (batch, seq_len, latent_dim)
            else: (batch, latent_dim) - pooled summary of temporal state
        """
        if x.dim() == 4 and x.shape[1] == 1:
            # (B, 1, C, T) -> (B, T, C)
            x = x.squeeze(1).transpose(1, 2)

        out, (hn, cn) = self.lstm(x)  # out: (B, seq_len, 2 * hidden_dim)

        if return_sequences:
            return self.proj(out)
        else:
            # Concatenate the last forward and first backward hidden states
            # hn is of shape (num_layers * 2, B, hidden_dim)
            forward_last = hn[-2, :, :]
            backward_last = hn[-1, :, :]
            last_hidden = torch.cat([forward_last, backward_last], dim=1)  # (B, 2 * hidden_dim)
            return self.proj(last_hidden)
