"""
eegnet.py
=========
EEGNet Architecture for Multichannel EEG Feature Extraction.
Adapted from Lawhern et al. (2018) for 19-channel EEG signals.
Incorporates temporal convolutions, depthwise spatial filtering across
cortical channels, and separable convolutions.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNetEncoder(nn.Module):
    """
    Compact spatial-temporal CNN encoder for EEG signals.

    Parameters:
        n_channels: Number of EEG electrode channels (default 19).
        n_timesteps: Number of time samples per window (default 500 for 2.0s @ 250 Hz).
        F1: Number of temporal filters (default 8).
        D: Depth multiplier for spatial depthwise convolution (default 2).
        F2: Number of pointwise filters (default 16).
        kernel_length: Length of temporal convolution kernel (default 64).
        latent_dim: Dimensionality of output latent embedding (default 32).
        dropout_rate: Dropout probability (default 0.25).
    """
    def __init__(
        self,
        n_channels: int = 19,
        n_timesteps: int = 500,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        latent_dim: int = 32,
        dropout_rate: float = 0.25
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.latent_dim = latent_dim

        # Block 1: Temporal Conv + Depthwise Spatial Conv
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial filter across all 19 EEG electrodes
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.separable_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False
        )
        self.separable_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Compute flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timesteps)
            x_test = self._forward_features(dummy)
            self.flat_dim = x_test.view(1, -1).shape[1]

        # Final projection to latent embedding
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU()
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, n_timesteps, n_channels) or (batch, 1, n_channels, n_timesteps)
        Output: (batch, latent_dim)
        """
        # Handle input layout
        if x.dim() == 3:
            # (batch, time, channels) -> (batch, 1, channels, time)
            x = x.transpose(1, 2).unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            # Ensure channel dim 1 is singleton
            raise ValueError(f"Expected 4D input of shape (B, 1, C, T), got {x.shape}")

        features = self._forward_features(x)
        flattened = features.view(features.size(0), -1)
        latent = self.fc(flattened)
        return latent
