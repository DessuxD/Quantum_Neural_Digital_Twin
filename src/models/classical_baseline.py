"""
classical_baseline.py
=====================
Classical-only baseline model (EEGNet + BiLSTM, no quantum circuit).
Used for ablation: quantifies the contribution of the VQC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eegnet import EEGNetEncoder
from .bilstm import BiLSTMTemporalEncoder
from .hybrid_twin import WaveformDecoder


class ClassicalDigitalTwin(nn.Module):
    """
    Ablation baseline: full EEGNet + BiLSTM + Decoder with NO quantum component.
    Matched parameter budget to the hybrid model as closely as possible.

    Parameters:
        n_channels: EEG channel count (default 19).
        n_timesteps: Window length (default 500).
        spatial_dim: EEGNet latent dim (default 24).
        temporal_dim: BiLSTM latent dim (default 24).
        fused_dim: Fusion projection dim (default 48).
        n_subjects: Number of subject classes (default 36).
        dropout: Dropout probability (default 0.2).
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_timesteps: int = 500,
        spatial_dim: int = 24,
        temporal_dim: int = 24,
        fused_dim: int = 48,
        n_subjects: int = 36,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.spatial_encoder = EEGNetEncoder(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            F1=8, D=2, F2=16,
            latent_dim=spatial_dim,
            dropout_rate=dropout,
        )

        self.temporal_encoder = BiLSTMTemporalEncoder(
            input_dim=n_channels,
            hidden_dim=32,
            num_layers=2,
            latent_dim=temporal_dim,
            dropout=dropout,
        )

        classical_dim = spatial_dim + temporal_dim
        self.fusion = nn.Sequential(
            nn.Linear(classical_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.predictor_decoder = WaveformDecoder(
            latent_dim=fused_dim,
            n_channels=n_channels,
            n_timesteps=n_timesteps,
        )
        self.recon_decoder = WaveformDecoder(
            latent_dim=fused_dim,
            n_channels=n_channels,
            n_timesteps=n_timesteps,
        )
        self.biometric_head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ELU(),
            nn.Linear(32, n_subjects),
        )

    def encode(self, x: torch.Tensor):
        z_s = self.spatial_encoder(x)
        z_t = self.temporal_encoder(x)
        z_f = self.fusion(torch.cat([z_s, z_t], dim=1))
        return {"spatial": z_s, "temporal": z_t, "fused": z_f}

    def forward(self, x: torch.Tensor, task: str = "predictive"):
        enc = self.encode(x)
        z = enc["fused"]
        results = {**enc}
        if task in ("predictive", "all"):
            results["x_next_pred"] = self.predictor_decoder(z)
        if task in ("reconstruction", "all"):
            results["x_recon"] = self.recon_decoder(z)
        if task in ("biometric", "all"):
            results["subject_logits"] = self.biometric_head(z)
        return results
