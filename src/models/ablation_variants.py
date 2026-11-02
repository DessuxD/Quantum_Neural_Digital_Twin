"""
ablation_variants.py
====================
Provides the specific model architectures required for the full 6-level
ablation hierarchy defined in Paper Section XII.E and Section VII.E:

1. EEGNetOnlyDigitalTwin: Spatial CNN only (no BiLSTM temporal recurrence, no quantum).
2. EEGNetBiLSTM_PCADigitalTwin: EEGNet + BiLSTM + 4-dimensional PCA bottleneck (no quantum).
3. QuantumNeuralDigitalTwin_PCA: EEGNet + BiLSTM + PCA projection (Eq. 9-10) + VQC (Eq. 11-14) (Paper Eq. 19).
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eegnet import EEGNetEncoder
from .bilstm import BiLSTMTemporalEncoder
from .hybrid_twin import WaveformDecoder
from .quantum_circuit import VariationalQuantumEncoder


class EEGNetOnlyDigitalTwin(nn.Module):
    """
    Ablation 1 (Paper XII.E.1): EEGNet spatial encoder alone.
    No BiLSTM temporal recurrence, no quantum circuit.
    """
    def __init__(
        self,
        n_channels: int = 19,
        n_timesteps: int = 500,
        latent_dim: int = 32,
        n_subjects: int = 36,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_encoder = EEGNetEncoder(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            F1=8, D=2, F2=16,
            latent_dim=latent_dim,
            dropout_rate=dropout,
        )
        self.predictor_decoder = WaveformDecoder(
            latent_dim=latent_dim,
            n_channels=n_channels,
            n_timesteps=n_timesteps,
        )
        self.recon_decoder = WaveformDecoder(
            latent_dim=latent_dim,
            n_channels=n_channels,
            n_timesteps=n_timesteps,
        )
        self.biometric_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ELU(),
            nn.Linear(32, n_subjects),
        )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.spatial_encoder(x)
        return {"spatial": z, "fused": z}

    def forward(self, x: torch.Tensor, task: str = "predictive") -> Dict[str, torch.Tensor]:
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


class EEGNetBiLSTM_PCADigitalTwin(nn.Module):
    """
    Ablation 3 (Paper XII.E.3): EEGNet + BiLSTM + 4-dimensional PCA bottleneck.
    Quantifies whether a 4-dimensional bottleneck hurts or helps classical representation
    in the absence of quantum Hilbert space transformations.
    """
    def __init__(
        self,
        n_channels: int = 19,
        n_timesteps: int = 500,
        spatial_dim: int = 16,
        temporal_dim: int = 16,
        pca_dim: int = 4,
        fused_dim: int = 32,
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
            hidden_dim=24,
            num_layers=2,
            latent_dim=temporal_dim,
            dropout=dropout,
        )

        classical_dim = spatial_dim + temporal_dim
        self.classical_fusion = nn.Sequential(
            nn.Linear(classical_dim, classical_dim),
            nn.LayerNorm(classical_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # PCA linear bottleneck (4 dimensions)
        self.pca_bottleneck = nn.Sequential(
            nn.Linear(classical_dim, pca_dim),
            nn.LayerNorm(pca_dim),
            nn.Linear(pca_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ELU(),
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

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_s = self.spatial_encoder(x)
        z_t = self.temporal_encoder(x)
        z_c = self.classical_fusion(torch.cat([z_s, z_t], dim=1))
        z_f = self.pca_bottleneck(z_c)
        return {"spatial": z_s, "temporal": z_t, "fused": z_f}

    def forward(self, x: torch.Tensor, task: str = "predictive") -> Dict[str, torch.Tensor]:
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


class QuantumNeuralDigitalTwin_PCA(nn.Module):
    """
    Paper Equations (9)-(14) & (19):
    EEG -> Preprocessing -> EEGNet -> BiLSTM -> PCA (z -> z') -> Angle Encoding -> VQC -> Decoder.
    Uses explicit orthogonal linear projection (PCA) to reduce classical feature dimension
    to q=4 angles before feeding the Variational Quantum Circuit.
    """
    def __init__(
        self,
        n_channels: int = 19,
        n_timesteps: int = 500,
        spatial_dim: int = 16,
        temporal_dim: int = 16,
        n_qubits: int = 4,
        n_quantum_layers: int = 1,
        quantum_dim: int = 8,
        fused_dim: int = 40,
        n_subjects: int = 36,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.spatial_encoder = EEGNetEncoder(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            F1=8, D=2, F2=16,
            latent_dim=spatial_dim,
            dropout_rate=dropout,
        )
        self.temporal_encoder = BiLSTMTemporalEncoder(
            input_dim=n_channels,
            hidden_dim=24,
            num_layers=2,
            latent_dim=temporal_dim,
            dropout=dropout,
        )

        classical_dim = spatial_dim + temporal_dim
        self.classical_fusion = nn.Sequential(
            nn.Linear(classical_dim, classical_dim),
            nn.LayerNorm(classical_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # PCA Projection to q dimensions: z in R^d -> z' in R^q (Paper Eq. 9-10)
        self.pca_projection = nn.Sequential(
            nn.Linear(classical_dim, n_qubits),
            nn.LayerNorm(n_qubits),
        )

        # Variational Quantum Encoder (Paper Eq. 11-14)
        self.quantum_encoder = VariationalQuantumEncoder(
            n_qubits=n_qubits,
            n_layers=n_quantum_layers,
            output_dim=quantum_dim,
        )

        # Hybrid classical-quantum fusion
        self.hybrid_fusion = nn.Sequential(
            nn.Linear(classical_dim + quantum_dim, fused_dim),
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

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_s = self.spatial_encoder(x)
        z_t = self.temporal_encoder(x)
        z_c = self.classical_fusion(torch.cat([z_s, z_t], dim=1))

        # PCA transformation -> Angle scaling in [-pi, pi]
        pca_features = self.pca_projection(z_c)
        q_angles = torch.pi * torch.tanh(pca_features)

        z_q = self.quantum_encoder(q_angles)
        z_f = self.hybrid_fusion(torch.cat([z_c, z_q], dim=1))

        return {
            "spatial": z_s,
            "temporal": z_t,
            "classical": z_c,
            "pca_features": pca_features,
            "quantum_angles": q_angles,
            "quantum": z_q,
            "fused": z_f,
        }

    def forward(self, x: torch.Tensor, task: str = "predictive") -> Dict[str, torch.Tensor]:
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
