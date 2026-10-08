"""
hybrid_twin.py
==============
Unified Quantum Neural Digital Twin Architecture.
Integrates:
1. EEGNet: Spatial-temporal convolutional cortical encoder.
2. BiLSTM: Bidirectional recurrent sequence dynamics encoder.
3. PennyLane VQC: Parameterized Variational Quantum Circuit in Hilbert space.
4. Classical-Quantum Fusion Layer.
5. Multitask Decoders:
   - Next-window autoregressive forecasting (X_t -> X_{t+1})
   - State reconstruction (autoencoding)
   - Subject biometric classification (36 subjects)
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eegnet import EEGNetEncoder
from .bilstm import BiLSTMTemporalEncoder
from .dimension_reduction import QuantumStateMapper
from .quantum_circuit import VariationalQuantumEncoder

class WaveformDecoder(nn.Module):
    """
    Decodes compressed latent state vectors into continuous multichannel EEG windows.
    Uses 1D transposed convolutions with residual refinement.
    """
    def __init__(
        self,
        latent_dim: int = 48,
        n_channels: int = 19,
        n_timesteps: int = 500
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps

        # Initial expansion from latent_dim -> (32 channels, 62 timesteps)
        self.init_timesteps = 62
        self.init_channels = 32
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.init_channels * self.init_timesteps),
            nn.ELU()
        )

        # Transposed convolutions to progressively upsample to 500 timesteps
        # 62 -> 125 -> 250 -> 500
        self.deconv = nn.Sequential(
            # 62 -> 125
            nn.ConvTranspose1d(32, 32, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # 125 -> 250
            nn.ConvTranspose1d(32, 24, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(24),
            nn.ELU(),
            # 250 -> 500
            nn.ConvTranspose1d(24, 19, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(19),
            nn.ELU(),
            # Final refinement 1D conv
            nn.Conv1d(19, 19, kernel_size=3, padding=1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, latent_dim)
        Output: (batch, n_timesteps, n_channels)
        """
        b = z.size(0)
        h = self.fc(z).view(b, self.init_channels, self.init_timesteps)  # (B, 32, 62)
        out = self.deconv(h)  # (B, 19, 500)

        # Ensure exact match to n_timesteps if slight padding difference
        if out.shape[2] != self.n_timesteps:
            out = F.interpolate(out, size=self.n_timesteps, mode="linear", align_corners=False)

        # Transpose to standard (batch, n_timesteps, n_channels)
        return out.transpose(1, 2)

class QuantumNeuralDigitalTwin(nn.Module):
    """
    End-to-End Quantum Neural Digital Twin.

    Parameters:
        n_channels: Number of EEG electrode channels (default 19).
        n_timesteps: Timesteps per window (default 500 for 2.0s @ 250 Hz).
        n_qubits: Number of quantum wires in VQC (default 4).
        n_quantum_layers: Variational depth of VQC (default 2).
        spatial_dim: Latent dimension of EEGNet encoder (default 24).
        temporal_dim: Latent dimension of BiLSTM encoder (default 24).
        quantum_dim: Projected dimension after quantum measurement (default 16).
        n_subjects: Number of subject classes for biometric head (default 36).
    """
    def __init__(
        self,
        n_channels: int = 19,
        n_timesteps: int = 500,
        n_qubits: int = 4,
        n_quantum_layers: int = 2,
        spatial_dim: int = 24,
        temporal_dim: int = 24,
        quantum_dim: int = 16,
        n_subjects: int = 36,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.quantum_dim = quantum_dim
        self.n_subjects = n_subjects

        # 1. Classical Spatial Encoder (EEGNet)
        self.spatial_encoder = EEGNetEncoder(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            F1=8,
            D=2,
            F2=16,
            latent_dim=spatial_dim,
            dropout_rate=dropout
        )

        # 2. Classical Temporal Sequential Encoder (BiLSTM)
        self.temporal_encoder = BiLSTMTemporalEncoder(
            input_dim=n_channels,
            hidden_dim=24,
            num_layers=2,
            latent_dim=temporal_dim,
            dropout=dropout
        )

        # Classical Fusion Layer
        classical_dim = spatial_dim + temporal_dim
        self.classical_fusion = nn.Sequential(
            nn.Linear(classical_dim, classical_dim),
            nn.LayerNorm(classical_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # 3. Quantum State Mapper (maps classical latent -> [-pi, pi]^n_qubits)
        self.quantum_mapper = QuantumStateMapper(
            in_features=classical_dim,
            n_qubits=n_qubits,
            scaling="tanh"
        )

        # 4. Variational Quantum Circuit (VQC) in Hilbert space
        self.quantum_encoder = VariationalQuantumEncoder(
            n_qubits=n_qubits,
            n_layers=n_quantum_layers,
            output_dim=quantum_dim,
            rotation_type="Y",
            entanglement="ring"
        )

        # 5. Hybrid Quantum-Classical Fusion
        fused_dim = classical_dim + quantum_dim
        self.hybrid_fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # 6. Decoders & Multitask Heads
        # Head A: Next-window autoregressive predictive digital twin decoder
        self.predictor_decoder = WaveformDecoder(
            latent_dim=fused_dim,
            n_channels=n_channels,
            n_timesteps=n_timesteps
        )

        # Head B: Autoencoder reconstruction decoder
        self.recon_decoder = WaveformDecoder(
            latent_dim=fused_dim,
            n_channels=n_channels,
            n_timesteps=n_timesteps
        )

        # Head C: Subject biometric classification head
        self.biometric_head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ELU(),
            nn.Linear(32, n_subjects)
        )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode raw EEG window into spatial, temporal, quantum, and fused representations.
        Input: (batch, n_timesteps, n_channels)
        """
        # Spatial encoding
        z_spatial = self.spatial_encoder(x)  # (B, spatial_dim)

        # Temporal sequential encoding
        z_temporal = self.temporal_encoder(x)  # (B, temporal_dim)

        # Classical fusion
        z_classical = self.classical_fusion(torch.cat([z_spatial, z_temporal], dim=1))

        # Quantum angle mapping
        q_angles = self.quantum_mapper(z_classical)  # (B, n_qubits)

        # Variational quantum circuit execution
        z_quantum = self.quantum_encoder(q_angles)  # (B, quantum_dim)

        # Hybrid fusion
        z_fused = self.hybrid_fusion(torch.cat([z_classical, z_quantum], dim=1))

        return {
            "spatial": z_spatial,
            "temporal": z_temporal,
            "classical": z_classical,
            "quantum_angles": q_angles,
            "quantum": z_quantum,
            "fused": z_fused
        }

    def forward(
        self,
        x: torch.Tensor,
        task: str = "predictive"
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-end forward execution.

        Parameters:
            x: Input EEG window of shape (batch, n_timesteps, n_channels).
            task: 'predictive' (forecasts next window X_{t+1}),
                  'reconstruction' (reconstructs current window X_t),
                  'all' (computes all heads).

        Returns:
            Dictionary containing representations and predicted outputs.
        """
        enc = self.encode(x)
        z_fused = enc["fused"]
        results = {**enc}

        if task in ("predictive", "all"):
            results["x_next_pred"] = self.predictor_decoder(z_fused)

        if task in ("reconstruction", "all"):
            results["x_recon"] = self.recon_decoder(z_fused)

        if task in ("biometric", "all"):
            results["subject_logits"] = self.biometric_head(z_fused)

        return results
