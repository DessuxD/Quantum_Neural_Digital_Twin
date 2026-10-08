"""
dimension_reduction.py
=======================
Dimension reduction and quantum state mapping utilities.
Provides:
1. Spatial PCA decomposition across the 19 EEG channels.
2. QuantumStateMapper: Differentiable neural projection layer mapping
   latent embeddings to bounded rotation angles in [-pi, pi] for quantum circuits.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

class SpatialEEG_PCA:
    """
    Principal Component Analysis fitted across multichannel EEG topography.
    Maps 19 spatial channels to N principal spatial components.
    """
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "SpatialEEG_PCA":
        """
        Fit PCA on EEG windows array of shape (N, T, C) or continuous array (N_samples, C).
        """
        if X.ndim == 3:
            N, T, C = X.shape
            flat_X = X.reshape(-1, C)
        else:
            flat_X = X
        self.pca.fit(flat_X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform (N, T, C) -> (N, T, n_components).
        """
        if not self.is_fitted:
            raise RuntimeError("SpatialEEG_PCA must be fitted before transform.")
        if X.ndim == 3:
            N, T, C = X.shape
            flat_X = X.reshape(-1, C)
            transformed = self.pca.transform(flat_X)
            return transformed.reshape(N, T, self.n_components)
        return self.pca.transform(X)

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Inverse transform (N, T, n_components) -> (N, T, C).
        """
        if not self.is_fitted:
            raise RuntimeError("SpatialEEG_PCA must be fitted before inverse_transform.")
        if X_pca.ndim == 3:
            N, T, K = X_pca.shape
            flat_k = X_pca.reshape(-1, K)
            recon = self.pca.inverse_transform(flat_k)
            return recon.reshape(N, T, self.pca.n_features_in_)
        return self.pca.inverse_transform(X_pca)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    @property
    def total_explained_variance(self) -> float:
        return float(np.sum(self.pca.explained_variance_ratio_))

class QuantumStateMapper(nn.Module):
    """
    Differentiable PyTorch module mapping classical latent feature vectors
    to bounded rotation angles [-pi, pi] for quantum angle encoding.
    """
    def __init__(
        self,
        in_features: int,
        n_qubits: int = 4,
        scaling: str = "tanh"
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.scaling = scaling

        self.projection = nn.Sequential(
            nn.Linear(in_features, n_qubits),
            nn.LayerNorm(n_qubits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps (batch_size, in_features) -> (batch_size, n_qubits) in [-pi, pi].
        """
        proj = self.projection(x)
        if self.scaling == "tanh":
            # Scale to [-pi, pi]
            angles = torch.pi * torch.tanh(proj)
        elif self.scaling == "arctan":
            # 2 * arctan maps (-inf, inf) smoothly to (-pi, pi)
            angles = 2.0 * torch.atan(proj)
        elif self.scaling == "sigmoid":
            # Scale to [0, 2*pi]
            angles = 2.0 * torch.pi * torch.sigmoid(proj) - torch.pi
        else:
            angles = proj
        return angles
