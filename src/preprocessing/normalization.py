"""
normalization.py
================
Channel-wise and subject-wise normalization methods for EEG signals.
Supports standard Z-Score standardization, IQR-based Robust scaling,
and global fitted normalizers to prevent data leakage.
"""

from typing import Union, Tuple, Optional
import numpy as np

class RobustScaler1D:
    """
    Robust channel-wise scaler using median and Interquartile Range (IQR).
    Resilient to large blink spikes and movement transients.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.medians: Optional[np.ndarray] = None
        self.iqrs: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray, axis: int = 0) -> "RobustScaler1D":
        """
        Fit median and IQR across specified axis.
        """
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        self.medians = np.median(data, axis=axis, keepdims=True)
        self.iqrs = (q75 - q25) / 1.349 + self.eps
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.medians is None or self.iqrs is None:
            raise RuntimeError("RobustScaler1D must be fitted before calling transform.")
        return (data - self.medians) / self.iqrs

    def fit_transform(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        return self.fit(data, axis=axis).transform(data)

def normalize_channels(
    data: np.ndarray,
    method: str = "zscore",
    axis: int = 0,
    eps: float = 1e-8
) -> Tuple[np.ndarray, dict]:
    """
    Apply channel-wise normalization along the temporal dimension (axis=0).

    Parameters:
        data: Multichannel EEG array (n_samples, n_channels).
        method: 'zscore' (mean 0, std 1) or 'robust' (median, IQR) or 'minmax'.
        axis: Time axis along which to compute statistics (default 0).
        eps: Small constant to avoid zero division.

    Returns:
        Tuple of (normalized_data, stats_dict).
    """
    if method == "zscore":
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True) + eps
        normalized = (data - mean) / std
        stats = {"mean": mean, "std": std, "method": "zscore"}

    elif method == "robust":
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        median = np.median(data, axis=axis, keepdims=True)
        iqr = (q75 - q25) / 1.349 + eps
        normalized = (data - median) / iqr
        stats = {"median": median, "iqr": iqr, "method": "robust"}

    elif method == "minmax":
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + eps)
        stats = {"min": min_val, "max": max_val, "method": "minmax"}

    else:
        raise ValueError(f"Unknown normalization method: {method}. Choose 'zscore', 'robust', or 'minmax'.")

    return normalized, stats
