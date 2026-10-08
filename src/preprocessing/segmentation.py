"""
segmentation.py
===============
Temporal sliding window segmentation for EEG continuous timeseries.
Maintains exact chronological ordering, sequence continuity, and timestamps
for training the Quantum Neural Digital Twin in predictive and generative modes.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

def create_sliding_windows(
    data: np.ndarray,
    fs: float = 250.0,
    window_sec: float = 2.0,
    stride_sec: float = 1.0,
    subject_id: str = "s00",
    subject_numeric_id: int = 0
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extract sliding temporal windows from a continuous EEG array.

    Parameters:
        data: Continuous EEG array (n_samples, n_channels).
        fs: Sampling frequency in Hz (default 250.0).
        window_sec: Window duration in seconds (default 2.0).
        stride_sec: Step size between consecutive windows in seconds (default 1.0).
        subject_id: String subject identifier (e.g. 's00').
        subject_numeric_id: Integer subject ID (e.g. 0).

    Returns:
        Tuple of:
            - windows_array: np.ndarray of shape (num_windows, window_samples, n_channels)
            - metadata_records: List of dictionaries detailing each window's temporal span.
    """
    n_samples, n_channels = data.shape
    window_samples = int(round(window_sec * fs))
    stride_samples = int(round(stride_sec * fs))

    if window_samples > n_samples:
        raise ValueError(f"Window size ({window_samples} samples) exceeds total recording length ({n_samples} samples).")
    if stride_samples <= 0:
        raise ValueError("Stride samples must be strictly positive.")

    windows_list = []
    metadata_list = []

    start = 0
    w_idx = 0
    while start + window_samples <= n_samples:
        end = start + window_samples
        window_chunk = data[start:end, :]  # shape: (window_samples, n_channels)
        windows_list.append(window_chunk)

        start_time = round(start / fs, 4)
        end_time = round(end / fs, 4)

        metadata_list.append({
            "subject_id": subject_id,
            "subject_numeric_id": subject_numeric_id,
            "window_idx": w_idx,
            "start_sample": start,
            "end_sample": end,
            "start_time_sec": start_time,
            "end_time_sec": end_time,
            "window_samples": window_samples,
            "num_channels": n_channels
        })

        start += stride_samples
        w_idx += 1

    windows_array = np.array(windows_list, dtype=np.float32)
    return windows_array, metadata_list

def extract_subject_windows(
    continuous_eeg: np.ndarray,
    fs: float = 250.0,
    window_sec: float = 2.0,
    stride_sec: float = 1.0,
    subject_id: str = "s00",
    subject_numeric_id: int = 0
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract temporal windows and construct autoregressive sequential transition pairs
    (X_current -> Y_next) for Neural Digital Twin sequence modeling.

    Returns:
        Tuple of:
            - X: windows array of shape (N, window_samples, n_channels)
            - Y_next: next-window array of shape (N - 1, window_samples, n_channels)
            - metadata_df: pd.DataFrame with window indices and timestamps.
    """
    windows, meta_list = create_sliding_windows(
        data=continuous_eeg,
        fs=fs,
        window_sec=window_sec,
        stride_sec=stride_sec,
        subject_id=subject_id,
        subject_numeric_id=subject_numeric_id
    )

    df_meta = pd.DataFrame(meta_list)

    # Next-window target: Y_next[t] = X[t+1]
    if len(windows) > 1:
        y_next = windows[1:].copy()
    else:
        y_next = np.empty((0, *windows.shape[1:]), dtype=np.float32)

    return windows, y_next, df_meta
