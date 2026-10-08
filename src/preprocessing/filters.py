"""
filters.py
==========
Digital filtering and resampling utilities for multichannel EEG signals.
Implements zero-phase (forward-backward) notch and Butterworth bandpass filtering,
as well as polyphase anti-aliased resampling.
"""

from typing import Tuple, Optional
import numpy as np
import scipy.signal as signal
from math import gcd

def apply_notch_filter(
    data: np.ndarray,
    fs: float = 500.0,
    freq: float = 50.0,
    q: float = 30.0,
    axis: int = 0
) -> np.ndarray:
    """
    Apply a zero-phase IIR notch filter to suppress powerline hum (default 50 Hz).

    Parameters:
        data: Multichannel EEG array (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        freq: Frequency to reject in Hz (default 50.0 Hz).
        q: Quality factor determining bandwidth (default 30.0).
        axis: Time axis along which to filter.

    Returns:
        Filtered EEG array of identical shape and type float64.
    """
    if freq <= 0 or freq >= (fs / 2.0):
        raise ValueError(f"Notch frequency {freq} Hz must be between 0 and Nyquist ({fs/2.0} Hz).")

    b, a = signal.iirnotch(w0=freq, Q=q, fs=fs)
    filtered = signal.filtfilt(b, a, data, axis=axis)
    return filtered

def apply_bandpass_filter(
    data: np.ndarray,
    fs: float = 500.0,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    order: int = 4,
    axis: int = 0
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Parameters:
        data: Multichannel EEG array (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        lowcut: Lower cutoff frequency in Hz (default 0.5 Hz).
        highcut: Upper cutoff frequency in Hz (default 45.0 Hz).
        order: Filter order (default 4th-order).
        axis: Time axis along which to filter.

    Returns:
        Bandpass-filtered EEG array of identical shape and type float64.
    """
    nyquist = 0.5 * fs
    if lowcut <= 0 or lowcut >= nyquist:
        raise ValueError(f"lowcut {lowcut} Hz must be between 0 and Nyquist ({nyquist} Hz).")
    if highcut <= lowcut or highcut >= nyquist:
        raise ValueError(f"highcut {highcut} Hz must be between lowcut ({lowcut} Hz) and Nyquist ({nyquist} Hz).")

    b, a = signal.butter(N=order, Wn=[lowcut, highcut], btype="bandpass", fs=fs)
    filtered = signal.filtfilt(b, a, data, axis=axis)
    return filtered

def resample_signal(
    data: np.ndarray,
    orig_fs: float = 500.0,
    target_fs: float = 250.0,
    axis: int = 0
) -> np.ndarray:
    """
    Resample multichannel EEG timeseries using polyphase filtering to prevent aliasing.

    Parameters:
        data: Multichannel EEG array (n_samples, n_channels).
        orig_fs: Source sampling rate in Hz (default 500.0).
        target_fs: Desired target sampling rate in Hz (default 250.0).
        axis: Time axis along which to resample.

    Returns:
        Resampled EEG array.
    """
    if orig_fs == target_fs:
        return data.copy()

    # Determine rational up/down factors
    int_orig = int(round(orig_fs * 10))
    int_target = int(round(target_fs * 10))
    g = gcd(int_target, int_orig)
    up = int_target // g
    down = int_orig // g

    resampled = signal.resample_poly(data, up=up, down=down, axis=axis)
    return resampled

def preprocess_continuous_eeg(
    data: np.ndarray,
    orig_fs: float = 500.0,
    target_fs: float = 250.0,
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    bandpass_order: int = 4
) -> Tuple[np.ndarray, float]:
    """
    Execute complete continuous filtering pipeline:
    1. 50 Hz powerline notch filter (zero-phase)
    2. 0.5 - 45 Hz Butterworth bandpass filter (zero-phase)
    3. Polyphase anti-aliased resampling to target_fs

    Parameters:
        data: Raw EEG array of shape (n_samples, n_channels).
        orig_fs: Original acquisition rate in Hz (500.0).
        target_fs: Target rate in Hz (250.0).
        notch_freq: Notch center frequency (50.0 Hz).
        notch_q: Notch quality factor (30.0).
        lowcut: Highpass cutoff (0.5 Hz).
        highcut: Lowpass cutoff (45.0 Hz).
        bandpass_order: Butterworth filter order (4).

    Returns:
        Tuple of (preprocessed_array, current_sampling_rate).
    """
    # 1. Notch filter
    out = apply_notch_filter(data, fs=orig_fs, freq=notch_freq, q=notch_q, axis=0)

    # 2. Bandpass filter
    out = apply_bandpass_filter(out, fs=orig_fs, lowcut=lowcut, highcut=highcut, order=bandpass_order, axis=0)

    # 3. Resample
    if target_fs != orig_fs:
        out = resample_signal(out, orig_fs=orig_fs, target_fs=target_fs, axis=0)
        current_fs = target_fs
    else:
        current_fs = orig_fs

    return out, current_fs
