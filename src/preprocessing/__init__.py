"""
src/preprocessing
=================
EEG Signal Preprocessing Module for the Quantum Neural Digital Twin.
Includes zero-phase notch and bandpass filters, anti-aliased resampling,
robust normalization, temporal sliding-window segmentation, and leak-free
subject-wise dataset splitting.
"""

from .filters import apply_notch_filter, apply_bandpass_filter, resample_signal, preprocess_continuous_eeg
from .normalization import normalize_channels, RobustScaler1D
from .segmentation import create_sliding_windows, extract_subject_windows


def run_preprocessing_pipeline(*args, **kwargs):
    """Lazily import and run the pipeline without interfering with ``python -m``."""
    from .pipeline import run_preprocessing_pipeline as run_pipeline

    return run_pipeline(*args, **kwargs)

__all__ = [
    "apply_notch_filter",
    "apply_bandpass_filter",
    "resample_signal",
    "preprocess_continuous_eeg",
    "normalize_channels",
    "RobustScaler1D",
    "create_sliding_windows",
    "extract_subject_windows",
    "run_preprocessing_pipeline",
]
