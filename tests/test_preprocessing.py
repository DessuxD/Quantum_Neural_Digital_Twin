"""
test_preprocessing.py
=====================
Unit tests for preprocessing filters, normalizers, and segmentation routines.
"""

import unittest
import numpy as np
from src.preprocessing.filters import apply_notch_filter, apply_bandpass_filter, resample_signal, preprocess_continuous_eeg
from src.preprocessing.normalization import normalize_channels, RobustScaler1D
from src.preprocessing.segmentation import create_sliding_windows, extract_subject_windows

class TestEEGPreprocessing(unittest.TestCase):

    def setUp(self):
        # Generate synthetic 10-second 19-channel EEG-like signal at 500 Hz
        self.fs = 500.0
        self.duration = 10.0
        self.n_samples = int(self.fs * self.duration)
        self.n_channels = 19
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)

        # Superposition of 10 Hz alpha wave, 50 Hz line hum, 0.1 Hz DC drift, and Gaussian noise
        signal_base = np.sin(2 * np.pi * 10.0 * t) + 0.5 * np.sin(2 * np.pi * 50.0 * t) + 2.0 * np.sin(2 * np.pi * 0.1 * t)
        noise = 0.1 * np.random.randn(self.n_samples)
        single_ch = signal_base + noise
        self.raw_data = np.repeat(single_ch[:, np.newaxis], self.n_channels, axis=1)

    def test_notch_filter(self):
        filtered = apply_notch_filter(self.raw_data, fs=self.fs, freq=50.0, q=30.0)
        self.assertEqual(filtered.shape, self.raw_data.shape)
        self.assertFalse(np.isnan(filtered).any())
        self.assertFalse(np.isinf(filtered).any())

    def test_bandpass_filter(self):
        filtered = apply_bandpass_filter(self.raw_data, fs=self.fs, lowcut=0.5, highcut=45.0, order=4)
        self.assertEqual(filtered.shape, self.raw_data.shape)
        self.assertFalse(np.isnan(filtered).any())
        # DC drift (mean) should be centered near 0
        self.assertAlmostEqual(float(np.mean(filtered)), 0.0, places=1)

    def test_resample(self):
        target_fs = 250.0
        resampled = resample_signal(self.raw_data, orig_fs=self.fs, target_fs=target_fs)
        expected_samples = int(self.n_samples * target_fs / self.fs)
        self.assertEqual(resampled.shape, (expected_samples, self.n_channels))
        self.assertFalse(np.isnan(resampled).any())

    def test_normalization_zscore(self):
        normed, stats = normalize_channels(self.raw_data, method="zscore")
        self.assertEqual(normed.shape, self.raw_data.shape)
        means = np.mean(normed, axis=0)
        stds = np.std(normed, axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)
        np.testing.assert_allclose(stds, 1.0, atol=1e-5)

    def test_normalization_robust(self):
        normed, stats = normalize_channels(self.raw_data, method="robust")
        self.assertEqual(normed.shape, self.raw_data.shape)
        medians = np.median(normed, axis=0)
        np.testing.assert_allclose(medians, 0.0, atol=1e-5)

    def test_sliding_windows(self):
        # 10s signal resampled to 250 Hz = 2500 samples
        resampled = resample_signal(self.raw_data, orig_fs=self.fs, target_fs=250.0)
        windows, meta = create_sliding_windows(resampled, fs=250.0, window_sec=2.0, stride_sec=1.0)
        # Expected windows: (2500 - 500)/250 + 1 = 9 windows
        self.assertEqual(windows.shape, (9, 500, self.n_channels))
        self.assertEqual(len(meta), 9)
        self.assertEqual(meta[0]["start_time_sec"], 0.0)
        self.assertEqual(meta[0]["end_time_sec"], 2.0)
        self.assertEqual(meta[1]["start_time_sec"], 1.0)
        self.assertEqual(meta[1]["end_time_sec"], 3.0)

    def test_extract_subject_windows_transition_pairs(self):
        resampled = resample_signal(self.raw_data, orig_fs=self.fs, target_fs=250.0)
        windows, y_next, df_meta = extract_subject_windows(resampled, fs=250.0, window_sec=2.0, stride_sec=1.0)
        self.assertEqual(len(windows), 9)
        self.assertEqual(len(y_next), 8)
        # Next-window transition: y_next[0] should match windows[1]
        np.testing.assert_allclose(y_next[0], windows[1])

if __name__ == "__main__":
    unittest.main()
