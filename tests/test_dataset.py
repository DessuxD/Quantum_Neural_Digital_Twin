"""
test_dataset.py
===============
Unit tests for EEGDigitalTwinDataset and DataLoader functionality.
"""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.data.dataset import EEGDigitalTwinDataset, get_dataloaders

class TestEEGDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        manifest_rows = []
        for split, subjects, count in (("train", (0, 1), 10), ("val", (24,), 3), ("test", (30,), 3)):
            windows = []
            targets = []
            subject_ids = []
            for sid in subjects:
                subject_windows = np.full((count, 500, 19), sid, dtype=np.float32)
                windows.append(subject_windows)
                targets.append(subject_windows[1:])
                subject_ids.extend([sid] * count)
                for window_idx in range(count):
                    manifest_rows.append({
                        "subject_id": f"s{sid:02d}",
                        "subject_numeric_id": sid,
                        "window_idx": window_idx,
                        "split": split,
                    })
            np.save(self.data_dir / f"{split}_windows.npy", np.concatenate(windows))
            np.save(self.data_dir / f"{split}_next_windows.npy", np.concatenate(targets))
            np.save(self.data_dir / f"{split}_subject_ids.npy", np.asarray(subject_ids, dtype=np.int32))
        pd.DataFrame(manifest_rows).to_csv(self.data_dir / "manifest.csv", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_predictive_mode_loading(self):
        train_ds = EEGDigitalTwinDataset(split="train", data_dir=self.data_dir, mode="predictive", layout="NTC")
        self.assertGreater(len(train_ds), 0)
        x, y, sub_id = train_ds[0]
        self.assertEqual(x.shape, (500, 19))
        self.assertEqual(y.shape, (500, 19))
        self.assertIsInstance(sub_id.item(), int)

    def test_eegnet_layout(self):
        val_ds = EEGDigitalTwinDataset(split="val", data_dir=self.data_dir, mode="biometric", layout="N1CT")
        x, sub_id = val_ds[0]
        # Should be (1, channels, time) = (1, 19, 500)
        self.assertEqual(x.shape, (1, 19, 500))

    def test_dataloaders_batching(self):
        train_loader, val_loader, test_loader = get_dataloaders(data_dir=self.data_dir, batch_size=16, mode="predictive", layout="NTC")
        batch_x, batch_y, batch_sub = next(iter(train_loader))
        self.assertEqual(batch_x.shape, (16, 500, 19))
        self.assertEqual(batch_y.shape, (16, 500, 19))
        self.assertEqual(batch_sub.shape, (16,))

if __name__ == "__main__":
    unittest.main()
