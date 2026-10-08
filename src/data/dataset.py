"""
dataset.py
==========
PyTorch Dataset and DataLoader wrappers for preprocessed EEG windows.
Supports predictive next-window forecasting, autoencoder reconstruction,
and subject biometric identification modes for the Quantum Neural Digital Twin.
"""

from pathlib import Path
from typing import Tuple, Optional, Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDigitalTwinDataset(Dataset):
    """
    PyTorch Dataset for EEG sliding windows.

    Modes:
        - 'predictive': Returns (X_t, X_{t+1}, subject_id) for autoregressive future-state forecasting.
        - 'reconstruction': Returns (X_t, X_t, subject_id) for autoencoder state compression.
        - 'biometric': Returns (X_t, subject_id) for biometric representation learning.

    Layout:
        - 'NTC': (batch, num_timesteps, num_channels) -> default
        - 'NCT': (batch, num_channels, num_timesteps) -> for 1D temporal convs / BiLSTM
        - 'N1CT': (batch, 1, num_channels, num_timesteps) -> standard EEGNet format
    """
    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        data_dir: Path = Path("./data/processed"),
        mode: Literal["predictive", "reconstruction", "biometric"] = "predictive",
        layout: Literal["NTC", "NCT", "N1CT"] = "NTC"
    ):
        data_dir = Path(data_dir).resolve()
        self.split = split
        self.mode = mode
        self.layout = layout

        if mode == "predictive":
            # For predictive mode, we need aligned (X_t, X_{t+1}) pairs
            # next_windows contains X_{t+1} for all t in [0, N_sub - 2]
            # To match lengths, we truncate X to the same number of samples
            y_next = np.load(data_dir / f"{split}_next_windows.npy")  # (M, T, C)
            x_curr = np.load(data_dir / f"{split}_windows.npy")       # (N, T, C)
            # Match count per subject (each subject has 61 windows, 60 transition pairs)
            # Reconstruct pairing accurately:
            # We filter windows that have a valid subsequent window
            manifest_path = data_dir / "manifest.csv"
            if manifest_path.exists():
                import pandas as pd
                manifest = pd.read_csv(manifest_path)
                split_manifest = manifest[manifest["split"] == split].reset_index(drop=True)
                # Next window exists if window_idx is not the maximum for that subject
                max_indices = split_manifest.groupby("subject_id")["window_idx"].transform("max")
                has_next = split_manifest["window_idx"] < max_indices
                x_curr = x_curr[has_next.values]
                sub_ids = split_manifest.loc[has_next.values, "subject_numeric_id"].values
            else:
                x_curr = x_curr[:len(y_next)]
                sub_ids = np.load(data_dir / f"{split}_subject_ids.npy")[:len(y_next)]

            self.X = torch.from_numpy(x_curr).float()
            self.Y = torch.from_numpy(y_next).float()
            self.subject_ids = torch.from_numpy(sub_ids).long()

        else:
            x_arr = np.load(data_dir / f"{split}_windows.npy")
            sub_arr = np.load(data_dir / f"{split}_subject_ids.npy")
            self.X = torch.from_numpy(x_arr).float()
            self.Y = self.X.clone()
            self.subject_ids = torch.from_numpy(sub_arr).long()

    def _format_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # Input tensor is (T, C)
        if self.layout == "NTC":
            return tensor  # (T, C)
        elif self.layout == "NCT":
            return tensor.transpose(0, 1)  # (C, T)
        elif self.layout == "N1CT":
            return tensor.transpose(0, 1).unsqueeze(0)  # (1, C, T)
        return tensor

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self._format_tensor(self.X[idx])
        sub_id = self.subject_ids[idx]

        if self.mode == "predictive":
            y = self._format_tensor(self.Y[idx])
            return x, y, sub_id
        elif self.mode == "reconstruction":
            y = self._format_tensor(self.Y[idx])
            return x, y, sub_id
        elif self.mode == "biometric":
            return x, sub_id

def get_dataloaders(
    data_dir: Path = Path("./data/processed"),
    batch_size: int = 32,
    mode: Literal["predictive", "reconstruction", "biometric"] = "predictive",
    layout: Literal["NTC", "NCT", "N1CT"] = "NTC",
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct train, val, and test PyTorch DataLoaders.
    """
    train_ds = EEGDigitalTwinDataset(split="train", data_dir=data_dir, mode=mode, layout=layout)
    val_ds = EEGDigitalTwinDataset(split="val", data_dir=data_dir, mode=mode, layout=layout)
    test_ds = EEGDigitalTwinDataset(split="test", data_dir=data_dir, mode=mode, layout=layout)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
