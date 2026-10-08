"""
pipeline.py
===========
End-to-end EEG signal preprocessing pipeline for the Quantum Neural Digital Twin.
Executes:
1. Zero-phase notch filtering (50 Hz powerline)
2. Butterworth bandpass filtering (0.5 - 45 Hz)
3. Polyphase anti-aliasing decimation / resampling (500 Hz -> 250 Hz)
4. Robust / Z-score normalization per channel
5. Temporal sliding window segmentation (2.0s window, 1.0s stride)
6. Autoregressive sequence transition pair creation (X_t -> X_{t+1})
7. Strict subject-wise train/val/test partitioning (Zero subject leakage)
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

try:
    from .filters import preprocess_continuous_eeg
    from .normalization import normalize_channels
    from .segmentation import extract_subject_windows
except ImportError:
    from src.preprocessing.filters import preprocess_continuous_eeg
    from src.preprocessing.normalization import normalize_channels
    from src.preprocessing.segmentation import extract_subject_windows

def run_preprocessing_pipeline(
    archive_dir: Path = Path("./archive"),
    output_dir: Path = Path("./data/processed"),
    orig_fs: float = 500.0,
    target_fs: float = 250.0,
    notch_freq: float = 50.0,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    window_sec: float = 2.0,
    stride_sec: float = 1.0,
    norm_method: str = "zscore",
    n_train_subjects: int = 24,
    n_val_subjects: int = 6,
    n_test_subjects: int = 6,
    save_continuous: bool = True
) -> Dict[str, Any]:
    """
    Execute full preprocessing workflow across all 36 subjects.
    """
    archive_dir = Path(archive_dir).resolve()
    output_dir = Path(output_dir).resolve()
    continuous_dir = output_dir / "continuous"
    windows_dir = output_dir / "windows"

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_continuous:
        continuous_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(archive_dir.glob("s*.csv")))
    total_subjects = len(csv_files)
    if total_subjects != (n_train_subjects + n_val_subjects + n_test_subjects):
        raise ValueError(
            f"Total subjects ({total_subjects}) does not match train ({n_train_subjects}) + "
            f"val ({n_val_subjects}) + test ({n_test_subjects})."
        )

    print(f"=== Starting EEG Preprocessing Pipeline ===")
    print(f"Source Directory:     {archive_dir} ({total_subjects} subjects)")
    print(f"Output Directory:     {output_dir}")
    print(f"Sampling Rate:        {orig_fs} Hz -> {target_fs} Hz")
    print(f"Filter Band:          Notch={notch_freq} Hz, Bandpass={lowcut} - {highcut} Hz")
    print(f"Temporal Windowing:   {window_sec}s window, {stride_sec}s stride (overlap={(1 - stride_sec/window_sec)*100:.0f}%)")
    print(f"Subject Partition:    Train={n_train_subjects}, Val={n_val_subjects}, Test={n_test_subjects}")

    # Determine subject-wise split
    subject_names = [f.stem for f in csv_files]
    train_subjects = subject_names[:n_train_subjects]
    val_subjects = subject_names[n_train_subjects : n_train_subjects + n_val_subjects]
    test_subjects = subject_names[n_train_subjects + n_val_subjects :]

    split_map = {}
    for s in train_subjects:
        split_map[s] = "train"
    for s in val_subjects:
        split_map[s] = "val"
    for s in test_subjects:
        split_map[s] = "test"

    split_data = {
        "train": {"X": [], "Y_next": [], "subject_ids": [], "meta": []},
        "val": {"X": [], "Y_next": [], "subject_ids": [], "meta": []},
        "test": {"X": [], "Y_next": [], "subject_ids": [], "meta": []}
    }

    all_window_metadata = []

    for idx, fpath in enumerate(csv_files):
        s_id = fpath.stem
        split = split_map[s_id]

        # 1. Load raw data
        raw_df = pd.read_csv(fpath, header=None)
        raw_arr = raw_df.values.astype(np.float64)

        # 2. Filtering & Resampling
        filtered_arr, current_fs = preprocess_continuous_eeg(
            data=raw_arr,
            orig_fs=orig_fs,
            target_fs=target_fs,
            notch_freq=notch_freq,
            lowcut=lowcut,
            highcut=highcut
        )

        # 3. Channel Normalization
        norm_arr, norm_stats = normalize_channels(filtered_arr, method=norm_method, axis=0)

        # Optional: save clean continuous array
        if save_continuous:
            cont_path = continuous_dir / f"{s_id}_clean.npy"
            np.save(cont_path, norm_arr.astype(np.float32))

        # 4. Temporal sliding window segmentation & transition pair extraction
        windows, y_next, df_meta = extract_subject_windows(
            continuous_eeg=norm_arr,
            fs=current_fs,
            window_sec=window_sec,
            stride_sec=stride_sec,
            subject_id=s_id,
            subject_numeric_id=idx
        )

        df_meta["split"] = split
        all_window_metadata.append(df_meta)

        split_data[split]["X"].append(windows)
        split_data[split]["Y_next"].append(y_next)
        split_data[split]["subject_ids"].append(np.full(len(windows), idx, dtype=np.int32))

        if (idx + 1) % 6 == 0 or (idx + 1) == total_subjects:
            print(f"Processed subject {idx + 1}/{total_subjects} ({s_id}) -> {len(windows)} windows")

    # Stack and save partitions
    manifest_df = pd.concat(all_window_metadata, ignore_index=True)
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nSaved global window manifest to: {manifest_path}")

    summary_stats = {
        "pipeline_version": "1.0.0",
        "original_sampling_rate_hz": orig_fs,
        "target_sampling_rate_hz": target_fs,
        "filter_parameters": {
            "notch_frequency_hz": notch_freq,
            "notch_q": 30.0,
            "bandpass_lowcut_hz": lowcut,
            "bandpass_highcut_hz": highcut,
            "bandpass_order": 4,
            "filter_type": "zero-phase forward-backward Butterworth"
        },
        "normalization": {
            "method": norm_method,
            "scope": "per-subject channel-wise"
        },
        "windowing": {
            "window_duration_seconds": window_sec,
            "stride_duration_seconds": stride_sec,
            "window_samples": int(round(window_sec * target_fs)),
            "overlap_percentage": round((1.0 - stride_sec / window_sec) * 100, 1)
        },
        "subject_splits": {
            "train": {"count": len(train_subjects), "subjects": train_subjects},
            "val": {"count": len(val_subjects), "subjects": val_subjects},
            "test": {"count": len(test_subjects), "subjects": test_subjects}
        },
        "split_shapes": {}
    }

    for split in ["train", "val", "test"]:
        X_split = np.vstack(split_data[split]["X"])
        Y_next_split = np.vstack(split_data[split]["Y_next"])
        sub_ids = np.concatenate(split_data[split]["subject_ids"])

        np.save(output_dir / f"{split}_windows.npy", X_split)
        np.save(output_dir / f"{split}_next_windows.npy", Y_next_split)
        np.save(output_dir / f"{split}_subject_ids.npy", sub_ids)

        summary_stats["split_shapes"][split] = {
            "windows_shape": list(X_split.shape),
            "next_windows_shape": list(Y_next_split.shape),
            "subject_ids_shape": list(sub_ids.shape),
            "num_windows": int(X_split.shape[0]),
            "num_samples_per_window": int(X_split.shape[1]),
            "num_channels": int(X_split.shape[2])
        }

        print(f"Split [{split.upper()}]: Windows={X_split.shape}, NextWindows={Y_next_split.shape}, SubIDs={sub_ids.shape}")

    # Save summary metadata
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=4)
    print(f"Saved preprocessing summary to: {summary_path}")
    print("=== EEG Preprocessing Completed Successfully ===")

    return summary_stats

def main():
    parser = argparse.ArgumentParser(description="Run EEG Preprocessing Pipeline for Quantum Neural Digital Twin.")
    parser.add_argument("--archive-dir", type=str, default="./archive", help="Path to archive dir.")
    parser.add_argument("--output-dir", type=str, default="./data/processed", help="Path to output processed dir.")
    parser.add_argument("--target-fs", type=float, default=250.0, help="Target sampling rate (Hz).")
    parser.add_argument("--notch-freq", type=float, default=50.0, help="Notch filter center frequency (Hz).")
    parser.add_argument("--lowcut", type=float, default=0.5, help="Bandpass lower cutoff (Hz).")
    parser.add_argument("--highcut", type=float, default=45.0, help="Bandpass upper cutoff (Hz).")
    parser.add_argument("--window-sec", type=float, default=2.0, help="Window duration in seconds.")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="Window stride in seconds.")
    parser.add_argument("--norm-method", type=str, default="zscore", help="Normalization method: zscore or robust.")
    args = parser.parse_args()

    run_preprocessing_pipeline(
        archive_dir=Path(args.archive_dir),
        output_dir=Path(args.output_dir),
        target_fs=args.target_fs,
        notch_freq=args.notch_freq,
        lowcut=args.lowcut,
        highcut=args.highcut,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        norm_method=args.norm_method
    )

if __name__ == "__main__":
    main()
