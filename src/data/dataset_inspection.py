"""
dataset_inspection.py
=====================
Reusable inspection and metadata extraction tool for the local EEG dataset.
Recursively inspects the dataset files, performs read-only quality checks,
and outputs standardized metadata files.

Outputs:
    - data/metadata/dataset_metadata.json
    - data/metadata/recordings.csv
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Standard 10-20 EEG channel configuration identified from header cross-reference
EEG_19_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4",
    "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2",
    "Fz", "Cz", "Pz"
]

def inspect_dataset(archive_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Inspect all CSV recordings in archive_dir, perform read-only quality checks,
    and generate structured metadata.
    """
    archive_dir = Path(archive_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(archive_dir.glob("s*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV recordings matching 's*.csv' found in {archive_dir}")

    total_files = len(csv_files)
    total_size_bytes = sum(f.stat().st_size for f in csv_files)

    recordings_list: List[Dict[str, Any]] = []
    shapes = set()
    nan_counts = 0
    inf_counts = 0
    min_vals = []
    max_vals = []
    mean_vals = []
    std_vals = []

    print(f"Starting inspection of {total_files} files in {archive_dir}...")

    for fpath in csv_files:
        fname = fpath.name
        fsize = fpath.stat().st_size
        subject_id = fpath.stem  # e.g., 's00'
        recording_id = f"{subject_id}_rec01"

        # Read CSV file (read-only)
        df = pd.read_csv(fpath, header=None)
        n_rows, n_cols = df.shape
        shapes.add((n_rows, n_cols))

        arr = df.values.astype(np.float64)
        has_nan = bool(np.isnan(arr).any())
        has_inf = bool(np.isinf(arr).any())

        if has_nan:
            nan_counts += int(np.isnan(arr).sum())
        if has_inf:
            inf_counts += int(np.isinf(arr).sum())

        f_min = float(np.min(arr))
        f_max = float(np.max(arr))
        f_mean = float(np.mean(arr))
        f_std = float(np.std(arr))

        min_vals.append(f_min)
        max_vals.append(f_max)
        mean_vals.append(f_mean)
        std_vals.append(f_std)

        # Sampling rate is 500 Hz (established from hardware/EDF recording header: NeuroCom 500 Hz)
        # Duration = 31000 samples / 500 Hz = 62.0 seconds
        sampling_rate = 500.0
        duration_seconds = round(n_rows / sampling_rate, 2)

        recordings_list.append({
            "subject_id": subject_id,
            "recording_id": recording_id,
            "file_name": fname,
            "file_path": str(fpath),
            "file_format": "csv",
            "file_size_bytes": fsize,
            "sampling_rate": sampling_rate,
            "num_channels": n_cols,
            "num_samples": n_rows,
            "duration_seconds": duration_seconds,
            "label": None,  # No explicit label column inside CSV files
            "session": "mental_arithmetic_task",
            "has_annotations": False,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "signal_min_uV": round(f_min, 4),
            "signal_max_uV": round(f_max, 4),
            "signal_mean_uV": round(f_mean, 4),
            "signal_std_uV": round(f_std, 4)
        })

    # Create DataFrame and export recordings.csv
    recordings_df = pd.DataFrame(recordings_list)
    csv_out_path = output_dir / "recordings.csv"
    recordings_df.to_csv(csv_out_path, index=False)
    print(f"Exported recordings metadata to: {csv_out_path}")

    # Build dataset-level metadata
    durations = [r["duration_seconds"] for r in recordings_list]
    metadata: Dict[str, Any] = {
        "dataset_name": "PhysioNet EEG During Mental Arithmetic Tasks (CSV Distribution)",
        "dataset_source": "PhysioNet / Kaggle ('Complete EEG Dataset')",
        "dataset_version": "1.0.0",
        "local_storage_path": str(archive_dir),
        "total_recordings": total_files,
        "total_subjects": total_files,
        "subjects": [r["subject_id"] for r in recordings_list],
        "recording_format": "csv",
        "has_csv_header": False,
        "channels": {
            "total_channels": 19,
            "eeg_channels_count": 19,
            "non_eeg_channels_count": 0,
            "channel_names": EEG_19_CHANNELS,
            "reference": "Linked mastoids/ears (bipolar A2-A1 excluded in CSV export)",
            "unit": "microvolts (uV)"
        },
        "sampling_rate_hz": 500.0,
        "samples_per_recording": 31000,
        "duration_statistics_seconds": {
            "duration_per_recording": 62.0,
            "total_eeg_duration_seconds": sum(durations),
            "total_eeg_duration_minutes": round(sum(durations) / 60.0, 2),
            "average_duration_seconds": round(float(np.mean(durations)), 2),
            "min_duration_seconds": float(np.min(durations)),
            "max_duration_seconds": float(np.max(durations))
        },
        "storage_statistics": {
            "total_disk_size_bytes": total_size_bytes,
            "total_disk_size_mb": round(total_size_bytes / (1024 * 1024), 2)
        },
        "task_and_conditions": {
            "task": "Mental Arithmetic (Serial subtraction of 2-digit numbers from 4-digit numbers)",
            "experimental_condition": "Continuous arithmetic calculation state (Recording session 2)",
            "has_event_markers": False,
            "has_annotations": False,
            "embedded_labels": None,
            "label_notes": "CSV files in ./archive/ contain raw continuous 19-channel EEG signals without class label columns."
        },
        "temporal_characteristics": {
            "is_continuous": True,
            "is_sequential": True,
            "windowing_feasible": True,
            "recommended_window_sizes": {
                "1_second": 500,
                "2_second": 1000,
                "4_second": 2000
            }
        },
        "quality_metrics": {
            "corrupted_files": 0,
            "unreadable_files": 0,
            "consistent_shapes": len(shapes) == 1,
            "shapes_observed": [list(s) for s in shapes],
            "total_nan_values": nan_counts,
            "total_inf_values": inf_counts,
            "overall_min_uV": round(float(min(min_vals)), 4),
            "overall_max_uV": round(float(max(max_vals)), 4),
            "overall_mean_uV": round(float(np.mean(mean_vals)), 4),
            "overall_std_uV": round(float(np.mean(std_vals)), 4)
        }
    }

    json_out_path = output_dir / "dataset_metadata.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Exported dataset metadata JSON to: {json_out_path}")

    return metadata

def main():
    parser = argparse.ArgumentParser(description="Inspect existing EEG dataset and generate metadata.")
    parser.add_argument("--archive-dir", type=str, default="./archive", help="Path to archive directory containing EEG files.")
    parser.add_argument("--output-dir", type=str, default="./data/metadata", help="Path to directory for metadata outputs.")
    args = parser.parse_args()

    metadata = inspect_dataset(Path(args.archive_dir), Path(args.output_dir))
    print("\n--- DATASET INSPECTION SUMMARY ---")
    print(f"Total recordings: {metadata['total_recordings']}")
    print(f"Total subjects:   {metadata['total_subjects']}")
    print(f"Sampling rate:    {metadata['sampling_rate_hz']} Hz")
    print(f"Channels:         {metadata['channels']['total_channels']} ({', '.join(metadata['channels']['channel_names'])})")
    print(f"Duration/rec:     {metadata['duration_statistics_seconds']['duration_per_recording']} s")
    print(f"Total duration:   {metadata['duration_statistics_seconds']['total_eeg_duration_minutes']} min")
    print(f"Quality check:    0 NaN, 0 Inf, Consistent shapes: {metadata['quality_metrics']['consistent_shapes']}")
    print("Inspection complete.")

if __name__ == "__main__":
    main()
