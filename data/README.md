# Data files

Raw EEG recordings are not distributed in this repository. The original [PhysioNet EDF release](https://physionet.org/content/eegmat/1.0.0/) is marked ODC-BY 1.0, while the Kaggle [converted CSV distribution](https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset) currently lists its license as unknown. The existing preprocessing pipeline reads the 36 headerless `s00.csv`–`s35.csv` files from top-level `archive/`; ensure you have permission to use or derive them before obtaining or sharing those files.

The files in `processed/` are generated with:

```bash
python -m src.preprocessing.pipeline --archive-dir archive --output-dir data/processed
```

Processed `.npy` arrays and continuous recordings are excluded from Git because they are reproducible and comparatively large. `metadata/` contains dataset and recording summaries; machine-specific paths have been replaced with repository-relative paths. The two EDF files formerly found under `data/raw/mental_arithmetic/` are not used by the CSV pipeline and are excluded from version control.
