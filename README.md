# Quantum Neural Digital Twin of the Human Brain

An EEG research prototype for learning EEG-derived latent states and forecasting the next 2-second EEG window. The model is a computational representation of EEG recordings; it is not a biological brain simulation, clinical system, or disease predictor. The quantum model is a hybrid comparator and is not expected to guarantee quantum advantage.

## Dataset

The project uses the PhysioNet EEG During Mental Arithmetic Tasks recordings (36 subjects, 19 channels, 500 Hz source sampling). The original [PhysioNet EDF release](https://physionet.org/content/eegmat/1.0.0/) is marked Open Data Commons Attribution License v1.0. The Kaggle page for the converted [CSV distribution](https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset) lists its license as **Unknown**. The current pipeline requires the 36 headerless `s00.csv`–`s35.csv` files; confirm your right to use or derive those CSVs before sharing them. Do not include the local raw files in this repository. The incomplete EDF files under `data/raw/mental_arithmetic/` are not used by the current pipeline. The recordings contain no fatigue labels.

## Setup

Use Python 3.11. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Reproduce preprocessing

```bash
python -m src.preprocessing.pipeline --archive-dir archive --output-dir data/processed
```

The pipeline filters at 50 Hz and 0.5–45 Hz, resamples 500 Hz to 250 Hz, normalizes per subject/channel, and creates 500-sample windows with 250-sample stride. Subjects are split by filename order: `s00–s23` train, `s24–s29` validation, `s30–s35` test. The checked-in processed arrays are excluded from Git; regenerate them from the separately obtained CSV data.

## Tests and evaluation

```bash
python -m unittest discover -s tests -v
python -m src.evaluation.evaluate_digital_twin
```

The evaluation loads the saved 20-epoch classical, 4Q/1L, 4Q/2L, and 6Q/1L checkpoints, compares them with persistence and training-target-mean baselines on the held-out subjects, and writes `results/digital_twin_evaluation.json` plus a subject-level figure. This is inference only; results are documented in `FINAL_PROJECT_REPORT.md`.

To retrain an experiment, see the argument options in `src/training/extended_training.py`. Simulated quantum circuits can be slow, especially during training.

## Current result

On the 360 transitions from test subjects `s30–s35`, the classical baseline scores MSE/MAE/cosine 0.8458/0.7148/0.3860. Hybrid results are 4Q/1L 0.8771/0.7314/0.3561, 4Q/2L 0.8819/0.7327/0.3405, and 6Q/1L 0.8690/0.7265/0.3563. The classical baseline performs better across these metrics. Models are not parameter matched (269,826 vs. approximately 220k parameters) and use different BiLSTM sizes. Each configuration was run once on this split.

## Repository notes

Raw EEG, processed arrays, model checkpoint binaries, and Python caches are excluded by `.gitignore`. Small source files, metadata, test code, documentation, experiment histories, and figures are intended to be shareable after a separate data/licensing and secret review. No license is declared for this project; add one only after the project owner chooses the terms.
