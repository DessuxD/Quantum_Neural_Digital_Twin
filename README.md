# Quantum Neural Digital Twin of the Human Brain

An EEG research prototype for learning EEG-derived latent states and forecasting the next 2-second EEG window. The model is a computational representation of EEG recordings; it is not a biological brain simulation, clinical system, or disease predictor. The quantum model is a hybrid comparator and is not expected to guarantee quantum advantage.

## Dataset

The project uses the PhysioNet EEG During Mental Arithmetic Tasks recordings (36 subjects, 19 channels, 500 Hz source sampling). The original [PhysioNet EDF release](https://physionet.org/content/eegmat/1.0.0/) is marked Open Data Commons Attribution License v1.0. The Kaggle page for the converted [CSV distribution](https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset) lists its license as **Unknown**. The current pipeline requires the 36 headerless `s00.csv`–`s35.csv` files; confirm your right to use or derive those CSVs before sharing them. Do not include the local raw files in this repository. The incomplete EDF files under `data/raw/mental_arithmetic/` are not used by the current pipeline. The recordings contain no fatigue labels.

## Where the project runs

GitHub hosts the source code and runs the lightweight unit tests in GitHub Actions. The EEG recordings, processed arrays, and trained checkpoint files are intentionally not in Git, so GitHub Actions and a fresh clone cannot run the full evaluation until you provide the data locally and train or otherwise obtain the checkpoints. The easiest reproducible path is to clone the repository and run it on the computer that has the dataset. GitHub Codespaces can also run it, but you must securely copy the data into the Codespace first; it is not loaded automatically from your PC.

## Setup (Windows PowerShell)

Clone the GitHub repository using GitHub Desktop or:

```powershell
git clone https://github.com/DessuxD/Quantum_Neural_Digital_Twin.git
cd Quantum_Neural_Digital_Twin
```

Use Python 3.11 from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, create and activate the environment with `python3 -m venv .venv` and `source .venv/bin/activate`, then install the requirements as above.

## Put the dataset in place

The preprocessing code expects 36 headerless CSV files named `s00.csv` through `s35.csv`. Put them in a local `archive/` folder at the repository root. That folder is gitignored, so these files stay on your computer and will not appear in GitHub Desktop's commit list. For example, if your files are in `D:\EEG\archive`, you can pass that directory directly to preprocessing instead of copying them. Do not commit or share the CSVs unless their redistribution rights are confirmed. The two EDF files under `data/raw/mental_arithmetic/` are incomplete for this pipeline and are not used.

Check that the expected files are present:

```powershell
(Get-ChildItem .\archive\s*.csv -File).Count
```

This should print `36`. If the dataset is elsewhere, use its full path in the preprocessing command below. In Git Bash or macOS/Linux, use `ls archive/s*.csv | wc -l`.

## Reproduce preprocessing

```powershell
python -m src.preprocessing.pipeline --archive-dir archive --output-dir data/processed
```

For a dataset stored elsewhere, for example:

```powershell
python -m src.preprocessing.pipeline --archive-dir "D:\EEG\archive" --output-dir data/processed
```

The pipeline filters at 50 Hz and 0.5–45 Hz, resamples 500 Hz to 250 Hz, normalizes per subject/channel, and creates 500-sample windows with 250-sample stride. Subjects are split by filename order: `s00–s23` train, `s24–s29` validation, `s30–s35` test. Generated processed arrays are excluded from Git; regenerate them from the separately obtained CSV data.

## Run tests, train, and evaluate

The tests use generated temporary data and do not need the EEG dataset. GitHub Actions runs this same test suite automatically when code is pushed or a pull request is opened:

```bash
python -m unittest discover -s tests -v
```

After preprocessing has produced the arrays in `data/processed/`, train the classical baseline and all three quantum variants (this can take a long time, especially on CPU):

```powershell
python -m src.training.extended_training --mode ablation --epochs 20 --runs classical_baseline quantum_4q_1l quantum_4q_2l quantum_6q_1l
```

Training writes checkpoints under `results/ablations/`; those binary files are also gitignored. Once training succeeds, evaluate the saved models:

```powershell
python -m src.evaluation.evaluate_digital_twin
```

Evaluation compares the checkpoints with persistence and training-target-mean baselines on held-out subjects and writes `results/digital_twin_evaluation.json` plus a subject-level figure. A fresh clone cannot run evaluation until the processed arrays exist and the required checkpoints have been trained. Previously reported metrics in `FINAL_PROJECT_REPORT.md` are historical results; reproducing them requires the same data, split, dependencies, and training setup.

To run a single configuration or adjust training, use `python -m src.training.extended_training --help`. Simulated quantum circuits can be slow, especially during training. Keep `archive/`, generated `.npy` arrays, and checkpoint files local; their ignore rules are deliberate.

## Current result

On the 360 transitions from test subjects `s30–s35`, the classical baseline scores MSE/MAE/cosine 0.8458/0.7148/0.3860. Hybrid results are 4Q/1L 0.8771/0.7314/0.3561, 4Q/2L 0.8819/0.7327/0.3405, and 6Q/1L 0.8690/0.7265/0.3563. The classical baseline performs better across these metrics. Models are not parameter matched (269,826 vs. approximately 220k parameters) and use different BiLSTM sizes. Each configuration was run once on this split.

## Repository notes

Raw EEG, processed arrays, model checkpoint binaries, and Python caches are excluded by `.gitignore`. Small source files, metadata, test code, documentation, experiment histories, and figures are intended to be shareable after a separate data/licensing and secret review. No license is declared for this project; add one only after the project owner chooses the terms.
