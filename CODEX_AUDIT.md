# Repository Audit

Audit and follow-up performed 2026-10-08 against files in the workspace. New 4Q/2L and 6Q/1L experiments were trained; no files under `archive/` were modified.

## Structure and implementation

- `src/preprocessing/filters.py`, `normalization.py`, `segmentation.py`, and `pipeline.py`: notch/bandpass filtering, resampling, normalization, windowing and split generation.
- `src/data/dataset.py`: PyTorch dataset and loaders; `dataset_inspection.py`: raw CSV inspection and metadata generation.
- `src/models/eegnet.py`, `bilstm.py`, `dimension_reduction.py`, `quantum_circuit.py`, `hybrid_twin.py`, and `classical_baseline.py`: encoders, quantum angle mapping/VQC, hybrid and classical forecast models.
- `src/training/train_digital_twin.py` and `extended_training.py`: initial and extended training.
- `src/evaluation/visualize.py`: plots. Added `src/evaluation/evaluate_digital_twin.py` for reproducible held-out forecasting and latent trajectory evaluation.
- `tests/`: preprocessing, dataset, and model tests. Dataset tests now use synthetic temporary inputs so they work without local processed arrays.
- Reports, processed metadata, checkpoints, histories and figures are present. README, requirements, ignore rules, CI, this audit, and the final project report were initially missing.

## Verified complete

- Source archive contains 36 `s00.csv`–`s35.csv` files; no source files were edited.
- Full preprocessing was regenerated into a temporary directory and all nine train/validation/test arrays compared bit-for-bit equal to the existing arrays. Shapes and finite-value checks passed.
- Regenerated manifest contains 2,196 windows; split subjects are 24/6/6 with no overlap. Pair counts are 1,440/360/360.
- Test suite passes 17/17 after making dataset tests self-contained.
- All four saved 20-epoch experiment checkpoints load and reproduce test metrics on 360 held-out transitions to within floating-point rounding.
- New 4Q/2L and 6Q/1L experiments completed using the existing training protocol and held-out split.
- The saved classical model scores better than the 4-qubit/1-layer model for MSE, MAE, and cosine similarity.
- Quantum circuit implementation: AngleEmbedding with Y rotations, trainable Rot gates, ring CNOT entanglement, Pauli-Z expectations for each qubit, and PennyLane TorchLayer autograd. The experiment uses 4 qubits and 1 variational layer.

## Findings and caveats

- `archive/` is the intended preprocessing input. `data/raw/mental_arithmetic/` contains two EDF files and is not used by this CSV pipeline.
- Existing PCA is a standalone utility and is not used in the model's learned feature-to-angle path. PCA variance claims do not describe the actual VQC input path.
- The hybrid experiment has 219,882 parameters; the classical model has 269,826. Their recurrent hidden sizes differ (24 vs. 32). The comparison is useful but not parameter matched.
- The older 5-epoch Phase 3 status/training reports describe a different model state and are superseded by the saved 20-epoch ablation records for comparison results.
- Extended training evaluation returns metric keys prefixed `val_` even when called on the test loader. The stored `test_metrics` fields therefore have legacy/misleading key names; their values reproduce the held-out test metrics. New evaluation output uses explicit metric names.
- Recomputed persistence baseline MSE is 2.0695 on current arrays; the previously quoted ~2.0061 is not reproduced. Training-target-mean baseline MSE reproduces at 1.0101.
- No fatigue labels exist. No disease, diagnostic, or clinical claims are supported.
- No `.git` directory exists at the workspace root. Branch, remotes, commit history, tracked files, and prior ignore policy cannot be inspected; commits/pushes were not possible.
- No project license is present; project code reuse terms remain unspecified.
- The official PhysioNet EDF dataset page marks its original data ODC-BY 1.0; the Kaggle converted CSV distribution used in this project lists its license as unknown. The existing CSV-only pipeline has no verified EDF-to-CSV conversion path, so users need an authorized CSV copy or a separately validated converter before fresh reproduction from EDF.

## Safe sharing guidance

Share source, tests, small metadata, histories, figures, and documentation after a secret review. Keep raw EEG, processed arrays, and checkpoint binaries out of Git by default. `archive/` is an external local input; `data/processed/` can be regenerated. Do not claim a GitHub-ready repository state until a Git root is initialized/located and the complete staged contents, size, secrets, and remote are reviewed.

## Follow-up completed

- Added a held-out evaluation script and JSON/figure outputs for all four trained models.
- Added README, dependency ranges, `.gitignore`, and GitHub Actions test workflow.
- Added final report and reconciled older documentation with verified experiment results and limitations.
