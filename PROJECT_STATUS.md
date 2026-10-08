# Project Status: Quantum Neural Digital Twin

## Current state

The preprocessing pipeline, classical baseline, and 4-qubit/1-layer hybrid comparator are implemented. Existing 20-epoch checkpoints have been evaluated on held-out subjects. This is a research prototype for EEG-derived latent representations and next-window forecasting, not a biological brain simulation or clinical system.

## Dataset and preprocessing

- PhysioNet EEG During Mental Arithmetic Tasks CSV distribution: 36 subjects, 19 channels, 500 Hz source sampling.
- The CSVs under `archive/` are the active source. The two EDF files under `data/raw/mental_arithmetic/` are not used by the current pipeline.
- Filter: 50 Hz notch and 0.5–45 Hz bandpass; resample to 250 Hz; per-subject/channel z-score.
- Windowing: 2 seconds (500 samples), 1-second stride.
- Split: train `s00–s23`; validation `s24–s29`; test `s30–s35`; 2,196 windows, 2,160 transitions.
- Regeneration from the current archive reproduced saved arrays bit-for-bit, with finite values and no subject overlap.
- Dataset has no fatigue labels.

## Models and evaluation

| Model | Components | Parameters |
|---|---|---:|
| Classical baseline | EEGNet + BiLSTM + classical fusion + next-window decoder | 269,826 |
| Hybrid 4Q/1L | EEGNet + BiLSTM + learned quantum angle map + PennyLane VQC + decoder | 219,882 |
| Hybrid 4Q/2L | Same, with 2 variational layers | 219,894 |
| Hybrid 6Q/1L | Same, with 6 qubits and 1 layer | 219,974 |

The models differ in parameter count and BiLSTM hidden size. On 360 held-out transitions, classical MSE/MAE/cosine are 0.8458/0.7148/0.3860; hybrid 4Q/1L 0.8771/0.7314/0.3561; 4Q/2L 0.8819/0.7327/0.3405; 6Q/1L 0.8690/0.7265/0.3563. The classical baseline is best on these metrics. PCA is not used by the learned feature-to-angle path.

See [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) for methods, baselines, limitations, and reproduction commands. See [CODEX_AUDIT.md](CODEX_AUDIT.md) for audit findings. Run `python -m src.evaluation.evaluate_digital_twin` to recompute metrics and latent-trajectory summaries from the saved checkpoints.

## Verification

- Unit suite: 17/17 passing.
- Preprocessing regeneration: all saved train/validation/test arrays reproduced bit-for-bit.
- Saved experiment checkpoints: loaded and recomputed metrics agree with the 20-epoch histories within rounding.
- 4Q/2L and 6Q/1L were each trained for 20 epochs and evaluated on the held-out subjects.

## Repository and publication status

README, dependency ranges, `.gitignore`, and a test workflow are present. The current workspace has no Git repository root or configured remote, so no commit/push or staged-file audit was possible. Raw EEG, processed arrays, and checkpoints should remain outside Git unless their distribution is explicitly intended and permitted. No project license has been selected.
