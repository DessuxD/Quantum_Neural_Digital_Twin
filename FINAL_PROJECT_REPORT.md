# Final Project Report

## 1. Project Overview

This project is a research prototype for EEG-derived latent state representation and one-step next-window forecasting. Its “Digital Twin” is a learned computational representation of EEG windows and their transitions. It is not a complete brain simulation, a clinical diagnostic system, or a disease predictor.

## 2. Dataset

The local source is the converted PhysioNet EEG During Mental Arithmetic Tasks CSV distribution: 36 subjects, 19 channels, 31,000 samples per subject, recorded at 500 Hz (62 seconds). The official [PhysioNet EDF release](https://physionet.org/content/eegmat/1.0.0/) is marked ODC-BY 1.0; the Kaggle [converted CSV distribution](https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset) lists its license as unknown. The source archive contains continuous EEG values without fatigue labels or event annotations. The incomplete pair of EDF files under `data/raw/mental_arithmetic/` is not used. Raw data must be acquired separately and is excluded from Git. The current code has no verified EDF-to-CSV conversion path.

## 3. Preprocessing

`src/preprocessing/pipeline.py` performs a 50 Hz notch, 0.5–45 Hz bandpass, polyphase resampling from 500 to 250 Hz, per-subject/channel z-score normalization, and 2-second windows at 1-second stride. Regeneration from the existing CSV archive produced arrays bit-for-bit equal to the saved processed arrays. All generated arrays were finite. Subject-wise split: train `s00–s23`, validation `s24–s29`, test `s30–s35`; no subject overlap. This yields 2,196 windows and 2,160 sequential transition pairs.

## 4. Classical Architecture

The comparator combines EEGNet and a two-layer bidirectional LSTM, then a classical fusion layer and waveform decoder to predict the next 500×19 window. The tested checkpoint has 269,826 trainable parameters. This is the classical comparator used in the saved 20-epoch experiment.

## 5. Quantum Architecture

The tested hybrid has EEGNet and BiLSTM feature encoders, learned feature fusion and angle projection, a 4-qubit/1-layer PennyLane VQC, and a classical decoder. It uses Y AngleEmbedding, trainable Rot gates, ring CNOT entanglement, and one Pauli-Z expectation per qubit. PennyLane TorchLayer integrates gradients with PyTorch. The tested checkpoint has 219,882 trainable parameters. The standalone PCA utility is not part of this model path.

## 6. Digital Twin Architecture

Each 2-second EEG window is encoded to a fused latent vector. The predictive decoder estimates the following 2-second window. Sequential held-out evaluation also measures adjacent changes in these encoded states within each test subject. These latent distances are descriptive properties of the model representation, not validated physiological or clinical measures.

## 7. Experimental Setup

The saved experiment trained on 24 subjects, selected by validation on 6 subjects, and evaluated on 6 held-out subjects. The reported comparator runs used 20 epochs and CPU training. Existing checkpoint inference was run on CPU for 360 test transitions. No retraining or new architecture ablations were run. The comparison is not parameter matched: the classical model has 269,826 parameters versus 219,882, and the BiLSTM hidden sizes differ (32 vs. 24).

## 8. Results

Recomputed checkpoint metrics on held-out transitions:

| Model | Test MSE | Test MAE | Cosine similarity | Parameters | Training time |
|---|---:|---:|---:|---:|---:|
| Classical baseline | 0.8458 | 0.7148 | 0.3860 | 269,826 | 1,703.78 s |
| Hybrid quantum (4Q/1L) | 0.8771 | 0.7314 | 0.3561 | 219,882 | 1,991.33 s |
| Hybrid quantum (4Q/2L) | 0.8819 | 0.7327 | 0.3405 | 219,894 | 725.31 s |
| Hybrid quantum (6Q/1L) | 0.8690 | 0.7265 | 0.3563 | 219,974 | 710.53 s |

In a single CPU inference run on this workspace, throughput was approximately 9.80 ms per transition for the classical model, 9.96 ms for 4Q/1L, 10.47 ms for 4Q/2L, and 9.66 ms for 6Q/1L. These timings are machine- and run-dependent.

Persistence baseline recomputed on the saved arrays: MSE 2.0695, MAE 1.1367. Training-target-mean baseline: MSE 1.0101, MAE 0.7864. A previous approximate persistence value of 2.0061 was not reproduced. Results are descriptive for this dataset and split.

## 9. Classical vs Quantum Comparison

The classical model performed better across all three reported forecasting metrics and trained faster in the saved runs. These experiments do not establish quantum advantage. The models also differ in parameter count and recurrent hidden size, limiting causal attribution to the VQC.

## 10. Ablation Results

The 4Q/2L variant did not improve on 4Q/1L for any reported test metric. The 6Q/1L variant modestly improved on 4Q/1L MSE and MAE, with a similar cosine score, but remained below the classical model. Each configuration has one run on this fixed subject split; these observations are not statistical evidence. The PCA utility has unit coverage but is not an ablation of the trained model.

## 11. Digital Twin Results

`src/evaluation/evaluate_digital_twin.py` writes `results/digital_twin_evaluation.json` and `results/figures/digital_twin_subject_mse.png`. It reports per-subject forecasting metrics, inference time, parameter count, and adjacent latent-state distances. These outputs summarize EEG state sequence modeling; they do not establish a validated cognitive state or biological twin. The generated output records this evaluation on the current machine and may vary slightly across hardware/software.

## 12. Limitations

1. The dataset is not fatigue-labelled; fatigue classification is unsupported.
2. The Digital Twin is an EEG-derived computational representation.
3. The task is next-window EEG forecasting, not disease prediction.
4. Quantum advantage is not established or guaranteed.
5. The shallow hybrid model underperforms the classical comparator on the saved metrics.
6. A small public dataset limits clinical and broader generalization claims.
7. Quantum simulation can be computationally expensive.
8. The prototype is not a clinical system or medical device.
9. The comparator runs are not parameter-matched.

## 13. Reproducibility

Use Python 3.11, install `requirements.txt`, place the separately obtained CSV recordings in `archive/`, then run:

```bash
python -m src.preprocessing.pipeline --archive-dir archive --output-dir data/processed
python -m unittest discover -s tests -v
python -m src.evaluation.evaluate_digital_twin
```

The official PhysioNet EDF release is marked ODC-BY 1.0, but the Kaggle converted CSV page lists its license as unknown. The current pipeline has no verified EDF conversion path; resolve this data-format/provenance step before claiming a fresh reproduction from official EDF files. Model training takes substantial CPU time and may vary across software/hardware. Full commands are in `README.md`.

## 14. GitHub Usage

The workspace root currently has no Git repository or configured remote, so no commit or push was made. `.gitignore` excludes raw recordings, processed arrays, and checkpoints. Before publishing, initialize or locate the intended repository, review the staged files and size, check for secrets and local paths, and confirm data redistribution terms. No project license was selected or added.

## 15. Future Work

- Repeat the comparison with controlled parameter budgets and identical training protocols.
- Add multi-seed uncertainty estimates and subject-level statistical analysis.
- Repeat promising comparisons with multiple seeds and a parameter-matched classical comparator.
- Validate the latent trajectory interpretation against independently defined, appropriately labelled measurements.
- Improve dataset provenance/licensing notes and publish a reproducible data preparation manifest.
