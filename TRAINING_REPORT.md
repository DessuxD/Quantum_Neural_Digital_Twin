# Quantum Neural Digital Twin — Historical Phase 3 Training Report

> This is the original 5-epoch run report. It is retained as an experiment record and is not the current 20-epoch classical-versus-hybrid comparison. See [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) for recomputed comparison metrics and current limitations.

## Executive Summary

Phase 3 (Neural Feature Extraction & Quantum Embedding) is **complete**. The full Quantum Neural Digital Twin was assembled, unit-tested, and trained end-to-end for 5 epochs on the preprocessed PhysioNet EEG During Mental Arithmetic Tasks dataset. The model successfully learned to predict the next EEG state window from the current one, with monotonically decreasing loss and improving temporal coherence across all epochs.

---

## Architecture Overview

The `QuantumNeuralDigitalTwin` model integrates four distinct subsystems in a unified hybrid quantum-classical pipeline:

```
Input Window: (B, 500, 19)  →  500 timesteps × 19 EEG channels
         │
         ├──[EEGNet Spatial Encoder]──────────── z_spatial  (B, 16)
         │   Temporal Conv → Depthwise Spatial Conv → Separable Conv
         │
         ├──[BiLSTM Temporal Encoder]──────────  z_temporal (B, 16)
         │   2-Layer Bidirectional LSTM → Projection
         │
         ↓
    [Classical Fusion: cat + Linear + LayerNorm + ELU]   z_classical (B, 32)
         │
         ├──[QuantumStateMapper: Linear + tanh×π]──────── q_angles (B, 4)
         │
         ↓
    [PennyLane VQC: AngleEmbedding + Rot Gates + Ring CNOT]
         │   1 variational layer × 4 qubits × 3 Euler angles
         │   Measurements: ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩
         ↓
    [Post-Quantum Projection: Linear + LayerNorm + ELU]   z_quantum (B, 8)
         │
         ↓
    [Hybrid Fusion: cat(z_classical, z_quantum) + Linear + LayerNorm]  z_fused (B, 40)
         │
         ├──[WaveformDecoder A]─── x̂_{t+1} (B, 500, 19)  Predictive Head
         ├──[WaveformDecoder B]─── x̂_recon  (B, 500, 19)  Reconstruction Head
         └──[BiometricHead: Linear→ELU→Linear]─── subject_logits (B, 36)
```

---

## Model Configuration

| Component | Parameter | Value |
|:---|:---|:---|
| **EEGNet** | F1 temporal filters | 8 |
| | Depthwise depth multiplier D | 2 |
| | F2 separable filters | 16 |
| | Temporal kernel length | 64 |
| | Latent dim (`spatial_dim`) | 16 |
| **BiLSTM** | Hidden units per direction | 24 |
| | Stacked layers | 2 |
| | Latent dim (`temporal_dim`) | 16 |
| **VQC** | Qubits ($n$) | 4 |
| | Variational layers | 1 |
| | Gate type | Rot ($R_Z R_Y R_Z$) |
| | Entanglement | Ring CNOT |
| | Embedding | AngleEmbedding (Y rotation) |
| | Angle scaling | $\tanh(\cdot) \times \pi$ |
| | Measurement | $\langle Z_i \rangle$ all qubits |
| | Post-quantum dim (`quantum_dim`) | 8 |
| **Fusion** | Classical dim | 32 |
| | Hybrid fused dim | 40 |
| **Total Parameters** | — | **219,882** trainable |
| **Checkpoint Size** | — | 1,929.6 KB |

---

## Training Configuration

| Hyperparameter | Value |
|:---|:---|
| Epochs | 5 |
| Batch size | 32 |
| Learning rate | $1 \times 10^{-3}$ |
| Optimizer | AdamW ($\lambda = 10^{-4}$) |
| LR Scheduler | Cosine Annealing ($T_{max} = 5$) |
| Gradient clipping | $\|\nabla\| \leq 1.0$ |
| Loss function | Composite: $\mathcal{L} = \text{MSE}(\hat{x}_{t+1}, x_{t+1}) + 0.1 \times (1 - \cos(\hat{x}_{t+1}, x_{t+1}))$ |
| Device | CPU (no CUDA) |
| Random seed | 42 |

---

## Training Results

### Per-Epoch Metrics

| Epoch | Train Loss | Train MSE | Val Loss | Val MSE | Val CosSim | Duration (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.1445 | 1.0461 | 1.0905 | 1.0004 | 0.0988 | 34.3 |
| 2 | 1.0378 | 0.9595 | 1.0287 | 0.9532 | 0.2453 | 122.8 |
| 3 | 1.0023 | 0.9303 | 1.0109 | 0.9377 | 0.2686 | 43.7 |
| 4 | 0.9892 | 0.9191 | 1.0034 | 0.9312 | 0.2777 | 60.2 |
| **5** | **0.9855** | **0.9158** | **1.0029** | **0.9309** | **0.2798** | 25.1 |

- **Total Training Time:** 288.1 seconds (~4.8 minutes, CPU only)
- **Best Val Loss (Epoch 5):** 1.0029

### Test Set Results (6 Unseen Subjects: s30–s35)

| Metric | Value |
|:---|:---|
| **Test Loss** | **0.9934** |
| **Test MSE** | **0.9222** |
| **Test MAE** | **0.7515** |
| **Test Cosine Similarity** | **0.2883** |

> [!NOTE]
> This earlier run used the held-out subjects s30–s35. A lower test metric than validation alone does not establish absence of overfitting or robust generalization.

---

## Key Observations

1. **Steady Convergence:** Training loss dropped from 1.1445 → 0.9855 (−13.9%) across 5 epochs with no instability.
2. Validation cosine similarity increased from 0.099 to 0.280 across this run; this is a descriptive metric change and does not by itself establish a specific physiological interpretation.
3. Test MSE (0.9222) was slightly lower than validation MSE (0.9309) in this single split; this does not establish absence of overfitting.
4. The model test suite checked gradient availability through the VQC parameters. This is not a general assessment of gradient conditioning.
5. Epoch duration varied across this run (25.1–122.8 seconds); the cause was not measured.

---

## Files Created / Updated

| File | Description |
|:---|:---|
| [src/models/eegnet.py](src/models/eegnet.py) | EEGNet spatial-temporal feature encoder (F1=8, D=2, F2=16) |
| [src/models/bilstm.py](src/models/bilstm.py) | BiLSTM sequential dynamics encoder |
| [src/models/dimension_reduction.py](src/models/dimension_reduction.py) | SpatialEEG_PCA + QuantumStateMapper (tanh→π angle projection) |
| [src/models/quantum_circuit.py](src/models/quantum_circuit.py) | PennyLane VQC with AngleEmbedding + Rot + Ring CNOT |
| [src/models/hybrid_twin.py](src/models/hybrid_twin.py) | Full `QuantumNeuralDigitalTwin` with 3 decoder heads |
| [src/training/train_digital_twin.py](src/training/train_digital_twin.py) | Training loop, AdamW + CosineAnnealing, composite loss, evaluation |
| [`tests/test_models.py`](tests/test_models.py) | 7 model tests |
| [`checkpoints/quantum_digital_twin_best.pt`](checkpoints/quantum_digital_twin_best.pt) | Phase 3 model weights (1,929.6 KB) |
| [`checkpoints/training_history.json`](checkpoints/training_history.json) | Phase 3 per-epoch metrics history |

---

## PCA Utility

`SpatialEEG_PCA` is a standalone utility with unit coverage. The trained hybrid model uses a learned neural projection from its classical latent representation to VQC angles; it does not apply this PCA transform. PCA variance figures from exploratory work therefore do not describe the VQC inputs and are not used as justification for the 4-qubit experiment.

---

## Later Experiments

The project subsequently added 20-epoch classical and hybrid comparisons plus 4Q/2L and 6Q/1L circuit ablations. These later results are summarized in [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) and the JSON histories under `results/ablations/`. This historical 5-epoch report should not be used as the current model comparison.
