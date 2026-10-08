"""Evaluate saved models and EEG-derived sequential Digital Twin trajectories.

This script performs inference only. It does not train models or modify data.
Run from the repository root with ``python -m src.evaluation.evaluate_digital_twin``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import EEGDigitalTwinDataset
from src.models.classical_baseline import ClassicalDigitalTwin
from src.models.hybrid_twin import QuantumNeuralDigitalTwin


ROOT = Path(__file__).resolve().parents[2]


def _load_models(device: torch.device) -> dict[str, torch.nn.Module]:
    quantum_path = ROOT / "results/ablations/quantum_4q_1l/quantum_4q_1l_best.pt"
    classical_path = ROOT / "results/ablations/classical_baseline/classical_baseline_best.pt"
    if not quantum_path.is_file() or not classical_path.is_file():
        raise FileNotFoundError("Expected 20-epoch ablation checkpoints under results/ablations/.")

    quantum_configs = {
        "quantum_4q_1l": (quantum_path, 4, 1),
        "quantum_4q_2l": (ROOT / "results/ablations/quantum_4q_2l/quantum_4q_2l_best.pt", 4, 2),
        "quantum_6q_1l": (ROOT / "results/ablations/quantum_6q_1l/quantum_6q_1l_best.pt", 6, 1),
    }
    models: dict[str, torch.nn.Module] = {}
    for name, (checkpoint_path, n_qubits, n_layers) in quantum_configs.items():
        if not checkpoint_path.is_file():
            continue
        q_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        q_model = QuantumNeuralDigitalTwin(
            n_channels=19,
            n_timesteps=500,
            n_subjects=36,
            n_qubits=n_qubits,
            n_quantum_layers=n_layers,
            spatial_dim=16,
            temporal_dim=16,
            quantum_dim=8,
        ).to(device)
        q_model.load_state_dict(q_ckpt["model_state_dict"])
        models[name] = q_model.eval()

    c_ckpt = torch.load(classical_path, map_location=device, weights_only=False)
    c_model = ClassicalDigitalTwin(
        n_channels=19,
        n_timesteps=500,
        n_subjects=36,
        spatial_dim=24,
        temporal_dim=24,
        fused_dim=48,
    ).to(device)
    c_model.load_state_dict(c_ckpt["model_state_dict"])
    models["classical_baseline"] = c_model.eval()
    return models


def _metric_values(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(np.mean(np.square(pred - target))),
        "mae": float(np.mean(np.abs(pred - target))),
        "cosine_similarity": float(
            F.cosine_similarity(
                torch.from_numpy(np.array(pred.reshape(len(pred), -1), copy=True)),
                torch.from_numpy(np.array(target.reshape(len(target), -1), copy=True)),
                dim=1,
            ).mean().item()
        ),
    }


def _evaluate_model(
    name: str,
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions: list[np.ndarray] = []
    latent_by_subject: dict[int, list[np.ndarray]] = {}
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(x), 16):
            xb = torch.from_numpy(x[offset : offset + 16]).to(device=device, dtype=torch.float32)
            out = model(xb, task="predictive")
            predictions.append(out["x_next_pred"].cpu().numpy())
            if isinstance(model, QuantumNeuralDigitalTwin):
                states = out["fused"].cpu().numpy()
            else:
                states = out["fused"].cpu().numpy()
            for sid, state in zip(subject_ids[offset : offset + len(states)], states):
                latent_by_subject.setdefault(int(sid), []).append(state)
    elapsed = time.perf_counter() - start
    pred = np.concatenate(predictions, axis=0)

    per_subject = {}
    for sid in sorted(np.unique(subject_ids)):
        mask = subject_ids == sid
        per_subject[f"s{sid:02d}"] = _metric_values(pred[mask], y[mask])

    state_step_distances = []
    for values in latent_by_subject.values():
        states = np.asarray(values)
        if len(states) > 1:
            state_step_distances.extend(np.linalg.norm(np.diff(states, axis=0), axis=1).tolist())

    result = {
        "metrics": _metric_values(pred, y),
        "per_subject": per_subject,
        "inference_seconds": elapsed,
        "inference_ms_per_transition": elapsed * 1000.0 / len(x),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_trajectory": {
            "subjects": len(latent_by_subject),
            "latent_dimension": int(next(iter(latent_by_subject.values()))[0].shape[0]),
            "mean_adjacent_latent_distance": float(np.mean(state_step_distances)),
            "median_adjacent_latent_distance": float(np.median(state_step_distances)),
            "interpretation": "Euclidean change between encoded EEG-window states; descriptive, not a clinical measure.",
        },
    }

    # Aggregate errors by transition index (one-second stride, 50% overlap).
    step_errors = []
    max_steps = max(int(np.sum(subject_ids == sid)) for sid in np.unique(subject_ids))
    for step in range(max_steps):
        positions = []
        seen = {}
        for i, sid in enumerate(subject_ids):
            n = seen.get(int(sid), 0)
            if n == step:
                positions.append(i)
            seen[int(sid)] = n + 1
        if positions:
            step_errors.append(float(np.mean(np.square(pred[positions] - y[positions]))))
    result["mse_by_transition_index"] = step_errors
    return result, pred


def run_evaluation(
    data_dir: Path = ROOT / "data/processed",
    output_dir: Path = ROOT / "results",
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    dataset = EEGDigitalTwinDataset(split="test", data_dir=data_dir, mode="predictive", layout="NTC")
    x = dataset.X.numpy()
    y = dataset.Y.numpy()
    subject_ids = dataset.subject_ids.numpy()
    train_targets = np.load(data_dir / "train_next_windows.npy", mmap_mode="r")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(train_targets).all()):
        raise ValueError("Found NaN or infinity in evaluation arrays.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _load_models(device)
    results: dict[str, Any] = {
        "evaluation": "held-out one-step next-window forecasting and EEG-derived latent trajectory summary",
        "split": "test",
        "device": str(device),
        "transitions": int(len(x)),
        "subjects": sorted(f"s{int(sid):02d}" for sid in np.unique(subject_ids)),
        "window_shape": list(x.shape[1:]),
        "models": {},
        "baselines": {},
        "saved_20_epoch_results": {},
    }

    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model_result, pred = _evaluate_model(name, model, x, y, subject_ids, device)
        results["models"][name] = model_result
        predictions[name] = pred

    results["baselines"]["persistence"] = _metric_values(x, y)
    train_mean = np.asarray(train_targets).mean(axis=0, dtype=np.float64).astype(np.float32)
    results["baselines"]["training_target_mean"] = _metric_values(np.broadcast_to(train_mean, y.shape), y)

    # Compare recomputed checkpoints with their stored test-result rows.
    for name in models:
        history_path = ROOT / f"results/ablations/{name}/{name}_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        stored = history["test_metrics"]
        results["saved_20_epoch_results"][name] = {
            "epochs": history["epochs_run"],
            "training_seconds": history["total_time_sec"],
            "stored_test_metrics_legacy_val_keys": {
                "mse": stored["val_mse"],
                "mae": stored["val_mae"],
                "cosine_similarity": stored["val_cosine_similarity"],
            },
            "recomputed_checkpoint_metrics": results["models"][name]["metrics"],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "digital_twin_evaluation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, pred in predictions.items():
        subject_means = []
        for sid in sorted(np.unique(subject_ids)):
            mask = subject_ids == sid
            subject_means.append(np.mean(np.square(pred[mask] - y[mask])))
        ax.plot(np.arange(1, len(subject_means) + 1), subject_means, marker="o", label=name.replace("_", " "))
    ax.set(xlabel="Held-out subject index (s30–s35)", ylabel="Subject mean next-window MSE", title="Sequential EEG Forecast Error by Held-out Subject")
    ax.set_xticks(range(1, len(np.unique(subject_ids)) + 1), [f"s{int(s):02d}" for s in sorted(np.unique(subject_ids))])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "digital_twin_subject_mse.png", dpi=150)
    plt.close(fig)
    return results


if __name__ == "__main__":
    summary = run_evaluation()
    print(json.dumps({"device": summary["device"], "transitions": summary["transitions"], "models": {k: v["metrics"] for k, v in summary["models"].items()}, "baselines": summary["baselines"]}, indent=2))
