"""
visualize.py
============
Phase 4 Inference & Visualisation.
Loads the best checkpoint and produces:
  1. Predicted vs Actual EEG waveforms (per channel, per subject).
  2. Per-channel MSE bar chart.
  3. Quantum angle distribution scatter plot.
  4. Training loss curves for all ablation runs.
All figures are saved to results/figures/.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.data.dataset import get_dataloaders
from src.models.hybrid_twin import QuantumNeuralDigitalTwin
from src.models.classical_baseline import ClassicalDigitalTwin

# Standard 19-channel 10-20 names (column order in the CSV / processed arrays)
CHANNEL_NAMES = [
    "Fp1","Fp2","F3","F4","F7","F8",
    "T3","T4","C3","C4","T5","T6",
    "P3","P4","O1","O2","Fz","Cz","Pz",
]


# ──────────────────────────────────────────────────────────────
# Utility: load model from checkpoint
# ──────────────────────────────────────────────────────────────

def load_quantum_model(ckpt_path: Path, device: torch.device) -> QuantumNeuralDigitalTwin:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {
        "n_qubits": 4, "n_quantum_layers": 1,
        "spatial_dim": 16, "temporal_dim": 16, "quantum_dim": 8,
    })
    model = QuantumNeuralDigitalTwin(
        n_channels=19, n_timesteps=500, n_subjects=36, **cfg
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_classical_model(ckpt_path: Path, device: torch.device) -> ClassicalDigitalTwin:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ClassicalDigitalTwin(
        n_channels=19, n_timesteps=500, n_subjects=36,
        spatial_dim=24, temporal_dim=24, fused_dim=48,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────
# Plot 1: Predicted vs Actual waveforms
# ──────────────────────────────────────────────────────────────

def plot_waveform_comparison(
    model: torch.nn.Module,
    data_dir: Path,
    out_dir: Path,
    split: str = "test",
    sample_idx: int = 0,
    n_channels_plot: int = 6,
    run_label: str = "Hybrid Quantum",
):
    """Plot predicted vs actual next-window EEG waveforms for selected channels."""
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=64,
                                         mode="predictive", layout="NTC")

    device = next(model.parameters()).device
    x_t, x_next, sub_ids = next(iter(test_loader))
    x_t = x_t.to(device)

    with torch.no_grad():
        out = model(x_t, task="predictive")
        pred = out["x_next_pred"]

    x_t_np = x_t[sample_idx].cpu().numpy()          # (500, 19)
    x_next_np = x_next[sample_idx].cpu().numpy()     # (500, 19)
    pred_np = pred[sample_idx].cpu().numpy()         # (500, 19)

    t = np.linspace(0, 2.0, 500)
    channels_to_plot = list(range(min(n_channels_plot, 19)))

    fig, axes = plt.subplots(n_channels_plot, 1, figsize=(14, 2.8 * n_channels_plot),
                              sharex=True)
    fig.suptitle(f"EEG Digital Twin — Predicted vs Actual Next Window\n"
                 f"({run_label}  |  Subject: s{sub_ids[sample_idx].item():02d})",
                 fontsize=13, fontweight="bold")

    for row, ch in enumerate(channels_to_plot):
        ax = axes[row]
        ax.plot(t, x_next_np[:, ch], color="#2196F3", linewidth=1.1, label="Actual $x_{t+1}$", alpha=0.85)
        ax.plot(t, pred_np[:, ch],   color="#FF5722", linewidth=1.0, label="Predicted $\\hat{x}_{t+1}$",
                linestyle="--", alpha=0.85)
        ax.set_ylabel(CHANNEL_NAMES[ch], fontsize=9, rotation=0, labelpad=32)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    fig_path = out_dir / f"waveform_pred_vs_actual_{run_label.replace(' ', '_').lower()}.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


# ──────────────────────────────────────────────────────────────
# Plot 2: Per-channel MSE bar chart
# ──────────────────────────────────────────────────────────────

def plot_per_channel_mse(
    results: Dict[str, dict],
    out_dir: Path,
):
    """
    results: {run_label: {"per_channel_mse": np.array(19,)}}
    """
    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(19)
    width = 0.8 / len(results)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, (label, data) in enumerate(results.items()):
        offset = (i - len(results) / 2 + 0.5) * width
        ax.bar(x + offset, data["per_channel_mse"], width=width * 0.9,
               label=label, color=colors[i % len(colors)], alpha=0.82)

    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Test MSE (per channel)", fontsize=10)
    ax.set_title("Per-Channel Prediction MSE — Ablation Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / "per_channel_mse.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


# ──────────────────────────────────────────────────────────────
# Plot 3: Quantum angle distribution
# ──────────────────────────────────────────────────────────────

def plot_quantum_angles(
    model: QuantumNeuralDigitalTwin,
    data_dir: Path,
    out_dir: Path,
    n_batches: int = 5,
):
    """Scatter/violin of VQC input angle distributions across test set."""
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=64,
                                         mode="predictive", layout="NTC")
    device = next(model.parameters()).device
    all_angles = []

    model.eval()
    with torch.no_grad():
        for i, (x_t, _, _) in enumerate(test_loader):
            if i >= n_batches:
                break
            enc = model.encode(x_t.to(device))
            all_angles.append(enc["quantum_angles"].cpu().numpy())

    angles = np.vstack(all_angles)   # (N, n_qubits)
    n_qubits = angles.shape[1]

    fig, axes = plt.subplots(1, n_qubits, figsize=(3.5 * n_qubits, 4), sharey=True)
    for q in range(n_qubits):
        ax = axes[q] if n_qubits > 1 else axes
        ax.hist(angles[:, q], bins=30, color="#673AB7", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Qubit {q}", fontsize=10)
        ax.set_xlabel("Rotation angle (rad)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-np.pi, np.pi)

    axes[0].set_ylabel("Count", fontsize=9)
    fig.suptitle("VQC Input Angle Distribution (Test Set)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_path = out_dir / "quantum_angle_distribution.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


# ──────────────────────────────────────────────────────────────
# Plot 4: Training loss curves
# ──────────────────────────────────────────────────────────────

def plot_loss_curves(
    ablation_dir: Path,
    out_dir: Path,
    run_names: Optional[List[str]] = None,
):
    """Read per-run history JSON files and overlay training/val loss curves."""
    colors = {"train": "#2196F3", "val": "#FF5722"}
    run_styles = ["-", "--", "-.", ":"]
    run_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    if run_names is None:
        run_names = [d.name for d in ablation_dir.iterdir() if d.is_dir()]

    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    fig_cos,  ax_cos  = plt.subplots(figsize=(10, 5))

    for i, run_name in enumerate(run_names):
        hist_path = ablation_dir / run_name / f"{run_name}_history.json"
        if not hist_path.exists():
            # Fallback: look in checkpoints for Phase 3 run
            hist_path = Path("checkpoints/training_history.json")
            if not hist_path.exists():
                continue

        with open(hist_path) as f:
            data = json.load(f)

        history = data.get("history", [])
        if not history:
            continue

        epochs = [h["epoch"] for h in history]
        train_loss = [h.get("train_loss", h.get("train_loss", None)) for h in history]
        val_loss   = [h.get("val_loss",   None) for h in history]
        val_cos    = [h.get("val_cosine_similarity", h.get("val_cosine_sim", 0)) for h in history]

        style = run_styles[i % len(run_styles)]
        color = run_colors[i % len(run_colors)]
        label = run_name.replace("_", " ")

        ax_loss.plot(epochs, train_loss, linestyle=style, color=color,
                     alpha=0.55, linewidth=1.2, label=f"{label} train")
        ax_loss.plot(epochs, val_loss,   linestyle=style, color=color,
                     alpha=1.0,  linewidth=1.8, marker="o", markersize=3,
                     label=f"{label} val")
        ax_cos.plot(epochs, val_cos, linestyle=style, color=color,
                    linewidth=1.8, marker="o", markersize=3, label=label)

    for ax, title, ylabel in [
        (ax_loss, "Training & Validation Loss Curves", "Composite Loss"),
        (ax_cos,  "Validation Cosine Similarity",     "CosSim"),
    ]:
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    fig_loss.tight_layout()
    fig_cos.tight_layout()
    p1 = out_dir / "loss_curves.png"
    p2 = out_dir / "cosine_similarity_curves.png"
    fig_loss.savefig(p1, dpi=130, bbox_inches="tight")
    fig_cos.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


# ──────────────────────────────────────────────────────────────
# Main: generate all figures using Phase 3 checkpoint
# ──────────────────────────────────────────────────────────────

def run_all_visualizations(
    data_dir: Path = Path("./data/processed"),
    ckpt_path: Path = Path("./checkpoints/quantum_digital_twin_best.pt"),
    ablation_dir: Path = Path("./results/ablations"),
    out_dir: Path = Path("./results/figures"),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("Loading Phase 3 quantum model checkpoint …")
    model = load_quantum_model(ckpt_path, device)

    print("\n[1/4] Waveform comparison …")
    plot_waveform_comparison(
        model=model,
        data_dir=data_dir,
        out_dir=out_dir,
        sample_idx=0,
        n_channels_plot=6,
        run_label="Hybrid Quantum (4q 1L)",
    )

    print("\n[2/4] Per-channel MSE …")
    # Compute per-channel MSE on test set
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=64,
                                         mode="predictive", layout="NTC")
    ch_sq_errors = np.zeros(19)
    n_total = 0
    model.eval()
    with torch.no_grad():
        for x_t, x_next, _ in test_loader:
            pred = model(x_t.to(device), task="predictive")["x_next_pred"]
            sq = ((pred.cpu().numpy() - x_next.numpy()) ** 2).mean(axis=1)  # (B, 19)
            ch_sq_errors += sq.sum(axis=0)
            n_total += x_t.size(0)
    per_channel_mse = ch_sq_errors / n_total

    plot_per_channel_mse(
        results={"Hybrid Quantum (4q 1L)": {"per_channel_mse": per_channel_mse}},
        out_dir=out_dir,
    )

    print("\n[3/4] Quantum angle distribution …")
    plot_quantum_angles(model=model, data_dir=data_dir, out_dir=out_dir)

    print("\n[4/4] Loss curves …")
    plot_loss_curves(
        ablation_dir=ablation_dir,
        out_dir=out_dir,
        run_names=None,
    )

    print("\nAll visualizations complete.")


if __name__ == "__main__":
    run_all_visualizations()
