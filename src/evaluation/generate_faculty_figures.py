"""
generate_faculty_figures.py
===========================
Generates comprehensive figures and publication-grade plots for the Faculty Defense:
1. ablation_benchmark_comparison.png: 7-model comparative bar chart (MSE, MAE, CosSim, Params).
2. digital_twin_sequential_trajectory.png: Complete BT_t = [z_t, s_t, p_t, h_t] trajectory across 60 windows.
3. transition_stability_comparison.png: Temporal prediction stability across consecutive window transitions.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "results/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_ablation_benchmark():
    summary_path = ROOT / "results/ablations/ablation_summary.json"
    if not summary_path.exists():
        print("Summary file not found.")
        return

    with open(summary_path) as f:
        data = json.load(f)

    labels = []
    mse_vals = []
    mae_vals = []
    cos_vals = []
    colors = []

    order = [
        ("classical_baseline", "Classical Baseline (EEGNet+BiLSTM)", "#00e676"),
        ("quantum_4q_1l_pca", "Hybrid: Quantum 4Q 1L + PCA (Eq. 19)", "#00e5ff"),
        ("quantum_6q_1l", "Hybrid: Quantum 6Q 1L", "#2979ff"),
        ("quantum_4q_1l", "Hybrid: Quantum 4Q 1L (Learned)", "#b388ff"),
        ("quantum_4q_2l", "Hybrid: Quantum 4Q 2L (Deep)", "#7c4dff"),
        ("eegnet_bilstm_pca", "Classical 4D PCA Bottleneck", "#ff9100"),
        ("eegnet_only", "EEGNet Only (Spatial CNN)", "#ff5252"),
    ]

    for k, name, col in order:
        if k in data:
            labels.append(name)
            mse_vals.append(data[k]["test_mse"])
            mae_vals.append(data[k]["test_mae"])
            cos_vals.append(data[k]["test_cosine_sim"])
            colors.append(col)

    y = np.arange(len(labels))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Quantum Neural Digital Twin — Controlled Ablation Benchmark (Unseen Test Subjects)", fontsize=13, fontweight="bold")

    # MSE
    bars1 = ax1.barh(y, mse_vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Test MSE (Lower is Better)", fontsize=10)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    for bar in bars1:
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.4f}", va="center", fontsize=8)

    # MAE
    bars2 = ax2.barh(y, mae_vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Test MAE (Lower is Better)", fontsize=10)
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    for bar in bars2:
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.4f}", va="center", fontsize=8)

    # Cosine Similarity
    bars3 = ax3.barh(y, cos_vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax3.set_xlabel("Cosine Coherence (Higher is Better)", fontsize=10)
    ax3.grid(axis="x", alpha=0.3, linestyle="--")
    for bar in bars3:
        ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.4f}", va="center", fontsize=8)

    plt.tight_layout()
    out_file = FIGURES_DIR / "ablation_benchmark_comparison.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_transition_stability():
    dt_eval_path = ROOT / "results/digital_twin_evaluation.json"
    if not dt_eval_path.exists():
        print("Digital Twin evaluation file not found.")
        return

    with open(dt_eval_path) as f:
        dt = json.load(f)

    if "classical_baseline" not in dt or "quantum_4q_1l" not in dt:
        return

    c_trans = dt["classical_baseline"].get("mse_by_transition_index", [])
    q_trans = dt["quantum_4q_1l"].get("mse_by_transition_index", [])
    q6_trans = dt.get("quantum_6q_1l", {}).get("mse_by_transition_index", [])

    if not c_trans or not q_trans:
        return

    x = np.arange(len(c_trans))

    plt.figure(figsize=(12, 4.5))
    plt.plot(x, c_trans, label="Classical Baseline (EEGNet+BiLSTM)", color="#00e676", linewidth=1.6)
    plt.plot(x, q_trans, label="Quantum Hybrid 4Q 1L", color="#ff9100", linewidth=1.6, linestyle="--")
    if q6_trans:
        plt.plot(x, q6_trans, label="Quantum Hybrid 6Q 1L", color="#00e5ff", linewidth=1.4, linestyle="-.")

    plt.axhline(np.mean(c_trans), color="#00e676", linestyle=":", alpha=0.6, label=f"Classical Mean MSE: {np.mean(c_trans):.3f}")
    plt.axhline(np.mean(q_trans), color="#ff9100", linestyle=":", alpha=0.6, label=f"Quantum Mean MSE: {np.mean(q_trans):.3f}")

    plt.title("Sequential Digital Twin Transition Stability across Consecutive Windows (t -> t+1)", fontsize=11, fontweight="bold")
    plt.xlabel("Sequential Transition Index (Window Step)", fontsize=10)
    plt.ylabel("Prediction MSE", fontsize=10)
    plt.legend(fontsize=9, loc="upper right")
    plt.grid(True, alpha=0.25)

    plt.tight_layout()
    out_file = FIGURES_DIR / "transition_stability_comparison.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    plot_ablation_benchmark()
    plot_transition_stability()
