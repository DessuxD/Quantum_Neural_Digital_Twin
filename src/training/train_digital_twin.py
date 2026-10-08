"""
train_digital_twin.py
=====================
Training and Evaluation Pipeline for the Quantum Neural Digital Twin.
Optimizes the hybrid model for autoregressive brain state forecasting (X_t -> X_{t+1})
using AdamW, Cosine Annealing, and a composite loss (MSE + Temporal Phase Correlation).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import get_dataloaders
from src.models.hybrid_twin import QuantumNeuralDigitalTwin

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CompositeDigitalTwinLoss(nn.Module):
    """
    Composite loss combining point-wise Mean Squared Error (MSE)
    with temporal phase correlation loss to preserve wave oscillations.
    """
    def __init__(self, alpha_mse: float = 1.0, beta_cosine: float = 0.1):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.beta_cosine = beta_cosine

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # pred, target shape: (B, T, C)
        mse = F.mse_loss(pred, target)

        # Flatten time and channel for cosine similarity: (B, T*C)
        b = pred.size(0)
        p_flat = pred.reshape(b, -1)
        t_flat = target.reshape(b, -1)
        cos_sim = F.cosine_similarity(p_flat, t_flat, dim=1).mean()
        cos_loss = 1.0 - cos_sim

        total_loss = self.alpha_mse * mse + self.beta_cosine * cos_loss
        metrics = {
            "loss": total_loss.item(),
            "mse": mse.item(),
            "cosine_similarity": cos_sim.item()
        }
        return total_loss, metrics

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_cos = 0.0
    total_samples = 0

    for x_t, x_next, _ in loader:
        x_t = x_t.to(device)
        x_next = x_next.to(device)
        batch_size = x_t.size(0)

        optimizer.zero_grad()
        out = model(x_t, task="predictive")
        pred_next = out["x_next_pred"]

        loss, metrics = criterion(pred_next, x_next)
        loss.backward()

        # Gradient clipping to stabilize hybrid quantum-classical optimization
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * batch_size
        running_mse += metrics["mse"] * batch_size
        running_cos += metrics["cosine_similarity"] * batch_size
        total_samples += batch_size

    return {
        "train_loss": running_loss / total_samples,
        "train_mse": running_mse / total_samples,
        "train_cosine_sim": running_cos / total_samples
    }

def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_mae = 0.0
    running_cos = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_t, x_next, _ in loader:
            x_t = x_t.to(device)
            x_next = x_next.to(device)
            batch_size = x_t.size(0)

            out = model(x_t, task="predictive")
            pred_next = out["x_next_pred"]

            loss, metrics = criterion(pred_next, x_next)
            mae = F.l1_loss(pred_next, x_next)

            running_loss += loss.item() * batch_size
            running_mse += metrics["mse"] * batch_size
            running_mae += mae.item() * batch_size
            running_cos += metrics["cosine_similarity"] * batch_size
            total_samples += batch_size

    return {
        "val_loss": running_loss / total_samples,
        "val_mse": running_mse / total_samples,
        "val_mae": running_mae / total_samples,
        "val_cosine_sim": running_cos / total_samples
    }

def run_training(
    data_dir: Path = Path("./data/processed"),
    checkpoint_dir: Path = Path("./checkpoints"),
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_qubits: int = 4,
    n_quantum_layers: int = 1,
    spatial_dim: int = 16,
    temporal_dim: int = 16,
    quantum_dim: int = 8,
    seed: int = 42
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # Data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        mode="predictive",
        layout="NTC"
    )

    print(f"Data Loaded: Train={len(train_loader.dataset)} pairs, Val={len(val_loader.dataset)} pairs, Test={len(test_loader.dataset)} pairs")

    # Instantiate model
    model = QuantumNeuralDigitalTwin(
        n_channels=19,
        n_timesteps=500,
        n_qubits=n_qubits,
        n_quantum_layers=n_quantum_layers,
        spatial_dim=spatial_dim,
        temporal_dim=temporal_dim,
        quantum_dim=quantum_dim,
        n_subjects=36
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with {total_params:,} trainable parameters ({n_qubits} qubits).")

    criterion = CompositeDigitalTwinLoss(alpha_mse=1.0, beta_cosine=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    print("\nStarting Training Loop:")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        ep_duration = time.time() - ep_start

        epoch_record = {
            "epoch": epoch,
            "duration_sec": round(ep_duration, 2),
            **train_metrics,
            **val_metrics
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:2d}/{epochs:2d} ({ep_duration:5.1f}s) | "
            f"Train Loss: {train_metrics['train_loss']:.4f} (MSE: {train_metrics['train_mse']:.4f}) | "
            f"Val Loss: {val_metrics['val_loss']:.4f} (MSE: {val_metrics['val_mse']:.4f}, CosSim: {val_metrics['val_cosine_sim']:.4f})"
        )

        # Save best checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_path = checkpoint_dir / "quantum_digital_twin_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": {
                    "n_qubits": n_qubits,
                    "n_quantum_layers": n_quantum_layers,
                    "spatial_dim": spatial_dim,
                    "temporal_dim": temporal_dim,
                    "quantum_dim": quantum_dim
                }
            }, best_path)

    # Final evaluation on unseen Test Subjects
    print("\n--- Evaluating Best Model on Unseen Test Subjects ---")
    best_ckpt = torch.load(checkpoint_dir / "quantum_digital_twin_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['val_loss']:.4f} | Test MSE: {test_metrics['val_mse']:.4f} | Test MAE: {test_metrics['val_mae']:.4f} | CosSim: {test_metrics['val_cosine_sim']:.4f}")

    total_duration = time.time() - start_time
    results = {
        "epochs": epochs,
        "total_training_time_sec": round(total_duration, 2),
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "history": history
    }

    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved training history to {history_path}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Train Quantum Neural Digital Twin.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--n-qubits", type=int, default=4, help="Number of qubits.")
    parser.add_argument("--n-layers", type=int, default=1, help="Quantum circuit depth.")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_qubits=args.n_qubits,
        n_quantum_layers=args.n_layers
    )

if __name__ == "__main__":
    main()
