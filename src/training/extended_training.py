"""
extended_training.py
====================
Phase 4: Extended training with:
  - Linear LR warmup + CosineAnnealingLR
  - Early stopping (patience-based on val loss)
  - Multi-run ablation driver:
      (A) Hybrid Quantum (4 qubits, 1 layer)   [best Phase 3 config]
      (B) Hybrid Quantum (4 qubits, 2 layers)
      (C) Hybrid Quantum (6 qubits, 1 layer)
      (D) Classical Baseline (no VQC)
"""

import os, sys, json, time, argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import get_dataloaders
from src.models.hybrid_twin import QuantumNeuralDigitalTwin
from src.models.classical_baseline import ClassicalDigitalTwin


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class WarmupCosineScheduler:
    """Linear warmup then CosineAnnealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


class EarlyStopper:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False   # keep going
        self.counter += 1
        return self.counter >= self.patience  # True = stop


class CompositeLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        b = pred.size(0)
        cos = F.cosine_similarity(pred.reshape(b, -1), target.reshape(b, -1), dim=1).mean()
        loss = self.alpha * mse + self.beta * (1.0 - cos)
        return loss, {"loss": loss.item(), "mse": mse.item(), "cosine_similarity": cos.item()}


# ──────────────────────────────────────────────
# Train / Eval loops
# ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "cosine_similarity": 0.0}
    n = 0
    for x_t, x_next, _ in loader:
        x_t, x_next = x_t.to(device), x_next.to(device)
        optimizer.zero_grad()
        out = model(x_t, task="predictive")
        loss, m = criterion(out["x_next_pred"], x_next)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = x_t.size(0)
        for k in totals:
            totals[k] += m[k] * bs
        n += bs
    return {f"train_{k}": v / n for k, v in totals.items()}


def evaluate(model, loader, criterion, device):
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "cosine_similarity": 0.0, "mae": 0.0}
    n = 0
    with torch.no_grad():
        for x_t, x_next, _ in loader:
            x_t, x_next = x_t.to(device), x_next.to(device)
            out = model(x_t, task="predictive")
            pred = out["x_next_pred"]
            loss, m = criterion(pred, x_next)
            mae = F.l1_loss(pred, x_next).item()
            bs = x_t.size(0)
            totals["loss"] += m["loss"] * bs
            totals["mse"] += m["mse"] * bs
            totals["cosine_similarity"] += m["cosine_similarity"] * bs
            totals["mae"] += mae * bs
            n += bs
    return {f"val_{k}": v / n for k, v in totals.items()}


# ──────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────

def run_single(
    run_name: str,
    model: nn.Module,
    data_dir: Path,
    out_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    warmup_epochs: int = 3,
    patience: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Run: {run_name}  |  Params: {total_params:,}  |  Device: {device}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, mode="predictive", layout="NTC"
    )

    criterion = CompositeLoss(alpha=1.0, beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, lr)
    stopper = EarlyStopper(patience=patience)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{run_name}_best.pt"
    history = []
    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        ep_t = time.time()
        current_lr = scheduler.step(epoch - 1)
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate(model, val_loader, criterion, device)
        ep_dur = time.time() - ep_t

        row = {"epoch": epoch, "lr": round(current_lr, 6), "duration_sec": round(ep_dur, 2),
               **train_m, **val_m}
        history.append(row)

        print(f"  Ep {epoch:3d}/{epochs} | lr={current_lr:.5f} | "
              f"Train Loss={train_m['train_loss']:.4f} MSE={train_m['train_mse']:.4f} | "
              f"Val Loss={val_m['val_loss']:.4f} MSE={val_m['val_mse']:.4f} "
              f"CosSim={val_m['val_cosine_similarity']:.4f} ({ep_dur:.1f}s)")

        if val_m["val_loss"] < best_val_loss:
            best_val_loss = val_m["val_loss"]
            torch.save({
                "epoch": epoch, "run_name": run_name,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "total_params": total_params,
            }, ckpt_path)

        if stopper(val_m["val_loss"]):
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    # Test evaluation
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_m = evaluate(model, test_loader, criterion, device)

    print(f"\n  Test  MSE={test_m['val_mse']:.4f}  MAE={test_m['val_mae']:.4f}  "
          f"CosSim={test_m['val_cosine_similarity']:.4f}")

    results = {
        "run_name": run_name,
        "total_params": total_params,
        "total_time_sec": round(time.time() - t0, 2),
        "epochs_run": len(history),
        "best_val_loss": best_val_loss,
        "test_metrics": test_m,
        "history": history,
    }

    with open(out_dir / f"{run_name}_history.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


# ──────────────────────────────────────────────
# Ablation Driver
# ──────────────────────────────────────────────

ABLATION_CONFIGS = {
    "quantum_4q_1l": dict(
        model_cls="quantum",
        n_qubits=4, n_quantum_layers=1,
        spatial_dim=16, temporal_dim=16, quantum_dim=8,
    ),
    "quantum_4q_2l": dict(
        model_cls="quantum",
        n_qubits=4, n_quantum_layers=2,
        spatial_dim=16, temporal_dim=16, quantum_dim=8,
    ),
    "quantum_6q_1l": dict(
        model_cls="quantum",
        n_qubits=6, n_quantum_layers=1,
        spatial_dim=16, temporal_dim=16, quantum_dim=8,
    ),
    "classical_baseline": dict(
        model_cls="classical",
        spatial_dim=24, temporal_dim=24, fused_dim=48,
    ),
}


def build_model(cfg: dict) -> nn.Module:
    if cfg["model_cls"] == "quantum":
        return QuantumNeuralDigitalTwin(
            n_channels=19, n_timesteps=500, n_subjects=36,
            n_qubits=cfg["n_qubits"],
            n_quantum_layers=cfg["n_quantum_layers"],
            spatial_dim=cfg["spatial_dim"],
            temporal_dim=cfg["temporal_dim"],
            quantum_dim=cfg["quantum_dim"],
        )
    else:
        return ClassicalDigitalTwin(
            n_channels=19, n_timesteps=500, n_subjects=36,
            spatial_dim=cfg["spatial_dim"],
            temporal_dim=cfg["temporal_dim"],
            fused_dim=cfg["fused_dim"],
        )


def run_ablations(
    data_dir: Path = Path("./data/processed"),
    out_dir: Path = Path("./results/ablations"),
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    warmup_epochs: int = 3,
    patience: int = 7,
    runs: list = None,
):
    if runs is None:
        runs = list(ABLATION_CONFIGS.keys())

    summary_path = out_dir / "ablation_summary.json"
    all_results = {}
    if summary_path.exists():
        try:
            all_results = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            all_results = {}
    for run_name in runs:
        cfg = ABLATION_CONFIGS[run_name]
        model = build_model(cfg)
        res = run_single(
            run_name=run_name,
            model=model,
            data_dir=data_dir,
            out_dir=out_dir / run_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            warmup_epochs=warmup_epochs,
            patience=patience,
        )
        all_results[run_name] = {
            "total_params": res["total_params"],
            "epochs_run": res["epochs_run"],
            "best_val_loss": round(res["best_val_loss"], 4),
            "test_mse": round(res["test_metrics"]["val_mse"], 4),
            "test_mae": round(res["test_metrics"]["val_mae"], 4),
            "test_cosine_sim": round(res["test_metrics"]["val_cosine_similarity"], 4),
            "total_time_sec": res["total_time_sec"],
        }

    # Summary table
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print(f"{'Run':<22} {'Params':>10} {'Ep':>4} {'BestVal':>8} {'TestMSE':>8} {'TestMAE':>8} {'CosSim':>8}")
    print("-"*70)
    for run_name, r in all_results.items():
        print(f"{run_name:<22} {r['total_params']:>10,} {r['epochs_run']:>4} "
              f"{r['best_val_loss']:>8.4f} {r['test_mse']:>8.4f} "
              f"{r['test_mae']:>8.4f} {r['test_cosine_sim']:>8.4f}")
    print("="*70)
    print(f"Saved: {summary_path}")
    return all_results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "ablation"], default="ablation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Subset of ablation runs. Choices: " + ", ".join(ABLATION_CONFIGS.keys()))
    args = parser.parse_args()

    if args.mode == "ablation":
        run_ablations(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_epochs=args.warmup,
            patience=args.patience,
            runs=args.runs,
        )
    else:
        cfg = ABLATION_CONFIGS["quantum_4q_1l"]
        model = build_model(cfg)
        run_single(
            run_name="quantum_4q_1l_extended",
            model=model,
            data_dir=Path("./data/processed"),
            out_dir=Path("./results/ablations/quantum_4q_1l"),
            epochs=args.epochs,
            lr=args.lr,
            warmup_epochs=args.warmup,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
