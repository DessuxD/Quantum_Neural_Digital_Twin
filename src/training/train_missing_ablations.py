"""
train_missing_ablations.py
==========================
Trains the remaining paper-specified ablations and evaluates the Biometric Identification Head:
1. eegnet_only (Paper XII.E.1)
2. eegnet_bilstm_pca (Paper XII.E.3)
3. quantum_4q_1l_pca (Paper Eq. 19 & XII.E.4)
4. Biometric Head Fine-Tuning & Evaluation (Paper Objective 5 & Table I)
"""

import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import get_dataloaders
from src.training.extended_training import run_single, ABLATION_CONFIGS, build_model


def train_and_eval_biometrics(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
) -> dict:
    """
    Trains the biometric classification head on subject IDs (0-35)
    and reports Top-1 and Top-3 accuracy on the unseen test subjects.
    """
    model.eval()
    # Freeze encoder parameters so we only train the biometric classification head
    for name, p in model.named_parameters():
        if "biometric_head" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for x_t, _, sub_id in train_loader:
            x_t = x_t.to(device)
            sub_id = sub_id.to(device)
            optimizer.zero_grad()
            out = model(x_t, task="biometric")
            loss = criterion(out["subject_logits"], sub_id)
            loss.backward()
            optimizer.step()

    # Evaluation on test set
    model.eval()
    top1_correct = 0
    top3_correct = 0
    total = 0
    with torch.no_grad():
        for x_t, _, sub_id in test_loader:
            x_t = x_t.to(device)
            sub_id = sub_id.to(device)
            logits = model(x_t, task="biometric")["subject_logits"]
            preds = torch.topk(logits, k=3, dim=1).indices
            top1_correct += (preds[:, 0] == sub_id).sum().item()
            for k in range(3):
                top3_correct += (preds[:, k] == sub_id).sum().item()
            total += sub_id.size(0)

    top1_acc = float(top1_correct / total) if total > 0 else 0.0
    top3_acc = float(top3_correct / total) if total > 0 else 0.0
    return {"top1_accuracy": round(top1_acc, 4), "top3_accuracy": round(top3_acc, 4), "samples": total}


def main():
    device = torch.device("cpu")
    data_dir = Path("./data/processed")
    ablations_dir = Path("./results/ablations")
    summary_path = ablations_dir / "ablation_summary.json"

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir, batch_size=32, mode="predictive", layout="NTC"
    )

    existing_summary = {}
    if summary_path.exists():
        with open(summary_path, "r") as f:
            existing_summary = json.load(f)

    runs_to_train = ["eegnet_only", "eegnet_bilstm_pca", "quantum_4q_1l_pca"]

    print("=" * 70)
    print("PHASE 4: TRAINING MISSING PAPER ABLATIONS")
    print("=" * 70)

    for run_name in runs_to_train:
        cfg = ABLATION_CONFIGS[run_name]
        model = build_model(cfg)
        out_sub = ablations_dir / run_name
        res = run_single(
            run_name=run_name,
            model=model,
            data_dir=data_dir,
            out_dir=out_sub,
            epochs=5,
            batch_size=32,
            lr=1e-3,
            warmup_epochs=1,
            patience=5,
        )

        existing_summary[run_name] = {
            "total_params": res["total_params"],
            "epochs_run": res["epochs_run"],
            "best_val_loss": round(res["best_val_loss"], 4),
            "test_mse": round(res["test_metrics"]["val_mse"], 4),
            "test_mae": round(res["test_metrics"]["val_mae"], 4),
            "test_cosine_sim": round(res["test_metrics"]["val_cosine_similarity"], 4),
            "total_time_sec": res["total_time_sec"],
        }

    # Evaluate Biometric Head for Classical vs Quantum models
    print("\n" + "=" * 70)
    print("EVALUATING BIOMETRIC HEAD (SUBJECT IDENTIFICATION)")
    print("=" * 70)

    biometric_results = {}
    for eval_run in ["classical_baseline", "quantum_4q_1l", "quantum_4q_1l_pca"]:
        ckpt_file = ablations_dir / eval_run / f"{eval_run}_best.pt"
        if not ckpt_file.exists():
            continue
        cfg = ABLATION_CONFIGS[eval_run]
        m = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_file, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"], strict=False)

        bio_acc = train_and_eval_biometrics(
            m, train_loader, test_loader, device, epochs=5
        )
        biometric_results[eval_run] = bio_acc
        print(f"Biometric Subject ID ({eval_run}): Top-1 Acc = {bio_acc['top1_accuracy']*100:.1f}%, Top-3 Acc = {bio_acc['top3_acc']*100 if 'top3_acc' in bio_acc else bio_acc['top3_accuracy']*100:.1f}%")

    with open(ablations_dir / "biometric_evaluation.json", "w") as f:
        json.dump(biometric_results, f, indent=4)

    # Save comprehensive summary
    with open(summary_path, "w") as f:
        json.dump(existing_summary, f, indent=4)

    print("\n" + "=" * 70)
    print("ALL ABLATIONS SUMMARY COMPLETE")
    print(f"{'Run':<24} {'Params':>10} {'Ep':>4} {'TestMSE':>8} {'TestMAE':>8} {'CosSim':>8}")
    print("-" * 70)
    for k, v in existing_summary.items():
        print(f"{k:<24} {v['total_params']:>10,} {v['epochs_run']:>4} {v['test_mse']:>8.4f} {v['test_mae']:>8.4f} {v['test_cosine_sim']:>8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
