"""
app.py
======
Interactive Web Dashboard & Faculty Defense Interface
for the Quantum Neural Digital Twin of the Human Brain.

Run with:
    python -m src.web.app
or
    python src/web/app.py
Access at: http://localhost:5000
"""

from __future__ import annotations
import os, sys, json, io, re
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.hybrid_twin import QuantumNeuralDigitalTwin
from src.models.classical_baseline import ClassicalDigitalTwin
from src.models.ablation_variants import (
    EEGNetOnlyDigitalTwin,
    EEGNetBiLSTM_PCADigitalTwin,
    QuantumNeuralDigitalTwin_PCA
)
from src.digital_twin.state import DigitalTwinUpdater, CHANNEL_NAMES

app = Flask(
    __name__,
    template_folder=str(ROOT / "src/web/templates"),
    static_folder=str(ROOT / "src/web/static")
)

DEVICE = torch.device("cpu")
CACHED_MODELS: Dict[str, torch.nn.Module] = {}
CACHED_UPDATERS: Dict[str, DigitalTwinUpdater] = {}


def load_model(model_name: str) -> Optional[torch.nn.Module]:
    """Load and cache model checkpoint."""
    if model_name in CACHED_MODELS:
        return CACHED_MODELS[model_name]

    abl_dir = ROOT / "results/ablations" / model_name
    ckpt_file = abl_dir / f"{model_name}_best.pt"

    # Fallback to general checkpoints if needed
    if not ckpt_file.exists():
        if model_name == "quantum_4q_1l":
            ckpt_file = ROOT / "checkpoints/quantum_digital_twin_best.pt"

    if not ckpt_file.exists():
        return None

    try:
        ckpt = torch.load(ckpt_file, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"]

        if model_name == "eegnet_only":
            m = EEGNetOnlyDigitalTwin(n_channels=19, n_timesteps=500, latent_dim=32)
        elif model_name == "eegnet_bilstm_pca":
            m = EEGNetBiLSTM_PCADigitalTwin(n_channels=19, n_timesteps=500, pca_dim=4, fused_dim=32)
        elif model_name == "quantum_4q_1l_pca":
            m = QuantumNeuralDigitalTwin_PCA(n_channels=19, n_timesteps=500, n_qubits=4, n_quantum_layers=1, quantum_dim=8)
        elif "classical" in model_name:
            m = ClassicalDigitalTwin(n_channels=19, n_timesteps=500, spatial_dim=24, temporal_dim=24, fused_dim=48)
        elif model_name == "quantum_6q_1l":
            m = QuantumNeuralDigitalTwin(n_channels=19, n_timesteps=500, n_qubits=6, n_quantum_layers=1, spatial_dim=16, temporal_dim=16, quantum_dim=8)
        elif model_name == "quantum_4q_2l":
            m = QuantumNeuralDigitalTwin(n_channels=19, n_timesteps=500, n_qubits=4, n_quantum_layers=2, spatial_dim=16, temporal_dim=16, quantum_dim=8)
        else:
            m = QuantumNeuralDigitalTwin(n_channels=19, n_timesteps=500, n_qubits=4, n_quantum_layers=1, spatial_dim=16, temporal_dim=16, quantum_dim=8)

        m.load_state_dict(state_dict, strict=False)
        m.to(DEVICE)
        m.eval()
        CACHED_MODELS[model_name] = m
        return m
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None


def get_updater(model_name: str, subject_id: str) -> Optional[DigitalTwinUpdater]:
    key = f"{model_name}_{subject_id}"
    if key in CACHED_UPDATERS:
        return CACHED_UPDATERS[key]
    m = load_model(model_name)
    if m is None:
        return None
    updater = DigitalTwinUpdater(m, DEVICE, subject_id=subject_id, mc_passes=3)
    CACHED_UPDATERS[key] = updater
    return updater


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    summary_file = ROOT / "results/ablations/ablation_summary.json"
    abl_summary = {}
    if summary_file.exists():
        with open(summary_file) as f:
            abl_summary = json.load(f)

    bio_file = ROOT / "results/ablations/biometric_evaluation.json"
    bio_summary = {}
    if bio_file.exists():
        with open(bio_file) as f:
            bio_summary = json.load(f)

    return jsonify({
        "project": "Quantum Neural Digital Twin of the Human Brain",
        "dataset": "PhysioNet EEG During Mental Arithmetic Tasks (36 Subjects, 19 Channels, 250 Hz)",
        "available_channels": CHANNEL_NAMES,
        "models_count": len(abl_summary),
        "ablations": abl_summary,
        "biometrics": bio_summary,
        "device": str(DEVICE),
    })


@app.route("/api/subjects")
def api_subjects():
    subjects = []
    # Check for custom uploaded subjects first
    cont_dir = ROOT / "data/processed/continuous"
    if cont_dir.exists():
        for custom_path in sorted(cont_dir.glob("custom_*_clean.npy")):
            cid = custom_path.stem.replace("_clean", "")
            disp_name = cid.replace("custom_", "").replace("_", " ").title()
            subjects.append({"id": cid, "split": "custom", "label": f"Custom: {disp_name}"})

    for s_idx in range(36):
        sid = f"s{s_idx:02d}"
        if s_idx < 24:
            split = "train"
        elif s_idx < 30:
            split = "val"
        else:
            split = "test"  # Unseen test subjects (s30-s35)
        subjects.append({"id": sid, "split": split, "label": f"Subject {s_idx:02d} ({split.upper()})"})
    return jsonify(subjects)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = file.filename.lower()
    raw_stem = Path(file.filename).stem
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_stem)[:30]
    if not safe_name:
        safe_name = "recording"

    try:
        if filename.endswith(".npy"):
            arr = np.load(file.stream)
            if arr.ndim != 2:
                return jsonify({"error": f"NPY array must be 2D, got shape {arr.shape}."}), 400
            if arr.shape[0] == 19 and arr.shape[1] > 19:
                arr = arr.T  # Transpose to (N, 19)
            elif arr.shape[1] != 19:
                if arr.shape[1] > 19:
                    arr = arr[:, :19]
                else:
                    pad = np.zeros((arr.shape[0], 19 - arr.shape[1]), dtype=arr.dtype)
                    arr = np.hstack([arr, pad])

        elif filename.endswith(".csv"):
            df = pd.read_csv(file.stream)
            col_map = {str(c).strip().lower(): c for c in df.columns}
            matched_cols = []
            for ch in CHANNEL_NAMES:
                if ch.lower() in col_map:
                    matched_cols.append(col_map[ch.lower()])

            if len(matched_cols) == 19:
                arr = df[matched_cols].values
            else:
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.shape[1] == 0:
                    return jsonify({"error": "CSV contains no numerical columns."}), 400
                if numeric_df.shape[1] >= 19:
                    arr = numeric_df.iloc[:, :19].values
                else:
                    arr = numeric_df.values
                    pad = np.zeros((arr.shape[0], 19 - arr.shape[1]), dtype=np.float32)
                    arr = np.hstack([arr, pad])
        else:
            return jsonify({"error": "Unsupported file format. Please upload a .csv or .npy file."}), 400

        # Replace NaN / Inf with 0.0
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure length (need at least 2 consecutive windows: 500 + 250 = 750 samples)
        if arr.shape[0] < 750:
            return jsonify({
                "error": f"Signal length is {arr.shape[0]} samples. Digital twin requires at least 750 samples (3.0 seconds at 250 Hz)."
            }), 400

        # Channel-wise standard Z-score normalization
        means = np.mean(arr, axis=0, keepdims=True)
        stds = np.std(arr, axis=0, keepdims=True)
        stds[stds == 0] = 1.0
        arr_normalized = (arr - means) / stds

        # Persist to continuous processed data
        cont_dir = ROOT / "data/processed/continuous"
        cont_dir.mkdir(parents=True, exist_ok=True)
        out_file = cont_dir / f"custom_{safe_name}_clean.npy"
        np.save(out_file, arr_normalized.astype(np.float32))

        subject_id = f"custom_{safe_name}"
        num_windows = (arr.shape[0] - 500) // 250

        return jsonify({
            "success": True,
            "subject_id": subject_id,
            "label": f"Custom: {safe_name.replace('_', ' ').title()}",
            "num_samples": int(arr.shape[0]),
            "num_windows": int(num_windows),
            "channels_used": 19,
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse and process EEG file: {str(e)}"}), 500


@app.route("/api/step", methods=["POST"])
def api_step():
    data = request.json or {}
    subject_id = data.get("subject_id", "s30")
    window_idx = int(data.get("window_idx", 0))
    model_name = data.get("model_name", "quantum_4q_1l")
    channel_idx = int(data.get("channel_idx", 0))  # 0 = Fp1

    cont_file = ROOT / f"data/processed/continuous/{subject_id}_clean.npy"
    if not cont_file.exists():
        return jsonify({"error": f"Subject continuous file {cont_file.name} not found"}), 404

    cont_data = np.load(cont_file)  # (15500, 19)
    max_windows = (cont_data.shape[0] - 500) // 250
    window_idx = max(0, min(window_idx, max_windows - 1))

    # Current window W_t
    w_t = cont_data[window_idx * 250 : window_idx * 250 + 500]
    # Ground truth next window W_{t+1}
    w_next = cont_data[(window_idx + 1) * 250 : (window_idx + 1) * 250 + 500]

    updater = get_updater(model_name, subject_id)
    if updater is None:
        return jsonify({"error": f"Model {model_name} could not be loaded"}), 400

    # Execute State Update: BT_{t+1} = U(BT_t, EEG_{t+1})
    twin_state = updater.update(w_t, ground_truth_next=w_next)

    pred_eeg = np.array(twin_state.predicted_next_eeg)  # (500, 19)
    t_axis = [round(float(t), 3) for t in np.linspace(0.0, 2.0, 500)]

    # Extract target channel waveform
    actual_ch = [round(float(v), 3) for v in w_next[:, channel_idx]]
    pred_ch   = [round(float(v), 3) for v in pred_eeg[:, channel_idx]]
    std_ch    = [round(float(twin_state.p_t.per_channel_uncertainty[channel_idx]), 3)] * 500

    # Upper and lower uncertainty bounds (MC Dropout +- 2 std)
    upper_bound = [round(pred_ch[i] + 2.0 * std_ch[i], 3) for i in range(500)]
    lower_bound = [round(pred_ch[i] - 2.0 * std_ch[i], 3) for i in range(500)]

    return jsonify({
        "subject_id": subject_id,
        "window_idx": window_idx,
        "max_windows": max_windows,
        "model_name": model_name,
        "channel_name": CHANNEL_NAMES[channel_idx],
        "channel_idx": channel_idx,
        "time_axis": t_axis,
        "waveform": {
            "actual": actual_ch,
            "predicted": pred_ch,
            "uncertainty_upper": upper_bound,
            "uncertainty_lower": lower_bound,
        },
        "digital_twin_state": twin_state.to_dict(),
    })


@app.route("/api/figures/<filename>")
def api_figures(filename):
    fig_dir = ROOT / "results/figures"
    return send_from_directory(fig_dir, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Digital Twin Web Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
