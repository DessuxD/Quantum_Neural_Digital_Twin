"""
state.py
========
Digital Twin State Representation & Sequential Update Engine.
Implements the formal mathematical formulation from Paper Section IX:

    BT_t = [z_t, s_t, p_t, h_t]   (Eq. 15)

Where:
    - z_t: Learned latent feature representation (hybrid or classical embedding)
    - s_t: Estimated brain/cognitive physiological state vector (frequency bands, amplitude envelope)
    - p_t: Prediction-confidence vector (derived via Monte Carlo Dropout epistemic uncertainty)
    - h_t: Historical recursive trajectory information & drift metrics

State Update Function:
    BT_{t+1} = U(BT_t, EEG_{t+1})  (Eq. 16)
Sequential Chain:
    EEG_1 -> BT_1 -> EEG_2 -> BT_2 -> ... -> EEG_t -> BT_t  (Eq. 17)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8",
    "T3", "T4", "C3", "C4", "T5", "T6",
    "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"
]

@dataclass
class PhysiologicalState:
    """Represents s_t: physiological frequency band power & spectral summary."""
    delta_power: float  # 0.5 - 4 Hz
    theta_power: float  # 4 - 8 Hz (mental workload)
    alpha_power: float  # 8 - 13 Hz (relaxation / attentional suppression)
    beta_power: float   # 13 - 30 Hz (active cognitive processing)
    gamma_power: float  # 30 - 45 Hz
    theta_beta_ratio: float  # Classic cognitive fatigue / workload biomarker
    mean_amplitude: float
    channel_variance: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_power": round(self.delta_power, 4),
            "theta_power": round(self.theta_power, 4),
            "alpha_power": round(self.alpha_power, 4),
            "beta_power": round(self.beta_power, 4),
            "gamma_power": round(self.gamma_power, 4),
            "theta_beta_ratio": round(self.theta_beta_ratio, 4),
            "mean_amplitude": round(self.mean_amplitude, 4),
            "channel_variance": [round(v, 4) for v in self.channel_variance],
        }

@dataclass
class ConfidenceVector:
    """Represents p_t: prediction confidence and epistemic uncertainty."""
    overall_confidence: float  # 0.0 (high uncertainty) to 1.0 (high certainty)
    per_channel_uncertainty: List[float]  # Standard deviation across MC Dropout passes
    per_channel_confidence: List[float]   # 1 / (1 + std)
    mean_epistemic_std: float
    cosine_coherence: float  # Temporal alignment score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_confidence": round(self.overall_confidence, 4),
            "mean_epistemic_std": round(self.mean_epistemic_std, 4),
            "cosine_coherence": round(self.cosine_coherence, 4),
            "per_channel_uncertainty": [round(u, 4) for u in self.per_channel_uncertainty],
            "per_channel_confidence": [round(c, 4) for c in self.per_channel_confidence],
        }

@dataclass
class HistoryVector:
    """Represents h_t: historical state trajectory and recursive dynamics."""
    time_index: int
    subject_id: str
    cumulative_drift: float  # Cumulative Euclidean distance traversed in latent space
    instantaneous_velocity: float  # Distance from z_{t-1} to z_t
    ema_latent: List[float]  # Exponential moving average of latent state
    state_stability_score: float  # Smoothness of the trajectory

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_index": self.time_index,
            "subject_id": self.subject_id,
            "cumulative_drift": round(self.cumulative_drift, 4),
            "instantaneous_velocity": round(self.instantaneous_velocity, 4),
            "state_stability_score": round(self.state_stability_score, 4),
            "ema_latent": [round(v, 4) for v in self.ema_latent[:8]],  # preview top dims
        }

@dataclass
class DigitalTwinState:
    """
    Formal Digital Twin State representation:
        BT_t = [z_t, s_t, p_t, h_t]
    """
    z_t: List[float]                    # Latent state vector (dim: 40 or 48)
    s_t: PhysiologicalState             # Biological/cognitive state summary
    p_t: ConfidenceVector               # Uncertainty and confidence vector
    h_t: HistoryVector                  # Trajectory history and temporal continuity
    predicted_next_eeg: Optional[List[List[float]]] = None  # (500 timesteps x 19 channels)
    quantum_angles: Optional[List[float]] = None            # Qubit rotation angles (if hybrid)
    quantum_expectations: Optional[List[float]] = None      # <Z_i> Pauli-Z expectations

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "z_latent_dim": len(self.z_t),
            "z_latent_sample": [round(v, 4) for v in self.z_t[:8]],
            "s_physiological": self.s_t.to_dict(),
            "p_confidence": self.p_t.to_dict(),
            "h_history": self.h_t.to_dict(),
        }
        if self.quantum_angles is not None:
            d["quantum_angles"] = [round(a, 4) for a in self.quantum_angles]
        if self.quantum_expectations is not None:
            d["quantum_expectations"] = [round(e, 4) for e in self.quantum_expectations]
        return d


class DigitalTwinUpdater:
    """
    Executes the sequential state update function:
        BT_{t+1} = U(BT_t, EEG_{t+1})

    Maintains subject-specific state memory, computes Monte Carlo Dropout
    uncertainty bounds, estimates physiological band powers, and updates trajectory.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        subject_id: str = "s00",
        mc_passes: int = 5,
        ema_alpha: float = 0.3,
        sampling_rate: float = 250.0,
    ):
        self.model = model
        self.device = device
        self.subject_id = subject_id
        self.mc_passes = mc_passes
        self.ema_alpha = ema_alpha
        self.sampling_rate = sampling_rate

        self.current_state: Optional[DigitalTwinState] = None
        self.time_step: int = 0
        self.previous_z: Optional[np.ndarray] = None
        self.ema_z: Optional[np.ndarray] = None
        self.cumulative_drift: float = 0.0
        self.velocity_history: List[float] = []

    def reset(self, subject_id: Optional[str] = None):
        """Reset internal history for a new subject recording."""
        if subject_id:
            self.subject_id = subject_id
        self.current_state = None
        self.time_step = 0
        self.previous_z = None
        self.ema_z = None
        self.cumulative_drift = 0.0
        self.velocity_history = []

    def _compute_band_powers(self, eeg_window: np.ndarray) -> PhysiologicalState:
        """
        Compute standard EEG frequency band powers using FFT:
        Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-45Hz).
        eeg_window: shape (T=500, C=19).
        """
        T, C = eeg_window.shape
        fft_vals = np.abs(np.fft.rfft(eeg_window, axis=0)) ** 2
        freqs = np.fft.rfftfreq(T, d=1.0 / self.sampling_rate)

        def get_band(fmin, fmax):
            mask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(mask):
                return 1e-6
            return float(np.mean(fft_vals[mask, :]))

        delta = get_band(0.5, 4.0)
        theta = get_band(4.0, 8.0)
        alpha = get_band(8.0, 13.0)
        beta  = get_band(13.0, 30.0)
        gamma = get_band(30.0, 45.0)

        tbr = theta / max(beta, 1e-6)
        mean_amp = float(np.mean(np.abs(eeg_window)))
        var_per_ch = [float(v) for v in np.var(eeg_window, axis=0)]

        return PhysiologicalState(
            delta_power=delta,
            theta_power=theta,
            alpha_power=alpha,
            beta_power=beta,
            gamma_power=gamma,
            theta_beta_ratio=tbr,
            mean_amplitude=mean_amp,
            channel_variance=var_per_ch,
        )

    def _mc_dropout_predict(
        self,
        x_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes Monte Carlo Dropout forward passes to estimate epistemic uncertainty.
        Returns:
            mean_pred: (1, 500, 19)
            std_pred:  (1, 500, 19)
            z_fused:   (1, D)
        """
        # Ensure dropout is active for stochastic sampling
        self.model.train()

        predictions = []
        z_fused_sample = None
        with torch.no_grad():
            for i in range(self.mc_passes):
                out = self.model(x_tensor, task="predictive")
                predictions.append(out["x_next_pred"])
                if z_fused_sample is None and "fused" in out:
                    z_fused_sample = out["fused"]

        # Back to eval mode
        self.model.eval()

        preds_stacked = torch.stack(predictions, dim=0)  # (mc_passes, 1, 500, 19)
        mean_pred = torch.mean(preds_stacked, dim=0)
        std_pred  = torch.std(preds_stacked, dim=0)

        # In case model outputs latent during deterministic encode
        with torch.no_grad():
            enc = self.model.encode(x_tensor)
            z_fused = enc["fused"]

        return mean_pred, std_pred, z_fused

    def update(
        self,
        eeg_window: np.ndarray,
        ground_truth_next: Optional[np.ndarray] = None
    ) -> DigitalTwinState:
        """
        Perform sequential update: BT_{t+1} = U(BT_t, EEG_{t+1})

        Parameters:
            eeg_window: numpy array of shape (500, 19)
            ground_truth_next: optional next window for validation metrics
        """
        self.time_step += 1
        x_tensor = torch.from_numpy(eeg_window).float().unsqueeze(0).to(self.device)

        # 1. Prediction & Epistemic Uncertainty Estimation (p_t)
        mean_pred, std_pred, z_tensor = self._mc_dropout_predict(x_tensor)
        z_np = z_tensor.squeeze(0).cpu().numpy()
        pred_np = mean_pred.squeeze(0).cpu().numpy()
        std_np  = std_pred.squeeze(0).cpu().numpy()

        per_ch_uncertainty = [float(u) for u in np.mean(std_np, axis=0)]
        per_ch_confidence  = [float(1.0 / (1.0 + u)) for u in per_ch_uncertainty]
        mean_std = float(np.mean(std_np))
        overall_conf = float(1.0 / (1.0 + mean_std))

        # Cosine coherence
        if ground_truth_next is not None:
            p_flat = pred_np.flatten()
            t_flat = ground_truth_next.flatten()
            cos_sim = float(np.dot(p_flat, t_flat) / (np.linalg.norm(p_flat) * np.linalg.norm(t_flat) + 1e-8))
        else:
            cos_sim = 1.0

        p_t = ConfidenceVector(
            overall_confidence=overall_conf,
            per_channel_uncertainty=per_ch_uncertainty,
            per_channel_confidence=per_ch_confidence,
            mean_epistemic_std=mean_std,
            cosine_coherence=cos_sim,
        )

        # 2. Physiological State Estimation (s_t)
        s_t = self._compute_band_powers(eeg_window)

        # 3. History Trajectory Update (h_t)
        if self.previous_z is None:
            velocity = 0.0
            self.ema_z = z_np.copy()
        else:
            velocity = float(np.linalg.norm(z_np - self.previous_z))
            self.cumulative_drift += velocity
            self.ema_z = (1.0 - self.ema_alpha) * self.ema_z + self.ema_alpha * z_np

        self.previous_z = z_np.copy()
        self.velocity_history.append(velocity)

        # Stability: inverse variance of velocity
        if len(self.velocity_history) > 2:
            vel_std = float(np.std(self.velocity_history[-10:]))
            stability = float(1.0 / (1.0 + vel_std))
        else:
            stability = 1.0

        h_t = HistoryVector(
            time_index=self.time_step,
            subject_id=self.subject_id,
            cumulative_drift=self.cumulative_drift,
            instantaneous_velocity=velocity,
            ema_latent=self.ema_z.tolist(),
            state_stability_score=stability,
        )

        # 4. Extract Quantum variables if available
        q_angles = None
        q_expectations = None
        with torch.no_grad():
            enc = self.model.encode(x_tensor)
            if "quantum_angles" in enc:
                q_angles = enc["quantum_angles"].squeeze(0).cpu().tolist()
            if "quantum" in enc:
                q_expectations = enc["quantum"].squeeze(0).cpu().tolist()

        # 5. Assemble Digital Twin State
        bt_state = DigitalTwinState(
            z_t=z_np.tolist(),
            s_t=s_t,
            p_t=p_t,
            h_t=h_t,
            predicted_next_eeg=pred_np.tolist(),
            quantum_angles=q_angles,
            quantum_expectations=q_expectations,
        )

        self.current_state = bt_state
        return bt_state
