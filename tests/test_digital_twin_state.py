"""
test_digital_twin_state.py
==========================
Unit tests for the formal Digital Twin state representation:
    BT_t = [z_t, s_t, p_t, h_t]
and sequential state updater:
    BT_{t+1} = U(BT_t, EEG_{t+1})
"""

import unittest
import numpy as np
import torch

from src.models.classical_baseline import ClassicalDigitalTwin
from src.digital_twin.state import DigitalTwinUpdater, DigitalTwinState, PhysiologicalState, ConfidenceVector, HistoryVector


class TestDigitalTwinState(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = ClassicalDigitalTwin(
            n_channels=19, n_timesteps=500, spatial_dim=16, temporal_dim=16, fused_dim=32
        )
        self.updater = DigitalTwinUpdater(
            model=self.model,
            device=self.device,
            subject_id="s30",
            mc_passes=2,
            sampling_rate=250.0
        )

    def test_state_initialization_and_first_step(self):
        w_t = np.random.randn(500, 19).astype(np.float32)
        state = self.updater.update(w_t)

        self.assertIsInstance(state, DigitalTwinState)
        self.assertEqual(len(state.z_t), 32)
        self.assertIsInstance(state.s_t, PhysiologicalState)
        self.assertIsInstance(state.p_t, ConfidenceVector)
        self.assertIsInstance(state.h_t, HistoryVector)

        self.assertEqual(state.h_t.time_index, 1)
        self.assertEqual(state.h_t.instantaneous_velocity, 0.0)
        self.assertEqual(state.h_t.cumulative_drift, 0.0)

    def test_sequential_update_trajectory_drift(self):
        w_t1 = np.random.randn(500, 19).astype(np.float32)
        w_t2 = np.random.randn(500, 19).astype(np.float32)

        s1 = self.updater.update(w_t1)
        s2 = self.updater.update(w_t2)

        self.assertEqual(s2.h_t.time_index, 2)
        self.assertGreater(s2.h_t.instantaneous_velocity, 0.0)
        self.assertGreater(s2.h_t.cumulative_drift, 0.0)
        self.assertGreater(s2.p_t.overall_confidence, 0.0)
        self.assertLessEqual(s2.p_t.overall_confidence, 1.0)

    def test_band_powers_computation(self):
        w_t = np.random.randn(500, 19).astype(np.float32)
        state = self.updater.update(w_t)

        self.assertGreater(state.s_t.alpha_power, 0.0)
        self.assertGreater(state.s_t.beta_power, 0.0)
        self.assertGreater(state.s_t.theta_beta_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
