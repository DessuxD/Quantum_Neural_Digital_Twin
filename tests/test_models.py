"""
test_models.py
==============
Unit tests for EEGNet, BiLSTM, Dimension Reduction, Quantum VQC, and Hybrid Digital Twin.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from src.models.dimension_reduction import SpatialEEG_PCA, QuantumStateMapper
from src.models.eegnet import EEGNetEncoder
from src.models.bilstm import BiLSTMTemporalEncoder
from src.models.quantum_circuit import VariationalQuantumEncoder
from src.models.hybrid_twin import QuantumNeuralDigitalTwin, WaveformDecoder

class TestModels(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.n_timesteps = 500
        self.n_channels = 19
        self.dummy_input = torch.randn(self.batch_size, self.n_timesteps, self.n_channels)

    def test_spatial_pca(self):
        pca_model = SpatialEEG_PCA(n_components=4)
        synthetic_data = np.random.randn(20, 500, 19)
        pca_model.fit(synthetic_data)
        self.assertTrue(pca_model.is_fitted)
        self.assertEqual(len(pca_model.explained_variance_ratio), 4)

        transformed = pca_model.transform(synthetic_data)
        self.assertEqual(transformed.shape, (20, 500, 4))

        recon = pca_model.inverse_transform(transformed)
        self.assertEqual(recon.shape, (20, 500, 19))

    def test_quantum_state_mapper(self):
        mapper = QuantumStateMapper(in_features=32, n_qubits=4, scaling="tanh")
        x = torch.randn(self.batch_size, 32)
        angles = mapper(x)
        self.assertEqual(angles.shape, (self.batch_size, 4))
        # Verify strict bounds [-pi, pi]
        self.assertTrue(torch.all(angles >= -torch.pi))
        self.assertTrue(torch.all(angles <= torch.pi))

        # Check gradient flow
        loss = angles.sum()
        loss.backward()
        self.assertIsNotNone(mapper.projection[0].weight.grad)

    def test_eegnet_encoder(self):
        encoder = EEGNetEncoder(
            n_channels=self.n_channels,
            n_timesteps=self.n_timesteps,
            latent_dim=24
        )
        out = encoder(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, 24))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(encoder.conv1.weight.grad)

    def test_bilstm_encoder(self):
        encoder = BiLSTMTemporalEncoder(
            input_dim=self.n_channels,
            hidden_dim=24,
            latent_dim=24
        )
        out = encoder(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, 24))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(encoder.lstm.weight_ih_l0.grad)

    def test_variational_quantum_encoder(self):
        vqc = VariationalQuantumEncoder(n_qubits=4, n_layers=2, output_dim=16)
        angles = torch.randn(self.batch_size, 4)
        q_out = vqc(angles)
        self.assertEqual(q_out.shape, (self.batch_size, 16))

        loss = q_out.sum()
        loss.backward()
        self.assertIsNotNone(vqc.q_layer.weights.grad)

    def test_waveform_decoder(self):
        decoder = WaveformDecoder(latent_dim=64, n_channels=self.n_channels, n_timesteps=self.n_timesteps)
        z = torch.randn(self.batch_size, 64)
        out = decoder(z)
        self.assertEqual(out.shape, (self.batch_size, self.n_timesteps, self.n_channels))

    def test_quantum_neural_digital_twin_end_to_end(self):
        model = QuantumNeuralDigitalTwin(
            n_channels=self.n_channels,
            n_timesteps=self.n_timesteps,
            n_qubits=4,
            n_quantum_layers=1,
            spatial_dim=16,
            temporal_dim=16,
            quantum_dim=8,
            n_subjects=36
        )

        # Forward predictive task
        res = model(self.dummy_input, task="all")
        self.assertIn("x_next_pred", res)
        self.assertIn("x_recon", res)
        self.assertIn("subject_logits", res)
        self.assertIn("quantum_angles", res)
        self.assertIn("quantum", res)

        self.assertEqual(res["x_next_pred"].shape, (self.batch_size, self.n_timesteps, self.n_channels))
        self.assertEqual(res["x_recon"].shape, (self.batch_size, self.n_timesteps, self.n_channels))
        self.assertEqual(res["subject_logits"].shape, (self.batch_size, 36))

        # Check multi-objective backpropagation
        target_next = torch.randn_like(self.dummy_input)
        target_subject = torch.tensor([0, 1, 2, 3])

        loss_pred = nn.functional.mse_loss(res["x_next_pred"], target_next)
        loss_recon = nn.functional.mse_loss(res["x_recon"], self.dummy_input)
        loss_bio = nn.functional.cross_entropy(res["subject_logits"], target_subject)
        total_loss = loss_pred + 0.5 * loss_recon + 0.1 * loss_bio

        total_loss.backward()

        # Check gradient flow to EEGNet and Quantum Layer
        self.assertIsNotNone(model.spatial_encoder.conv1.weight.grad)
        self.assertIsNotNone(model.temporal_encoder.lstm.weight_ih_l0.grad)
        self.assertIsNotNone(model.quantum_encoder.q_layer.weights.grad)

if __name__ == "__main__":
    unittest.main()
