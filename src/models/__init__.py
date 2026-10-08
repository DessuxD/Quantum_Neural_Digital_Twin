"""
src/models
==========
Neural, Quantum, and Hybrid Architectures for the Quantum Neural Digital Twin.
"""

from .eegnet import EEGNetEncoder
from .bilstm import BiLSTMTemporalEncoder
from .dimension_reduction import SpatialEEG_PCA, QuantumStateMapper
from .quantum_circuit import VariationalQuantumEncoder, build_vqc_qnode
from .hybrid_twin import QuantumNeuralDigitalTwin, WaveformDecoder
from .classical_baseline import ClassicalDigitalTwin

__all__ = [
    "EEGNetEncoder",
    "BiLSTMTemporalEncoder",
    "SpatialEEG_PCA",
    "QuantumStateMapper",
    "VariationalQuantumEncoder",
    "build_vqc_qnode",
    "QuantumNeuralDigitalTwin",
    "WaveformDecoder",
    "ClassicalDigitalTwin",
]
