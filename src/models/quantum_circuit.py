"""
quantum_circuit.py
==================
Parameterized Variational Quantum Circuit (VQC) using PennyLane.
Implements:
1. Angle Embedding mapping classical features to quantum Hilbert space.
2. Parametrized rotation gates (Rot) and entangling CNOT layers.
3. Multi-wire Pauli-Z expectation value measurements.
4. Seamless PyTorch autograd integration via qml.qnn.TorchLayer.
"""

from typing import Dict, Tuple, Optional
import pennylane as qml
import torch
import torch.nn as nn

def build_vqc_qnode(
    n_qubits: int = 4,
    n_layers: int = 2,
    rotation_type: str = "Y",
    entanglement: str = "ring"
):
    """
    Construct a PennyLane QNode with PyTorch backpropagation interface.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # 1. Quantum State Preparation (Angle Embedding)
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation_type)

        # 2. Variational Entangling Layers
        for l in range(n_layers):
            # Parametric single-qubit rotations
            for i in range(n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            # Entangling CNOT gates
            if entanglement == "ring":
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            elif entanglement == "linear":
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

        # 3. Measurement: Pauli-Z expectation values on all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return circuit, weight_shapes

class VariationalQuantumEncoder(nn.Module):
    """
    Hybrid PyTorch layer incorporating PennyLane VQC with input/output mappings.

    Parameters:
        n_qubits: Number of quantum wires/qubits (default 4).
        n_layers: Depth of variational quantum layers (default 2).
        output_dim: Dimension of post-quantum classical projection (default 16).
        rotation_type: Angle embedding rotation ('Y', 'X', or 'Z').
    """
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        output_dim: int = 16,
        rotation_type: str = "Y",
        entanglement: str = "ring"
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_dim = output_dim

        # Build QNode and TorchLayer
        circuit, weight_shapes = build_vqc_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_type=rotation_type,
            entanglement=entanglement
        )
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Optional linear expansion/projection after quantum measurement
        self.post_quantum = nn.Sequential(
            nn.Linear(n_qubits, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU()
        )

    def forward(self, quantum_angles: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch_size, n_qubits) angles in [-pi, pi]
        Output: (batch_size, output_dim) quantum-transformed feature representations
        """
        # Quantum forward pass: (batch_size, n_qubits)
        q_out = self.q_layer(quantum_angles)
        # Classical post-processing: (batch_size, output_dim)
        out = self.post_quantum(q_out)
        return out
