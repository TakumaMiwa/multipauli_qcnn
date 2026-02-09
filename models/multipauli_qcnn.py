"""Multi-Pauli QCNN module for training.

This module defines a 9-qubit QCNN feature extractor for 3x3 image windows and a
PyTorch model that can be trained for classification tasks.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

__all__ = [
    "create_qcnn_9qubit",
    "MultiPauliQCNN",
    "build_model",
    "load_model_for_inference",
    "predict_logits",
    "predict_proba",
]


def _ring_entanglement(circuit: QuantumCircuit, qubits: Iterable[int]) -> None:
    """Apply ring CNOT entanglement over the given qubit indices."""
    qubits = list(qubits)
    for i, src in enumerate(qubits):
        dst = qubits[(i + 1) % len(qubits)]
        circuit.cx(src, dst)


def create_qcnn_9qubit(
    *,
    n_conv1: int = 18,
    n_pool1: int = 6,
    n_conv2: int = 8,
    n_pool2: int = 4,
    n_conv3: int = 4,
    n_pool3: int = 2,
) -> TorchConnector:
    """Build a 9-qubit QCNN block with configurable trainable parameter counts.

    Input:
      - 3x3 patch flattened to 9 values.
    Output:
      - [<X>, <Y>, <Z>] expectation values on qubit 0.
    """

    n_weights = n_conv1 + n_pool1 + n_conv2 + n_pool2 + n_conv3 + n_pool3
    x = ParameterVector("x", 9)
    w = ParameterVector("w", n_weights)

    qc = QuantumCircuit(9)

    # Data encoding
    for i in range(9):
        qc.ry(x[i], i)
    qc.barrier()

    wi = 0

    # Convolution block 1: local rotations + ring entanglement
    for q in range(9):
        qc.ry(w[wi], q)
        wi += 1
    for q in range(9):
        qc.rz(w[wi], q)
        wi += 1
    _ring_entanglement(qc, range(9))
    qc.barrier()

    # Pooling block 1: 9 -> 3 (to qubits 0, 3, 6)
    qc.crx(w[wi], 1, 0)
    qc.crx(w[wi + 1], 2, 0)
    qc.crx(w[wi + 2], 4, 3)
    qc.crx(w[wi + 3], 5, 3)
    qc.crx(w[wi + 4], 7, 6)
    qc.crx(w[wi + 5], 8, 6)
    wi += n_pool1
    qc.barrier()

    # Convolution block 2 on active qubits (0,3,6)
    active_3: List[int] = [0, 3, 6]
    for q in active_3:
        qc.ry(w[wi], q)
        wi += 1
    for q in active_3:
        qc.rz(w[wi], q)
        wi += 1
    _ring_entanglement(qc, active_3)
    qc.ry(w[wi], 0)
    qc.rz(w[wi + 1], 3)
    wi += n_conv2
    qc.barrier()

    # Pooling block 2: 3 -> 2 (to qubits 0, 6)
    qc.crx(w[wi], 3, 0)
    qc.crx(w[wi + 1], 3, 6)
    qc.ry(w[wi + 2], 0)
    qc.ry(w[wi + 3], 6)
    wi += n_pool2
    qc.barrier()

    # Convolution block 3 on qubits (0, 6)
    qc.ry(w[wi], 0)
    qc.ry(w[wi + 1], 6)
    qc.cx(0, 6)
    qc.rz(w[wi + 2], 0)
    qc.rz(w[wi + 3], 6)
    wi += n_conv3
    qc.barrier()

    # Pooling block 3: 2 -> 1 (to qubit 0)
    qc.crx(w[wi], 6, 0)
    qc.ry(w[wi + 1], 0)
    wi += n_pool3

    assert wi == n_weights, "Weight indexing mismatch in QCNN construction"

    observables = [
        SparsePauliOp.from_list([("IIIIIIIIX", 1.0)]),
        SparsePauliOp.from_list([("IIIIIIIIY", 1.0)]),
        SparsePauliOp.from_list([("IIIIIIIIZ", 1.0)]),
    ]

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=list(x),
        weight_params=list(w),
        observables=observables,
        estimator=Estimator(),
        input_gradients=False,
    )

    return TorchConnector(qnn)


class MultiPauliQCNN(nn.Module):
    """Trainable Multi-Pauli QCNN model using 3x3 windows and 9 qubits per patch."""

    def __init__(
        self,
        *,
        image_size: int = 28,
        window_size: int = 3,
        stride: int = 1,
        n_classes: int = 10,
        n_conv1: int = 18,
        n_pool1: int = 6,
        n_conv2: int = 8,
        n_pool2: int = 4,
        n_conv3: int = 4,
        n_pool3: int = 2,
    ) -> None:
        super().__init__()

        if window_size != 3:
            raise ValueError("This implementation is fixed to window_size=3 (9 qubits).")
        if stride <= 0:
            raise ValueError("stride must be >= 1")
        if image_size < window_size:
            raise ValueError("image_size must be >= window_size")

        self.image_size = image_size
        self.window_size = window_size
        self.stride = stride

        self.n_patches_per_side = ((image_size - window_size) // stride) + 1
        self.n_patches = self.n_patches_per_side**2
        self.features_per_patch = 3

        self.qcnn = create_qcnn_9qubit(
            n_conv1=n_conv1,
            n_pool1=n_pool1,
            n_conv2=n_conv2,
            n_pool2=n_pool2,
            n_conv3=n_conv3,
            n_pool3=n_pool3,
        )

        n_features = self.n_patches * self.features_per_patch
        self.fc = nn.Linear(n_features, n_classes)

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 3x3 patches with configurable stride.

        Args:
            x: Tensor of shape (B, 1, H, W).

        Returns:
            Tensor of shape (B, N, 9), where N is number of patches.
        """
        ws = self.window_size
        st = self.stride
        patches = x.unfold(2, ws, st).unfold(3, ws, st)
        patches = patches.contiguous().view(x.shape[0], -1, ws * ws)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patches(x)
        _, n_patches, _ = patches.shape

        patches_scaled = patches * math.pi

        all_features = []
        for i in range(n_patches):
            patch_input = patches_scaled[:, i, :]
            bloch = self.qcnn(patch_input)
            all_features.append(bloch)

        features = torch.cat(all_features, dim=1)
        logits = self.fc(features)
        return logits


def build_model(**kwargs) -> MultiPauliQCNN:
    """Create a MultiPauliQCNN model.

    This helper exists so inference/training scripts can simply import this module
    and call `build_model(...)` without touching internals.
    """

    return MultiPauliQCNN(**kwargs)


def load_model_for_inference(
    checkpoint_path: str,
    *,
    map_location: Optional[str | torch.device] = "cpu",
    eval_mode: bool = True,
    **model_kwargs,
) -> MultiPauliQCNN:
    """Load model weights and return a ready-to-use model instance.

    Args:
        checkpoint_path: Path to a state_dict checkpoint.
        map_location: `torch.load` map location.
        eval_mode: If True, call `model.eval()` before returning.
        **model_kwargs: Arguments forwarded to `MultiPauliQCNN` constructor.
    """

    model = build_model(**model_kwargs)
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict)
    if eval_mode:
        model.eval()
    return model


@torch.no_grad()
def predict_logits(model: MultiPauliQCNN, x: torch.Tensor) -> torch.Tensor:
    """Run inference and return logits."""

    return model(x)


@torch.no_grad()
def predict_proba(model: MultiPauliQCNN, x: torch.Tensor) -> torch.Tensor:
    """Run inference and return class probabilities."""

    logits = predict_logits(model, x)
    return torch.softmax(logits, dim=1)