"""Baseline Multi-Pauli QCNN module with channel-wise second-stage Z readout.

This baseline keeps Pauli-X/Y/Z expectation values from the first quantum layer as
three independent channels. In the second layer, each channel is processed
separately and only Pauli-Z expectation values are passed to the final linear
classifier.
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
    "create_qcnn_9qubit_xyz",
    "create_qcnn_9qubit_z",
    "MultiPauliChannelZBaselineQCNN",
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


def _build_9qubit_qcnn_circuit(
    *,
    n_conv1: int,
    n_pool1: int,
    n_conv2: int,
    n_pool2: int,
    n_conv3: int,
    n_pool3: int,
    input_params: ParameterVector,
    weight_params: ParameterVector,
) -> QuantumCircuit:
    """Build the shared 9-qubit QCNN circuit body."""
    qc = QuantumCircuit(9)

    for i in range(9):
        qc.ry(input_params[i], i)
    qc.barrier()

    wi = 0

    for q in range(9):
        qc.ry(weight_params[wi], q)
        wi += 1
    for q in range(9):
        qc.rz(weight_params[wi], q)
        wi += 1
    _ring_entanglement(qc, range(9))
    qc.barrier()

    qc.crx(weight_params[wi], 1, 0)
    qc.crx(weight_params[wi + 1], 2, 0)
    qc.crx(weight_params[wi + 2], 4, 3)
    qc.crx(weight_params[wi + 3], 5, 3)
    qc.crx(weight_params[wi + 4], 7, 6)
    qc.crx(weight_params[wi + 5], 8, 6)
    wi += n_pool1
    qc.barrier()

    active_3: List[int] = [0, 3, 6]
    for q in active_3:
        qc.ry(weight_params[wi], q)
        wi += 1
    for q in active_3:
        qc.rz(weight_params[wi], q)
        wi += 1
    _ring_entanglement(qc, active_3)
    qc.ry(weight_params[wi], 0)
    qc.rz(weight_params[wi + 1], 3)
    wi += n_conv2
    qc.barrier()

    qc.crx(weight_params[wi], 3, 0)
    qc.crx(weight_params[wi + 1], 3, 6)
    qc.ry(weight_params[wi + 2], 0)
    qc.ry(weight_params[wi + 3], 6)
    wi += n_pool2
    qc.barrier()

    qc.ry(weight_params[wi], 0)
    qc.ry(weight_params[wi + 1], 6)
    qc.cx(0, 6)
    qc.rz(weight_params[wi + 2], 0)
    qc.rz(weight_params[wi + 3], 6)
    wi += n_conv3
    qc.barrier()

    qc.crx(weight_params[wi], 6, 0)
    qc.ry(weight_params[wi + 1], 0)
    wi += n_pool3

    assert wi == len(weight_params), "Weight indexing mismatch in QCNN construction"
    return qc


def create_qcnn_9qubit_xyz(
    *,
    n_conv1: int = 18,
    n_pool1: int = 6,
    n_conv2: int = 8,
    n_pool2: int = 4,
    n_conv3: int = 4,
    n_pool3: int = 2,
) -> TorchConnector:
    """Build a 9-qubit QCNN that outputs [<X>, <Y>, <Z>] on qubit 0."""

    n_weights = n_conv1 + n_pool1 + n_conv2 + n_pool2 + n_conv3 + n_pool3
    x = ParameterVector("x", 9)
    w = ParameterVector("w", n_weights)

    qc = _build_9qubit_qcnn_circuit(
        n_conv1=n_conv1,
        n_pool1=n_pool1,
        n_conv2=n_conv2,
        n_pool2=n_pool2,
        n_conv3=n_conv3,
        n_pool3=n_pool3,
        input_params=x,
        weight_params=w,
    )

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


def create_qcnn_9qubit_z(
    *,
    n_conv1: int = 18,
    n_pool1: int = 6,
    n_conv2: int = 8,
    n_pool2: int = 4,
    n_conv3: int = 4,
    n_pool3: int = 2,
) -> TorchConnector:
    """Build a 9-qubit QCNN that outputs only <Z> on qubit 0."""

    n_weights = n_conv1 + n_pool1 + n_conv2 + n_pool2 + n_conv3 + n_pool3
    x = ParameterVector("x", 9)
    w = ParameterVector("w", n_weights)

    qc = _build_9qubit_qcnn_circuit(
        n_conv1=n_conv1,
        n_pool1=n_pool1,
        n_conv2=n_conv2,
        n_pool2=n_pool2,
        n_conv3=n_conv3,
        n_pool3=n_pool3,
        input_params=x,
        weight_params=w,
    )

    observables = [SparsePauliOp.from_list([("IIIIIIIIZ", 1.0)])]

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=list(x),
        weight_params=list(w),
        observables=observables,
        estimator=Estimator(),
        input_gradients=False,
    )

    return TorchConnector(qnn)


class MultiPauliChannelZBaselineQCNN(nn.Module):
    """Baseline model with channel-wise second quantum layer and Z-only readout."""

    def __init__(
        self,
        *,
        image_size: int = 28,
        window_size_l1: int = 3,
        stride_l1: int = 1,
        window_size_l2: int = 3,
        stride_l2: int = 1,
        n_classes: int = 10,
        n_conv1_l1: int = 18,
        n_pool1_l1: int = 6,
        n_conv2_l1: int = 8,
        n_pool2_l1: int = 4,
        n_conv3_l1: int = 4,
        n_pool3_l1: int = 2,
        n_conv1_l2: int = 18,
        n_pool1_l2: int = 6,
        n_conv2_l2: int = 8,
        n_pool2_l2: int = 4,
        n_conv3_l2: int = 4,
        n_pool3_l2: int = 2,
    ) -> None:
        super().__init__()

        if window_size_l1 != 3 or window_size_l2 != 3:
            raise ValueError("This implementation is fixed to 3x3 windows (9 qubits).")
        if stride_l1 <= 0 or stride_l2 <= 0:
            raise ValueError("stride_l1 and stride_l2 must be >= 1")
        if image_size < window_size_l1:
            raise ValueError("image_size must be >= window_size_l1")

        self.image_size = image_size
        self.window_size_l1 = window_size_l1
        self.stride_l1 = stride_l1
        self.window_size_l2 = window_size_l2
        self.stride_l2 = stride_l2

        self.n_patches_side_l1 = ((image_size - window_size_l1) // stride_l1) + 1
        if self.n_patches_side_l1 < window_size_l2:
            raise ValueError("First-layer feature map is too small for second-layer 3x3 patches.")
        self.n_patches_l1 = self.n_patches_side_l1**2

        self.n_patches_side_l2 = ((self.n_patches_side_l1 - window_size_l2) // stride_l2) + 1
        self.n_patches_l2_per_channel = self.n_patches_side_l2**2
        self.n_channels_l2 = 3

        self.qcnn_l1 = create_qcnn_9qubit_xyz(
            n_conv1=n_conv1_l1,
            n_pool1=n_pool1_l1,
            n_conv2=n_conv2_l1,
            n_pool2=n_pool2_l1,
            n_conv3=n_conv3_l1,
            n_pool3=n_pool3_l1,
        )

        self.qcnn_l2 = create_qcnn_9qubit_z(
            n_conv1=n_conv1_l2,
            n_pool1=n_pool1_l2,
            n_conv2=n_conv2_l2,
            n_pool2=n_pool2_l2,
            n_conv3=n_conv3_l2,
            n_pool3=n_pool3_l2,
        )

        n_features = self.n_channels_l2 * self.n_patches_l2_per_channel
        self.fc = nn.Linear(n_features, n_classes)

    @staticmethod
    def _extract_patches(x: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
        """Extract patches from a 4D tensor (B, C, H, W) to (B, N, C*window_size^2)."""
        patches = x.unfold(2, window_size, stride).unfold(3, window_size, stride)
        patches = patches.contiguous().view(x.shape[0], -1, x.shape[1] * window_size * window_size)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches_l1 = self._extract_patches(x, self.window_size_l1, self.stride_l1)
        patches_l1 = patches_l1 * math.pi

        l1_features = []
        for i in range(self.n_patches_l1):
            patch_input = patches_l1[:, i, :]
            xyz = self.qcnn_l1(patch_input)
            l1_features.append(xyz)

        l1_stacked = torch.stack(l1_features, dim=1)
        channel_maps = l1_stacked.transpose(1, 2).contiguous().view(
            x.shape[0],
            self.n_channels_l2,
            self.n_patches_side_l1,
            self.n_patches_side_l1,
        )

        channel_outputs = []
        for channel in range(self.n_channels_l2):
            channel_map = channel_maps[:, channel : channel + 1, :, :]
            patches_l2 = self._extract_patches(channel_map, self.window_size_l2, self.stride_l2)

            z_features = []
            for i in range(self.n_patches_l2_per_channel):
                patch_input = patches_l2[:, i, :]
                z_value = self.qcnn_l2(patch_input)
                z_features.append(z_value)

            channel_outputs.append(torch.cat(z_features, dim=1))

        features = torch.cat(channel_outputs, dim=1)
        logits = self.fc(features)
        return logits


def build_model(**kwargs) -> MultiPauliChannelZBaselineQCNN:
    """Create a MultiPauliChannelZBaselineQCNN model instance."""

    return MultiPauliChannelZBaselineQCNN(**kwargs)


def load_model_for_inference(
    checkpoint_path: str,
    *,
    map_location: Optional[str | torch.device] = "cpu",
    eval_mode: bool = True,
    **model_kwargs,
) -> MultiPauliChannelZBaselineQCNN:
    """Load model weights and return a ready-to-use model instance."""

    model = build_model(**model_kwargs)
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict)
    if eval_mode:
        model.eval()
    return model


@torch.no_grad()
def predict_logits(model: MultiPauliChannelZBaselineQCNN, x: torch.Tensor) -> torch.Tensor:
    """Run inference and return logits."""

    return model(x)


@torch.no_grad()
def predict_proba(model: MultiPauliChannelZBaselineQCNN, x: torch.Tensor) -> torch.Tensor:
    """Run inference and return class probabilities."""

    logits = predict_logits(model, x)
    return torch.softmax(logits, dim=1)
