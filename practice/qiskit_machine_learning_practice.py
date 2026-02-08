# pip install qiskit-machine-learning

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# 1量子ビットの例：入力 x と重み w で回路を作り、<X>,<Y>,<Z> を出力にする
x = ParameterVector("x", 1)
w = ParameterVector("w", 2)

qc = QuantumCircuit(1)
qc.ry(x[0], 0)
qc.ry(w[0], 0)
qc.rz(w[1], 0)

observables = [
    SparsePauliOp.from_list([("X", 1.0)]),
    SparsePauliOp.from_list([("Y", 1.0)]),
    SparsePauliOp.from_list([("Z", 1.0)]),
]

estimator = Estimator()

qnn = EstimatorQNN(
    circuit=qc,
    input_params=list(x),
    weight_params=list(w),
    observables=observables,
    estimator=estimator,
    input_gradients=True,  # PyTorch側で x にも勾配が欲しい場合
)

torch_layer = TorchConnector(qnn)  # PyTorch Module化

# PyTorch の forward
x_t = torch.tensor([0.3], dtype=torch.float32)  # 入力
bloch = torch_layer(x_t)  # shape=(3,) で [<X>,<Y>,<Z>] が返る（勾配も追える）
print(bloch)
