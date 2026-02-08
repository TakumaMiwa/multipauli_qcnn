# pip install torch numpy qiskit qiskit-aer

import numpy as np
import torch

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ----------------------------
# 1) 測定基底の回転（X/Y/Z を Z測定に変換）
#    X: H, Y: Sdg -> H, Z: 何もしない
# ----------------------------
def _append_measure_in_pauli_basis(qc: QuantumCircuit, qubit: int, pauli: str, cbit: int = 0):
    if pauli == "X":
        qc.h(qubit)
    elif pauli == "Y":
        qc.sdg(qubit)
        qc.h(qubit)
    elif pauli == "Z":
        pass
    else:
        raise ValueError("pauli must be one of 'X','Y','Z'")
    qc.measure(qubit, cbit)


def _counts_to_expectation_z(counts: dict, shots: int) -> float:
    """
    1 classical bit の測定結果 {'0': n0, '1': n1} から <Z> = P(0) - P(1) を計算
    """
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    return (n0 - n1) / shots


def estimate_single_qubit_paulis(
    base_qc: QuantumCircuit,
    measured_qubit: int = 0,
    shots: int = 8192,
    backend=None,
) -> torch.Tensor:
    """
    base_qc（測定なし）に対して、同じ状態を何度も準備して
    <X>,<Y>,<Z> をショットから推定して返す。
    """
    if backend is None:
        backend = AerSimulator()

    exps = []
    for pauli in ["X", "Y", "Z"]:
        # base_qc を 1 classical bit 付きの回路に埋め込む
        qc = QuantumCircuit(base_qc.num_qubits, 1)
        qc.compose(base_qc, inplace=True)
        qc.barrier()
        _append_measure_in_pauli_basis(qc, measured_qubit, pauli, cbit=0)

        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts(0)  # 1本だけなので index=0
        exps.append(_counts_to_expectation_z(counts, shots))

    return torch.tensor(exps, dtype=torch.float64)  # [<X>,<Y>,<Z>]


# ----------------------------
# 2) Blochベクトル(<X>,<Y>,<Z>) から 1量子ビット密度行列を復元
#    rho = 1/2 (I + rx X + ry Y + rz Z)
# ----------------------------
def bloch_to_density_matrix(r: torch.Tensor) -> torch.Tensor:
    """
    r: shape (3,) = [rx, ry, rz]
    return: rho (2x2) complex128 torch tensor
    """
    rx, ry, rz = r.tolist()

    I = torch.eye(2, dtype=torch.complex128)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    rho = 0.5 * (I + rx * X + ry * Y + rz * Z)

    # ショットノイズ等で物理的でない（負の固有値など）ことがあるので
    # 近い物理状態へ射影（簡易：固有値を 0 でクリップして正規化）
    rho = (rho + rho.conj().T) / 2  # Hermitian に寄せる
    evals, evecs = torch.linalg.eigh(rho)
    evals = torch.clamp(evals.real, min=0.0)
    s = evals.sum()
    if s.item() > 0:
        evals = evals / s
    rho_phys = (evecs * evals) @ evecs.conj().T
    return rho_phys


def density_matrix_to_pure_angles(rho: torch.Tensor):
    """
    rho を「最も重い固有ベクトル」で純粋状態近似し、
    |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    の (theta, phi) を返す。
    """
    evals, evecs = torch.linalg.eigh(rho)
    idx = torch.argmax(evals.real).item()
    psi = evecs[:, idx]  # shape (2,)

    # global phase を揃える（psi[0] を実数に）
    if torch.abs(psi[0]) > 1e-12:
        psi = psi * torch.exp(-1j * torch.angle(psi[0]))
    a = psi[0].real.clamp(-1.0, 1.0)
    theta = 2.0 * torch.acos(a)
    phi = torch.angle(psi[1]) if torch.abs(psi[1]) > 1e-12 else torch.tensor(0.0)

    return float(theta.item()), float(phi.item()), psi


def prepare_state_circuit(theta: float, phi: float) -> QuantumCircuit:
    """
    |0> に RY(theta) → RZ(phi) を当てて状態を準備（グローバル位相は無視）
    """
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return qc


# ----------------------------
# 3) “QCNNっぽい” 1層ブロック（例）
#    入力は 1量子ビット状態（q0）＋補助量子ビット（q1=|0>）
# ----------------------------
def layer_block(theta_in: float, phi_in: float, params: list[float]) -> QuantumCircuit:
    """
    params: 6個くらいの回転パラメータの例
    出力として q0 を見る（q1 は測らずに捨てた扱い）
    """
    assert len(params) == 6
    qc = QuantumCircuit(2)

    # 入力状態を q0 に準備
    qc.ry(theta_in, 0)
    qc.rz(phi_in, 0)

    # q1 は |0>
    # 例: 2量子ビットの “畳み込みっぽい” ブロック
    qc.ry(params[0], 0)
    qc.rz(params[1], 0)
    qc.ry(params[2], 1)
    qc.rz(params[3], 1)
    qc.cx(0, 1)
    qc.ry(params[4], 0)
    qc.ry(params[5], 1)
    qc.cx(1, 0)

    return qc


def run_demo():
    backend = AerSimulator()
    shots = 4096

    # 入力（例）
    theta, phi = 0.8, 0.3

    # 2層分のパラメータ（例）
    params_l1 = [0.20, -0.40, 0.10, 0.70, -0.30, 0.50]
    params_l2 = [-0.10, 0.20, 0.40, -0.20, 0.60, -0.50]

    for li, params in enumerate([params_l1, params_l2], start=1):
        base = layer_block(theta, phi, params)

        # q0 だけ X/Y/Z 測定して Bloch ベクトル推定
        r = estimate_single_qubit_paulis(base, measured_qubit=0, shots=shots, backend=backend)
        rho = bloch_to_density_matrix(r)
        theta, phi, psi = density_matrix_to_pure_angles(rho)

        print(f"[Layer {li}] <X,Y,Z> = {r.tolist()}")
        print(f"[Layer {li}] rho =\n{rho}")
        print(f"[Layer {li}] reprepare angles: theta={theta:.6f}, phi={phi:.6f}")
        print("-" * 60)

    # 最終層の出力状態を Zで読む（例）
    qc_out = QuantumCircuit(1, 1)
    qc_out.ry(theta, 0)
    qc_out.rz(phi, 0)
    qc_out.measure(0, 0)
    tqc = transpile(qc_out, backend)
    counts = backend.run(tqc, shots=shots).result().get_counts(0)
    z_exp = _counts_to_expectation_z(counts, shots)
    print("[Final] Z expectation (approx) =", z_exp)


if __name__ == "__main__":
    run_demo()
