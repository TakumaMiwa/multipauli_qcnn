# pip install torch numpy qiskit qiskit-machine-learning torchvision

import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from torchvision import datasets, transforms


# ----------------------------
# 1) 4量子ビット QCNN ブロック（2×2 パッチ → <X>,<Y>,<Z> on qubit 0）
#    Conv → Pool → Conv → Pool で 4 qubit → 1 qubit に畳み込み
# ----------------------------
def create_qcnn_4qubit(n_conv1=4, n_pool1=2, n_conv2=4, n_pool2=2) -> TorchConnector:
    """
    4量子ビット QCNN ブロックを EstimatorQNN + TorchConnector で構築。
    入力: 2×2 パッチの 4 ピクセル値
    出力: qubit 0 の [<X>, <Y>, <Z>]（Blochベクトル成分）
    """
    n_weights = n_conv1 + n_pool1 + n_conv2 + n_pool2  # 12
    x = ParameterVector("x", 4)    # 入力: 4 pixels
    w = ParameterVector("w", n_weights)

    qc = QuantumCircuit(4)

    # === データエンコーディング: RY(pixel * pi) ===
    for i in range(4):
        qc.ry(x[i], i)

    qc.barrier()

    # === Convolution Layer 1 (4 qubit 全体) ===
    wi = 0
    qc.ry(w[wi], 0); qc.ry(w[wi+1], 1)
    qc.cx(0, 1)
    qc.ry(w[wi+2], 2); qc.ry(w[wi+3], 3)
    qc.cx(2, 3)
    qc.cx(1, 2)  # 隣接ペア間の結合
    qc.cx(3, 0)  # 循環結合
    wi += n_conv1

    qc.barrier()

    # === Pooling Layer 1: 4 → 2 (q1, q3 の情報を q0, q2 に集約) ===
    qc.crx(w[wi],   1, 0)
    qc.crx(w[wi+1], 3, 2)
    wi += n_pool1

    qc.barrier()

    # === Convolution Layer 2 (q0, q2 のみ活性) ===
    qc.ry(w[wi], 0); qc.ry(w[wi+1], 2)
    qc.cx(0, 2)
    qc.ry(w[wi+2], 0); qc.ry(w[wi+3], 2)
    wi += n_conv2

    qc.barrier()

    # === Pooling Layer 2: 2 → 1 (q2 の情報を q0 に集約) ===
    qc.crx(w[wi], 2, 0)
    qc.ry(w[wi+1], 0)
    wi += n_pool2

    # === qubit 0 の Pauli 期待値を測定（little-endian: 右端が q0） ===
    observables = [
        SparsePauliOp.from_list([("IIIX", 1.0)]),
        SparsePauliOp.from_list([("IIIY", 1.0)]),
        SparsePauliOp.from_list([("IIIZ", 1.0)]),
    ]

    estimator = Estimator()

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=list(x),
        weight_params=list(w),
        observables=observables,
        estimator=estimator,
        input_gradients=False,
    )

    return TorchConnector(qnn)


# ----------------------------
# 2) Multi-Pauli QCNN モデル（MNIST 10クラス分類）
# ----------------------------
class MultiPauliQCNN(nn.Module):
    """
    MNIST 画像 (28×28) を 2×2 ウィンドウで走査し、
    各パッチを 4 量子ビット QCNN で処理して Bloch ベクトル [<X>,<Y>,<Z>] を取得。
    全パッチの特徴量をベクトルに変形後、nn.Linear で 10 クラス分類。
    """
    def __init__(self, image_size=28, window_size=2, n_classes=10):
        super().__init__()
        self.image_size = image_size
        self.window_size = window_size
        self.n_patches_per_side = image_size // window_size  # 14
        self.n_patches = self.n_patches_per_side ** 2        # 196
        self.features_per_patch = 3                          # <X>, <Y>, <Z>

        # 量子畳み込み層（全パッチで重み共有）
        self.qcnn = create_qcnn_4qubit()

        # 古典全結合層: 196 patches × 3 features = 588 → 10 classes
        n_features = self.n_patches * self.features_per_patch
        self.fc = nn.Linear(n_features, n_classes)

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 28, 28) → patches: (B, 196, 4)
        2×2 ウィンドウ（stride=2）でパッチ分割し、各パッチを 4 次元ベクトルに
        """
        ws = self.window_size
        # unfold で 2×2 パッチを取り出す
        patches = x.unfold(2, ws, ws).unfold(3, ws, ws)  # (B, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(x.shape[0], -1, ws * ws)  # (B, 196, 4)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patches(x)           # (B, 196, 4)
        B, N, _ = patches.shape

        # ピクセル値 [0, 1] → [0, π] にスケーリング（RY エンコーディング用）
        patches_scaled = patches * np.pi

        # 各パッチを量子回路に通す（パッチ間で重み共有）
        all_features = []
        for i in range(N):
            patch_input = patches_scaled[:, i, :]   # (B, 4)
            bloch = self.qcnn(patch_input)           # (B, 3): [<X>,<Y>,<Z>]
            all_features.append(bloch)

        # 全パッチの特徴量をベクトルに変形
        features = torch.cat(all_features, dim=1)   # (B, 196×3 = 588)

        # 古典全結合層で 10 クラス分類
        logits = self.fc(features)                   # (B, 10)
        return logits


# ----------------------------
# 3) MNIST 推論デモ（訓練なし・初期パラメータ）
# ----------------------------
def run_demo():
    print("=" * 60)
    print("Multi-Pauli QCNN — MNIST 推論デモ（初期パラメータ）")
    print("=" * 60)

    # MNIST テストデータ読み込み
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform,
    )

    # モデル構築（訓練なし・初期パラメータのまま）
    model = MultiPauliQCNN(image_size=28, window_size=2, n_classes=10)
    model.eval()

    print(f"\nモデル構成:")
    print(f"  画像サイズ   : 28×28")
    print(f"  ウィンドウ   : 2×2 (stride=2)")
    print(f"  パッチ数     : {model.n_patches} ({model.n_patches_per_side}×{model.n_patches_per_side})")
    print(f"  量子ビット数 : 4 (per patch)")
    print(f"  パッチ特徴量 : {model.features_per_patch} (<X>,<Y>,<Z>)")
    print(f"  全特徴量     : {model.n_patches * model.features_per_patch}")
    print(f"  出力クラス数 : 10")
    print(f"  QCNN 重み数  : {sum(p.numel() for p in model.qcnn.parameters())}")
    print(f"  Linear 重み数: {sum(p.numel() for p in model.fc.parameters())}")
    print("-" * 60)

    # 5 枚の画像で推論
    n_samples = 5
    for idx in range(n_samples):
        image, label = dataset[idx]
        image = image.unsqueeze(0)  # (1, 1, 28, 28)

        print(f"\n[Sample {idx+1}] 真のラベル: {label}")

        with torch.no_grad():
            logits = model(image)                          # (1, 10)
            probs = torch.softmax(logits, dim=1).squeeze() # (10,)
            pred = torch.argmax(probs).item()

        print(f"  予測ラベル : {pred}")
        print(f"  確率分布   : {[f'{p:.4f}' for p in probs.tolist()]}")

    print("\n" + "=" * 60)
    print("※ 訓練していないため、予測はランダムに近い結果です。")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
