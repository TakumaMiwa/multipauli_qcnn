from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import sys
sys.path.append(".")
from multipauli_qcnn.models.multipauli_qcnn import MultiPauliQCNN


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        help="Use GPU (CUDA) if available.",
    )
    group.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Force CPU execution.",
    )
    parser.set_defaults(use_gpu=None)
    return parser.parse_args()


def main(use_gpu: bool | None = None) -> None:
    seed = 42
    torch.manual_seed(seed)

    has_cuda = torch.cuda.is_available()
    if use_gpu is False:
        device = torch.device("cpu")
    elif use_gpu is True:
        if has_cuda:
            device = torch.device("cuda")
        else:
            print("`--use-gpu` was specified, but CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if has_cuda else "cpu")

    output_dir = Path("multipauli_qcnn_output")
    checkpoint_path = output_dir / "checkpoints" / "one_channel_mnist.pt"
    logs_path = output_dir / "logs" / "logs.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    train_full = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    train_indices = indices[:2000]
    val_indices = indices[2000:2500]

    train_subset = Subset(train_full, train_indices)
    val_subset = Subset(train_full, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
    )

    model = MultiPauliQCNN(
        image_size=28,
        window_size=2,
        stride=2,
        n_classes=10,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    print("Starting training...")
    for epoch in range(1, 21):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        print(
            f"Epoch {epoch:02d}/10 | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )

    with logs_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Training logs saved to: {logs_path}")


if __name__ == "__main__":
    args = parse_args()
    main(use_gpu=args.use_gpu)
