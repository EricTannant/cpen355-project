from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.config import ensure_directories, load_config, fix_seed, validate_baseline_constraints
from src.dataset import create_dataloaders
from src.models import build_model


def resolve_device(device_name: str, gpu_id: int = 0) -> torch.device:
    requested = device_name.lower()
    wants_cuda = requested in {"auto", "cuda"} or requested.startswith("cuda:")

    if wants_cuda:
        if torch.cuda.is_available():
            if requested.startswith("cuda:"):
                try:
                    chosen_gpu = int(requested.split(":", maxsplit=1)[1])
                except ValueError as exc:
                    raise ValueError(f"Invalid CUDA device specifier: {device_name}") from exc
            else:
                chosen_gpu = int(gpu_id)

            num_gpus = torch.cuda.device_count()
            if chosen_gpu < 0 or chosen_gpu >= num_gpus:
                raise ValueError(
                    f"Requested gpu_id {chosen_gpu}, but only {num_gpus} GPU(s) are available."
                )

            device = torch.device(f"cuda:{chosen_gpu}")
            print(f"Using device: {device}")
            return device

        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if requested == "cpu":
        print("Using device: cpu")
        return torch.device("cpu")

    raise ValueError(
        "training.device must be one of: cuda, auto, cpu, or cuda:<index>."
    )


def compute_class_weights(train_csv: Path, num_classes: int) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().reindex(range(num_classes), fill_value=1)
    weights = 1.0 / counts.values.astype("float32")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.size(0))

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def run_training(config_path: str) -> None:
    config = load_config(config_path)
    validate_baseline_constraints(config)
    ensure_directories(config)

    seed = int(config["project"]["seed"])
    fix_seed(seed)

    data_cfg = config["data"]
    train_cfg = config["training"]

    processed_dir = Path(data_cfg["processed_dir"])
    with (processed_dir / "label_to_index.json").open("r", encoding="utf-8") as handle:
        label_to_index = json.load(handle)

    num_classes = len(label_to_index)
    if num_classes != 8:
        raise ValueError(f"Expected 8 configured breeds, found {num_classes} classes.")

    train_loader, val_loader, _ = create_dataloaders(
        processed_dir=processed_dir,
        image_size=int(data_cfg["image_size"]),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
    )

    device = resolve_device(
        str(train_cfg.get("device", "cuda")),
        gpu_id=int(train_cfg.get("gpu_id", 0)),
    )
    model = build_model(
        model_name=str(train_cfg["model_name"]),
        num_classes=num_classes,
        freeze_backbone=bool(train_cfg.get("freeze_backbone", False)),
    ).to(device)

    class_weights = None
    if bool(train_cfg.get("use_class_weights", True)):
        class_weights = compute_class_weights(processed_dir / "train.csv", num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["early_stopping_patience"])
    best_val_acc = 0.0
    stale_epochs = 0
    history = []

    # Use checkpoint_dir from config if available, otherwise create model-specific directory
    if "paths" in config and "checkpoint_dir" in config["paths"]:
        checkpoints_dir = Path(config["paths"]["checkpoint_dir"])
    else:
        model_name = str(train_cfg["model_name"])
        checkpoints_dir = Path("outputs/checkpoints") / model_name

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / "best.pt"
    last_path = checkpoints_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step(val_acc)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        print(json.dumps(entry))

        state = {
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "config": config,
        }
        torch.save(state, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stale_epochs = 0
            torch.save(state, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping triggered.")
                break

    with (Path("outputs/metrics") / "train_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cat/dog breed classifier.")
    parser.add_argument("--config", default="configs/resnet50.yaml")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
