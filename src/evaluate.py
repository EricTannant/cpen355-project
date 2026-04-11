from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import validate_baseline_constraints
from src.dataset import create_test_dataloader
from src.train import resolve_device


def run_evaluation(config_path: str, checkpoint_path: str | None) -> None:
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_baseline_constraints(config)

    data_cfg = config["data"]
    train_cfg = config["training"]

    processed_dir = Path(data_cfg["processed_dir"])
    if checkpoint_path is None and "paths" in config and "checkpoint_dir" in config["paths"]:
        checkpoint_path = str(Path(config["paths"]["checkpoint_dir"]) / "best.pt")
    elif checkpoint_path is None:
        checkpoint_path = "outputs/checkpoints/best.pt"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        if "paths" in config and "checkpoint_dir" in config["paths"]:
            alt_path = Path(config["paths"]["checkpoint_dir"]) / "best.pt"
            if alt_path.exists():
                checkpoint_path = alt_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    label_to_index = checkpoint["label_to_index"]
    index_to_label = {int(v): k for k, v in label_to_index.items()}

    from src.models import build_model

    model = build_model(
        model_name=str(train_cfg["model_name"]),
        num_classes=len(label_to_index),
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = resolve_device(
        str(train_cfg.get("device", "cuda")),
        gpu_id=int(train_cfg.get("gpu_id", 0)),
    )
    model = model.to(device)
    model.eval()

    test_loader = create_test_dataloader(
        processed_dir=processed_dir,
        image_size=int(data_cfg["image_size"]),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
    )

    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_pred.extend(preds)
            all_true.extend(labels.numpy().tolist())

    labels_sorted = sorted(index_to_label.keys())
    label_names = [index_to_label[idx] for idx in labels_sorted]

    report = classification_report(
        all_true,
        all_pred,
        labels=labels_sorted,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_true, all_pred, labels=labels_sorted)
    acc = accuracy_score(all_true, all_pred)

    metrics = {
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report,
    }

    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_name = Path(config_path).stem

    # Keep shared filenames for backwards compatibility.
    with (metrics_dir / "eval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    # Also write model/config-specific files so runs do not overwrite each other.
    with (metrics_dir / f"eval_metrics_{run_name}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(metrics_dir / "confusion_matrix.csv")
    cm_df.to_csv(metrics_dir / f"confusion_matrix_{run_name}.csv")

    print(
        json.dumps(
            {
                "accuracy": acc,
                "macro_f1": metrics["macro_f1"],
                "checkpoint": str(checkpoint_path),
                "metrics_file": str(metrics_dir / f"eval_metrics_{run_name}.json"),
                "confusion_matrix_file": str(metrics_dir / f"confusion_matrix_{run_name}.csv"),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on test split.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    run_evaluation(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
