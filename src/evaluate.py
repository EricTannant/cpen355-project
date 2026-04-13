from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import fix_seed, validate_baseline_constraints
from src.dataset import create_test_dataloader
from src.train import resolve_device


def _resolve_model_from_checkpoint(
    runtime_config: Mapping[str, object],
    checkpoint: Mapping[str, object],
    num_classes: int,
):
    from src.models import build_model

    runtime_training = runtime_config.get("training", {})
    if not isinstance(runtime_training, Mapping):
        runtime_training = {}

    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, Mapping):
        checkpoint_config = {}

    checkpoint_training = checkpoint_config.get("training", {})
    if not isinstance(checkpoint_training, Mapping):
        checkpoint_training = {}

    model_name = str(
        checkpoint_training.get("model_name", runtime_training.get("model_name", "custom_cnn"))
    )
    freeze_backbone = bool(
        checkpoint_training.get("freeze_backbone", runtime_training.get("freeze_backbone", False))
    )

    model_kwargs: dict[str, object] = {}
    if model_name.lower() in {"custom_cnn", "cnn"}:
        tuned_cfg = checkpoint_config.get("fine_tuning_model", {})
        if not isinstance(tuned_cfg, Mapping):
            tuned_cfg = {}

        fallback_candidate = checkpoint_config.get("fine_tuning_candidate", {})
        if not isinstance(fallback_candidate, Mapping):
            fallback_candidate = {}

        cnn_source: Mapping[str, object] = tuned_cfg if tuned_cfg else fallback_candidate
        if cnn_source:
            if "conv_channels" in cnn_source:
                model_kwargs["conv_channels"] = tuple(cnn_source["conv_channels"])
            if "hidden_dim" in cnn_source:
                model_kwargs["hidden_dim"] = int(cnn_source["hidden_dim"])
            if "feature_dropout" in cnn_source:
                model_kwargs["feature_dropout"] = float(cnn_source["feature_dropout"])
            if "classifier_dropout" in cnn_source:
                model_kwargs["classifier_dropout"] = float(cnn_source["classifier_dropout"])

    return build_model(
        model_name=model_name,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        model_kwargs=model_kwargs,
    )


def run_evaluation(config_path: str, checkpoint_path: str | None) -> None:
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_baseline_constraints(config)
    fix_seed(int(config["project"]["seed"]))

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

    model = _resolve_model_from_checkpoint(
        runtime_config=config,
        checkpoint=checkpoint,
        num_classes=len(label_to_index),
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
    parser.add_argument("--config", default="configs/resnet50.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    run_evaluation(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
