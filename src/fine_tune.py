from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config import ensure_directories, load_config, fix_seed, validate_baseline_constraints
from src.dataset import create_dataloaders
from src.models import CNN
from src.train import compute_class_weights, resolve_device, run_epoch


TRAINING_KEYS = {
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "early_stopping_patience",
}

MODEL_KEYS = {
    "conv_channels",
    "hidden_dim",
    "feature_dropout",
    "classifier_dropout",
}


def default_search_space() -> dict[str, tuple[Any, ...]]:
    return {
        "learning_rate": (1e-3, 5e-4),
        "weight_decay": (1e-4, 2e-4),
        "batch_size": (16, 32, 64),
        "epochs": (10, 15, 20),
        "early_stopping_patience": (4, 10),
        "conv_channels": ((32, 64, 128, 256), (32, 64, 128, 128)),
        "hidden_dim": (128, 256),
        "feature_dropout": (0.3, 0.4),
        "classifier_dropout": (0, 0.1, 0.2, 0.3),
    }


def normalize_search_space(
    search_space: Mapping[str, Sequence[Any]] | None,
) -> dict[str, tuple[Any, ...]]:
    if search_space is None:
        return default_search_space()

    normalized: dict[str, tuple[Any, ...]] = {}
    for key, values in search_space.items():
        if isinstance(values, (str, bytes)):
            normalized[key] = (values,)
            continue

        candidate_values = tuple(values)
        if not candidate_values:
            raise ValueError(f"Search space for {key} must contain at least one value.")
        normalized[key] = candidate_values

    return normalized


def iter_search_space(search_space: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(search_space.keys())
    value_grid = [tuple(search_space[key]) for key in keys]
    return [dict(zip(keys, values)) for values in product(*value_grid)]


def resolve_tuning_space(config: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    tuning_cfg = config.get("fine_tuning", {})
    if not isinstance(tuning_cfg, Mapping):
        raise ValueError("fine_tuning must be a mapping when present in the config.")

    search_space = tuning_cfg.get("search_space")
    if search_space is not None and not isinstance(search_space, Mapping):
        raise ValueError("fine_tuning.search_space must be a mapping when present.")

    if search_space is None:
        return default_search_space()

    return normalize_search_space(search_space)


def _candidate_value(candidate: Mapping[str, Any], key: str, default: Any) -> Any:
    return candidate.get(key, default)


def build_cnn_from_candidate(
    num_classes: int,
    candidate: Mapping[str, Any],
) -> CNN:
    return CNN(
        num_classes=num_classes,
        conv_channels=tuple(_candidate_value(candidate, "conv_channels", (32, 64, 128, 256))),
        hidden_dim=int(_candidate_value(candidate, "hidden_dim", 128)),
        feature_dropout=float(_candidate_value(candidate, "feature_dropout", 0.4)),
        classifier_dropout=float(_candidate_value(candidate, "classifier_dropout", 0.2)),
    )


def prepare_candidate_config(
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    tuned_config = copy.deepcopy(dict(config))
    training_cfg = tuned_config["training"]

    for key in TRAINING_KEYS:
        if key in candidate:
            training_cfg[key] = candidate[key]

    tuned_config["training"] = training_cfg
    tuned_config["fine_tuning_model"] = {
        key: candidate[key]
        for key in MODEL_KEYS
        if key in candidate
    }
    tuned_config["fine_tuning_candidate"] = dict(candidate)
    return tuned_config


def train_cnn_candidate(
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    trial_dir: Path,
    device: torch.device,
    trial_index: int,
    objective: str,
) -> dict[str, Any]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    processed_dir = Path(data_cfg["processed_dir"])

    seed = int(config["project"]["seed"]) + trial_index
    fix_seed(seed)

    batch_size = int(_candidate_value(candidate, "batch_size", train_cfg["batch_size"]))
    epochs = int(_candidate_value(candidate, "epochs", train_cfg["epochs"]))
    learning_rate = float(_candidate_value(candidate, "learning_rate", train_cfg["learning_rate"]))
    weight_decay = float(_candidate_value(candidate, "weight_decay", train_cfg["weight_decay"]))
    patience = int(
        _candidate_value(candidate, "early_stopping_patience", train_cfg["early_stopping_patience"])
    )

    train_loader, val_loader, _ = create_dataloaders(
        processed_dir=processed_dir,
        image_size=int(data_cfg["image_size"]),
        batch_size=batch_size,
        num_workers=int(data_cfg["num_workers"]),
    )

    label_file = processed_dir / "label_to_index.json"
    with label_file.open("r", encoding="utf-8") as handle:
        label_to_index = json.load(handle)

    num_classes = len(label_to_index)
    model = build_cnn_from_candidate(num_classes=num_classes, candidate=candidate).to(device)

    class_weights = None
    if bool(train_cfg.get("use_class_weights", True)):
        class_weights = compute_class_weights(processed_dir / "train.csv", num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if objective == "val_acc" else "min",
        factor=0.5,
        patience=2,
    )

    trial_dir.mkdir(parents=True, exist_ok=True)
    best_path = trial_dir / "best.pt"
    last_path = trial_dir / "last.pt"

    history: list[dict[str, Any]] = []
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_score = float("-inf") if objective == "val_acc" else float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            training=False,
        )
        score = val_acc if objective == "val_acc" else val_loss
        scheduler.step(score)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)

        state = {
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "config": prepare_candidate_config(config, candidate),
        }
        torch.save(state, last_path)

        is_better = score > best_score if objective == "val_acc" else score < best_score
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_score = score
            stale_epochs = 0
            torch.save(state, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    result = {
        "candidate": dict(candidate),
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "checkpoint_dir": str(trial_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }

    with (trial_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (trial_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return result


def optimize_cnn_values(
    config_path: str | Path,
    search_space: Mapping[str, Sequence[Any]] | None = None,
    max_trials: int | None = None,
    objective: str = "val_acc",
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_baseline_constraints(config)
    ensure_directories(config)

    train_cfg = config["training"]
    if str(train_cfg["model_name"]).lower() not in {"cnn", "custom_cnn"}:
        raise ValueError("CNN fine-tuning only supports model_name = cnn or custom_cnn.")

    device = resolve_device(
        str(train_cfg.get("device", "cuda")),
        gpu_id=int(train_cfg.get("gpu_id", 0)),
    )

    resolved_space = normalize_search_space(search_space) if search_space is not None else resolve_tuning_space(config)
    candidates = iter_search_space(resolved_space)
    if not candidates:
        raise ValueError("Search space produced no candidate configurations.")

    if max_trials is not None:
        candidates = candidates[:max_trials]

    objective_key = objective.lower()
    if objective_key not in {"val_acc", "val_loss"}:
        raise ValueError("objective must be either 'val_acc' or 'val_loss'.")

    maximize = objective_key == "val_acc"

    checkpoints_root = Path(
        config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints/cnn")
    ) / "fine_tune"
    metrics_root = Path("outputs/metrics")
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    trial_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_score = float("-inf") if maximize else float("inf")

    for trial_index, candidate in enumerate(candidates, start=1):
        trial_dir = checkpoints_root / f"trial_{trial_index:03d}"
        result = train_cnn_candidate(
            config=config,
            candidate=candidate,
            trial_dir=trial_dir,
            device=device,
            trial_index=trial_index,
            objective=objective_key,
        )

        score = float(result["best_score"])
        result_with_score = dict(result)
        result_with_score["score"] = score
        result_with_score["objective"] = objective_key
        trial_results.append(result_with_score)

        if best_result is None:
            best_result = result_with_score
            best_score = score
            continue

        is_better = score > best_score if maximize else score < best_score
        if is_better:
            best_result = result_with_score
            best_score = score

    summary = {
        "objective": objective_key,
        "best_score": best_score,
        "best_result": best_result,
        "trials": trial_results,
    }

    summary_path = metrics_root / "cnn_fine_tune_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        json.dumps(
            {
                "objective": objective_key,
                "best_score": best_score,
                "best_checkpoint": None if best_result is None else best_result["best_checkpoint"],
                "summary_file": str(summary_path),
            },
            indent=2,
        )
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the custom CNN model.")
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--objective", default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--max-trials", type=int, default=None)
    args = parser.parse_args()

    optimize_cnn_values(
        config_path=args.config,
        max_trials=args.max_trials,
        objective=args.objective,
    )


if __name__ == "__main__":
    main()