from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn as nn
from optuna.exceptions import TrialPruned
from torch.optim import AdamW

from src.config import ensure_directories, load_config, fix_seed, validate_baseline_constraints
from src.dataset import create_dataloaders
from src.models import build_model
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
    "freeze_backbone",
}


def _is_cnn_model(model_name: str) -> bool:
    return model_name.lower() in {"cnn", "custom_cnn"}


def default_search_space(model_name: str) -> dict[str, tuple[Any, ...]]:
    if _is_cnn_model(model_name):
        return {
            "learning_rate": (1e-3, 5e-4, 3e-4),
            "weight_decay": (1e-4, 2e-4, 5e-4, 1e-3),
            "batch_size": (16, 32, 64),
            "epochs": (15, 20, 25),
            "early_stopping_patience": (5, 10),
            "conv_channels": ((32, 64, 128, 256), (32, 64, 128, 128), (32, 64, 128, 256, 256)),
            "hidden_dim": (128, 192, 256),
            "feature_dropout": (0.2, 0.3, 0.4),
            "classifier_dropout": (0.1, 0.2, 0.3),
        }

    return {
        "learning_rate": (1e-4, 2e-4, 5e-4),
        "weight_decay": (5e-5, 1e-4, 5e-4),
        "batch_size": (16, 32, 64),
        "epochs": (15, 20, 25),
        "early_stopping_patience": (5, 10),
    }


def normalize_search_space(
    search_space: Mapping[str, Sequence[Any]] | None,
    model_name: str,
) -> dict[str, tuple[Any, ...]]:
    if search_space is None:
        return default_search_space(model_name)

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


def resolve_tuning_space(config: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    model_name = str(config["training"]["model_name"])
    tuning_cfg = config.get("fine_tuning", {})
    if not isinstance(tuning_cfg, Mapping):
        raise ValueError("fine_tuning must be a mapping when present in the config.")

    search_space = tuning_cfg.get("search_space")
    if search_space is not None and not isinstance(search_space, Mapping):
        raise ValueError("fine_tuning.search_space must be a mapping when present.")

    if search_space is None:
        return default_search_space(model_name)

    return normalize_search_space(search_space, model_name=model_name)


def _candidate_value(candidate: Mapping[str, Any], key: str, default: Any) -> Any:
    return candidate.get(key, default)


def build_model_from_candidate(
    model_name: str,
    num_classes: int,
    candidate: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
) -> nn.Module:
    model_name = model_name.lower()
    freeze_backbone = bool(_candidate_value(candidate, "freeze_backbone", train_cfg.get("freeze_backbone", False)))

    model_kwargs: dict[str, Any] = {}
    if _is_cnn_model(model_name):
        model_kwargs = {
            "conv_channels": tuple(_candidate_value(candidate, "conv_channels", (32, 64, 128, 256))),
            "hidden_dim": int(_candidate_value(candidate, "hidden_dim", 128)),
            "feature_dropout": float(_candidate_value(candidate, "feature_dropout", 0.4)),
            "classifier_dropout": float(_candidate_value(candidate, "classifier_dropout", 0.2)),
        }

    return build_model(
        model_name=model_name,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        model_kwargs=model_kwargs,
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


def train_model_candidate(
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    trial_dir: Path,
    device: torch.device,
    trial_index: int,
    objective: str,
    trial: optuna.Trial | None = None,
    prune_warmup_epochs: int = 3,
) -> dict[str, Any]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_name = str(train_cfg["model_name"]).lower()
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
    model = build_model_from_candidate(
        model_name=model_name,
        num_classes=num_classes,
        candidate=candidate,
        train_cfg=train_cfg,
    ).to(device)

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
    pruned = False

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

        if trial is not None:
            trial.report(float(score), step=epoch)
            if epoch >= prune_warmup_epochs and trial.should_prune():
                pruned = True
                break

    if pruned:
        raise TrialPruned(f"Trial pruned at epoch {epoch}.")

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
    n_trials: int | None = 30,
    max_trials: int | None = None,
    objective: str = "val_acc",
    top_k_checkpoints: int = 3,
    study_name: str | None = None,
    storage: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_baseline_constraints(config)
    ensure_directories(config)

    train_cfg = config["training"]
    model_name = str(train_cfg["model_name"]).lower()
    supported_models = {"cnn", "custom_cnn", "resnet18", "resnet50"}
    if model_name not in supported_models:
        raise ValueError(
            "Fine-tuning supports model_name in {cnn, custom_cnn, resnet18, resnet50}."
        )

    device = resolve_device(
        str(train_cfg.get("device", "cuda")),
        gpu_id=int(train_cfg.get("gpu_id", 0)),
    )

    resolved_space = (
        normalize_search_space(search_space, model_name=model_name)
        if search_space is not None
        else resolve_tuning_space(config)
    )
    if not resolved_space:
        raise ValueError("Search space produced no candidate configurations.")

    trial_budget = int(max_trials if max_trials is not None else n_trials)
    if trial_budget <= 0:
        raise ValueError("Trial count must be positive.")

    objective_key = objective.lower()
    if objective_key not in {"val_acc", "val_loss"}:
        raise ValueError("objective must be either 'val_acc' or 'val_loss'.")

    maximize = objective_key == "val_acc"
    direction = "maximize" if maximize else "minimize"

    print(
        json.dumps(
            {
                "event": "fine_tune_start",
                "objective": objective_key,
                "n_trials": trial_budget,
                "model_name": model_name,
                "device": str(device),
            },
            indent=2,
        )
    )

    checkpoints_root = Path(
        config.get("paths", {}).get("checkpoint_dir", f"outputs/checkpoints/{model_name}")
    ) / "fine_tune"
    metrics_root = Path("outputs/metrics")
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    def suggest_value(
        trial: optuna.Trial,
        key: str,
        options: Sequence[Any],
    ) -> Any:
        if not options:
            raise ValueError(f"Search space for {key} must contain at least one value.")
        return trial.suggest_categorical(key, list(options))

    tuning_cfg = config.get("fine_tuning", {})
    optuna_cfg: Mapping[str, Any] = {}
    if isinstance(tuning_cfg, Mapping):
        nested_optuna = tuning_cfg.get("optuna", {})
        if isinstance(nested_optuna, Mapping):
            optuna_cfg = nested_optuna

    startup_trials = int(optuna_cfg.get("startup_trials", min(8, max(4, trial_budget // 5))))
    warmup_epochs = int(optuna_cfg.get("warmup_epochs", 3))
    interval_steps = int(optuna_cfg.get("pruning_interval_steps", 1))

    sampler = optuna.samplers.TPESampler(seed=int(config["project"]["seed"]))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(0, startup_trials),
        n_warmup_steps=max(0, warmup_epochs),
        interval_steps=max(1, interval_steps),
    )

    computed_study_name = study_name or f"{model_name}_fine_tune"
    study = optuna.create_study(
        study_name=computed_study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=storage is not None,
    )

    trial_records: dict[int, dict[str, Any]] = {}

    def objective_fn(trial: optuna.Trial) -> float:
        trial_number = trial.number + 1
        candidate: dict[str, Any] = {}
        for key, values in resolved_space.items():
            candidate[key] = suggest_value(trial, key, values)

        trial_dir = checkpoints_root / f"trial_{trial_number:03d}"
        print(
            json.dumps(
                {
                    "event": "trial_start",
                    "trial": trial_number,
                    "candidate": candidate,
                },
                indent=2,
            )
        )

        result = train_model_candidate(
            config=config,
            candidate=candidate,
            trial_dir=trial_dir,
            device=device,
            trial_index=trial_number,
            objective=objective_key,
            trial=trial,
            prune_warmup_epochs=warmup_epochs,
        )

        score = float(result["best_score"])
        result_with_score = dict(result)
        result_with_score["score"] = score
        result_with_score["objective"] = objective_key
        result_with_score["trial_number"] = trial_number
        trial_records[trial.number] = result_with_score
        trial.set_user_attr("result", result_with_score)

        print(
            json.dumps(
                {
                    "event": "trial_end",
                    "trial": trial_number,
                    "score": score,
                    "best_epoch": result["best_epoch"],
                    "best_val_acc": result["best_val_acc"],
                    "best_val_loss": result["best_val_loss"],
                },
                indent=2,
            )
        )
        return score

    study.optimize(objective_fn, n_trials=trial_budget, gc_after_trial=True)

    trial_results: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            result = trial.user_attrs.get("result", trial_records.get(trial.number, {}))
            if isinstance(result, Mapping):
                entry = dict(result)
            else:
                entry = {}
            # Keep summary compact: epoch-level logs stay in per-trial history.json files.
            entry.pop("history", None)
            entry.setdefault("trial_number", trial.number + 1)
            entry["state"] = "COMPLETE"
            trial_results.append(entry)
            continue

        entry = {
            "trial_number": trial.number + 1,
            "state": str(trial.state.name),
            "params": dict(trial.params),
            "score": float(trial.value) if trial.value is not None else None,
        }
        trial_results.append(entry)

    complete_results = [
        result for result in trial_results
        if str(result.get("state", "")).upper() == "COMPLETE"
    ]

    best_result: dict[str, Any] | None = None
    best_score = float("-inf") if maximize else float("inf")
    if complete_results:
        sorted_results = sorted(
            complete_results,
            key=lambda item: float(item["score"]),
            reverse=maximize,
        )
        best_result = sorted_results[0]
        best_score = float(best_result["score"])

        keep_limit = max(1, int(top_k_checkpoints))
        keep_trial_numbers = {int(item["trial_number"]) for item in sorted_results[:keep_limit]}
        for item in sorted_results[keep_limit:]:
            trial_num = int(item["trial_number"])
            trial_dir = checkpoints_root / f"trial_{trial_num:03d}"
            for artifact_name in ("best.pt", "last.pt"):
                artifact_path = trial_dir / artifact_name
                if artifact_path.exists() and trial_num not in keep_trial_numbers:
                    artifact_path.unlink()
            if trial_dir.exists() and not any(trial_dir.iterdir()):
                shutil.rmtree(trial_dir, ignore_errors=True)

    best_hyperparameters = None
    if best_result is not None:
        candidate = best_result.get("candidate")
        if isinstance(candidate, Mapping):
            best_hyperparameters = dict(candidate)

    summary = {
        "objective": objective_key,
        "direction": direction,
        "model_name": model_name,
        "study_name": computed_study_name,
        "n_trials_requested": trial_budget,
        "n_trials_completed": len(complete_results),
        "best_score": best_score,
        "best_hyperparameters": best_hyperparameters,
        "best_result": best_result,
        "trials": trial_results,
    }

    summary_path = metrics_root / f"{model_name}_fine_tune_summary.json"
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
    parser = argparse.ArgumentParser(description="Fine-tune CNN/ResNet models with Optuna.")
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--objective", default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--top-k-checkpoints", type=int, default=3)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    args = parser.parse_args()

    optimize_cnn_values(
        config_path=args.config,
        n_trials=args.n_trials,
        max_trials=args.max_trials,
        objective=args.objective,
        top_k_checkpoints=args.top_k_checkpoints,
        study_name=args.study_name,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()