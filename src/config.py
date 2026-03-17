from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping at the top level.")

    return config


def ensure_directories(config: Dict[str, Any]) -> None:
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    checkpoints_dir = Path("outputs/checkpoints")
    metrics_dir = Path("outputs/metrics")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)


def validate_baseline_constraints(config: Dict[str, Any]) -> None:
    image_size = int(config["data"].get("image_size", 224))
    if image_size != 224:
        raise ValueError(
            "This baseline enforces data.image_size = 224 for all pipelines."
        )
