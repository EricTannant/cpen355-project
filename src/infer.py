from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image

from src.config import fix_seed, validate_baseline_constraints
from src.dataset import build_transform
from src.models import build_model
from src.train import resolve_device


def _resolve_model_from_checkpoint(
    runtime_config: Mapping[str, object],
    checkpoint: Mapping[str, object],
    num_classes: int,
):
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


def run_inference(config_path: str, checkpoint_path: str, image_path: str, top_k: int) -> None:
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_baseline_constraints(config)
    fix_seed(int(config["project"]["seed"]))

    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        if "paths" in config and "checkpoint_dir" in config["paths"]:
            alt_path = Path(config["paths"]["checkpoint_dir"]) / "best.pt"
            if alt_path.exists():
                checkpoint_path_obj = alt_path
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")

    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(
            "Image not found: "
            f"{image_path_obj}. "
            "Provide a valid path, for example one from data/processed/test.csv"
        )

    checkpoint = torch.load(str(checkpoint_path_obj), map_location="cpu")
    label_to_index = checkpoint["label_to_index"]
    index_to_label = {int(v): k for k, v in label_to_index.items()}

    model = _resolve_model_from_checkpoint(
        runtime_config=config,
        checkpoint=checkpoint,
        num_classes=len(label_to_index),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = resolve_device(str(config["training"]["device"]))
    model = model.to(device)
    model.eval()

    transform = build_transform(image_size=int(config["data"]["image_size"]), is_train=False)
    image = Image.open(image_path_obj).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        values, indices = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1)

    results = []
    for score, idx in zip(values[0].cpu().tolist(), indices[0].cpu().tolist()):
        results.append({"breed": index_to_label[int(idx)], "probability": float(score)})

    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict cat or dog breed for one image.")
    parser.add_argument("--config", default="configs/resnet50.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    run_inference(args.config, args.checkpoint, args.image, args.top_k)


if __name__ == "__main__":
    main()
