from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.config import fix_seed, validate_baseline_constraints
from src.dataset import build_transform
from src.models import build_model
from src.train import resolve_device


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

    model = build_model(
        model_name=str(config["training"]["model_name"]),
        num_classes=len(label_to_index),
        freeze_backbone=False,
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
