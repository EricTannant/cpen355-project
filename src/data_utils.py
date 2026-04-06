from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_full_metadata(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for image_path in iter_image_files(raw_dir):
        label = "_".join(image_path.stem.split("_")[:-1]).strip()
        if not label:
            continue
        rows.append({"image_path": str(image_path.resolve()), "breed": label})

    if not rows:
        raise RuntimeError(
            "No images found under data/raw. Download and extract dataset first."
        )

    metadata = pd.DataFrame(rows).drop_duplicates()
    metadata = metadata[metadata["breed"].astype(str).str.len() > 0]
    metadata = metadata.sort_values(["breed", "image_path"]).reset_index(drop=True)
    return metadata


def validate_selected_breeds(selected_breeds: list[str], available_breeds: list[str]) -> None:
    if len(selected_breeds) != 8:
        raise ValueError(
            f"Exactly 8 breeds are required, got {len(selected_breeds)}. "
            "Set data.selected_breeds in configs/baseline.yaml."
        )

    if len(set(selected_breeds)) != len(selected_breeds):
        raise ValueError("Selected breeds contain duplicates. Provide 8 unique breed names.")

    missing = sorted(set(selected_breeds) - set(available_breeds))
    if missing:
        raise ValueError(
            "Configured breeds not found in dataset metadata: " + ", ".join(missing)
        )
