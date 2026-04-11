from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kaggle.api.kaggle_api_extended import KaggleApi

from src.config import load_config


def download_dataset(config_path: str) -> None:
    config = load_config(config_path)
    dataset_id = config["data"]["dataset_id"]
    raw_dir = Path(config["data"]["raw_dir"])
    raw_parent = Path(config["data"]["raw_dir"]).parent
    raw_parent.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset=dataset_id,
        path=str(raw_parent),
        unzip=True,
        quiet=False,
    )

    nested_dir = raw_dir / "images"
    if nested_dir.exists() and nested_dir.is_dir():
        # Move everything from 'images/images/*' up to 'images/*'
        for file_path in nested_dir.iterdir():
            shutil.move(str(file_path), str(raw_dir))
        # Remove the now-empty extra folder
        nested_dir.rmdir()

    print(f"Downloaded and extracted {dataset_id} into {raw_parent}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Download full Kaggle cat-breeds dataset.")
    parser.add_argument("--config", default="configs/resnet50.yaml")
    args = parser.parse_args()
    download_dataset(args.config)


if __name__ == "__main__":
    main()
