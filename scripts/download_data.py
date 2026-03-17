from __future__ import annotations

import argparse
from pathlib import Path
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
    raw_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset=dataset_id,
        path=str(raw_dir),
        unzip=True,
        quiet=False,
    )

    print(f"Downloaded and extracted {dataset_id} into {raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download full Kaggle cat-breeds dataset.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    download_dataset(args.config)


if __name__ == "__main__":
    main()
