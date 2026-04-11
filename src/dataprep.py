from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ensure_directories, load_config, fix_seed, validate_baseline_constraints
from src.data_utils import build_full_metadata, validate_selected_breeds


def create_splits(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["breed"],
        random_state=seed,
    )

    val_fraction_of_temp = val_ratio / (1.0 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        stratify=temp_df["breed"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def run_dataprep(config_path: str) -> None:
    config = load_config(config_path)
    validate_baseline_constraints(config)
    ensure_directories(config)

    seed = int(config["project"]["seed"])
    fix_seed(seed)
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(data_cfg["processed_dir"])

    full_metadata = build_full_metadata(raw_dir)
    full_metadata.to_csv(processed_dir / "metadata_full.csv", index=False)

    available_breeds = sorted(full_metadata["breed"].unique().tolist())
    selected_breeds = data_cfg.get("selected_breeds", [])
    validate_selected_breeds(selected_breeds, available_breeds)

    filtered = full_metadata[full_metadata["breed"].isin(selected_breeds)].copy()
    filtered = filtered.reset_index(drop=True)

    label_to_index = {breed: idx for idx, breed in enumerate(sorted(selected_breeds))}
    filtered["label"] = filtered["breed"].map(label_to_index)

    train_df, val_df, test_df = create_splits(
        filtered,
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        seed=seed,
    )

    filtered.to_csv(processed_dir / "metadata_filtered.csv", index=False)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    with (processed_dir / "label_to_index.json").open("w", encoding="utf-8") as handle:
        json.dump(label_to_index, handle, indent=2)

    summary = {
        "num_images_full": int(len(full_metadata)),
        "num_images_filtered": int(len(filtered)),
        "selected_breeds": sorted(selected_breeds),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "image_size": int(data_cfg["image_size"]),
    }
    with (processed_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Dataprep complete.")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cat and dog breed dataset splits.")
    parser.add_argument(
        "--config",
        default="configs/resnet50.yaml",
        help="Path to config file.",
    )
    args = parser.parse_args()
    run_dataprep(args.config)


if __name__ == "__main__":
    main()
