from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CatBreedDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_size: int, is_train: bool) -> None:
        self.df = pd.read_csv(csv_path)
        if "image_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"CSV {csv_path} must contain image_path and label columns.")

        self.transform = build_transform(image_size=image_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        return image, label


def build_transform(image_size: int = 224, is_train: bool = False):
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_dataloaders(
    processed_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
):
    processed_dir = Path(processed_dir)

    train_dataset = CatBreedDataset(
        csv_path=processed_dir / "train.csv", image_size=image_size, is_train=True
    )
    val_dataset = CatBreedDataset(
        csv_path=processed_dir / "val.csv", image_size=image_size, is_train=False
    )
    test_dataset = CatBreedDataset(
        csv_path=processed_dir / "test.csv", image_size=image_size, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
