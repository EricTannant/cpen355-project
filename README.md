# CPEN355 Project

PyTorch pipeline for 8-breed cat/dog image classification using the Kaggle dataset:
`zippyz/cats-and-dogs-breeds-classification-oxford-dataset`

## Overview

This project supports:

- Data download and preprocessing for a custom number of selected breeds (default 8).
- Training with `custom_cnn`, `resnet18`, or `resnet50` models.
- Optuna fine-tuning.
- Evaluation with metrics + confusion matrix to identify hardest to classify breeds.
- Single-image inference with top-k predictions for specific testing.

## Project Layout

```
configs/
  cnn.yaml
  resnet18.yaml
  resnet50.yaml
scripts/
  download_data.py
  prepare_data.py
src/
  train.py
  fine_tune.py
  evaluate.py
  infer.py
data/
  raw/
  processed/
outputs/
  checkpoints/
  metrics/
```

## Setup (Windows PowerShell)

1. Create and activate the virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Configure Kaggle credentials.

- Put `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`. (On Linux `~\.kaggle\kaggle.json`)
- Ensure permissions are private.

## Configure Breeds

Set exactly the number of breeds in `data.selected_breeds` in your chosen config file (for example `configs/cnn.yaml`).

## End-to-End Quick Start

Use one config consistently (for example `configs/cnn.yaml`):

```powershell
python scripts/download_data.py --config configs/cnn.yaml
python scripts/prepare_data.py --config configs/cnn.yaml
python -m src.train --config configs/cnn.yaml
python -m src.evaluate --config configs/cnn.yaml --checkpoint outputs/checkpoints/cnn/best.pt
python -m src.infer --config configs/cnn.yaml --checkpoint outputs/checkpoints/cnn/best.pt --image data/raw/images/pug_82.jpg --top-k 3
```

## Train by Model

CNN:

```powershell
python -m src.train --config configs/cnn.yaml
```

ResNet18:

```powershell
python -m src.train --config configs/resnet18.yaml
```

ResNet50:

```powershell
python -m src.train --config configs/resnet50.yaml
```

## Fine-Tuning

Run fine-tuning by selecting the config, value to maximize and number of runs:

```powershell
python -m src.fine_tune --config configs/cnn.yaml --objective val_acc --max-trials 30
```

Outputs:

- Trial checkpoints: `outputs/checkpoints/cnn/fine_tune/trial_XXX/`
- Summary file: `outputs/metrics/custom_cnn_fine_tune_summary.json`

Use a fine-tuned trial checkpoint for eval/inference:

```powershell
python -m src.evaluate --config configs/cnn.yaml --checkpoint outputs/checkpoints/cnn/fine_tune/trial_101/best.pt
python -m src.infer --config configs/cnn.yaml --checkpoint outputs/checkpoints/cnn/fine_tune/trial_101/best.pt --image data/raw/images/pug_10.jpg --top-k 3
```

## Main Output Files

Data prep (`data/processed/`):

- `metadata_full.csv`
- `metadata_filtered.csv`
- `train.csv`, `val.csv`, `test.csv`
- `label_to_index.json`
- `split_summary.json`

Training checkpoints:

- CNN: `outputs/checkpoints/cnn/`
- ResNet18: `outputs/checkpoints/resnet18/`
- ResNet50: `outputs/checkpoints/resnet50/`

Evaluation metrics (`outputs/metrics/`):

- `eval_metrics.json`
- `eval_metrics_<config_name>.json`
- `confusion_matrix.csv`
- `confusion_matrix_<config_name>.csv`

## Troubleshooting

- Training/evaluation defaults to CUDA and falls back to CPU when CUDA is unavailable.
- `training.gpu_id` controls which GPU index is used.

## Colab

Use the `project_workflow.ipynb` notebook in Google Colab and run the provided cells in order.

If using Kaggle in Colab, place `kaggle.json` in Google Drive (`Drive/kaggle/kaggle.json`) and mount Drive before running data download.