# CPEN355 Project

Baseline PyTorch pipeline for an 8-breed cat classifier using the Kaggle dataset:
oxford cats and dogs dataset (dataset id: zippyz/cats-and-dogs-breeds-classification-oxford-dataset)

## What is implemented

- Full dataset download helper (downloads all breeds, no filtering).
- Config-driven dataprep that requires exactly 8 selected breeds.
- Deterministic stratified train/val/test split generation.
- Fixed image preprocessing to 224x224 for train, eval, and inference.
- Transfer-learning training script (default: pretrained ResNet50).
- Evaluation script with classification metrics and confusion matrix export.
- Single-image inference CLI with top-k probabilities.

## Project structure

```
configs/
	baseline.yaml
data/
	raw/
	processed/
outputs/
	checkpoints/
	metrics/
scripts/
	download_data.py
	prepare_data.py
src/
	config.py
	data_utils.py
	dataprep.py
	dataset.py
	models.py
	train.py
	evaluate.py
	infer.py
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set up Kaggle credentials on Windows:

- Put `kaggle.json` in `%USERPROFILE%\.kaggle\kaggle.json`.
- Ensure file permissions are private.

## Configure selected breeds

Edit `configs/baseline.yaml` and set `data.selected_breeds` to exactly 8 breed names
that exist in the downloaded dataset.

Example:

```yaml
data:
	selected_breeds:
		- Sphynx
		- Siamese
		- Russian_Blue
		- Bengal
		- pug
		- great_pyrenees
		- shiba_inu
		- yorkshire_terrier
```

Note: dataprep fails fast if the list is not exactly 8 unique valid breeds.

## Run workflow

1. Download full dataset:

```powershell
python scripts/download_data.py --config configs/baseline.yaml
```

2. Prepare filtered subset and splits:

```powershell
python scripts/prepare_data.py --config configs/baseline.yaml
```

Outputs written to `data/processed/` include:

- `metadata_full.csv`
- `metadata_filtered.csv`
- `train.csv`, `val.csv`, `test.csv`
- `label_to_index.json`
- `split_summary.json`

3. Train baseline model:

```powershell
python -m src.train --config configs/baseline.yaml
```

4. Evaluate best checkpoint:

```powershell
python -m src.evaluate --config configs/baseline.yaml --checkpoint outputs/checkpoints/best.pt
```

5. Run inference on one image:

```powershell
python -m src.infer --config configs/baseline.yaml --checkpoint outputs/checkpoints/best.pt --image path\to\cat.jpg --top-k 3
```

## Notes

- Image shape is fixed to 224x224 via torchvision `Resize((224, 224))`.
- The model expects 8 classes for baseline runs.
- Evaluation artifacts are saved under `outputs/metrics/`.
- Training defaults to `training.device: cuda` and falls back to CPU if CUDA is unavailable.
- Single-GPU training is controlled with `training.gpu_id` (default `0`).