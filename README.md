# CPEN355 Project

PyTorch pipeline for 8-breed pet image classification using the Kaggle dataset:
`zippyz/cats-and-dogs-breeds-classification-oxford-dataset`

## What is implemented

- Full dataset download helper (downloads all breeds, no filtering).
- Config-driven dataprep that requires exactly 8 selected breeds.
- Deterministic stratified train/val/test split generation.
- Fixed image preprocessing to 224x224 for train, eval, and inference.
- Training for both a custom CNN and transfer-learning ResNet50.
- Evaluation script with classification metrics and confusion matrix export.
- Single-image inference CLI with top-k probabilities.

## Project structure

```
configs/
	baseline.yaml
	cnn.yaml
	resnet50.yaml
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

## Local run order

Run in this order:

1. Download raw dataset:

```powershell
python scripts/download_data.py --config configs/baseline.yaml
```

2. Prepare filtered metadata and splits (required before train/evaluate):

```powershell
python scripts/prepare_data.py --config configs/baseline.yaml
```

3. Train model.

4. Evaluate model.

## Train and evaluate each model

### CNN

Train:

```powershell
python -m src.train --config configs/cnn.yaml
```

Evaluate:

```powershell
python -m src.evaluate --config configs/cnn.yaml --checkpoint outputs/checkpoints/cnn/best.pt
```

### ResNet50 baseline

Train:

```powershell
python -m src.train --config configs/baseline.yaml
```

Evaluate:

```powershell
python -m src.evaluate --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline/best.pt
```

### ResNet50 extended config

Train:

```powershell
python -m src.train --config configs/resnet50.yaml
```

Evaluate:

```powershell
python -m src.evaluate --config configs/resnet50.yaml --checkpoint outputs/checkpoints/resnet50/best.pt
```

## Output files

Data prep writes to `data/processed/`:

- `metadata_full.csv`
- `metadata_filtered.csv`
- `train.csv`, `val.csv`, `test.csv`
- `label_to_index.json`
- `split_summary.json`

Model checkpoints:

- CNN: `outputs/checkpoints/cnn/`
- Baseline ResNet50: `outputs/checkpoints/baseline/`
- ResNet50 config: `outputs/checkpoints/resnet50/`

Metrics:

- `outputs/metrics/eval_metrics.json`
- `outputs/metrics/confusion_matrix.csv`

## Colab workflow

Upload the `project_workflow.ipynb` notebook to Google colab and run the cells.

Note: Either manually create the ~/.kaggle/kaggle.json file or add it to your Google Drive under Drive/kaggle/kaggle.json which will be automatically cloned via the notebook cells.

## Notes

- Image shape is fixed to 224x224 via torchvision `Resize((224, 224))`.
- The current validation rule requires exactly 8 classes.
- Evaluation artifacts are saved under `outputs/metrics/`.
- Training defaults to `training.device: cuda` and falls back to CPU if CUDA is unavailable.
- Single-GPU training is controlled with `training.gpu_id` (default `0`).