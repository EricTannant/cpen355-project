# CPEN355 Project - Cat and Dog Breed Classification

Pipeline for image classification using PyTorch and the Kaggle Oxford-style breed dataset:
zippyz/cats-and-dogs-breeds-classification-oxford-dataset

## 1. What This Project Does

The project provides an end-to-end workflow for:

- Downloading and preparing dataset splits.
- Training three model families: custom_cnn, resnet18, resnet50.
- Fine-tuning with Optuna (TPE sampler + median pruning).
- Evaluating with accuracy/F1 and confusion matrix export.
- Running single-image inference with top-k predictions.

## 2. Repository Structure

```text
configs/
  cnn.yaml
  resnet18.yaml
  resnet50.yaml
scripts/
  download_data.py
  prepare_data.py
  data_visualization.py
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
project_workflow.ipynb
```

## 3. Environment Setup

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

3. Setup Kaggle credentials.

- Put `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`. (On Linux `~\.kaggle\kaggle.json`)
- Ensure permissions are private.

## 4. Configuration

Set exactly the number of breeds in `data.selected_breeds` in your chosen config file (for example `configs/cnn.yaml`).

- Choose one config and use it consistently per experiment/fine-tune run (for example, configs/cnn.yaml).
- data.selected_breeds controls the classes included.
- At least 2 breeds are required.
- If you change selected_breeds, re-run data preparation before training/evaluation.

## 5. Reproducible End-to-End Run

```powershell
python scripts/download_data.py --config configs/cnn.yaml
python scripts/prepare_data.py --config configs/cnn.yaml
python -m src.train --config configs/cnn.yaml
python -m src.evaluate --config configs/cnn.yaml --checkpoint outputs/checkpoints/cnn/best.pt
python -m src.infer --config configs/cnn.yaml --checkpoint outputs/checkpoints/cnn/best.pt --image data/raw/images/pug_82.jpg --top-k 3
```

## 6. Train Each Model

```powershell
python -m src.train --config configs/cnn.yaml
python -m src.train --config configs/resnet18.yaml
python -m src.train --config configs/resnet50.yaml
```

## 7. Fine-Tuning

Edit specific model configs with desired fine-tuning ranges.

- `configs/cnn.yaml`
- `configs/resnet18.yaml`
- `configs/resnet50.yaml`

Then run tuning for any supported config:

```powershell
python -m src.fine_tune --config configs/cnn.yaml --objective val_acc --max-trials 30
```

Useful options:

- --objective val_acc or --objective val_loss
- --max-trials (number of trial runs to check)
- --top-k-checkpoints N (keeps best N trial checkpoints)
- --study-name my_study

Typical outputs:

- Trial checkpoints: outputs/checkpoints/<model_name>/fine_tune/trial_XXX/
- Tuning summary: outputs/metrics/<model_name>_fine_tune_summary.json

Evaluate or infer from a tuned checkpoint:

```powershell
python -m src.evaluate --config configs/cnn.yaml --checkpoint outputs/checkpoints/cnn/fine_tune/trial_001/best.pt
python -m src.infer --config configs/cnn.yaml --checkpoint outputs/checkpoints/cnn/fine_tune/trial_001/best.pt --image data/raw/images/pug_10.jpg --top-k 3
```

## 8. Evaluation and Metric Interpretation

Evaluation writes:

- eval_metrics.json and eval_metrics_<config_name>.json
- confusion_matrix.csv and confusion_matrix_<config_name>.csv

Metric meanings for report writing:

- Precision: of predicted positives, how many were correct.
- Recall: of actual positives, how many were found.
- F1-score: harmonic mean of precision and recall.

## 9. Confusion Matrix Visualization

Use the visualization helper script:

```powershell
python scripts/data_visualization.py --csv outputs/metrics/confusion_matrix_resnet50.csv
```

Helpful options:

- --normalize (row-normalized matrix)
- --save outputs/metrics/confusion_matrix_resnet50.png

## 10. Inference Output Format

Inference returns JSON including runtime and top-k predictions:

```json
{
  "inference_time_seconds": 0.012,
  "inference_time_ms": 12.0,
  "predictions": [
    {
      "breed": "Persian",
      "probability": 0.93
    }
  ]
}
```

To run inference on CPU, set training.device: cpu in your config and run src.infer with that config.

## 11. Output Artifacts

Data artifacts (data/processed):

- metadata_full.csv
- metadata_filtered.csv
- train.csv, val.csv, test.csv
- label_to_index.json
- split_summary.json

Model artifacts (outputs/checkpoints):

- <model_name>/best.pt
- <model_name>/last.pt
- <model_name>/fine_tune/trial_XXX/best.pt (for tuned runs)

Metrics artifacts (outputs/metrics):

- eval_metrics_<config_name>.json
- confusion_matrix_<config_name>.csv
- <model_name>_fine_tune_summary.json (if tuning was run)

## 12. Google Colab Usage

project_workflow.ipynb can be run in Google Colab.

If using Kaggle from Colab:

- Place `kaggle.json` in `Drive/kaggle/kaggle.json`.
- Mount Drive before running download cells.
