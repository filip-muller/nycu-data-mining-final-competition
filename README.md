# Natural disaster prediction

Kaggle competition for predicting the natural disasters in the next 5 weeks based on historical weather data.

A part of the Data Mining course at NYCU during spring 2026.

Kaggle competition [link](https://www.kaggle.com/competitions/data-mining-2026-final-project) (restricted access)

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The requirements file includes:

- core data tools: `numpy`, `pandas`
- plotting and clustering tools: `matplotlib`, `scipy`
- notebook widgets: `ipywidgets`
- deep learning: `torch`
- optional experiment packages: `polars`, `pyarrow`, `scikit-learn`, `lightgbm`, `xgboost`

Most scripts expect the competition files under `data/`:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

If you are running on Kaggle, copy the read-only input files into a writable working folder first:

```python
from pathlib import Path
import shutil

input_dir = Path("/kaggle/input/datasets/cheukhongtang/drought")
work_dir = Path("/kaggle/working")
data_dir = work_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

for name in ["train.csv", "test.csv", "sample_submission.csv"]:
    shutil.copy2(input_dir / name, data_dir / name)
```

Then run scripts from `/kaggle/working` or pass `--cfg data_dir=/kaggle/working/data` to `deep_drought.py`.

## Main Pipeline

### 1. Cluster regions

Build region-level climate/drought clusters:

```bash
python cluster_regions.py
```

This reads `data/train.csv` and writes:

```text
region_features.csv
region_clusters.csv
region_dendrogram.png
```

`deep_drought.py` uses `region_clusters.csv` if it is present.

### 2. Train the deep drought model

Train the Transformer-based model and write a submission:

```bash
python deep_drought.py train
```

Useful quick test:

```bash
python deep_drought.py train --cfg n_epochs=1 --cfg samples_per_region_per_epoch=5 --cfg use_tta=false
```

Example with explicit output folders:

```bash
python deep_drought.py train \
  --cfg data_dir=data \
  --cfg checkpoint_dir=checkpoints \
  --cfg submission_path=submissions/submission_deep_drought.csv
```

Config values can be overridden with repeated `--cfg key=value` arguments.

### 3. Predict from a saved checkpoint

```bash
python deep_drought.py predict --checkpoint checkpoints/YOUR_CHECKPOINT.pt
```

Optional custom output path:

```bash
python deep_drought.py predict \
  --checkpoint checkpoints/YOUR_CHECKPOINT.pt \
  --output submissions/submission_deep_drought.csv
```

### 4. Postprocess a submission

Scale, zero, and round prediction values:

```bash
python postprocess_submission.py submissions/submission_deep_drought_002.csv 0.9 0.6 0.49
```

Arguments are:

```text
input_csv multiplier zero_threshold closeness_to_integer_threshold
```

Manual-threshold mode:

```bash
python postprocess_submission.py --manual submissions/submission_deep_drought_002.csv
```

Dry run mode creates the output, prints stats, then deletes it:

```bash
python postprocess_submission.py --dry submissions/submission_deep_drought_002.csv 0.9 0.6 0.49
```

## Baselines And Utilities

Create a monthly-average baseline submission:

```bash
python predict_monthly_averages.py
```

Print per-column and overall averages for a submission:

```bash
python calculate_submission_average.py submissions/submission.csv
```

Multiply all numeric values in a CSV by a constant:

```bash
python modify_values.py submissions/submission.csv 0.9
```

Replace zero values in `data/sample_submission.csv` with the constant configured inside `replace_values.py`:

```bash
python replace_values.py
```

Compute the average drought score over a configured training-window range:

```bash
python score_window_average.py
```

For `replace_values.py`, `score_window_average.py`, and `predict_monthly_averages.py`, edit the constants at the top of the file if you want different input paths, windows, quantiles, or replacement values.

## Notebooks

- `deep_drought_ablation_study.ipynb`: ablation study notebook for the deep drought model. Before running it, replace the data file paths in the notebook with your own local or Kaggle dataset paths.
- `group_eda.ipynb`: exploratory data analysis notebook for grouped/region-level analysis. Before running it, replace the data file paths in the notebook with your own local or Kaggle dataset paths.
- `lightgbm.ipynb`: LightGBM model notebook. Before running it, replace the data file paths in the notebook with your own local or Kaggle dataset paths.
- `one-d-cnn-drought-kaggle(1).ipynb`: 1D CNN Kaggle model notebook. Before running it, replace the data file paths in the notebook with your own local or Kaggle dataset paths.
- `cluster_exploration.ipynb`: visual sanity checks for `region_features.csv` and `region_clusters.csv`. Run `cluster_regions.py` first.
- `region_exploration.ipynb`: region-level exploratory analysis. Replace the data file paths with your own local or Kaggle dataset paths before running.
