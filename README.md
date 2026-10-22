# Spotter Freight Rate Prediction Challenge

This repository is a reproducible solution for the Machine Learning Engineer. It trains from `data/train_test.csv`, predicts every load in `data/validation.csv`, and validates the required submission files.

## Deliverables

- `output/validation_predictions.csv` and `output/december_chart_inputs.csv` contain the completed prediction files.
- `output/candidate_december.png` contains the December prediction chart.
- `validation_predictions.csv` — the required 12,000-row submission with exactly `load_id,predicted_rate`.
- `report.pdf` — validation approach, model rationale, data-quality treatment, model comparison, and December chart.

## Setup and final-model run

Use Python 3.15. From the repository root in PowerShell:

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_predict.py
python score.py --predictions output/validation_predictions.csv --december-predictions output/december_chart_inputs.csv
```

`train_predict.py` creates the completed CSV files in `output/` and saves model artifacts in `models/`. Use its output-path options to change these defaults.

## Approach

- Target: `posted_rate`.
- Split: train on dates before 2025-10-01 (43,147 rows) and evaluate on October 2025 (4,853 rows). This chronological holdout estimates future prediction performance without training on later records.
- Features: derive month, day, weekday, and day of year from `date`; convert invalid negative weights to positive integers; ordinal-encode pickup, delivery, and equipment.
- Final model: `HistGradientBoostingRegressor` with absolute-error loss. It models nonlinear tabular relationships and natively supports missing numeric values.
- Excluded fields: `load_id` is used only to align template rows. `quote_signal` is omitted because its definition and prediction-time availability are undocumented.
- December scenario: because it lacks coordinates and `market_index`, a separate pipeline uses only fields shared with the labeled data instead of inventing values.

## Validation and assessment checks

The scripts report holdout MAE, RMSE, and R-squared. These are internal October estimates.

`score.py` verifies that the final CSV has exactly 12,000 required IDs and positive numeric rates. It also validates the 31 fixed December rows and writes `scorer_results/candidate_december.png`.
