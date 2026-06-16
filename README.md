# Yogyank Entitlement Score - Fixed Baseline Training Pipeline

## Time

* Start Time (IST): `11:00`
* End Time (IST): `12:12`
* Approximate Time Spent: `72 mins`

## Setup

Install dependencies:

```bash
pip install pandas scikit-learn xgboost joblib
```

Run:

```bash
python fixed_yogyank_training.py
```

## Generated Files

```text
artifacts/
├── pipeline.pkl
├── feature_list.json
└── validation_summary.json
```

## Completed

* Removed leakage feature (`defaulted_in_next_12_months`)
* Separated business policy from model
* Replaced random split with temporal split (2022+2023 → 2024)
* Added preprocessing pipeline with imputation and encoding
* Saved the complete training pipeline and metadata artifacts
* Added schema validation and evaluation metrics

## Skipped

* Hyperparameter tuning
* Walk-forward cross validation
* SHAP/reason codes due to time and dependency constraints

## Assumptions

* `historical_repayment_score` is available at application time.
* `annual_income_inr` refers to prior FY income.
* `rainfall_deviation_pct` and `ndvi_score` are computed before application.
* `defaulted_in_next_12_months` is a future outcome and therefore excluded.

## Validation

A temporal holdout strategy was used.

* Train: 2022, 2023
* Test: 2024

This better simulates real deployment than a random shuffle split. I trust the result more than the original baseline, although the estimate is limited by having only a single holdout year.
