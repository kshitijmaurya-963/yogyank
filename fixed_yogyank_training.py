
"""
Yogyank Entitlement Score - Fixed Baseline Training Script
Author - Kshitij Maurya
Notes - Fixed issues from the v1 draft
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

MODEL_VERSION = "1.0.0"
RANDOM_SEED = 42
TEMPORAL_CUTOFF_YEAR = 2024
PM_KISAN_PENALTY = 150
ARTIFACTS_DIR = "artifacts"

NUMERIC_FEATURES = [
    "land_area_acres",
    "historical_repayment_score",
    "annual_income_inr",
    "liability_ratio_pct",
    "rainfall_deviation_pct",
    "ndvi_score",
]

CATEGORICAL_FEATURES = [
    "crop_type",
    "pm_kisan_status",
    "irrigation_type",
    "land_ownership",
    "soil_type",
    "sales_channel",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "target_entitlement_score"
REQUIRED_COLUMNS = ALL_FEATURES + ["application_year", TARGET_COL]


def load_data(path="farmer_scoring_sample_yogyank_round1.csv"):
    return pd.read_csv(path)


def validate_schema(df):
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Schema violation - missing columns: {missing}")


def temporal_split(df, cutoff=TEMPORAL_CUTOFF_YEAR):
    return (
        df[df["application_year"] < cutoff].copy(),
        df[df["application_year"] == cutoff].copy(),
    )


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
    ])

    model = xgb.XGBRegressor(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def evaluate(pipeline, X, y):
    preds = pipeline.predict(X)
    metrics = {
        "r2": float(r2_score(y, preds)),
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(root_mean_squared_error(y, preds)),
        "n_test": int(len(y)),
    }
    return metrics, preds

def apply_business_policy(scores, X):
    result = pd.DataFrame({
        "model_score": scores,
        "policy_adjustment": 0.0,
        "final_score": scores.copy(),
    })

    mask = X["pm_kisan_status"].values == "No"
    result.loc[mask, "policy_adjustment"] -= PM_KISAN_PENALTY
    result.loc[mask, "final_score"] -= PM_KISAN_PENALTY
    return result


def save_artifacts(pipeline, metrics):
    out = Path(ARTIFACTS_DIR)
    out.mkdir(exist_ok=True)

    joblib.dump(pipeline, out / "pipeline.pkl")

    with open(out / "feature_list.json", "w") as f:
        json.dump({
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "all_features": ALL_FEATURES,
            "target": TARGET_COL,
            "excluded_features": [{
                "name": "defaulted_in_next_12_months",
                "reason": "future information leakage",
            }],
        }, f, indent=2)

    with open(out / "validation_summary.json", "w") as f:
        json.dump({
            "model_version": MODEL_VERSION,
            "split_strategy": "temporal",
            "train_years": [2022, 2023],
            "test_year": 2024,
            "metrics": metrics,
        }, f, indent=2)


def train_model(data_path="farmer_scoring_sample_yogyank_round1.csv"):
    df = load_data(data_path)
    validate_schema(df)

    train_df, test_df = temporal_split(df)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET_COL]

    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COL]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics, raw_preds = evaluate(pipeline, X_test, y_test)
    final_scores = apply_business_policy(raw_preds, X_test)

    save_artifacts(pipeline, metrics)

    print("Validation Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return final_scores


if __name__ == "__main__":
    train_model()
