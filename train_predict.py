from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

DATE_COL = "date"
TARGET = "posted_rate"
CATEGORICAL_FULL = ["pickup", "delivery", "equipment"]
MODEL_LEARNING_RATE = 0.05
MODEL_MAX_LEAF_NODES = 63
FULL_FEATURES = [
    "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
    "delivery_lon", "distance", "equipment", "weight", "month", "day",
    "dayofweek", "dayofyear", "market_index",
]
DECEMBER_FEATURES = [
    "pickup", "delivery", "distance", "equipment", "weight", "month",
    "day", "dayofweek", "dayofyear",
]
EXPECTED_IDS = {f"TE-{i:06d}" for i in range(1, 12001)}


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result[DATE_COL], errors="coerce")
    if dates.isna().any():
        raise ValueError("Found invalid dates.")
    result[DATE_COL] = dates
    result["month"] = dates.dt.month
    result["day"] = dates.dt.day
    result["dayofweek"] = dates.dt.dayofweek
    result["dayofyear"] = dates.dt.dayofyear
    return result


def normalize_negative_weights(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result = frame.copy()
    negative_weights = result["weight"].lt(0)
    negative_count = int(negative_weights.sum())
    result.loc[negative_weights, "weight"] = (
        result.loc[negative_weights, "weight"].abs().astype(int)
    )
    return result, negative_count


def make_model(features: list[str], iterations: int, seed: int = 42):
    categorical_features = [name for name in features if name in CATEGORICAL_FULL]
    numeric_features = [name for name in features if name not in CATEGORICAL_FULL]
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                ),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ],
        verbose_feature_names_out=False,
    )
    categorical_mask = [True] * len(categorical_features) + [False] * len(numeric_features)
    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=MODEL_LEARNING_RATE,
        max_iter=iterations,
        max_leaf_nodes=MODEL_MAX_LEAF_NODES,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        early_stopping=False,
        random_state=seed,
    )
    return make_pipeline(preprocessing, estimator)


def validate_holdout(train: pd.DataFrame) -> dict[str, float | int | str]:
    # Hold out the final month available in the labeled data to simulate forecasting.
    data = add_date_features(train)
    split_date = pd.Timestamp("2025-10-01")
    fit = data[data[DATE_COL] < split_date].copy()
    holdout = data[data[DATE_COL] >= split_date].copy()
    if fit.empty or holdout.empty:
        raise ValueError("Time-based split produced an empty partition.")

    iterations = 500
    model = make_model(FULL_FEATURES, iterations=iterations)
    model.fit(fit[FULL_FEATURES], fit[TARGET])
    predictions = np.maximum(model.predict(holdout[FULL_FEATURES]), 0.01)
    baseline = np.repeat(float(fit[TARGET].median()), len(holdout))
    errors = predictions - holdout[TARGET].to_numpy()
    absolute_errors = np.abs(errors)
    squared_errors = errors**2
    severe_errors = absolute_errors > 500
    return {
        "split": "Train: dates before 2025-10-01; holdout: 2025-10-01 through 2025-10-31",
        "train_rows": int(len(fit)),
        "holdout_rows": int(len(holdout)),
        "iterations": iterations,
        "learning_rate": MODEL_LEARNING_RATE,
        "max_leaf_nodes": MODEL_MAX_LEAF_NODES,
        "baseline_mae": float(mean_absolute_error(holdout[TARGET], baseline)),
        "baseline_rmse": float(mean_squared_error(holdout[TARGET], baseline) ** 0.5),
        "mae": float(mean_absolute_error(holdout[TARGET], predictions)),
        "rmse": float(mean_squared_error(holdout[TARGET], predictions) ** 0.5),
        "r2": float(r2_score(holdout[TARGET], predictions)),
        "p95_absolute_error": float(np.quantile(absolute_errors, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute_errors, 0.99)),
        "errors_over_500": int(severe_errors.sum()),
        "squared_error_share_over_500": float(
            squared_errors[severe_errors].sum() / squared_errors.sum()
        ),
    }


def train_predict(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train_test.csv"
    validation_path = data_dir / "validation.csv"
    template_path = data_dir / "validation_predictions_template.csv"
    december_path = data_dir / "december_chart_inputs.csv"

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    template = pd.read_csv(template_path)
    december = pd.read_csv(december_path)

    required_train = {
        "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
        "delivery_lon", "distance", "equipment", "weight", DATE_COL,
        "market_index", "quote_signal", TARGET, "load_id"
    }
    if not required_train.issubset(train.columns):
        raise ValueError(f"Training file missing columns: {sorted(required_train - set(train.columns))}")
    if "load_id" not in validation or "load_id" not in template:
        raise ValueError("Validation and template must contain load_id.")
    if len(validation) != 12000 or len(template) != 12000:
        raise ValueError("Validation and template must each contain 12,000 rows.")
    if set(validation["load_id"].astype(str)) != EXPECTED_IDS:
        raise ValueError("Validation load IDs do not match expected TE-000001 through TE-012000.")
    if set(template["load_id"].astype(str)) != EXPECTED_IDS:
        raise ValueError("Template load IDs do not match expected TE-000001 through TE-012000.")

    train, negative_train_weights = normalize_negative_weights(train)
    validation, negative_validation_weights = normalize_negative_weights(validation)
    december, negative_december_weights = normalize_negative_weights(december)
    print(
        "Weight normalization: converted "
        f"{negative_train_weights} training, {negative_validation_weights} validation, and "
        f"{negative_december_weights} December negative values to positive integers."
    )

    metrics = validate_holdout(train)
    print("Chronological holdout validation")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    train_features = add_date_features(train)
    validation_features = add_date_features(validation)
    december_features = add_date_features(december)

    # chronological ablation did not improve holdout metrics when it was included.
    final_model = make_model(FULL_FEATURES, iterations=int(metrics["iterations"]))
    final_model.fit(train_features[FULL_FEATURES], train_features[TARGET])
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_out)
    validation_rates = np.maximum(final_model.predict(validation_features[FULL_FEATURES]), 0.01)
    prediction_map = pd.DataFrame({
        "load_id": validation["load_id"].astype(str),
        "predicted_rate": validation_rates,
    })
    # Reindex to the supplied template order, matching by load_id rather than row position.
    output = template[["load_id"]].copy()
    output["load_id"] = output["load_id"].astype(str)
    output = output.merge(prediction_map, on="load_id", how="left", validate="one_to_one")
    if output["predicted_rate"].isna().any():
        raise ValueError("Some template IDs did not receive predictions.")
    output = output[["load_id", "predicted_rate"]]
    output_path = Path(args.predictions_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    # December inputs lack coordinates, market_index and quote_signal, so train a
    # second model using only shared features.
    december_model = make_model(DECEMBER_FEATURES, iterations=int(metrics["iterations"]))
    december_model.fit(train_features[DECEMBER_FEATURES], train_features[TARGET])
    december_model_out = Path(args.december_model_out)
    december_model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(december_model, december_model_out)
    december_rates = np.maximum(december_model.predict(december_features[DECEMBER_FEATURES]), 0.01)
    december_output = december.copy()
    december_output["predicted_rate"] = december_rates
    december_output_path = Path(args.december_out)
    december_output_path.parent.mkdir(parents=True, exist_ok=True)
    december_output.to_csv(december_output_path, index=False)

    print(f"Wrote {len(output):,} validation predictions to {output_path}")
    print(f"Wrote {len(december_output):,} December predictions to {december_output_path}")
    print(f"Saved full-feature model to {model_out}")
    print(f"Saved December-feature model to {december_model_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train freight-rate models and generate assessment predictions.")
    parser.add_argument("--data-dir", default="data", help="Directory containing the four assessment CSV files.")
    parser.add_argument("--predictions-out", default="output/validation_predictions.csv")
    parser.add_argument("--december-out", default="output/december_chart_inputs.csv")
    parser.add_argument("--model-out", default="models/freight_rate_model.joblib")
    parser.add_argument("--december-model-out", default="models/december_freight_rate_model.joblib")
    args = parser.parse_args()
    train_predict(args)


if __name__ == "__main__":
    main()
