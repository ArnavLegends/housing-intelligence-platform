"""
Train an XGBoost regression model for housing price prediction.

Change DATASET_PATH below to point at your housing CSV file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Dataset path — update this when using a different CSV file or location.
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).resolve().parent / "House Price India.csv"

TARGET_COLUMN = "Price"
DROP_COLUMNS = {"id", "Date"}
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_FILENAME = "housing_model.joblib"
PREPROCESSOR_FILENAME = "preprocessor.joblib"
METADATA_FILENAME = "training_metadata.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load housing data from CSV and perform basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    logger.info("Loaded %s rows and %s columns from %s", len(df), len(df.columns), path)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target; drop identifier / non-predictive columns."""
    columns_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    feature_df = df.drop(columns=[TARGET_COLUMN, *columns_to_drop], errors="ignore")
    target = df[TARGET_COLUMN]
    return feature_df, target


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for categorical and numeric columns."""
    categorical_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in feature_df.columns if col not in categorical_cols]

    transformers = []

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )
        logger.info("Categorical columns for encoding: %s", categorical_cols)
    else:
        logger.info("No categorical columns detected; using numeric features only.")

    if numeric_cols:
        transformers.append(("numeric", "passthrough", numeric_cols))
        logger.info("Numeric columns: %s", numeric_cols)

    if not transformers:
        raise ValueError("No usable feature columns found after preprocessing setup.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Create sklearn pipeline with preprocessing and XGBoost regressor."""
    regressor = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
    }


def save_artifacts(
    pipeline: Pipeline,
    feature_columns: list[str],
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Persist model, preprocessor, and metadata for inference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / MODEL_FILENAME
    preprocessor_path = output_dir / PREPROCESSOR_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline.named_steps["preprocessor"], preprocessor_path)
    joblib.dump(
        {
            "target_column": TARGET_COLUMN,
            "drop_columns": sorted(DROP_COLUMNS),
            "feature_columns": feature_columns,
            "metrics": metrics,
        },
        metadata_path,
    )

    logger.info("Saved full pipeline to %s", model_path)
    logger.info("Saved preprocessor to %s", preprocessor_path)
    logger.info("Saved metadata to %s", metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train housing price regression model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to housing dataset CSV (default: DATASET_PATH in train.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory to save trained model and preprocessing artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of data reserved for evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        df = load_dataset(args.dataset)
        features, target = split_features_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=args.test_size,
            random_state=RANDOM_STATE,
        )

        preprocessor = build_preprocessor(features)
        pipeline = build_model_pipeline(preprocessor)

        logger.info("Training XGBoost model on %s samples...", len(X_train))
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        print("\nModel evaluation (test set)")
        print(f"R2 Score : {metrics['r2']:.4f}")
        print(f"MAE      : {metrics['mae']:.2f}")
        print(f"RMSE     : {metrics['rmse']:.2f}\n")

        save_artifacts(pipeline, features.columns.tolist(), metrics, args.output_dir)
        logger.info("Training completed successfully.")
        return 0

    except Exception:
        logger.exception("Training failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
