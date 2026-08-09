"""Shared model inference utilities for V1/V2 application surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "housing_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "training_metadata.joblib"
MODEL_VERSION = "1.0.0"


@dataclass(frozen=True)
class PredictionResult:
    predicted_price: float
    model_version: str


class HousingInferenceService:
    """Loads the trained pipeline once and runs aligned feature prediction."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline: Any = None
        self.metadata: dict[str, Any] = {}
        self.feature_columns: list[str] = []
        self.is_ready = False

    def load(self) -> None:
        missing = [
            str(path)
            for path in (self.model_path, self.metadata_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required artifact(s). Run `python train.py` first.\n"
                + "\n".join(f"  - {path}" for path in missing)
            )

        self.pipeline = joblib.load(self.model_path)
        self.metadata = joblib.load(self.metadata_path)
        self.feature_columns = self.metadata.get("feature_columns", [])

        if not self.feature_columns:
            raise ValueError("training_metadata.joblib does not contain feature_columns.")

        self.is_ready = True

    def unload(self) -> None:
        self.pipeline = None
        self.metadata = {}
        self.feature_columns = []
        self.is_ready = False

    def build_feature_frame(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        row = {column: payload.get(column) for column in self.feature_columns}
        missing_values = [column for column, value in row.items() if value is None]
        if missing_values:
            raise ValueError(f"Missing required feature(s): {', '.join(missing_values)}")

        return pd.DataFrame([row])

    def predict(self, payload: Mapping[str, Any]) -> PredictionResult:
        if not self.is_ready or self.pipeline is None:
            self.load()

        input_df = self.build_feature_frame(payload)
        prediction = float(self.pipeline.predict(input_df)[0])

        return PredictionResult(
            predicted_price=round(prediction, 2),
            model_version=MODEL_VERSION,
        )


@lru_cache(maxsize=1)
def get_inference_service() -> HousingInferenceService:
    service = HousingInferenceService()
    service.load()
    return service
