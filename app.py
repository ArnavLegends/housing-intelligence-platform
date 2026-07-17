"""
Housing Intelligence Platform — production FastAPI inference service.

Run the API locally with uvicorn:

    # From the project root (housing-intelligence-platform/)
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Production example (multiple workers, no auto-reload):

    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

Interactive docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "housing_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
METADATA_PATH = ARTIFACTS_DIR / "training_metadata.joblib"

MODEL_VERSION = "1.0.0"
APP_TITLE = "Housing Intelligence Platform"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AppState:
    """Holds loaded model artifacts for the lifetime of the application."""

    pipeline: Any = None
    preprocessor: Any = None
    metadata: dict[str, Any] = {}
    feature_columns: list[str] = []
    is_ready: bool = False


state = AppState()

# Keys must match training feature column names (PredictionRequest aliases).
PREDICTION_REQUEST_EXAMPLE: dict[str, float | int] = {
    "number of bedrooms": 4,
    "number of bathrooms": 2.5,
    "living area": 2920,
    "lot area": 4000,
    "number of floors": 1.5,
    "waterfront present": 0,
    "number of views": 0,
    "condition of the house": 5,
    "grade of the house": 8,
    "Area of the house(excluding basement)": 1910,
    "Area of the basement": 1010,
    "Built Year": 1909,
    "Renovation Year": 0,
    "Postal Code": 122004,
    "Lattitude": 52.8878,
    "Longitude": -114.47,
    "living_area_renov": 2470,
    "lot_area_renov": 4000,
    "Number of schools nearby": 2,
    "Distance from the airport": 51,
}


class PredictionRequest(BaseModel):
    """Validated housing feature payload for price prediction."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_mode="serialization",
        json_schema_extra={"examples": [PREDICTION_REQUEST_EXAMPLE]},
    )

    number_of_bedrooms: float = Field(..., alias="number of bedrooms", gt=0, le=20)
    number_of_bathrooms: float = Field(..., alias="number of bathrooms", gt=0, le=20)
    living_area: float = Field(..., alias="living area", gt=0)
    lot_area: float = Field(..., alias="lot area", gt=0)
    number_of_floors: float = Field(..., alias="number of floors", gt=0, le=10)
    waterfront_present: int = Field(..., alias="waterfront present", ge=0, le=1)
    number_of_views: float = Field(..., alias="number of views", ge=0)
    condition_of_the_house: float = Field(..., alias="condition of the house", ge=1, le=5)
    grade_of_the_house: float = Field(..., alias="grade of the house", ge=1, le=13)
    area_excluding_basement: float = Field(
        ..., alias="Area of the house(excluding basement)", ge=0
    )
    area_of_basement: float = Field(..., alias="Area of the basement", ge=0)
    built_year: float = Field(..., alias="Built Year", ge=1800, le=2100)
    renovation_year: float = Field(..., alias="Renovation Year", ge=0, le=2100)
    postal_code: float = Field(..., alias="Postal Code", gt=0)
    lattitude: float = Field(..., alias="Lattitude", ge=-90, le=90)
    longitude: float = Field(..., alias="Longitude", ge=-180, le=180)
    living_area_renov: float = Field(..., alias="living_area_renov", ge=0)
    lot_area_renov: float = Field(..., alias="lot_area_renov", ge=0)
    number_of_schools_nearby: float = Field(
        ..., alias="Number of schools nearby", ge=0
    )
    distance_from_airport: float = Field(
        ..., alias="Distance from the airport", ge=0
    )

    @field_validator("renovation_year")
    @classmethod
    def validate_renovation_year(cls, value: float, info) -> float:
        built_year = info.data.get("built_year")
        if built_year is not None and value > 0 and value < built_year:
            raise ValueError("Renovation Year must be greater than or equal to Built Year.")
        return value


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted housing price in INR.")
    model_version: str = Field(..., description="Deployed model version identifier.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class RootResponse(BaseModel):
    service: str
    version: str
    model_version: str
    endpoints: dict[str, str]


def load_artifacts() -> None:
    """Load model, preprocessor, and metadata from disk."""
    missing = [
        str(path)
        for path in (MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required artifact(s). Run `python train.py` first.\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    state.pipeline = joblib.load(MODEL_PATH)
    state.preprocessor = joblib.load(PREPROCESSOR_PATH)
    state.metadata = joblib.load(METADATA_PATH)
    state.feature_columns = state.metadata.get("feature_columns", [])

    if not state.feature_columns:
        raise ValueError("training_metadata.joblib does not contain feature_columns.")

    state.is_ready = True
    logger.info("Loaded model from %s", MODEL_PATH)
    logger.info("Loaded preprocessor from %s", PREPROCESSOR_PATH)
    logger.info("Loaded metadata from %s", METADATA_PATH)


def build_feature_frame(request: PredictionRequest) -> pd.DataFrame:
    """Convert validated request into a single-row DataFrame aligned with training columns."""
    record = request.model_dump(by_alias=True)
    row = {column: record.get(column) for column in state.feature_columns}

    missing_values = [column for column, value in row.items() if value is None]
    if missing_values:
        raise ValueError(f"Missing required feature(s): {', '.join(missing_values)}")

    return pd.DataFrame([row])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts on startup and release references on shutdown."""
    try:
        load_artifacts()
    except Exception:
        logger.exception("Failed to load model artifacts during startup.")
        raise
    yield
    state.pipeline = None
    state.preprocessor = None
    state.metadata = {}
    state.feature_columns = []
    state.is_ready = False
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=APP_TITLE,
    description="Predict housing prices using a trained XGBoost regression model.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid request payload.",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )


@app.get("/", response_model=RootResponse, tags=["General"])
def root() -> RootResponse:
    return RootResponse(
        service=APP_TITLE,
        version=MODEL_VERSION,
        model_version=MODEL_VERSION,
        endpoints={
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health() -> HealthResponse:
    if not state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(
    request: PredictionRequest = Body(
        ...,
        openapi_examples={
            "sample_property": {
                "summary": "Sample property (matches training CSV columns)",
                "value": PREDICTION_REQUEST_EXAMPLE,
            }
        },
    ),
) -> PredictionResponse:
    if not state.is_ready or state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    try:
        input_df = build_feature_frame(request)
        prediction = float(state.pipeline.predict(input_df)[0])

        if prediction < 0:
            logger.warning("Model returned a negative price prediction: %s", prediction)

        return PredictionResponse(
            predicted_price=round(prediction, 2),
            model_version=MODEL_VERSION,
        )
    except ValueError as exc:
        logger.warning("Prediction rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal error.",
        ) from exc
