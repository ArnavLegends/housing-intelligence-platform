# V1 Architecture

## Overview

Housing Intelligence Platform V1 is a compact Python application with three main execution surfaces:

- `streamlit_app.py` for the interactive frontend.
- `app.py` for the FastAPI inference service.
- `train.py` for model training and artifact generation.

The repository also contains one CSV dataset, saved model artifacts, Streamlit configuration, screenshots, dependency declarations, and README documentation.

## Repository Components

```text
housing-intelligence-platform/
|-- House Price India.csv
|-- train.py
|-- app.py
|-- streamlit_app.py
|-- requirements.txt
|-- README.md
|-- .streamlit/config.toml
|-- artifacts/
|   |-- housing_model.joblib
|   |-- preprocessor.joblib
|   `-- training_metadata.joblib
`-- screenshots/
```

## Actual Streamlit Prediction Flow

```text
User
-> Streamlit sidebar inputs
-> build_payload()
-> run_prediction_flow()
-> fetch_prediction()
-> load_model_pipeline()
-> artifacts/housing_model.joblib
-> pipeline.predict()
-> predicted price displayed in Streamlit
```

Important V1 fact: Streamlit does not currently call FastAPI for inference. `streamlit_app.py` defines `API_URL` and `PREDICT_ENDPOINT`, but `fetch_prediction()` loads the local model artifact and runs prediction directly.

## FastAPI Flow

```text
External API client
-> POST /predict
-> PredictionRequest validation
-> build_feature_frame()
-> state.pipeline.predict()
-> PredictionResponse
```

FastAPI loads artifacts during application startup:

- `artifacts/housing_model.joblib`
- `artifacts/preprocessor.joblib`
- `artifacts/training_metadata.joblib`

The implemented endpoints are:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service metadata and endpoint list. |
| `/health` | GET | Model readiness check. |
| `/predict` | POST | Single-property price prediction. |

## Training Flow

```text
House Price India.csv
-> train.py load_dataset()
-> split_features_target()
-> drop id and Date
-> train_test_split(test_size=0.2, random_state=42)
-> ColumnTransformer preprocessing
-> XGBRegressor
-> evaluation metrics
-> saved artifacts
```

## Analytics Flow

Streamlit analytics reads `House Price India.csv` directly. It does not request analytics data from FastAPI.

Analytics includes:

- dataset overview
- missing-value summary
- basic statistics
- interactive filters
- price histogram
- price boxplot
- correlation heatmap
- top correlations with price
- CSV export

## Explainability Flow

Feature importance and SHAP are computed in Streamlit from local model artifacts.

SHAP flow:

```text
filtered dataset sample
-> model feature columns
-> pipeline preprocessor.transform()
-> shap.TreeExplainer(regressor)
-> global SHAP importance chart
-> SHAP impact scatter chart
```

The current SHAP view is global over filtered dataset samples. It is not a local explanation for a specific user-submitted prediction.

## Known Architecture Limitations

- Streamlit bypasses FastAPI for prediction.
- Inference artifact loading is duplicated in Streamlit and FastAPI.
- The feature schema is repeated across training metadata, FastAPI request models, Streamlit defaults, and Streamlit input widgets.
- `streamlit_app.py` is monolithic.
- No tests were found.
- No CI or deployment manifest was found.
- Dataset, artifact, and model paths are hardcoded relative to the repository.
