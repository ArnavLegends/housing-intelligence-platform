# Housing Intelligence Platform V1 Engineering Audit

Date: 2026-08-09  
Scope: repository audit, architecture trace, dataset/ML/backend/frontend/product review, V2 traceability, and documentation plan.  
Constraint: audit and documentation only. No application code, UI, training, model, or prediction pipeline changes were made.

## 1. Executive Summary

V1 is a compact Streamlit + FastAPI + XGBoost project with a working local prediction path, dataset analytics, feature importance, SHAP-style global explanation charts, screenshots, and saved model artifacts. It is stronger than a static student website because it contains an actual training script, saved artifacts, an inference API, and an interactive Streamlit application.

The most important architectural finding is that Streamlit does **not** currently use FastAPI for inference. `streamlit_app.py` defines `API_URL` and imports `requests`, but `fetch_prediction()` loads `artifacts/housing_model.joblib` directly and calls `model.predict()` in-process. FastAPI is implemented in `app.py`, but it is a parallel inference surface rather than the active frontend boundary.

The most important product/data finding is that dataset provenance is **UNVERIFIED**. The repository names the data `House Price India.csv` and displays prices as INR, but the actual latitude/longitude range is approximately `52.3859` to `53.0076` latitude and `-114.709` to `-113.505` longitude, which is not India. Postal codes are `122003` to `122072`, which look Indian/Gurugram-like, creating an unresolved geographic mismatch. Until provenance is verified, this project should not claim to be a genuine Indian residential valuation product.

The most important ML finding is that the saved metadata reports `R2=0.893353`, `MAE=63533.23`, and `RMSE=122590.98`, matching the README's rounded values. These values were verified from `artifacts/training_metadata.joblib`, not recomputed from the saved model because the current sandbox does not have `joblib`, `sklearn`, or `xgboost` installed. The training script uses a single random 80/20 split and does not perform cross-validation, geographic holdout validation, leakage testing, uncertainty estimation, or calibration.

V1 is a good technical prototype, but it is not yet a trustworthy real-estate decision product. The V2 priority should be provenance, API-first inference, UX simplification around real user journeys, clearer human-readable explanations, uncertainty/valuation ranges, and tests.

## 2. Repository Audit

### Structure

```text
housing-intelligence-platform/
|-- .gitignore
|-- .streamlit/config.toml
|-- README.md
|-- requirements.txt
|-- House Price India.csv
|-- train.py
|-- app.py
|-- streamlit_app.py
|-- artifacts/
|   |-- housing_model.joblib
|   |-- preprocessor.joblib
|   `-- training_metadata.joblib
`-- screenshots/
    |-- correlation_heatmap.png
    |-- dashboard_overview.png
    |-- feature_importance.png
    |-- model_performance.png
    |-- prediction_interface.png
    |-- price_analytics.png
    `-- shap_explainability.png
```

### Important Files

| File | What it does | Evidence |
|---|---|---|
| `README.md` | Public project description, feature list, screenshots, setup, training, API instructions, reported metrics. | `README.md:1-139` |
| `requirements.txt` | Runtime dependencies for pandas, numpy, scikit-learn, xgboost, FastAPI, Streamlit, Plotly, SHAP, requests. | `requirements.txt` |
| `House Price India.csv` | Primary dataset used by training and analytics. | `train.py:27`, `streamlit_app.py:26` |
| `train.py` | Loads CSV, drops `id` and `Date`, splits data, builds preprocessing + XGBoost pipeline, evaluates, saves artifacts. | `train.py:47-222` |
| `app.py` | FastAPI inference service with request validation, artifact loading, `/`, `/health`, and `/predict`. | `app.py:84-311` |
| `streamlit_app.py` | Streamlit frontend with sidebar inputs, prediction display, analytics tabs, feature importance, SHAP, About page. | `streamlit_app.py:1-1227` |
| `.streamlit/config.toml` | Streamlit theme/server/browser settings. | `.streamlit/config.toml` |
| `artifacts/housing_model.joblib` | Saved sklearn pipeline with preprocessing + XGBoost. Full load not executed in this audit due missing ML runtime packages. | `train.py:144` |
| `artifacts/preprocessor.joblib` | Saved preprocessing step only. Full load not executed in this audit due missing sklearn runtime. | `train.py:145` |
| `artifacts/training_metadata.joblib` | Saved metadata: target, dropped columns, feature columns, metrics. Read successfully via pickle. | `train.py:146-154` |
| `screenshots/*.png` | README/demo screenshots for prediction, analytics, explainability, and performance. | `README.md:33-54` |

### Entry Points

| Entry point | Purpose | Current status |
|---|---|---|
| `streamlit run streamlit_app.py` | Main frontend application. | Implemented. |
| `uvicorn app:app --reload` | FastAPI service. | Implemented. |
| `python train.py` | Model training and artifact generation. | Implemented, not run during this audit. |

### Tests

No test files, pytest config, CI config, or test directories were found.

### Deployment

README links a Streamlit Cloud demo and describes `API_URL`, but there is no `Dockerfile`, `docker-compose.yml`, GitHub Actions workflow, Streamlit secrets template, deployment manifest, or hosted FastAPI deployment config in the repository.

## 3. Actual V1 Architecture

### Verified Execution Flow

```text
User
-> Streamlit sidebar inputs
-> streamlit_app.py build_payload()
-> streamlit_app.py run_prediction_flow()
-> streamlit_app.py fetch_prediction()
-> local joblib pipeline loaded from artifacts/housing_model.joblib
-> pipeline preprocessing step
-> XGBoost regressor
-> predicted price displayed in Streamlit
```

### FastAPI Flow

```text
External API client
-> FastAPI /predict
-> Pydantic PredictionRequest validation
-> build_feature_frame()
-> state.pipeline.predict()
-> PredictionResponse
```

### Key Architecture Facts

| Question | Finding |
|---|---|
| Does Streamlit actually call FastAPI for inference? | No. `fetch_prediction()` loads the local model and predicts directly. No `requests.post()` call exists. |
| Where is the model loaded? | FastAPI: `load_artifacts()` loads `housing_model.joblib` on app startup. Streamlit: `load_model_pipeline()` loads `housing_model.joblib` as a cached resource. |
| Where does preprocessing occur? | Inside the saved sklearn pipeline. Training builds `ColumnTransformer` + `XGBRegressor`; inference calls `pipeline.predict()`. |
| How are artifacts loaded? | `joblib.load()` in both `app.py` and `streamlit_app.py`. Metadata was independently readable via pickle. |
| How does validation occur? | FastAPI uses Pydantic field bounds and a renovation-year validator. Streamlit uses widget min/max values and a renovation-before-built check. |
| How does analytics obtain data? | Streamlit reads `House Price India.csv` directly via `pd.read_csv()`. |
| How does SHAP obtain data? | Streamlit samples up to 500 filtered CSV rows, transforms them with the pipeline preprocessor, then runs `shap.TreeExplainer` on the regressor. |

Evidence:

- Streamlit endpoint constants exist: `streamlit_app.py:30-31`.
- Direct local model prediction occurs in `fetch_prediction()`: `streamlit_app.py:225-235`.
- Streamlit prediction flow calls `fetch_prediction(payload)`: `streamlit_app.py:1144-1180`.
- FastAPI loads artifacts on lifespan startup: `app.py:147-190`.
- FastAPI prediction calls `state.pipeline.predict(input_df)`: `app.py:289-299`.
- Analytics reads CSV directly: `streamlit_app.py:259-263`, `streamlit_app.py:1063-1072`.
- SHAP loads local model and transforms local sample: `streamlit_app.py:423-488`.

## 4. Dataset Audit

### Basic Dataset Facts

| Item | Value |
|---|---|
| Exact filename | `House Price India.csv` |
| Rows | 14,620 |
| Columns | 23 |
| Target column | `Price` |
| Feature columns used by model | 20 |
| Dropped columns | `id`, `Date` |
| Missing values | 0 in every column |
| Full duplicate rows | 0 |
| Duplicate IDs | 0 |
| Date encoding | Numeric values 42491 to 42734; if Excel serial dates, 2016-05-01 to 2016-12-30 |

### Columns and Types

| Column | Type | Model role |
|---|---|---|
| `id` | int64 | Dropped |
| `Date` | int64 | Dropped |
| `number of bedrooms` | int64 | Feature |
| `number of bathrooms` | float64 | Feature |
| `living area` | int64 | Feature |
| `lot area` | int64 | Feature |
| `number of floors` | float64 | Feature |
| `waterfront present` | int64 | Feature |
| `number of views` | int64 | Feature |
| `condition of the house` | int64 | Feature |
| `grade of the house` | int64 | Feature |
| `Area of the house(excluding basement)` | int64 | Feature |
| `Area of the basement` | int64 | Feature |
| `Built Year` | int64 | Feature |
| `Renovation Year` | int64 | Feature |
| `Postal Code` | int64 | Feature |
| `Lattitude` | float64 | Feature |
| `Longitude` | float64 | Feature |
| `living_area_renov` | int64 | Feature |
| `lot_area_renov` | int64 | Feature |
| `Number of schools nearby` | int64 | Feature |
| `Distance from the airport` | int64 | Feature |
| `Price` | int64 | Target |

### Target Distribution

| Statistic | Price |
|---|---:|
| Count | 14,620 |
| Mean | 538,932.22 |
| Std dev | 367,532.38 |
| Min | 78,000 |
| 1% | 154,585.50 |
| 5% | 210,000 |
| 25% | 320,000 |
| Median | 450,000 |
| 75% | 645,000 |
| 95% | 1,150,000 |
| 99% | 1,950,000 |
| Max | 7,700,000 |
| Skew | 4.2693 |

Price bins:

| Price range | Rows |
|---|---:|
| <= 200,000 | 571 |
| 200,001-400,000 | 5,417 |
| 400,001-600,000 | 4,416 |
| 600,001-800,000 | 2,236 |
| 800,001-1,000,000 | 1,001 |
| 1,000,001-1,500,000 | 646 |
| 1,500,001-2,000,000 | 202 |
| 2,000,001-10,000,000 | 131 |

### Feature Ranges and Suspicious Values

| Feature | Observed range / issue |
|---|---|
| `number of bedrooms` | 1 to 33. One row has 33 bedrooms with 1.75 bathrooms and 1,620 sq ft, likely an outlier or data error. |
| `number of bathrooms` | 0.5 to 8.0. |
| `living area` | 370 to 13,540 sq ft. |
| `lot area` | 520 to 1,074,218 sq ft. |
| `number of floors` | 1.0, 1.5, 2.0, 2.5, 3.0, 3.5. This is understandable in some markets but may confuse normal users. |
| `waterfront present` | 112 true rows, 14,508 false rows. Highly imbalanced. |
| `number of views` | 0 to 4. Definition is not documented. |
| `condition of the house` | 1 to 5. Definition is not documented. |
| `grade of the house` | 4 to 13. Definition is not documented and appears dataset-specific. |
| `Built Year` | 1900 to 2015. |
| `Renovation Year` | 0 or 1900 to 2015. 666 nonzero rows. |
| `Postal Code` | 122003 to 122072; 70 unique values. |
| `Lattitude` | 52.3859 to 53.0076. Column is misspelled as `Lattitude`. |
| `Longitude` | -114.709 to -113.505. |
| `Number of schools nearby` | Only 1, 2, or 3; no zero values. |
| `Distance from the airport` | 50 to 80 km. Very narrow and definition/source are undocumented. |

Internal consistency:

- `living area == Area of the house(excluding basement) + Area of the basement` for all rows.
- No nonzero `Renovation Year` values are earlier than `Built Year`.
- No `Built Year` or `Renovation Year` values are after 2016.

### Indian Dataset Provenance

Status: **UNVERIFIED**.

The repository does not contain a dataset card, source URL, license, data dictionary, collection method, geography explanation, or currency conversion note. The filename says India and postal codes resemble `122xxx`, but latitude/longitude values around `52` and `-114` do not correspond to India. The product also labels prices as INR in both API and Streamlit, but the target values are not explained.

Recommendation: before V2 product work, create a `docs/ml/dataset-card.md` that verifies original source, license, geography, currency, units, feature definitions, preprocessing applied before this repository received the CSV, and whether coordinates/postal codes were transformed or synthetic.

## 5. ML Audit

### Training Methodology

| Item | Finding |
|---|---|
| Algorithm | `XGBRegressor` |
| Target | `Price` |
| Dropped columns | `id`, `Date` |
| Preprocessing | Object/category/bool columns one-hot encoded; numeric columns passed through. Current CSV is all numeric, so effectively numeric passthrough. |
| Split | `train_test_split(..., test_size=0.2, random_state=42)` |
| Train rows | 11,696 inferred from 80% of 14,620 |
| Test rows | 2,924 inferred from 20% of 14,620 |
| Random seed | 42 |
| Saved artifacts | full pipeline, preprocessor, metadata |

### Hyperparameters

From `train.py:100-118`:

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.9 |
| `random_state` | 42 |
| `n_jobs` | -1 |
| `objective` | `reg:squarederror` |

### Metrics

Saved metadata reports:

| Metric | Value |
|---|---:|
| R2 | 0.8933532238006592 |
| MAE | 63,533.23046875 |
| RMSE | 122,590.98094068747 |

Verification status: artifact-verified from `artifacts/training_metadata.joblib`; not independently recomputed from `housing_model.joblib` because the current audit runtime lacks `joblib`, `sklearn`, and `xgboost`. No retraining was performed.

### Risks and Limitations

| Risk | Severity | Rationale |
|---|---|---|
| Dataset provenance unknown | Critical | A real-estate product cannot claim geography/currency credibility without source, license, and unit clarity. |
| Possible geography leakage | High | `Postal Code`, `Lattitude`, and `Longitude` can dominate predictions and may not generalize outside the dataset geography. |
| Single random split | High | Real-estate data is spatial and temporal. Random splitting can overstate performance if nearby/similar homes are split across train and test. |
| No temporal validation | High | `Date` is dropped, so the model is not tested on future-market generalization. |
| No cross-validation | Medium | Reported score depends on one split. |
| No outlier handling | Medium | Example: 33 bedrooms with 1,620 sq ft. |
| No uncertainty interval | Medium | A point estimate alone is not enough for decision support. |
| No error segmentation | Medium | No performance by price band, postal code, property size, year built, waterfront status, or high-value homes. |
| No model card | Medium | Metrics, limitations, intended use, and data constraints are not documented formally. |

Target leakage: no direct duplicate of `Price` was found among named features, and `id`/`Date` are dropped. However, leakage-like overfitting risk remains through highly location-specific fields and derived/renovated area fields if the dataset construction process used post-sale or neighborhood price information. This is **UNVERIFIED** without provenance.

## 6. Backend Audit

### Implemented API

| Endpoint | Method | Response model | Purpose |
|---|---|---|---|
| `/` | GET | `RootResponse` | Service metadata and endpoint list. |
| `/health` | GET | `HealthResponse` | Model readiness check. |
| `/predict` | POST | `PredictionResponse` | Price inference. |

### Request Validation

`PredictionRequest` validates all 20 model features using Pydantic aliases that match CSV column names. It enforces broad bounds for bedrooms, bathrooms, floors, waterfront, condition, grade, years, coordinates, and nonnegative area/context fields. It also rejects renovation years earlier than built years.

### Response Schema

`PredictionResponse` returns:

- `predicted_price`
- `model_version`

The field description states INR, but currency provenance is **UNVERIFIED**.

### Error Handling

FastAPI has custom handlers for:

- request validation errors: HTTP 422
- explicit `HTTPException`
- unhandled exceptions: HTTP 500

Prediction-specific errors include unloaded model, missing feature frame values, negative prediction warning, and generic prediction failure.

### Backend Problems

| Issue | Severity | Evidence |
|---|---|---|
| API is not the Streamlit inference boundary | High | Streamlit predicts locally in `streamlit_app.py:225-235`. |
| Duplicate inference logic/artifact loading | High | Both `app.py` and `streamlit_app.py` load `housing_model.joblib`. |
| API schema exposes raw dataset columns | Medium | Request aliases include `Lattitude`, `grade of the house`, `living_area_renov`, etc. |
| No explicit API versioned route | Medium | `MODEL_VERSION` exists, but route is `/predict`, not `/v1/predict`. |
| No batch prediction/report endpoint | Low for V1 | Current API only supports one prediction. |
| No observability beyond logs | Medium | No request IDs, metrics, latency tracking, or inference audit trail. |
| Currency claim unverified | High | API says predicted price in INR but dataset currency is undocumented. |

## 7. Frontend / UX Audit

### Current Navigation

Streamlit uses:

- Sidebar property input form.
- Main tabs: `Property Details`, `Prediction`, `Analytics`, `About`.
- Analytics subtabs: `Overview`, `Price Analytics`, `Correlations`, `Feature Importance`, `Model Performance`, `SHAP`.

### Current Prediction Form

The sidebar asks users to provide 20 raw model features:

- bedrooms, bathrooms, living area, lot area, floors
- waterfront, views, condition, grade
- area excluding basement, basement area
- built year, renovation year
- postal code, latitude, longitude
- renovated living/lot area
- schools nearby, airport distance

### UX Assessment by User Type

| User | Fit | Assessment |
|---|---|---|
| Homeowner | Yellow/Red | Can enter basic home facts, but latitude, longitude, grade, views, and renovated-area fields are confusing. No confidence range or comparable homes. |
| Seller | Yellow/Red | Gets a point estimate but not pricing strategy, listing range, improvement levers, or market positioning. |
| Buyer | Red | No search/comparison workflow, affordability context, neighborhood interpretation, or over/under-valued signal. |
| Investor | Red | No yield, rent, appreciation, risk, comparable market slices, or portfolio comparison. |
| Non-technical user | Red | UI exposes raw ML/dataset fields and model diagnostics rather than user concepts. |
| Analyst/student reviewer | Green/Yellow | Analytics, correlations, feature importance, and SHAP are useful for demonstrating model behavior. |

### UI Exposes Dataset/ML Concepts

The following fields should be redesigned or moved behind advanced settings in V2:

- `Latitude` / `Longitude`
- `Grade (1-13)`
- `Views` as an unexplained 0-4 index
- `Condition (1-5)` without definitions
- `Waterfront` for a claimed Indian dataset without geographic proof
- `Renovation year (0 if none)`
- `Living area after renovation`
- `Lot area after renovation`
- `Postal code` when geography is unresolved
- `Floors` values such as 1.5 and 2.5
- `Distance from airport (km)` without source or address-based derivation

### Product Color Classification

| Area | Status | Reason |
|---|---|---|
| Basic prediction interaction | Yellow | Works technically, but uses raw features and point estimate only. |
| Analytics dashboard | Green/Yellow | Good prototype analytics; needs market framing and provenance. |
| Model performance display | Yellow | Shows metrics but lacks segmented evaluation and caveats. |
| About page | Yellow | Explains stack, but claims API-backed architecture that current Streamlit bypasses. |
| Nontechnical trust | Red | No provenance, uncertainty, comparable evidence, or explanation in plain language. |

## 8. Explainability Audit

### Current Feature Importance

Feature importance is pulled from `pipeline.named_steps["regressor"].feature_importances_` and labels are generated from the preprocessor output names. Since all current CSV model features are numeric, the transformed feature names should correspond one-to-one with the original 20 feature columns.

### Current SHAP

SHAP is calculated globally over a sample of up to 500 filtered dataset rows:

```text
filtered_df
-> feature_sample of model feature columns
-> pipeline preprocessor.transform()
-> shap.TreeExplainer(regressor)
-> mean absolute SHAP bar chart
-> SHAP impact scatter
```

Current SHAP explains the transformed model representation, not a specific user's submitted property. For this dataset, transformed and original features likely align because all features are numeric, but the code is written for preprocessed features generally.

### Explainability Gaps

| Gap | Severity | Rationale |
|---|---|---|
| No local explanation for the user's prediction | High | Users need to know why their estimate moved up/down. |
| Feature labels are raw dataset names | Medium | Names like `living_area_renov` and `Lattitude` are not user-friendly. |
| SHAP values are not translated into plain language | Medium | Nontechnical users will not understand SHAP magnitude/direction unaided. |
| No caveat about correlations vs causality | Medium | Real-estate users may over-interpret model explanations. |
| No geographic/provenance caveat near explanations | High | SHAP explanations can seem authoritative despite unresolved dataset identity. |

Safe natural-language translation later:

- Size-related explanations can be translated cautiously.
- Age/renovation explanations can be translated cautiously if feature definitions are documented.
- Location explanations should remain cautious until provenance is verified.
- `grade`, `views`, `waterfront`, schools, and airport distance need data dictionary definitions before being user-facing advice.

## 9. Product Audit

### Intended V2 Philosophy

The intended product direction is to help users make better residential real-estate decisions, not merely output a predicted price. Against that bar, V1 is best understood as an ML demonstration/prototype with useful analytics, not yet as a decision-support product.

### Product Classification

| Area | Status | Rationale |
|---|---|---|
| Product positioning | Yellow | The "housing intelligence" framing is promising, but current implementation still behaves like a price prediction demo. |
| Target users | Red | Personas are not encoded in navigation, workflows, copy, or outputs. |
| User journeys | Red | No Sell, Buy, Investor, Market Explorer, property comparison, or report journey. |
| Decision support | Red | V1 provides a point estimate and technical analytics, but no action-oriented guidance. |
| Prediction usefulness | Yellow | Point prediction can be useful as a prototype, but lacks range, caveats, comparables, and provenance. |
| Trust | Red | Dataset provenance, currency, geography, and model validity are not documented. |
| Limitations communication | Red | The app does not clearly warn users about data scope or suitable use. |
| Analytics | Yellow | Charts are implemented, but they are framed as dataset/model diagnostics rather than market insight. |
| Explainability | Yellow | SHAP and feature importance exist, but are technical and global rather than user-specific. |
| Professional portfolio signal | Yellow/Green | Strong enough to demonstrate engineering initiative, but the next step should be credibility and product discipline. |

### Product Bottom Line

V1 should retain its working prototype foundation, but V2 should shift the center of gravity from "enter model features and see a price" to "choose a real-estate decision and receive evidence, valuation context, and next-step guidance."

## 10. Engineering Audit

### Strengths

| Area | Status | Evidence |
|---|---|---|
| Clear small-repo layout | Green | Few files, easy to inspect. |
| Training script saves full pipeline | Green | Prevents inference/training preprocessing drift in principle. |
| FastAPI validation | Green | Pydantic bounds and renovation validator are implemented. |
| Streamlit caching | Green | Dataset/model/SHAP functions use Streamlit caching. |
| Artifacts committed | Yellow | Demo can run after dependencies install, but artifact provenance/versioning is limited. |

### Maintainability Problems

| Issue | Severity | Evidence |
|---|---|---|
| `streamlit_app.py` is monolithic | High | 1,227 lines containing styling, state, inputs, prediction, analytics, SHAP, About. |
| Inference boundary duplicated/bypassed | High | Streamlit and FastAPI both load artifacts independently. |
| Hardcoded data paths | Medium | Dataset/artifacts are fixed relative paths. |
| Hardcoded feature schema in multiple places | High | Defaults, payload builder, API schema, widgets, and metadata must all remain aligned. |
| No tests | High | No unit/integration/API/UI tests found. |
| No CI/CD | Medium | No workflows or deployment verification. |
| No data/model documentation | Critical | Dataset provenance and model limitations are not documented. |
| Partial local `.venv` | Low/Medium | `.venv` exists but contains only `Include`, not a working environment. |
| Runtime assumptions not pinned | Medium | Dependencies use lower bounds only; artifact compatibility may break across library versions. |
| Encoding/mojibake in docs/UI text | Medium | README/source contain corrupted characters for emoji, arrows, rupee sign, en dash. |

## 11. Complete V1 Feature Inventory

| Feature | Status | Evidence/file | Quality | V2 action |
|---|---|---|---|---|
| Streamlit frontend | Implemented | `streamlit_app.py` | Functional prototype | Keep, modularize later |
| Sidebar property form | Implemented | `streamlit_app.py:491-641` | Raw model inputs | Redesign UX |
| Local Streamlit prediction | Implemented | `streamlit_app.py:225-235` | Works if artifacts/deps exist | Replace with API-first or shared service |
| FastAPI inference API | Implemented | `app.py` | Good V1 service | Make actual inference boundary |
| API validation | Implemented | `app.py:84-126` | Useful but raw schema | Map user schema to model schema |
| Health check | Implemented | `app.py:256-268` | Basic | Keep and expand |
| Model training script | Implemented | `train.py` | Reproducible single split | Add validation strategy, model card |
| XGBoost model | Implemented | `train.py:100-118`, artifacts | Good prototype | Retain until data strategy clarified |
| Saved model artifact | Implemented | `artifacts/housing_model.joblib` | Present | Version and document |
| Saved preprocessor artifact | Implemented | `artifacts/preprocessor.joblib` | Present | Version and document |
| Saved metadata artifact | Implemented | `artifacts/training_metadata.joblib` | Useful | Expand metadata |
| Dataset overview | Implemented | `streamlit_app.py:716-768` | Useful for analysts | Move toward market overview |
| Interactive filters | Implemented | `streamlit_app.py:320-377` | Basic | Expand with user-oriented filters |
| Price analytics | Implemented | `streamlit_app.py:770-820` | Good prototype | Add market language |
| Correlation heatmap | Implemented | `streamlit_app.py:822-886` | Technical | Keep under advanced analytics |
| Feature importance | Implemented | `streamlit_app.py:889-923` | Technical | Translate for users |
| Model performance dashboard | Implemented | `streamlit_app.py:925-954` | Basic | Add segmentation and caveats |
| SHAP global explanation | Implemented | `streamlit_app.py:956-1038` | Technical | Add local explanations |
| CSV export | Implemented | `streamlit_app.py:1041-1051` | Useful | Keep if privacy/source allowed |
| About page | Implemented | `streamlit_app.py:1117-1141` | Basic | Correct architecture/provenance |
| Screenshots | Implemented | `screenshots/*.png` | Good README support | Refresh after V2 |
| Product landing page | Missing | none | N/A | Add in V2 if needed |
| Sell workflow | Missing | none | N/A | Add |
| Buy workflow | Missing | none | N/A | Add |
| Investor workflow | Missing | none | N/A | Add |
| Property report | Missing | none | N/A | Add |
| Property comparison | Missing | none | N/A | Add |
| Valuation range | Missing | none | N/A | Add |
| Uncertainty estimation | Missing | none | N/A | Add |
| Tests | Missing | none | N/A | Add |

## 12. V1 Limitations

### Critical

- Dataset provenance is **UNVERIFIED**.
- Indian geography/currency claims are not supported by repository evidence.
- Coordinates conflict with India.
- Streamlit bypasses FastAPI, so the documented architecture is inaccurate.
- No tests protect model input schema, API behavior, or Streamlit prediction behavior.

### High

- UI requires raw ML/dataset inputs unsuitable for normal homeowners/buyers/sellers.
- Single random train/test split may overstate real-world performance.
- No spatial or temporal validation.
- No local explanation for an individual prediction.
- No valuation range, uncertainty, or confidence communication.
- Hardcoded duplicate schemas across training, API, and Streamlit.

### Medium

- Monolithic Streamlit file.
- Lower-bound dependency versions only.
- No CI/deployment config.
- No model card, dataset card, or data dictionary.
- No segmented performance by geography, price band, or property type.
- Mojibake/encoding issues in README/source text.
- Outliers and suspicious features are not flagged or handled.

### Low

- `.agents` directory is empty.
- `.venv` appears incomplete in the audited workspace.
- Screenshots are present but not tied to release/version docs.

## 13. V2 Requirements Traceability

| V2 direction | Status | Evidence / note |
|---|---|---|
| Product landing page | Missing | Current first screen is app tabs, not a product landing page. |
| User personas | Missing | No explicit persona modeling. |
| Sell Property workflow | Missing | No seller journey. |
| Buy Property workflow | Missing | No buyer journey. |
| Investor workflow | Missing | No investor journey. |
| Market Explorer | Partial | Analytics dashboard exists but is dataset/model-centric. |
| Simplified property inputs | Missing | Inputs expose all raw model features. |
| Advanced settings | Missing | No basic/advanced separation. |
| Better prediction results | Partial | Point estimate exists. No range, comparables, or caveats. |
| Valuation range | Missing | Only point prediction. |
| Human-readable explanations | Missing | Feature importance/SHAP are technical charts. |
| Improved SHAP | Partial | Global SHAP exists; local/user prediction SHAP missing. |
| Improved analytics | Partial | Charts exist; market insights not yet productized. |
| Market insights | Partial | Price distribution/correlation exist, but no interpreted insight layer. |
| Property report | Missing | No generated report. |
| Property comparison | Missing | No comparison workflow. |
| Uncertainty estimation | Missing | No interval/calibration. |
| Testing | Missing | No tests found. |
| Documentation | Partial | README exists; no docs hierarchy. |
| Modular architecture | Missing | Streamlit is monolithic; schemas duplicated. |
| API-first inference | Partial | API exists but Streamlit bypasses it. |
| Dataset provenance | Missing | No source/license/card. |
| Professional UX | Partial | Visual polish exists, but UX is not decision-oriented. |

## 14. Documentation Structure

Recommended structure:

```text
docs/
|-- README.md
|-- product/
|   |-- vision.md
|   |-- personas.md
|   |-- user-journeys.md
|   |-- v2-requirements.md
|   `-- ux-principles.md
|-- architecture/
|   |-- system-overview.md
|   |-- v1-architecture.md
|   |-- v2-target-architecture.md
|   |-- api-first-inference.md
|   `-- deployment.md
|-- ml/
|   |-- dataset-card.md
|   |-- data-dictionary.md
|   |-- model-card.md
|   |-- training-procedure.md
|   |-- evaluation.md
|   |-- explainability.md
|   `-- limitations.md
|-- research/
|   |-- real-estate-valuation-background.md
|   |-- comparable-sales-methods.md
|   |-- uncertainty-estimation.md
|   `-- market-data-sources.md
|-- decisions/
|   |-- adr-0001-api-first-inference.md
|   |-- adr-0002-dataset-provenance-policy.md
|   |-- adr-0003-v2-persona-workflows.md
|   `-- adr-0004-explainability-strategy.md
|-- releases/
|   |-- v1-audit.md
|   |-- v1-known-limitations.md
|   `-- v2-roadmap.md
`-- audits/
    `-- v1-engineering-audit-2026-08-09.md
```

Recommended content:

| File | Purpose |
|---|---|
| `docs/product/vision.md` | Product statement: who it serves and what decisions it improves. |
| `docs/product/personas.md` | Homeowner, seller, buyer, investor, analyst personas. |
| `docs/product/user-journeys.md` | Sell, buy, invest, explore-market flows. |
| `docs/product/v2-requirements.md` | Prioritized V2 requirements and acceptance criteria. |
| `docs/architecture/system-overview.md` | Current system diagram and runtime responsibilities. |
| `docs/architecture/api-first-inference.md` | Contract for frontend/API/model boundaries. |
| `docs/ml/dataset-card.md` | Source, license, geography, time range, units, limitations. |
| `docs/ml/data-dictionary.md` | Every column definition and allowed ranges. |
| `docs/ml/model-card.md` | Intended use, metrics, evaluation, caveats, ethical risks. |
| `docs/ml/evaluation.md` | Validation methodology, segmented metrics, residual analysis. |
| `docs/ml/explainability.md` | Feature importance and SHAP interpretation strategy. |
| `docs/decisions/*.md` | Architecture decision records for major V2 choices. |
| `docs/releases/v2-roadmap.md` | Implementation order, milestones, release criteria. |

## 15. Recommended V2 Implementation Order

1. Freeze V1 facts in documentation.
   - Keep this audit as the baseline.
   - Add dataset card, data dictionary, and model card before product claims expand.

2. Resolve dataset provenance.
   - Verify source, license, geography, currency, and feature definitions.
   - Decide whether V2 is Indian real estate, synthetic/demo data, or another market.

3. Correct the inference boundary.
   - Make Streamlit call FastAPI or extract a shared inference service used by both.
   - Add contract tests around schema, preprocessing, and prediction response.

4. Simplify the prediction UX.
   - Create basic property inputs normal users can answer.
   - Move raw model fields into advanced settings or derive them from user-friendly inputs.

5. Add valuation range and caveats.
   - Use validation residuals or model ensemble/quantile strategy later.
   - Display point estimate as one part of a decision aid, not the whole product.

6. Build persona workflows.
   - Seller: listing range, pricing confidence, improvement levers.
   - Buyer: over/under valuation, affordability/context, comparison.
   - Investor: risk, opportunity, comparable segments.

7. Improve explainability.
   - Add local explanation for current submitted property.
   - Translate safe features into plain language.
   - Keep technical SHAP under advanced analytics.

8. Upgrade evaluation.
   - Add cross-validation, spatial/temporal validation, price-band metrics, residual plots.
   - Flag known outliers and invalid input regions.

9. Modularize.
   - Split Streamlit views, schema/config, inference client, analytics, explainability, and formatting.
   - Avoid broad refactors until tests exist.

10. Add deployment and CI.
    - Pin dependency versions or define compatible ranges carefully.
    - Add automated tests and deployment health checks.

## 16. Risks and Unresolved Questions

| Question / risk | Status |
|---|---|
| What is the original dataset source? | UNVERIFIED |
| Is the dataset actually Indian housing data? | UNVERIFIED; current coordinates strongly conflict with India. |
| What currency is `Price`? | UNVERIFIED |
| What are the units for airport distance? | Labeled km in Streamlit, but source is UNVERIFIED. |
| What does `grade of the house` mean? | UNVERIFIED |
| What does `number of views` mean? | UNVERIFIED |
| Why is `Lattitude` misspelled? | Existing column name; origin unknown. |
| Are postal codes real, transformed, or synthetic? | UNVERIFIED |
| Are latitude/longitude transformed, synthetic, or from another region? | UNVERIFIED |
| Can saved artifacts be loaded in a clean install today? | UNVERIFIED in this audit runtime; dependencies were not installed. |
| Does the Streamlit Cloud demo currently work? | UNVERIFIED; not checked in this repository-only audit. |
| Are README screenshots current relative to code? | UNVERIFIED visually; files exist. |

## Bottom Line

V1 should be presented as a functioning ML prototype with interactive analytics, not yet as a trusted residential real-estate intelligence product. The strongest V2 move is not more features first; it is to establish data credibility, fix the architecture truth gap, and reshape the experience around real user decisions.
