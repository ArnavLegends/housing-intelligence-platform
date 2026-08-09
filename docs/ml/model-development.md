# Model Development

## V1 Model Purpose

The V1 model predicts the `Price` target from housing feature columns in `House Price India.csv`. It is implemented as an XGBoost regression pipeline trained by `train.py`.

## Dataset Used by Training

Training uses:

```text
House Price India.csv
```

The dataset contains:

- 14,620 rows.
- 23 columns.
- `Price` as the target column.
- `id` and `Date` as dropped columns.
- 20 feature columns used for modeling.

Dataset provenance, geography, and currency are UNVERIFIED in the repository.

## Feature Columns

The saved metadata lists these model features:

- `number of bedrooms`
- `number of bathrooms`
- `living area`
- `lot area`
- `number of floors`
- `waterfront present`
- `number of views`
- `condition of the house`
- `grade of the house`
- `Area of the house(excluding basement)`
- `Area of the basement`
- `Built Year`
- `Renovation Year`
- `Postal Code`
- `Lattitude`
- `Longitude`
- `living_area_renov`
- `lot_area_renov`
- `Number of schools nearby`
- `Distance from the airport`

## Preprocessing

`train.py` builds a `ColumnTransformer`.

The preprocessing logic is:

- categorical, object, and boolean columns are one-hot encoded with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- numeric columns are passed through.
- remainder columns are dropped.

The audited CSV columns are numeric, so V1 effectively passes numeric features through the preprocessor.

## Model Algorithm

V1 uses `XGBRegressor` with:

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

## Train/Test Methodology

`train.py` uses:

```text
train_test_split(test_size=0.2, random_state=42)
```

With 14,620 rows, this corresponds to an inferred split of:

- 11,696 training rows.
- 2,924 test rows.

No cross-validation, temporal validation, spatial validation, or segmented validation is implemented in V1.

## Saved Artifacts

Training saves:

| Artifact | Purpose |
|---|---|
| `artifacts/housing_model.joblib` | Full preprocessing + XGBoost pipeline. |
| `artifacts/preprocessor.joblib` | Preprocessing step only. |
| `artifacts/training_metadata.joblib` | Target column, dropped columns, feature columns, and metrics. |

## V1 Model Development Limitations

- Dataset provenance is UNVERIFIED.
- Currency and geography assumptions are UNVERIFIED.
- Location fields may limit generalization.
- A single random split may overstate real-world performance for spatial housing data.
- No uncertainty estimation is implemented.
- No model card existed before this documentation baseline.
- No automated tests were found for training, preprocessing, or inference schema compatibility.
