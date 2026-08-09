# Model Evaluation

## V1 Evaluation Method

V1 evaluation is performed in `train.py` after fitting the XGBoost pipeline. The script uses an 80/20 random train/test split with `random_state=42`.

The evaluation metrics are computed on the held-out test set:

- R2 score.
- Mean absolute error.
- Root mean squared error.

No retraining or metric recomputation was performed while creating this documentation baseline.

## Reported Metrics

The saved metadata in `artifacts/training_metadata.joblib` reports:

| Metric | Value |
|---|---:|
| R2 | 0.8933532238006592 |
| MAE | 63,533.23046875 |
| RMSE | 122,590.98094068747 |

The README reports rounded versions of the same values:

| Metric | README value |
|---|---:|
| R2 | 0.893 |
| MAE | 63,533 |
| RMSE | 122,590 |

## Verification Status

The metrics were verified from `artifacts/training_metadata.joblib`.

They were not independently recomputed from `artifacts/housing_model.joblib` during the original audit because the audit runtime did not include `joblib`, `sklearn`, or `xgboost`.

## What V1 Evaluation Covers

V1 covers one basic regression evaluation on one random test split.

It verifies that the trained model can produce a numeric performance summary under the training script's chosen split.

## What V1 Evaluation Does Not Cover

V1 does not include:

- cross-validation
- temporal holdout validation
- geographic holdout validation
- error analysis by postal code
- error analysis by price band
- error analysis by property size
- error analysis by renovation status
- calibration checks
- uncertainty intervals
- residual plots
- production monitoring
- automated regression tests

## Evaluation Limitations

The evaluation result should be treated as prototype-level evidence. For a real residential real-estate product, a single random split is not enough to establish trust because housing prices are location-sensitive and time-sensitive.

The repository does not verify whether the dataset is genuinely Indian, what currency `Price` represents, or whether the location fields are real, transformed, or synthetic. Those unresolved questions limit how confidently the V1 metrics can be interpreted.
