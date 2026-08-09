# Explainability

## V1 Explainability Features

V1 implements two model explainability views in Streamlit:

- XGBoost feature importance.
- Global SHAP explainability on filtered dataset samples.

Both are available under the Streamlit Analytics section.

## Feature Importance

Feature importance is read from:

```text
pipeline.named_steps["regressor"].feature_importances_
```

The Streamlit app gets transformed feature names from the pipeline preprocessor. Because the audited CSV uses numeric columns, the transformed feature names should correspond to the original model feature columns.

Feature importance is displayed as a top-10 horizontal bar chart.

## SHAP

V1 computes SHAP values in Streamlit.

The implemented flow is:

```text
filtered dataset
-> sample up to 500 records
-> select model feature columns
-> apply pipeline preprocessor
-> create shap.TreeExplainer(regressor)
-> compute SHAP values
-> display global mean absolute SHAP importance
-> display SHAP impact scatter chart
```

## What SHAP Explains in V1

The current SHAP implementation explains model behavior over a sample of filtered dataset records.

It does not explain the specific prediction submitted by the user in the sidebar. The explanation is global for the selected dataset slice, not local for the current property.

## User Understandability

V1 explanations are technical. They expose raw dataset and model feature names such as:

- `Lattitude`
- `Longitude`
- `grade of the house`
- `living_area_renov`
- `lot_area_renov`
- `number of views`

These names are useful for model inspection, but they are not yet translated into normal real-estate language.

## Current Limitations

- No local SHAP explanation for the user's predicted property.
- No natural-language explanation of why the predicted price is high or low.
- No explanation caveats near geography or dataset provenance.
- No plain-language definitions for dataset-specific fields.
- No distinction between correlation, feature importance, and causal interpretation.

## Safe Interpretation

V1 SHAP and feature importance should be treated as model diagnostics. They should not be treated as proof of causal drivers of home value.

Location-related explanation is especially limited because dataset provenance, geography, and currency are UNVERIFIED.
