# Product Vision

## V1 Purpose

Housing Intelligence Platform V1 attempts to provide an interactive housing price prediction and analytics prototype. It combines a Streamlit frontend, a saved XGBoost regression model, a FastAPI inference service, dataset analytics, model performance display, feature importance, and SHAP-style explainability.

V1 is best described as a functioning ML prototype for housing price estimation and model exploration. It is not yet a complete residential real-estate decision-support product.

## Intended Users in V1

V1 is most useful for technical reviewers, analysts, and portfolio evaluators who want to see that the project contains a real dataset, a training workflow, saved model artifacts, an API surface, and an interactive frontend.

For normal homeowners, sellers, buyers, and investors, V1 is limited. The interface asks for raw model and dataset features such as latitude, longitude, grade, view count, renovated area, postal code, and airport distance. Those inputs are not yet shaped around real user decisions.

## What V1 Provides

- A property input form in Streamlit.
- A point estimate for predicted house price.
- A property details summary of submitted inputs.
- Dataset overview and missing-value summary.
- Price distribution and price spread charts.
- Numeric correlation analysis.
- XGBoost feature importance.
- Global SHAP explainability on filtered dataset samples.
- Model performance metrics from saved training metadata.
- CSV export for filtered dataset slices.
- A FastAPI service with health and prediction endpoints.

## What V1 Does Not Yet Provide

- Verified dataset provenance.
- Verified geography and currency assumptions.
- Buyer, seller, homeowner, or investor workflows.
- Valuation ranges or uncertainty estimates.
- Comparable properties.
- Property reports.
- Property comparison.
- Human-readable local explanations for a user's prediction.
- API-first frontend inference.
- Tests or CI.

## Product Position

V1 should be presented as a working prototype and documentation baseline. Its value is that it proves the project can connect data, training, model artifacts, backend inference, and frontend analytics.

V1 should not be presented as a trusted production real-estate valuation system. The dataset provenance is UNVERIFIED, and the current UI is centered on model features rather than user decision workflows.
