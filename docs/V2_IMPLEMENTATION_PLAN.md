# Housing Intelligence Platform - V2 Implementation Plan

## 1. Project Objective

Housing Intelligence Platform is a flagship AI engineering project for USA MS applications. The project should demonstrate strong product judgment, machine learning engineering, software architecture, explainability, and deployment discipline.

The final product should feel like a real residential property intelligence product, not a student ML demo. It should help a user understand a property, estimate value, inspect drivers, and decide what to do next.

The final application should ultimately be deployed publicly through Streamlit Cloud so admissions committees, recruiters, and users can test it directly.

## 2. Current V1

V1 already implements a working housing price prediction prototype:

- XGBoost house-price prediction.
- Streamlit frontend.
- FastAPI API.
- Dataset analytics.
- Feature importance.
- SHAP explainability.
- Model performance display.
- Saved model artifacts and training metadata.

V1 model metrics are documented in the existing V1 audit and release baseline. The saved training metadata reports:

| Metric | Value |
|---|---:|
| R2 | 0.8933532238006592 |
| MAE | 63,533.23046875 |
| RMSE | 122,590.98094068747 |

V1 limitations are documented in:

- `docs/audits/v1-engineering-audit-2026-08-09.md`
- `docs/product/vision.md`
- `docs/architecture/architecture-v1.md`
- `docs/ml/model-development.md`
- `docs/ml/evaluation.md`
- `docs/ml/explainability.md`
- `docs/releases/v1.0.0.md`

Key V1 limitations include raw model-centric inputs, limited product workflows, no valuation range, no local human-readable explanation, no tests, and unresolved dataset provenance.

## 3. Completed V2 Work

### V2 Git Branch

Current V2 branch:

```text
v2-product-engineering
```

### V2 Inference Foundation

The shared inference implementation now lives in:

```text
ml/inference.py
```

Both Streamlit and FastAPI use the same inference service.

Current intended inference flow:

```text
Streamlit
-> shared inference service
-> artifacts/housing_model.joblib
-> prediction
```

```text
FastAPI
-> shared inference service
-> artifacts/housing_model.joblib
-> prediction
```

Important deployment decision:

Streamlit must not depend on a separately hosted FastAPI server. The final application should remain independently deployable on Streamlit Cloud.

FastAPI remains part of the engineering/API layer and should reuse the same inference implementation.

## 4. Important Project Decisions

### Dataset

Keep the current dataset for V2.

Do not spend V2 effort correcting or replacing the current dataset.

Dataset replacement, stronger provenance work, geographically richer data, and multi-market expansion can be considered in a later version.

### Deployment

Final production/demo deployment target:

```text
Streamlit Cloud
```

Do not introduce unnecessary remote backend infrastructure.

### Product Philosophy

The product should answer:

```text
What decision can the user make after seeing this?
```

Avoid adding features merely because they demonstrate technology.

### Engineering Philosophy

Prefer:

- modularity
- maintainability
- reproducibility
- tests
- explainability
- clear architecture
- incremental versioning

Avoid:

- unnecessary microservices
- unnecessary databases
- unnecessary AI/LLM features
- technology for technology's sake

## 5. V2 Main Roadmap

### Phase 1 - Product/UI Foundation

- Redesign landing/home experience.
- Establish clear navigation.
- Introduce product-first positioning.
- Make Sell Property the primary working workflow.
- Add Buy Property as a future/coming-soon workflow.
- Add Invest as a future/coming-soon workflow.
- Add Explore Market as a future/coming-soon workflow.
- Improve visual hierarchy.
- Reduce sidebar clutter.
- Improve terminology.
- Preserve current working functionality.

### Phase 2 - Property Valuation UX

Create a user-friendly property form.

Prefer user concepts such as:

- bedrooms
- bathrooms
- living area
- lot area
- property age
- property condition
- construction quality
- renovation
- location

Dataset/model-specific fields should not dominate the normal user experience.

Advanced model-specific fields can exist under an Advanced section where necessary.

### Phase 3 - Prediction Results

Replace the simple prediction output with:

- Estimated Property Value
- valuation range when methodologically justified
- property summary
- key value drivers
- human-readable explanation
- model/AI explanation
- appropriate disclaimer

Use consistent Indian number formatting where the current product uses INR.

### Phase 4 - Explainability

Keep technical explainability:

- SHAP
- feature importance

Also provide a user-facing translation:

```text
Why is this property valued this way?
```

Distinguish between:

- normal user explanation
- advanced technical explanation

### Phase 5 - Analytics / Market Intelligence

Redesign current analytics from raw data-science dashboards into decision-support experiences.

Potential sections:

- Market Overview
- Property Insights
- Price Drivers
- Market Relationships
- Advanced Analytics

Do not remove useful technical analytics. Reorganize and explain them.

### Phase 6 - About / Product Documentation

Create a professional About experience covering:

- product purpose
- users
- architecture
- ML pipeline
- explainability
- analytics
- technology
- limitations
- roadmap

### Phase 7 - Engineering Quality

Add:

- focused unit tests
- API tests
- prediction regression tests
- validation
- error handling
- modularization
- configuration cleanup
- Streamlit Cloud compatibility verification

### Phase 8 - Advanced V2/V2.5 Features

Potential future features:

- property report
- valuation interval/uncertainty
- property comparison
- Market Explorer
- renovation impact estimation

Do not implement these before the core V2 experience is stable.

## 6. Future V3

Keep these outside the immediate V2 scope:

- buyer workflow
- investor workflow
- geographically richer datasets
- multiple datasets
- multi-market support
- stronger geographic/temporal validation
- model comparison/research
- advanced market intelligence

## 7. Current Priority

The next implementation task is:

```text
V2 Product/UI Foundation
```

Specifically:

1. landing/home experience
2. product navigation
3. Sell Property primary workflow
4. Buy/Invest/Explore placeholders
5. improved page hierarchy
6. cleaner terminology
7. better visual structure

Do not start with model improvements.

Do not start with dataset changes.

Do not start with remote FastAPI deployment.

## 8. Coding/Development Rules

For every future task:

- implement one focused milestone at a time
- inspect only the files necessary for that milestone
- do not perform broad repository audits unless specifically requested
- do not change unrelated functionality
- preserve Streamlit Cloud compatibility
- test changes before claiming success
- commit meaningful milestones separately
- document important architectural decisions

## 9. Git Checkpoints

Current branch:

```text
v2-product-engineering
```

Already completed commits:

- V1 documentation baseline
- V2 shared inference refactor

Future milestones should be committed separately.

## 10. Definition of V2 Done

V2 should ultimately provide:

- professional product-style UI
- clear property valuation workflow
- simplified user inputs
- working ML prediction
- human-readable AI explanation
- technical explainability
- useful analytics
- robust error handling
- basic automated testing
- clean architecture
- comprehensive GitHub documentation
- public Streamlit deployment
- clear V1 to V2 development history

## 11. Do Not Forget

The goal is not:

```text
Add as many features as possible.
```

The goal is:

```text
Turn the existing working ML project into a coherent, useful, maintainable residential property intelligence product.
```

Every feature must be evaluated for:

- user value
- engineering value
- ML value
- portfolio value
- credibility

## Current State -> Next Action

Current state:

- V1 is documented.
- V2 shared inference foundation is complete.
- Streamlit and FastAPI share `ml/inference.py`.
- Streamlit must remain directly deployable on Streamlit Cloud.
- The current dataset remains in use for V2.

Next action:

Start Phase 1: Product/UI Foundation. Build the landing/home experience, product navigation, Sell Property primary workflow, Buy/Invest/Explore placeholders, improved hierarchy, cleaner terminology, and better visual structure while preserving current working prediction and analytics functionality.
