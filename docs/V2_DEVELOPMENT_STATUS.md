
Housing Intelligence Platform — V2 Development Status
Status: V2 implementation in progress
Active branch: v2-product-engineering
Baseline: V1 remains on main

Current state
The repository originally shipped as a working V1 housing-price ML prototype with:

XGBoost regression

Streamlit frontend

FastAPI inference API

dataset analytics

feature importance

SHAP explainability

saved model artifacts

Streamlit Cloud deployment

V1 is documented as a technical prototype rather than a complete residential property decision-support product.

V2 direction
V2 is transforming the project into an AI-powered Residential Property Intelligence Platform.

The product should help a user:

understand a property,

estimate its value,

understand important value drivers,

inspect supporting market/model information,

decide what to do next.

The primary user workflow for V2 is Sell Property / Property Valuation.

Buyer and Investor workflows remain future functionality and must not be represented as implemented until they actually exist.

Completed V2 foundation
The V2 branch has already established a shared inference layer:

ml/inference.py
Both Streamlit and FastAPI use the same inference implementation.

Deployment decision:

Streamlit must remain independently deployable on Streamlit Cloud.

Streamlit must not require a separately hosted FastAPI service.

FastAPI remains an API/engineering surface and reuses the shared inference implementation.

V2 roadmap
Phase 1 - Product/UI Foundation
Phase 1 is in progress. Phase 1A - Application Shell & Home - is complete.

Scope:

product-oriented landing page

clear primary navigation

Sell Property as the main working workflow

Buy Property placeholder

Invest placeholder

Explore Market placeholder / limited experience

improved visual hierarchy

reduced sidebar clutter

user-facing terminology

preservation of existing prediction and analytics functionality

Phase 2 - Property Valuation UX
user-friendly property inputs

property age instead of requiring raw year where appropriate

understandable condition and construction-quality terminology

location-oriented input design

advanced/model-specific inputs separated from the normal workflow

Phase 3 - Prediction Results
Estimated Property Value

valuation range when methodologically justified

property summary

key value drivers

human-readable explanation

advanced technical explanation

appropriate limitations/disclaimer

Phase 4 - Explainability
retain SHAP and feature importance for technical users

add local explanation for the submitted property

translate model behavior into understandable real-estate language

clearly distinguish model explanation from causal claims

Phase 5 - Analytics / Market Intelligence
Redesign current analytics from raw data-science dashboards into decision-support experiences.

Potential sections:

Market Overview

Property Insights

Price Drivers

Market Relationships

Advanced Analytics

Do not remove useful technical analytics. Reorganize and explain them.

Phase 6 - About / Product Documentation
Create a professional About experience covering:

product purpose

users

architecture

ML pipeline

explainability

analytics

technology

limitations

roadmap

Phase 7 - Engineering Quality
Add:

focused unit tests

API tests

prediction regression tests

validation

error handling

modularization

configuration cleanup

Streamlit Cloud compatibility verification

Phase 8 - Advanced V2/V2.5 Features
Potential future features:

property report

valuation interval/uncertainty

property comparison

Market Explorer

renovation impact estimation

Do not implement these before the core V2 experience is stable.

V3 - outside immediate scope
Keep these outside the immediate V2 scope:

buyer workflow

investor workflow

geographically richer datasets

multiple datasets

multi-market support

stronger geographic/temporal validation

model comparison/research

advanced market intelligence

Current Priority
The current implementation priority is:

V2 Product/UI Foundation
Phase 1A has established the application shell and Home experience.

The next implementation milestone is:

V2 Phase 1B - Sell Property / Valuation UX
Coding/Development Rules
For every future task:

implement one focused milestone at a time

inspect only the files necessary for that milestone

do not perform broad repository audits unless specifically requested

do not change unrelated functionality

preserve Streamlit Cloud compatibility

test changes before claiming success

commit meaningful milestones separately

document important architectural decisions

Git Checkpoints
Current branch:

v2-product-engineering
Completed checkpoints include:

V1 documentation baseline

V2 shared inference refactor

V2 implementation roadmap

V2 development status/checkpoint documentation

V2 Phase 1A application shell

Future milestones should be committed separately.

Definition of V2 Done
V2 should ultimately provide:

professional product-style UI

clear property valuation workflow

simplified user inputs

working ML prediction

human-readable AI explanation

technical explainability

useful analytics

robust error handling

basic automated testing

clean architecture

comprehensive GitHub documentation

public Streamlit deployment

clear V1 to V2 development history

Do Not Forget
The goal is not:

Add as many features as possible.
The goal is:

Turn the existing working ML project into a coherent, useful, maintainable residential property intelligence product.
Every feature must be evaluated for:

user value

engineering value

ML value

portfolio value

credibility

Phase 1A - Product/UI Foundation
Status: Completed

Phase 1A established the V2 product application shell without changing the underlying ML, dataset, inference, API, or analytics implementation.

Implemented
Home is now the default experience.

Persistent navigation:

Home

Valuation

Market Intelligence

Model Insights

About

Product-oriented Home page with:

"What are you here to do?"

Sell Property

Buy Property

Invest

Explore Market

Sell Property is the primary functional workflow.

Start Valuation routes to the existing working prediction experience.

Buy Property, Invest, and Explore Market are explicitly presented as Coming Soon.

Existing valuation inputs and prediction behaviour were preserved.

Existing analytics, SHAP, feature importance, correlations, and model performance remain accessible through Model Insights.

About content was minimally adapted for the V2 application shell and shared local inference direction.

Intentionally unchanged
XGBoost model

training pipeline

dataset

saved artifacts

shared inference implementation

FastAPI API

SHAP calculations

feature-importance calculations

analytics calculations

dependencies

external services

Validation
py_compile streamlit_app.py passed.

Structural navigation marker check passed.

git diff --check passed.

Full Streamlit runtime testing was not performed because Streamlit was not installed in the execution environment.

Files changed
streamlit_app.py
docs/V2_DEVELOPMENT_STATUS.md
docs/releases/v2-phase-1a.md
Commit
Phase 1A was implemented as a separate milestone commit:

bf85ba702594f228d813141d10fe7e9b07a197f7
Commit message:

feat: establish V2 product application shell
The Phase 1A commit is currently being integrated with the remote V2 documentation history before the branch is pushed.

Current state -> Next Action
Current state:

V1 is documented.

V2 shared inference foundation is complete.

Streamlit and FastAPI share ml/inference.py.

Streamlit must remain directly deployable on Streamlit Cloud.

The current dataset remains in use for V2.

Phase 1A application shell is complete.

Home is now the default product entry point.

Sell Property is the primary working user journey.

Buy Property, Invest, and Explore Market remain non-functional future workflows.

Next action:

Start Phase 1B - Sell Property / Valuation UX.

Phase 1B should redesign the existing property-entry experience around user-facing residential property concepts while preserving the underlying model feature schema and working inference path.

Phase 1B should not introduce:

model changes

dataset changes

valuation intervals

buyer functionality

investor functionality

Market Explorer functionality

LLM features

remote backend infrastructure

See docs/V2_IMPLEMENTATION_PLAN.md for the complete roadmap.