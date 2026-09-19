# Housing Intelligence Platform — V2 Development Status

**Status:** V2 implementation in progress  
**Active branch:** `v2-product-engineering`  
**Baseline:** V1 remains on `main`

## Current state

The repository originally shipped as a working V1 housing-price ML prototype with:

- XGBoost regression
- Streamlit frontend
- FastAPI inference API
- dataset analytics
- feature importance
- SHAP explainability
- saved model artifacts
- Streamlit Cloud deployment

V1 is documented as a technical prototype rather than a complete residential property decision-support product.

## V2 direction

V2 is transforming the project into an **AI-powered Residential Property Intelligence Platform**.

The product should help a user:

1. understand a property,
2. estimate its value,
3. understand important value drivers,
4. inspect supporting market/model information,
5. decide what to do next.

The primary user workflow for V2 is **Sell Property / Property Valuation**.

Buyer and Investor workflows remain future functionality and must not be represented as implemented until they actually exist.

## Completed V2 foundation

The V2 branch has already established a shared inference layer:

```text
ml/inference.py
```

Both Streamlit and FastAPI use the same inference implementation.

Deployment decision:

- Streamlit must remain independently deployable on Streamlit Cloud.
- Streamlit must not require a separately hosted FastAPI service.
- FastAPI remains an API/engineering surface and reuses the shared inference implementation.

## V2 roadmap

### Phase 1 — Product/UI Foundation

Current implementation phase.

Scope:

- product-oriented landing page
- clear primary navigation
- Sell Property as the main working workflow
- Buy Property placeholder
- Invest placeholder
- Explore Market placeholder / limited experience
- improved visual hierarchy
- reduced sidebar clutter
- user-facing terminology
- preservation of existing prediction and analytics functionality

### Phase 2 — Property Valuation UX

- user-friendly property inputs
- property age instead of requiring raw year where appropriate
- understandable condition and construction-quality terminology
- location-oriented input design
- advanced/model-specific inputs separated from the normal workflow

### Phase 3 — Prediction Results

- Estimated Property Value
- valuation range only when a defensible methodology exists
- property summary
- key value drivers
- human-readable explanation
- advanced technical explanation
- appropriate limitations/disclaimer

### Phase 4 — Explainability

- retain SHAP and feature importance for technical users
- add local explanation for the submitted property
- translate model behavior into understandable real-estate language
- clearly distinguish model explanation from causal claims

### Phase 5 — Market Intelligence

Reframe current analytics into decision-support experiences:

- Market Overview
- Property Insights
- Price Drivers
- Market Relationships
- Advanced Analytics

### Phase 6 — Product/About Documentation

Document:

- purpose
- users
- architecture
- ML pipeline
- explainability
- analytics
- technology
- limitations
- roadmap

### Phase 7 — Engineering Quality

- focused unit tests
- API tests
- prediction regression tests
- validation/error handling
- modularization
- configuration cleanup
- Streamlit Cloud compatibility verification

### Phase 8 — V2/V2.5 Extensions

Potential later work:

- property report
- defensible valuation intervals
- property comparison
- Market Explorer
- renovation impact estimation

## V3 — outside immediate scope

- full buyer workflow
- full investor workflow
- richer/multiple datasets
- multi-market support
- stronger geographic and temporal validation
- model comparison/research
- advanced market intelligence

## Engineering rules

Every meaningful milestone follows:

```text
Plan
→ Implement
→ Test
→ Review
→ Document
→ Commit
→ Push
→ Continue
```

Rules:

- keep milestones bounded
- preserve working functionality
- avoid unrelated refactors
- do not add technology for demonstration value alone
- do not claim an implementation before it exists and is tested
- document important architectural/product decisions
- keep Git history understandable

## Current next milestone

**V2 Phase 1 — Product/UI Foundation**

The implementation should begin with the home experience and navigation, then introduce the Sell Property workflow while preserving the existing ML functionality.

See `docs/V2_IMPLEMENTATION_PLAN.md` for the complete roadmap.
