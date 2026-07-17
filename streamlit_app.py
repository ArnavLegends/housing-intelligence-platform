"""
Housing Intelligence Platform — Streamlit frontend.

Run locally:
    streamlit run streamlit_app.py

Set API_URL for deployment (default: http://127.0.0.1:8000).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

APP_TITLE = "Housing Intelligence Platform"
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "House Price India.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "housing_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "training_metadata.joblib"
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_ENDPOINT = f"{DEFAULT_API_URL}/predict"
TARGET_COLUMN = "Price"
DROP_COLUMNS = {"id", "Date"}
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

DEFAULT_PROPERTY: dict[str, float | int] = {
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


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1240px;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        div[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        div[data-testid="stSidebar"] .stButton > button {
            background: #2563eb;
            color: white !important;
            border: none;
            font-weight: 600;
            width: 100%;
        }
        div[data-testid="stSidebar"] .stButton > button:hover {
            background: #1d4ed8;
            color: white !important;
        }
        .hero-title {
            font-size: 2.25rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .section-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        }
        .section-card h3 {
            margin-top: 0;
            color: #0f172a;
        }
        .kpi-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-top: 3px solid #2563eb;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            min-height: 96px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }
        .kpi-label {
            color: #475569;
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .kpi-help {
            color: #64748b;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }
        .analytics-caption {
            color: #64748b;
            margin-bottom: 0.75rem;
        }
        .analytics-title {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.25rem;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: 0.65rem 0.9rem;
            font-weight: 600;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    return f"₹{value:,.2f}"


def build_payload() -> dict[str, float | int]:
    return {
        "number of bedrooms": st.session_state.bedrooms,
        "number of bathrooms": st.session_state.bathrooms,
        "living area": st.session_state.living_area,
        "lot area": st.session_state.lot_area,
        "number of floors": st.session_state.floors,
        "waterfront present": st.session_state.waterfront,
        "number of views": st.session_state.views,
        "condition of the house": st.session_state.condition,
        "grade of the house": st.session_state.grade,
        "Area of the house(excluding basement)": st.session_state.area_excl_basement,
        "Area of the basement": st.session_state.basement_area,
        "Built Year": st.session_state.built_year,
        "Renovation Year": st.session_state.renovation_year,
        "Postal Code": st.session_state.postal_code,
        "Lattitude": st.session_state.latitude,
        "Longitude": st.session_state.longitude,
        "living_area_renov": st.session_state.living_area_renov,
        "lot_area_renov": st.session_state.lot_area_renov,
        "Number of schools nearby": st.session_state.schools_nearby,
        "Distance from the airport": st.session_state.airport_distance,
    }


def parse_api_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"Request failed with status {response.status_code}."

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        messages = []
        for item in detail:
            loc = " → ".join(str(part) for part in item.get("loc", []))
            msg = item.get("msg", "Invalid value")
            messages.append(f"{loc}: {msg}" if loc else msg)
        return "; ".join(messages) if messages else "Invalid request payload."

    errors = payload.get("errors")
    if isinstance(errors, list):
        messages = []
        for item in errors:
            loc = " → ".join(str(part) for part in item.get("loc", []))
            msg = item.get("msg", "Invalid value")
            messages.append(f"{loc}: {msg}" if loc else msg)
        return "; ".join(messages) if messages else "Invalid request payload."

    return f"Request failed with status {response.status_code}."


def fetch_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    model = load_model_pipeline()

    input_df = pd.DataFrame([payload])

    prediction = float(model.predict(input_df)[0])

    return {
        "predicted_price": round(prediction, 2),
        "model_version": "1.0.0"
    }

def format_currency_compact(value: float) -> str:
    return f"₹{value:,.0f}"


def format_number(value: float | int) -> str:
    return f"{value:,.0f}"


def render_kpi_card(label: str, value: str, help_text: str | None = None) -> None:
    help_markup = f'<div class="kpi-help">{help_text}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {help_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@st.cache_data(show_spinner=False)
def load_training_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    return joblib.load(METADATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model_pipeline() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def get_feature_columns(df: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
    metadata_features = metadata.get("feature_columns")
    if metadata_features:
        return [column for column in metadata_features if column in df.columns]

    excluded_columns = {TARGET_COLUMN, *DROP_COLUMNS}
    return [column for column in df.columns if column not in excluded_columns]


def get_numeric_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(DROP_COLUMNS), errors="ignore").select_dtypes(
        include=["number"]
    )


def apply_analytics_filters(
    df: pd.DataFrame,
    price_range: tuple[int, int],
    bedroom_range: tuple[int, int],
    bathroom_range: tuple[float, float],
    grade_range: tuple[int, int],
) -> pd.DataFrame:
    filtered_df = df.copy()
    filtered_df = filtered_df[
        filtered_df[TARGET_COLUMN].between(price_range[0], price_range[1])
    ]
    filtered_df = filtered_df[
        filtered_df["number of bedrooms"].between(bedroom_range[0], bedroom_range[1])
    ]
    filtered_df = filtered_df[
        filtered_df["number of bathrooms"].between(
            bathroom_range[0], bathroom_range[1]
        )
    ]
    filtered_df = filtered_df[
        filtered_df["grade of the house"].between(grade_range[0], grade_range[1])
    ]
    return filtered_df


def render_analytics_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Interactive Filters")
    st.caption("Filter the dataset before reviewing analytics and downloads.")

    price_min = int(df[TARGET_COLUMN].min())
    price_max = int(df[TARGET_COLUMN].max())
    bedroom_min = int(df["number of bedrooms"].min())
    bedroom_max = int(df["number of bedrooms"].max())
    bathroom_min = float(df["number of bathrooms"].min())
    bathroom_max = float(df["number of bathrooms"].max())
    grade_min = int(df["grade of the house"].min())
    grade_max = int(df["grade of the house"].max())

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        price_range = st.slider(
            "Price range",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
            format="₹%d",
        )
        bathroom_range = st.slider(
            "Bathrooms",
            min_value=bathroom_min,
            max_value=bathroom_max,
            value=(bathroom_min, bathroom_max),
            step=0.25,
        )

    with filter_col2:
        bedroom_range = st.slider(
            "Bedrooms",
            min_value=bedroom_min,
            max_value=bedroom_max,
            value=(bedroom_min, bedroom_max),
        )
        grade_range = st.slider(
            "Grade",
            min_value=grade_min,
            max_value=grade_max,
            value=(grade_min, grade_max),
        )

    filtered_df = apply_analytics_filters(
        df=df,
        price_range=price_range,
        bedroom_range=bedroom_range,
        bathroom_range=bathroom_range,
        grade_range=grade_range,
    )

    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} records after active filters."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return filtered_df


def get_transformed_feature_names(
    pipeline: Any, feature_columns: list[str]
) -> list[str]:
    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor")
    if preprocessor is None:
        return feature_columns

    try:
        names = preprocessor.get_feature_names_out(feature_columns)
    except Exception:
        names = feature_columns

    cleaned_names = []
    for name in names:
        cleaned_name = str(name)
        if "__" in cleaned_name:
            cleaned_name = cleaned_name.split("__", 1)[1]
        cleaned_names.append(cleaned_name)
    return cleaned_names


def get_feature_importance(
    pipeline: Any, metadata: dict[str, Any]
) -> pd.DataFrame:
    regressor = getattr(pipeline, "named_steps", {}).get("regressor")
    importances = getattr(regressor, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    feature_columns = metadata.get("feature_columns", [])
    feature_names = get_transformed_feature_names(pipeline, feature_columns)
    if len(feature_names) != len(importances):
        feature_names = [f"Feature {index + 1}" for index in range(len(importances))]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.asarray(importances, dtype=float),
        }
    )
    return importance_df.sort_values("importance", ascending=False)


@st.cache_data(show_spinner=False)
def compute_shap_summary(
    feature_sample: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    try:
        import shap
    except Exception as exc:
        return None, None, f"SHAP is not installed or could not be imported: {exc}"

    try:
        pipeline = joblib.load(MODEL_PATH)
        preprocessor = pipeline.named_steps["preprocessor"]
        regressor = pipeline.named_steps["regressor"]

        transformed_sample = preprocessor.transform(
            feature_sample[list(feature_columns)]
        )
        if hasattr(transformed_sample, "toarray"):
            transformed_sample = transformed_sample.toarray()

        transformed_sample = np.asarray(transformed_sample)
        feature_names = get_transformed_feature_names(pipeline, list(feature_columns))
        if len(feature_names) != transformed_sample.shape[1]:
            feature_names = [
                f"Feature {index + 1}"
                for index in range(transformed_sample.shape[1])
            ]

        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(transformed_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)
        mean_abs_values = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_values,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        impact_rows = []
        top_features = importance_df.head(8)["feature"].tolist()
        feature_index = {feature: index for index, feature in enumerate(feature_names)}
        for feature in top_features:
            index = feature_index[feature]
            impact_rows.append(
                pd.DataFrame(
                    {
                        "feature": feature,
                        "shap_value": shap_values[:, index],
                        "feature_value": transformed_sample[:, index],
                    }
                )
            )

        impact_df = (
            pd.concat(impact_rows, ignore_index=True)
            if impact_rows
            else pd.DataFrame(columns=["feature", "shap_value", "feature_value"])
        )
        return importance_df, impact_df, None
    except Exception as exc:
        return None, None, f"SHAP calculation failed: {exc}"


def render_sidebar_inputs() -> None:
    with st.sidebar:
        st.markdown("### Property Inputs")
        st.caption("Enter house features to generate a price estimate.")

        st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=20,
            value=int(DEFAULT_PROPERTY["number of bedrooms"]),
            key="bedrooms",
        )
        st.number_input(
            "Bathrooms",
            min_value=0.5,
            max_value=20.0,
            value=float(DEFAULT_PROPERTY["number of bathrooms"]),
            step=0.5,
            key="bathrooms",
        )
        st.number_input(
            "Living area (sq ft)",
            min_value=100,
            value=int(DEFAULT_PROPERTY["living area"]),
            step=50,
            key="living_area",
        )
        st.number_input(
            "Lot area (sq ft)",
            min_value=100,
            value=int(DEFAULT_PROPERTY["lot area"]),
            step=50,
            key="lot_area",
        )
        st.number_input(
            "Floors",
            min_value=1.0,
            max_value=10.0,
            value=float(DEFAULT_PROPERTY["number of floors"]),
            step=0.5,
            key="floors",
        )
        st.selectbox(
            "Waterfront",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=int(DEFAULT_PROPERTY["waterfront present"]),
            key="waterfront",
        )
        st.number_input(
            "Views",
            min_value=0,
            value=int(DEFAULT_PROPERTY["number of views"]),
            key="views",
        )
        st.slider(
            "Condition (1–5)",
            min_value=1,
            max_value=5,
            value=int(DEFAULT_PROPERTY["condition of the house"]),
            key="condition",
        )
        st.slider(
            "Grade (1–13)",
            min_value=1,
            max_value=13,
            value=int(DEFAULT_PROPERTY["grade of the house"]),
            key="grade",
        )
        st.number_input(
            "Area excluding basement (sq ft)",
            min_value=0,
            value=int(DEFAULT_PROPERTY["Area of the house(excluding basement)"]),
            step=50,
            key="area_excl_basement",
        )
        st.number_input(
            "Basement area (sq ft)",
            min_value=0,
            value=int(DEFAULT_PROPERTY["Area of the basement"]),
            step=50,
            key="basement_area",
        )
        st.number_input(
            "Built year",
            min_value=1800,
            max_value=2100,
            value=int(DEFAULT_PROPERTY["Built Year"]),
            key="built_year",
        )
        st.number_input(
            "Renovation year (0 if none)",
            min_value=0,
            max_value=2100,
            value=int(DEFAULT_PROPERTY["Renovation Year"]),
            key="renovation_year",
        )
        st.number_input(
            "Postal code",
            min_value=1,
            value=int(DEFAULT_PROPERTY["Postal Code"]),
            key="postal_code",
        )
        st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(DEFAULT_PROPERTY["Lattitude"]),
            format="%.4f",
            key="latitude",
        )
        st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(DEFAULT_PROPERTY["Longitude"]),
            format="%.4f",
            key="longitude",
        )
        st.number_input(
            "Living area after renovation (sq ft)",
            min_value=0,
            value=int(DEFAULT_PROPERTY["living_area_renov"]),
            step=50,
            key="living_area_renov",
        )
        st.number_input(
            "Lot area after renovation (sq ft)",
            min_value=0,
            value=int(DEFAULT_PROPERTY["lot_area_renov"]),
            step=50,
            key="lot_area_renov",
        )
        st.number_input(
            "Schools nearby",
            min_value=0,
            value=int(DEFAULT_PROPERTY["Number of schools nearby"]),
            key="schools_nearby",
        )
        st.number_input(
            "Distance from airport (km)",
            min_value=0,
            value=int(DEFAULT_PROPERTY["Distance from the airport"]),
            key="airport_distance",
        )

        st.divider()
        predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)

    if predict_clicked:
        st.session_state.run_prediction = True


def render_property_details_section() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Property Details")
    st.caption("Review the property configuration used for inference.")

    payload = build_payload()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Layout & Size**")
        st.write(f"Bedrooms: **{payload['number of bedrooms']}**")
        st.write(f"Bathrooms: **{payload['number of bathrooms']}**")
        st.write(f"Living area: **{payload['living area']:,} sq ft**")
        st.write(f"Lot area: **{payload['lot area']:,} sq ft**")
        st.write(f"Floors: **{payload['number of floors']}**")

    with col2:
        st.markdown("**Quality & Features**")
        st.write(
            f"Waterfront: **{'Yes' if payload['waterfront present'] else 'No'}**"
        )
        st.write(f"Views: **{payload['number of views']}**")
        st.write(f"Condition: **{payload['condition of the house']}/5**")
        st.write(f"Grade: **{payload['grade of the house']}/13**")
        st.write(
            f"Basement area: **{payload['Area of the basement']:,} sq ft**"
        )

    with col3:
        st.markdown("**Location & Context**")
        st.write(f"Built: **{int(payload['Built Year'])}**")
        renovation = int(payload["Renovation Year"])
        st.write(f"Renovated: **{renovation if renovation else 'Not renovated'}**")
        st.write(f"Postal code: **{int(payload['Postal Code'])}**")
        st.write(
            f"Coordinates: **{payload['Lattitude']:.4f}, {payload['Longitude']:.4f}**"
        )
        st.write(f"Schools nearby: **{int(payload['Number of schools nearby'])}**")
        st.write(f"Airport distance: **{payload['Distance from the airport']} km**")

    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_section() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Prediction")
    st.caption("Estimated market value from the deployed XGBoost model.")

    if st.session_state.get("prediction_error"):
        st.error(st.session_state.prediction_error)

    if st.session_state.get("prediction_result"):
        result = st.session_state.prediction_result
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                label="Predicted House Price",
                value=format_currency(result["predicted_price"]),
            )

        with metric_col2:
            st.metric(
                label="Model Version",
                value=result["model_version"],
            )
    else:
        st.info("Configure property details in the sidebar and click **Predict Price**.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_dataset_overview(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    st.subheader("Dataset Overview")
    total_missing = int(filtered_df.isna().sum().sum())

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4, gap="small")
    with kpi_col1:
        render_kpi_card(
            "Total Records",
            format_number(len(filtered_df)),
            "Filtered rows",
        )
    with kpi_col2:
        render_kpi_card("Features", format_number(len(feature_columns)), "Model inputs")
    with kpi_col3:
        render_kpi_card(
            "Dataset Shape",
            f"{filtered_df.shape[0]:,} x {filtered_df.shape[1]:,}",
            "Rows x columns",
        )
    with kpi_col4:
        render_kpi_card("Missing Values", format_number(total_missing), "Filtered data")

    st.markdown("#### Missing Values Summary")
    missing_summary = (
        filtered_df.isna()
        .sum()
        .rename("missing_values")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing_summary["missing_percent"] = (
        missing_summary["missing_values"] / max(len(filtered_df), 1) * 100
    ).round(2)

    if total_missing == 0:
        st.success("No missing values found in the filtered dataset.")
    st.dataframe(missing_summary, use_container_width=True, height=180)

    st.markdown("#### Basic Statistics")
    stats_columns = [
        column
        for column in [*feature_columns, TARGET_COLUMN]
        if column in filtered_df
    ]
    stats_df = filtered_df[stats_columns].describe().T.round(2)
    st.dataframe(stats_df, use_container_width=True, height=260)

    st.caption(f"Full source dataset contains {len(df):,} records.")


def render_price_analytics(filtered_df: pd.DataFrame) -> None:
    st.subheader("Price Analytics")
    price_series = filtered_df[TARGET_COLUMN]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4, gap="small")
    with metric_col1:
        render_kpi_card("Mean Price", format_currency_compact(price_series.mean()))
    with metric_col2:
        render_kpi_card("Median Price", format_currency_compact(price_series.median()))
    with metric_col3:
        render_kpi_card("Minimum Price", format_currency_compact(price_series.min()))
    with metric_col4:
        render_kpi_card("Maximum Price", format_currency_compact(price_series.max()))

    chart_col1, chart_col2 = st.columns([1.35, 1], gap="medium")
    with chart_col1:
        histogram = px.histogram(
            filtered_df,
            x=TARGET_COLUMN,
            nbins=50,
            title="Price Distribution",
            labels={TARGET_COLUMN: "Price (INR)"},
            color_discrete_sequence=["#2563eb"],
        )
        histogram.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
            bargap=0.04,
            yaxis_title="Number of properties",
        )
        histogram.update_xaxes(tickprefix="₹")
        st.plotly_chart(histogram, use_container_width=True, config=PLOTLY_CONFIG)

    with chart_col2:
        boxplot = px.box(
            filtered_df,
            x=TARGET_COLUMN,
            points="outliers",
            title="Price Spread",
            labels={TARGET_COLUMN: "Price (INR)"},
            color_discrete_sequence=["#14b8a6"],
        )
        boxplot.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        boxplot.update_xaxes(tickprefix="₹")
        st.plotly_chart(boxplot, use_container_width=True, config=PLOTLY_CONFIG)


def render_correlation_analysis(filtered_df: pd.DataFrame) -> None:
    st.subheader("Correlation Analysis")
    numeric_df = get_numeric_analysis_frame(filtered_df)
    if TARGET_COLUMN not in numeric_df.columns or len(numeric_df) < 2:
        st.info("Correlation analysis needs at least two rows with numeric price data.")
        return

    correlation_df = numeric_df.corr(numeric_only=True)
    heatmap = go.Figure(
        data=go.Heatmap(
            z=correlation_df.values,
            x=correlation_df.columns,
            y=correlation_df.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            colorbar=dict(title="Correlation"),
        )
    )
    heatmap.update_layout(
        title="Numeric Feature Correlation Heatmap",
        template="plotly_white",
        height=650,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    st.plotly_chart(heatmap, use_container_width=True, config=PLOTLY_CONFIG)

    price_correlations = (
        correlation_df[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN], errors="ignore")
        .dropna()
    )
    top_correlations = price_correlations.reindex(
        price_correlations.abs().sort_values(ascending=False).head(10).index
    )
    top_correlation_df = (
        top_correlations.rename("correlation")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    st.markdown("#### Top Features Correlated with Price")
    top_correlation_chart = px.bar(
        top_correlation_df.sort_values("correlation"),
        x="correlation",
        y="feature",
        orientation="h",
        color="correlation",
        color_continuous_scale="RdBu",
        range_color=(-1, 1),
        labels={"correlation": "Correlation with Price", "feature": ""},
    )
    top_correlation_chart.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_showscale=False,
    )
    st.plotly_chart(
        top_correlation_chart,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    st.dataframe(top_correlation_df.round(4), use_container_width=True, height=220)


def render_feature_importance(metadata: dict[str, Any]) -> None:
    st.subheader("Feature Importance")
    try:
        pipeline = load_model_pipeline()
        importance_df = get_feature_importance(pipeline, metadata)
    except Exception as exc:
        st.warning(f"Feature importance is unavailable: {exc}")
        return

    if importance_df.empty:
        st.info("The trained model does not expose feature importance values.")
        return

    top_features = importance_df.head(10).sort_values("importance")
    importance_chart = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        text="importance",
        title="Top 10 XGBoost Feature Importances",
        labels={"importance": "Importance", "feature": ""},
        color="importance",
        color_continuous_scale="Blues",
    )
    importance_chart.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    importance_chart.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=20, r=30, t=60, b=20),
        coloraxis_showscale=False,
        xaxis_range=[0, max(top_features["importance"].max() * 1.18, 0.01)],
    )
    st.plotly_chart(importance_chart, use_container_width=True, config=PLOTLY_CONFIG)


def render_model_performance(
    df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    st.subheader("Model Performance Dashboard")
    metrics = metadata.get("metrics", {})

    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(
        5,
        gap="small",
    )
    with perf_col1:
        render_kpi_card("R² Score", f"{metrics.get('r2', 0):.4f}", "Test set")
    with perf_col2:
        render_kpi_card(
            "MAE",
            format_currency_compact(metrics.get("mae", 0)),
            "Avg error",
        )
    with perf_col3:
        render_kpi_card(
            "RMSE",
            format_currency_compact(metrics.get("rmse", 0)),
            "Large-error weighted",
        )
    with perf_col4:
        render_kpi_card("Dataset Size", format_number(len(df)), "Training CSV")
    with perf_col5:
        render_kpi_card("Algorithm", "XGBoost", "Regressor")


def render_shap_explainability(
    filtered_df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    st.subheader("SHAP Explainability")
    st.caption(
        "Global model explanations are calculated on a cached sample of filtered records."
    )

    if not feature_columns:
        st.info("SHAP needs the saved training feature list to align model inputs.")
        return

    sample_size = min(len(filtered_df), 500)
    if sample_size < 2:
        st.info("SHAP explainability needs at least two filtered records.")
        return

    feature_sample = filtered_df[list(feature_columns)].sample(
        n=sample_size,
        random_state=42,
    )
    with st.spinner("Calculating SHAP explanations..."):
        shap_importance_df, shap_impact_df, shap_error = compute_shap_summary(
            feature_sample=feature_sample,
            feature_columns=tuple(feature_columns),
        )

    if shap_error:
        st.info(shap_error)
        return

    if shap_importance_df is None or shap_impact_df is None:
        st.info("SHAP explanations are unavailable for this model.")
        return

    shap_col1, shap_col2 = st.columns([1, 1.2])
    with shap_col1:
        global_importance = shap_importance_df.head(10).sort_values("mean_abs_shap")
        shap_bar = px.bar(
            global_importance,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Global SHAP Importance",
            labels={"mean_abs_shap": "Mean absolute SHAP value", "feature": ""},
            color="mean_abs_shap",
            color_continuous_scale="Teal",
        )
        shap_bar.update_layout(
            template="plotly_white",
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(shap_bar, use_container_width=True, config=PLOTLY_CONFIG)

    with shap_col2:
        shap_summary = px.scatter(
            shap_impact_df,
            x="shap_value",
            y="feature",
            color="feature_value",
            title="SHAP Summary Visualization",
            labels={
                "shap_value": "SHAP value impact on prediction",
                "feature": "",
                "feature_value": "Feature value",
            },
            color_continuous_scale="Viridis",
        )
        shap_summary.update_layout(
            template="plotly_white",
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        shap_summary.add_vline(
            x=0,
            line_width=1,
            line_dash="dash",
            line_color="#64748b",
        )
        st.plotly_chart(shap_summary, use_container_width=True, config=PLOTLY_CONFIG)


def render_download_section(filtered_df: pd.DataFrame) -> None:
    st.markdown("#### Download Filtered Dataset")
    st.caption("Export the currently filtered dataset view as a CSV file.")
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="filtered_housing_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_analytics_section() -> None:
    st.markdown('<div class="analytics-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="analytics-caption">'
        "Explore the housing dataset, model behavior, and filtered market slices."
        "</p>",
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading analytics assets..."):
            df = load_dataset()
            metadata = load_training_metadata()
    except Exception as exc:
        st.error(f"Could not load analytics assets: {exc}")
        return

    feature_columns = get_feature_columns(df, metadata)
    filtered_df = render_analytics_filters(df)

    if filtered_df.empty:
        st.warning("No records match the active filters. Widen the filters to continue.")
        return

    (
        tab_overview,
        tab_price,
        tab_correlations,
        tab_importance,
        tab_performance,
        tab_shap,
    ) = st.tabs(
        [
            "Overview",
            "Price Analytics",
            "Correlations",
            "Feature Importance",
            "Model Performance",
            "SHAP",
        ]
    )

    with tab_overview:
        render_dataset_overview(df, filtered_df, feature_columns)
        st.divider()
        render_download_section(filtered_df)

    with tab_price:
        render_price_analytics(filtered_df)

    with tab_correlations:
        render_correlation_analysis(filtered_df)

    with tab_importance:
        render_feature_importance(metadata)

    with tab_performance:
        render_model_performance(df, metadata)

    with tab_shap:
        render_shap_explainability(filtered_df, feature_columns)


def render_about_section() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("About")
    st.markdown(
        f"""
        **{APP_TITLE}** helps analysts and homebuyers estimate residential property
        prices using machine learning. The frontend combines real-time prediction with
        portfolio-ready dataset analytics, model diagnostics, and explainability views.

        **How it works**
        1. Enter property attributes in the sidebar for prediction.
        2. Explore the Analytics tab for dataset and model insights.
        3. Download filtered data slices for offline review.

        **Stack**
        - Frontend: Streamlit + Plotly
        - API: FastAPI + XGBoost
        - Backend endpoint: `{PREDICT_ENDPOINT}`

        **Deployment**
        Set the `API_URL` environment variable to point the frontend at your hosted API
        (for example, `https://api.example.com`).
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


def run_prediction_flow() -> None:
    if not st.session_state.pop("run_prediction", False):
        return

    st.session_state.prediction_error = None
    st.session_state.prediction_result = None

    payload = build_payload()
    if (
        payload["Renovation Year"] > 0
        and payload["Renovation Year"] < payload["Built Year"]
    ):
        st.session_state.prediction_error = (
            "Renovation year must be greater than or equal to built year."
        )
        return

    try:
        with st.spinner("Generating price prediction..."):
            st.session_state.prediction_result = fetch_prediction(payload)
    except requests.exceptions.ConnectionError:
        st.session_state.prediction_error = (
            f"Could not connect to the API at `{DEFAULT_API_URL}`. "
            "Ensure the FastAPI server is running (`uvicorn app:app --reload`)."
        )
    except requests.exceptions.Timeout:
        st.session_state.prediction_error = (
            "The prediction request timed out. Please try again."
        )
    except requests.exceptions.RequestException as exc:
        st.session_state.prediction_error = f"Network error: {exc}"
    except RuntimeError as exc:
        st.session_state.prediction_error = str(exc)
    except (KeyError, TypeError, ValueError):
        st.session_state.prediction_error = (
            "Received an unexpected response from the API."
        )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "prediction_error" not in st.session_state:
        st.session_state.prediction_error = None

    render_sidebar_inputs()
    run_prediction_flow()

    st.markdown(f'<p class="hero-title">{APP_TITLE}</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        "AI-powered housing price estimation with real-time model inference."
        "</p>",
        unsafe_allow_html=True,
    )

    tab_property, tab_prediction, tab_analytics, tab_about = st.tabs(
        ["Property Details", "Prediction", "Analytics", "About"]
    )

    with tab_property:
        render_property_details_section()

    with tab_prediction:
        render_prediction_section()

    with tab_analytics:
        render_analytics_section()

    with tab_about:
        render_about_section()


if __name__ == "__main__":
    main()
