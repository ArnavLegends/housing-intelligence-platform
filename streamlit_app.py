"""
Housing Intelligence Platform — Streamlit frontend.

Run locally:
    streamlit run streamlit_app.py

Set API_URL for deployment (default: http://127.0.0.1:8000).
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

APP_TITLE = "Housing Intelligence Platform"
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_ENDPOINT = f"{DEFAULT_API_URL}/predict"

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
            max-width: 1100px;
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
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }
        .section-card h3 {
            margin-top: 0;
            color: #0f172a;
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
    response = requests.post(
        PREDICT_ENDPOINT,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if response.ok:
        return response.json()

    raise RuntimeError(parse_api_error(response))


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


def render_about_section() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("About")
    st.markdown(
        f"""
        **{APP_TITLE}** helps analysts and homebuyers estimate residential property
        prices using machine learning. The frontend collects structured property features,
        sends them to a FastAPI inference service, and displays the model output in real time.

        **How it works**
        1. Enter property attributes in the sidebar.
        2. Submit a prediction request to the backend API.
        3. Review the estimated price and active model version.

        **Stack**
        - Frontend: Streamlit
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

    tab_property, tab_prediction, tab_about = st.tabs(
        ["Property Details", "Prediction", "About"]
    )

    with tab_property:
        render_property_details_section()

    with tab_prediction:
        render_prediction_section()

    with tab_about:
        render_about_section()


if __name__ == "__main__":
    main()
