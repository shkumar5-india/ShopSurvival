import streamlit as st
import pandas as pd
import joblib
import json
import os
from xgboost import XGBClassifier

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="ShopSurvival",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# GLOBAL STYLES (UI ONLY)
# ---------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --brand: #4F46E5;
        --brand-dark: #3730A3;
        --brand-soft: #EEF2FF;
        --ink: #0F172A;
        --muted: #64748B;
        --line: #E2E8F0;
        --surface: #FFFFFF;
        --bg: #F8FAFC;
        --ok: #059669;
        --ok-soft: #ECFDF5;
        --bad: #DC2626;
        --bad-soft: #FEF2F2;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp { background: var(--bg); }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(135deg, #312E81 0%, #4F46E5 55%, #7C3AED 100%);
        border-radius: 20px;
        padding: 40px 44px;
        color: #FFFFFF;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -20px rgba(79, 70, 229, 0.55);
        margin-bottom: 28px;
    }
    .hero::after {
        content: "";
        position: absolute;
        right: -80px; top: -80px;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 8px 0;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .hero p {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.86);
        max-width: 640px;
        margin: 0;
        line-height: 1.6;
    }
    .hero-stats { display: flex; gap: 32px; margin-top: 26px; flex-wrap: wrap; }
    .hero-stat b { display: block; font-size: 1.35rem; font-weight: 700; }
    .hero-stat span { font-size: 0.8rem; color: rgba(255,255,255,0.75); }

    /* ---------- Section headings ---------- */
    .section-head { display: flex; align-items: center; gap: 12px; margin: 6px 0 14px 0; }
    .step {
        width: 30px; height: 30px; border-radius: 9px;
        background: var(--brand-soft); color: var(--brand);
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 14px;
    }
    .section-title { font-size: 1.2rem; font-weight: 700; color: var(--ink); margin: 0; letter-spacing: -0.01em; }
    .section-sub { font-size: 0.85rem; color: var(--muted); margin: 0; }

    .card-label {
        font-size: 0.78rem; font-weight: 700; color: var(--brand);
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
    }
    .card-title { font-size: 1.02rem; font-weight: 600; color: var(--ink); margin-bottom: 14px; }

    /* ---------- Form card ---------- */
    [data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 28px 28px 22px 28px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px -12px rgba(15,23,42,0.08);
    }

    /* ---------- Inputs ---------- */
    label, .stNumberInput label p, .stSelectbox label p, .stMultiSelect label p {
        font-weight: 500 !important;
        color: #334155 !important;
        font-size: 0.88rem !important;
    }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border-color: var(--line) !important;
        background: #FFFFFF !important;
        color: var(--ink) !important;
    }
    .stNumberInput input:focus {
        border-color: var(--brand) !important;
        box-shadow: 0 0 0 3px rgba(79,70,229,0.15) !important;
    }
    span[data-baseweb="tag"] {
        background: var(--brand-soft) !important;
        color: var(--brand-dark) !important;
        border-radius: 8px !important;
        font-weight: 500;
    }

    /* ---------- Buttons ---------- */
    .stFormSubmitButton button, .stButton button {
        background: linear-gradient(135deg, #4F46E5, #6D28D9) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.4rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.01em;
        box-shadow: 0 10px 20px -8px rgba(79,70,229,0.6);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stFormSubmitButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 26px -8px rgba(79,70,229,0.7);
    }

    /* ---------- Result ---------- */
    .result-card {
        border-radius: 18px;
        padding: 28px 32px;
        display: flex; align-items: center; justify-content: space-between;
        gap: 20px; flex-wrap: wrap;
        border: 1px solid;
        margin-bottom: 18px;
    }
    .result-open  { background: var(--ok-soft); border-color: #A7F3D0; }
    .result-closed{ background: var(--bad-soft); border-color: #FECACA; }
    .result-eyebrow { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
    .result-status { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; margin: 2px 0 0 0; }
    .result-open .result-status   { color: var(--ok); }
    .result-closed .result-status { color: var(--bad); }
    .result-pill {
        padding: 8px 18px; border-radius: 999px; font-weight: 700; font-size: 0.9rem;
        background: #FFFFFF; border: 1px solid var(--line); color: var(--ink);
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        height: 100%;
    }
    .metric-label { font-size: 0.82rem; font-weight: 600; color: var(--muted); }
    .metric-value { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 4px 0 12px 0; color: var(--ink); }
    .meter { width: 100%; height: 10px; background: #EEF2F7; border-radius: 999px; overflow: hidden; }
    .meter > div { height: 100%; border-radius: 999px; }
    .meter-ok  > div { background: linear-gradient(90deg, #34D399, #059669); }
    .meter-bad > div { background: linear-gradient(90deg, #F87171, #DC2626); }

    .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 22px 24px;
        margin-top: 18px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    .disclaimer {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        padding: 14px 18px;
        border-radius: 12px;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-top: 18px;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #0F172A;
        border-right: none;
    }
    [data-testid="stSidebar"] * { color: #CBD5E1; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
    .side-brand { font-size: 1.3rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; }
    .side-item {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .side-item b { color: #FFFFFF; display: block; margin-bottom: 2px; font-size: 0.82rem; }

    .app-footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.82rem;
        padding: 26px 0 4px 0;
        border-top: 1px solid var(--line);
        margin-top: 36px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# FILE PATHS
# ---------------------------------

MODEL_DIR = "shopsurvival_models"

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR, "preprocessor.pkl"
)

XGB_PATH = os.path.join(
    MODEL_DIR, "xgb_model.json"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR, "features.json"
)

# ---------------------------------
# LOAD MODEL AND FEATURES
# ---------------------------------

@st.cache_resource
def load_model():

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    xgb_model = XGBClassifier()

    xgb_model.load_model(
        XGB_PATH
    )

    return preprocessor, xgb_model


if not all(os.path.exists(path) for path in [
    PREPROCESSOR_PATH,
    XGB_PATH,
    FEATURES_PATH
]):
    st.error(
        "Model files missing. Please check "
        "the shopsurvival_models folder."
    )
    st.stop()

with open(FEATURES_PATH, "r") as file:
    features = json.load(file)

preprocessor, model = load_model()

# ---------------------------------
# FEATURE DEFINITIONS
# ---------------------------------

numeric_defaults = {
    "rating_x_reviews": 0.0,
    "review_count": 0,
    "num_categories": 1,
    "years_in_business": 1.0,
    "num_checkins": 0,
    "has_checkin": 0,
    "latitude": 0.0,
    "longitude": 0.0,
    "pcpi": 0.0,
    "poverty_rate": 0.0,
    "median_household_income": 0.0,
    "unemployment_rate": 0.0,
    "avg_weekly_wages": 0.0
}

category_features = [
    feature for feature in features
    if feature.startswith("cat_")
]

# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.markdown(
        '<div class="side-brand">🏪 ShopSurvival</div>',
        unsafe_allow_html=True
    )
    st.caption("Business status intelligence")

    st.markdown("&nbsp;", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="side-item"><b>MODEL</b>XGBoost Classifier</div>
        <div class="side-item"><b>INPUT FEATURES</b>{len(features)} total · {len(category_features)} category flags</div>
        <div class="side-item"><b>OUTPUT</b>Recorded status: Open / Closed, with class probabilities</div>
        <div class="side-item"><b>DATA SCOPE</b>Trained on US business records</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("**How to use**")
    st.markdown(
        "1. Fill in business details\n"
        "2. Pick one or more categories\n"
        "3. Click **Analyze Business**"
    )

# ---------------------------------
# APPLICATION HEADER
# ---------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI-Powered Analytics</div>
        <h1>ShopSurvival</h1>
        <p>AI-Powered Business Status Analysis. Analyze business information and
        estimate its recorded Open or Closed status using Machine Learning.</p>
        <div class="hero-stats">
            <div class="hero-stat"><b>XGBoost</b><span>Prediction engine</span></div>
            <div class="hero-stat"><b>Real-time</b><span>Instant inference</span></div>
            <div class="hero-stat"><b>Probabilistic</b><span>Open vs Closed scores</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# BUSINESS INPUT FORM
# ---------------------------------

st.markdown(
    """
    <div class="section-head">
        <div class="step">1</div>
        <div>
            <p class="section-title">Business Information</p>
            <p class="section-sub">Operational, location and local economic indicators</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("business_form"):

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown(
            '<div class="card-label">Performance</div>'
            '<div class="card-title">Business activity &amp; engagement</div>',
            unsafe_allow_html=True
        )

        rating_reviews = st.number_input(
            "Rating × Reviews",
            min_value=0.0,
            value=0.0
        )

        review_count = st.number_input(
            "Review Count",
            min_value=0,
            value=0,
            step=1
        )

        num_categories = st.number_input(
            "Number of Categories",
            min_value=0,
            value=1,
            step=1
        )

        years = st.number_input(
            "Years in Business",
            min_value=0.0,
            value=1.0
        )

        checkins = st.number_input(
            "Number of Check-ins",
            min_value=0,
            value=0,
            step=1
        )

        has_checkin = st.selectbox(
            "Has Check-in?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col2:

        st.markdown(
            '<div class="card-label">Location &amp; Economy</div>'
            '<div class="card-title">Geography &amp; local indicators</div>',
            unsafe_allow_html=True
        )

        latitude = st.number_input(
            "Latitude",
            value=0.0,
            format="%.6f"
        )

        longitude = st.number_input(
            "Longitude",
            value=0.0,
            format="%.6f"
        )

        pcpi = st.number_input(
            "Per Capita Personal Income",
            min_value=0.0,
            value=0.0
        )

        poverty = st.number_input(
            "Poverty Rate",
            min_value=0.0,
            value=0.0
        )

        income = st.number_input(
            "Median Household Income",
            min_value=0.0,
            value=0.0
        )

        unemployment = st.number_input(
            "Unemployment Rate",
            min_value=0.0,
            value=0.0
        )

        wages = st.number_input(
            "Average Weekly Wages",
            min_value=0.0,
            value=0.0
        )

    st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:18px 0 14px 0;'>",
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-head" style="margin-top:0;">
            <div class="step">2</div>
            <div>
                <p class="section-title">Business Categories</p>
                <p class="section-sub">Select every category that applies</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_categories = st.multiselect(
        "Select Business Categories",
        options=category_features,
        format_func=lambda x: x.replace("cat_", "")
    )

    submitted = st.form_submit_button(
        "Analyze Business",
        use_container_width=True
    )

# ---------------------------------
# PREDICTION
# ---------------------------------

if submitted:

    input_data = {
        feature: 0 for feature in features
    }

    numeric_values = {
        "rating_x_reviews": rating_reviews,
        "review_count": review_count,
        "num_categories": num_categories,
        "years_in_business": years,
        "num_checkins": checkins,
        "has_checkin": has_checkin,
        "latitude": latitude,
        "longitude": longitude,
        "pcpi": pcpi,
        "poverty_rate": poverty,
        "median_household_income": income,
        "unemployment_rate": unemployment,
        "avg_weekly_wages": wages
    }

    for feature, value in numeric_values.items():

        if feature in input_data:
            input_data[feature] = value

    for category in selected_categories:

        if category in input_data:
            input_data[category] = 1

    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    try:

        processed_input = preprocessor.transform(
            input_df
        )

        prediction = model.predict(
            processed_input
        )[0]

        probabilities = model.predict_proba(
            processed_input
        )[0]

        classes = list(model.classes_)

        open_probability = (
            probabilities[classes.index(1)] * 100
        )

        closed_probability = (
            probabilities[classes.index(0)] * 100
        )

        # ---------------------------------
        # DISPLAY RESULTS
        # ---------------------------------

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-head">
                <div class="step">3</div>
                <div>
                    <p class="section-title">Business Prediction Results</p>
                    <p class="section-sub">Model output for the information provided</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        confidence = max(open_probability, closed_probability)

        if prediction == 1:

            st.markdown(
                f"""
                <div class="result-card result-open">
                    <div>
                        <div class="result-eyebrow">Predicted Business Status</div>
                        <div class="result-status">✅ OPEN</div>
                    </div>
                    <div class="result-pill">Confidence {confidence:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-card result-closed">
                    <div>
                        <div class="result-eyebrow">Predicted Business Status</div>
                        <div class="result-status">⛔ CLOSED</div>
                    </div>
                    <div class="result-pill">Confidence {confidence:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Open Probability</div>
                    <div class="metric-value">{open_probability:.2f}%</div>
                    <div class="meter meter-ok"><div style="width:{open_probability:.2f}%"></div></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Closed Probability</div>
                    <div class="metric-value">{closed_probability:.2f}%</div>
                    <div class="meter meter-bad"><div style="width:{closed_probability:.2f}%"></div></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------------------------------
        # PROBABILITY CHART
        # ---------------------------------

        chart_col, summary_col = st.columns([1, 1], gap="large")

        with chart_col:

            st.markdown(
                '<div class="panel" style="margin-top:18px;">'
                '<div class="card-label">Distribution</div>'
                '<div class="card-title">Business Status Probability</div>',
                unsafe_allow_html=True
            )

            chart_data = pd.DataFrame({
                "Status": ["Open", "Closed"],
                "Probability": [
                    open_probability,
                    closed_probability
                ]
            })

            st.bar_chart(
                chart_data,
                x="Status",
                y="Probability",
                color="#4F46E5"
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------
        # BUSINESS SUMMARY
        # ---------------------------------

        with summary_col:

            st.markdown(
                '<div class="panel" style="margin-top:18px;">'
                '<div class="card-label">Summary</div>'
                '<div class="card-title">Business Information Summary</div>',
                unsafe_allow_html=True
            )

            summary = pd.DataFrame({
                "Feature": [
                    "Review Count",
                    "Years in Business",
                    "Number of Check-ins",
                    "Selected Categories"
                ],
                "Value": [
                    str(review_count),
                    str(years),
                    str(checkins),
                    ", ".join(
                        category.replace("cat_", "")
                        for category in selected_categories
                    ) or "None"
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------
        # DISCLAIMER
        # ---------------------------------

        st.markdown(
            """
            <div class="disclaimer">
                <b>Disclaimer.</b> This application predicts recorded business
                status using the trained dataset. It does not guarantee future
                business survival. The training data is based on US businesses.
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )

        st.write(
            "Please verify that the model, "
            "preprocessor and feature files "
            "were exported from the same training pipeline."
        )

# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown(
    '<div class="app-footer">ShopSurvival &nbsp;|&nbsp; AI &amp; Machine Learning Project</div>',
    unsafe_allow_html=True
)