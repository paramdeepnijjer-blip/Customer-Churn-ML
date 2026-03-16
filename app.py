import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="📉",
    layout="centered"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0d0f14; }

.hero {
    background: linear-gradient(135deg, #0d0f14 0%, #111827 100%);
    border: 1px solid #1e2433;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #f1f5f9;
    margin: 0 0 0.5rem;
    letter-spacing: -0.03em;
}
.hero p {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0;
}

.metric-card {
    background: #111827;
    border: 1px solid #1e2433;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 600;
    letter-spacing: -0.04em;
    line-height: 1;
}
.metric-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.3rem;
}

.risk-high   { color: #f87171; }
.risk-medium { color: #fbbf24; }
.risk-low    { color: #34d399; }

.verdict-box {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
    border: 1px solid;
}
.verdict-target {
    background: rgba(251,191,36,0.08);
    border-color: rgba(251,191,36,0.3);
    color: #fbbf24;
}
.verdict-skip {
    background: rgba(100,116,139,0.08);
    border-color: rgba(100,116,139,0.2);
    color: #94a3b8;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2433;
}

.stSlider > div { padding-top: 0.25rem; }
div[data-testid="stForm"] { border: none; padding: 0; }

.built-by {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #1e2433;
    margin-top: 3rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ── Model training (cached so it only runs once) ──────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df.columns = df.columns.str.strip()
    df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocess = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    base = LogisticRegression(max_iter=5000, class_weight="balanced")
    model = Pipeline(steps=[
        ("prep", preprocess),
        ("cal", CalibratedClassifierCV(base, method="isotonic", cv=5))
    ])
    model.fit(X_train, y_train)
    return model, X.columns.tolist(), num_cols, cat_cols


model, feature_cols, num_cols, cat_cols = train_model()

# ── Business logic ────────────────────────────────────────────────────────────
LTV        = 800
OFFER_COST = 100
SAVE_RATE  = 0.30
THRESHOLD  = 0.42


def expected_value(prob):
    return (prob * SAVE_RATE * LTV) - OFFER_COST


def risk_band(prob):
    if prob >= 0.65:
        return "High", "risk-high"
    elif prob >= 0.35:
        return "Medium", "risk-medium"
    return "Low", "risk-low"


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📉 Churn Risk Predictor</h1>
  <p>Enter customer details to get churn probability, risk band, and retention ROI.</p>
</div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):

    st.markdown('<div class="section-label">Account Info</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with col2:
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    with col3:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0,
                                        value=float(tenure * monthly_charges),
                                        step=10.0)

    st.markdown('<div class="section-label">Contract & Billing</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with col5:
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with col6:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown('<div class="section-label">Services</div>', unsafe_allow_html=True)
    col7, col8, col9 = st.columns(3)
    with col7:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    with col8:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    with col9:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
    col10, col11, col12, col13 = st.columns(4)
    with col10:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col11:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col12:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with col13:
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
    multiple_lines = st.selectbox("Multiple Lines",
                                  ["No", "Yes", "No phone service"])
    streaming_movies = st.selectbox("Streaming Movies",
                                    ["Yes", "No", "No internet service"])

    submitted = st.form_submit_button("→ Predict Churn Risk", use_container_width=True)

if submitted:
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])
    prob = model.predict_proba(input_df)[0][1]
    ev = expected_value(prob)
    band, band_class = risk_band(prob)
    should_target = prob >= THRESHOLD

    st.markdown("---")
    st.markdown("### Prediction Results")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Churn Probability</div>
          <div class="metric-value {band_class}">{prob:.0%}</div>
          <div class="metric-sub">Calibrated score</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Risk Band</div>
          <div class="metric-value {band_class}">{band}</div>
          <div class="metric-sub">Based on probability</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        ev_color = "#34d399" if ev > 0 else "#f87171"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Expected Value</div>
          <div class="metric-value" style="color:{ev_color}">${ev:+.0f}</div>
          <div class="metric-sub">Per retention offer</div>
        </div>""", unsafe_allow_html=True)

    if should_target:
        st.markdown(f"""
        <div class="verdict-box verdict-target">
          <strong>✓ Send retention offer</strong><br>
          <span style="font-size:0.85rem">Churn probability ({prob:.0%}) exceeds the {THRESHOLD:.0%} threshold.
          Expected value of targeting this customer: <strong>${ev:+.0f}</strong>.</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-box verdict-skip">
          <strong>✗ Skip — offer not cost-effective</strong><br>
          <span style="font-size:0.85rem">Churn probability ({prob:.0%}) is below the {THRESHOLD:.0%} threshold.
          Sending an offer would cost more than the expected retention value.</span>
        </div>""", unsafe_allow_html=True)

    with st.expander("How is this calculated?"):
        st.markdown(f"""
        **Model:** Logistic Regression with isotonic probability calibration (ROC-AUC 0.84)

        **Threshold:** {THRESHOLD:.0%} — the probability above which targeting a customer is profitable

        **Expected Value formula:**
        ```
        EV = (churn_probability × save_rate × LTV) − offer_cost
           = ({prob:.2f} × {SAVE_RATE} × ${LTV}) − ${OFFER_COST}
           = ${ev:+.2f}
        ```

        **Assumptions:** Customer LTV = ${LTV} · Offer cost = ${OFFER_COST} · Save rate = {SAVE_RATE:.0%}
        """)

st.markdown('<div class="built-by">BUILT BY PARAMDEEP NIJJER · MS DATA SCIENCE · BOSTON UNIVERSITY</div>',
            unsafe_allow_html=True)
