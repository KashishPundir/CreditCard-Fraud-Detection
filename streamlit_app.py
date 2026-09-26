import streamlit as st
import requests

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        h1 {
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #6b7280;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            padding: 0.5rem;
        }

        div.stButton > button {
            width: 100%;
            height: 3rem;
            font-size: 1.05rem;
            font-weight: 600;
        }

        .result-box {
            padding: 1rem;
            border-radius: 10px;
            margin-top: 0.5rem;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection")
st.markdown(
    '<div class="subtitle">'
    'Real-time transaction analysis using an XGBoost fraud detection model'
    '</div>',
    unsafe_allow_html=True
)

API_URL = "https://fraud-detection-j5nt.onrender.com/predict"
HEALTH_URL = "https://fraud-detection-j5nt.onrender.com/health"

# --------------------------------------------------
# API Status
# --------------------------------------------------

try:
    health_response = requests.get(HEALTH_URL, timeout=2)

    if health_response.status_code == 200:
        health_data = health_response.json()
        st.success(
            f"🟢 API Online • {health_data['model']} • "
            f"Version {health_data['version']}"
        )
    else:
        st.warning("🟡 API is reachable but unhealthy.")

except requests.exceptions.RequestException:
    st.error(
        "🔴 API Offline — Start FastAPI on "
        "http://127.0.0.1:8000"
    )
    
# --------------------------------------------------
# Demo Transactions
# --------------------------------------------------

FRAUD_EXAMPLE = {
    "Time": 406.0,
    "V1": -2.312226542,
    "V2": 1.951992011,
    "V3": -1.609850732,
    "V4": 3.997905588,
    "V5": -0.522187865,
    "V6": -1.426545319,
    "V7": -2.537387306,
    "V8": 1.391657248,
    "V9": -2.770089277,
    "V10": -2.772272145,
    "V11": 3.202033207,
    "V12": -2.899907388,
    "V13": -0.595221881,
    "V14": -4.289253782,
    "V15": 0.389724120,
    "V16": -1.140747180,
    "V17": -2.830055674,
    "V18": -0.016822468,
    "V19": 0.416955705,
    "V20": 0.126910559,
    "V21": 0.517232371,
    "V22": -0.035049369,
    "V23": -0.465211076,
    "V24": 0.320198199,
    "V25": 0.044519167,
    "V26": 0.177839798,
    "V27": 0.261145003,
    "V28": -0.143275875,
    "Amount": 0.0
}

LEGITIMATE_EXAMPLE = {
    "Time": 1000.0,
    "V1": 1.2,
    "V2": 0.3,
    "V3": 0.8,
    "V4": -0.2,
    "V5": 0.4,
    "V6": -0.1,
    "V7": 0.2,
    "V8": -0.05,
    "V9": 0.6,
    "V10": 0.1,
    "V11": -0.3,
    "V12": 0.2,
    "V13": -0.1,
    "V14": 0.4,
    "V15": 0.2,
    "V16": -0.2,
    "V17": 0.1,
    "V18": 0.3,
    "V19": -0.1,
    "V20": 0.05,
    "V21": -0.02,
    "V22": 0.1,
    "V23": -0.05,
    "V24": 0.02,
    "V25": 0.1,
    "V26": -0.05,
    "V27": 0.02,
    "V28": 0.01,
    "Amount": 50.0
}

# --------------------------------------------------
# Demo Controls
# --------------------------------------------------

demo_col1, demo_col2 = st.columns(2)

with demo_col1:
    if st.button("🧪 Load Fraud Example", use_container_width=True):
        for key, value in FRAUD_EXAMPLE.items():
            if key.startswith("V"):
                st.session_state[key.lower()] = value
            else:
                st.session_state[key] = value
        st.rerun()

with demo_col2:
    if st.button("🧪 Load Demo Transaction", use_container_width=True):
        for key, value in LEGITIMATE_EXAMPLE.items():
            if key.startswith("V"):
                st.session_state[key.lower()] = value
            else:
                st.session_state[key] = value
        st.rerun()

# --------------------------------------------------
# Top Transaction Inputs
# --------------------------------------------------

top_left, top_mid, top_right = st.columns([1, 1, 1])

with top_left:
    time = st.number_input(
        "Transaction Time",
        value=st.session_state.get("Time", 406.0),
        key="Time"
    )

with top_mid:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=st.session_state.get("Amount", 0.0),
        format="%.2f",
        key="Amount"
    )

with top_right:
    st.metric(
        "Model",
        "XGBoost"
    )

# --------------------------------------------------
# V Features
# --------------------------------------------------

features = {}

with st.expander("⚙️ Advanced Model Features (V1–V28)", expanded=False):

    feature_columns = st.columns(4)

    for i in range(1, 29):

        column = feature_columns[(i - 1) % 4]

        with column:
            feature_key = f"v{i}"

            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=st.session_state.get(feature_key, 0.0),
                format="%.4f",
                key=feature_key
            )

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

st.markdown("")

predict_clicked = st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_clicked:

    transaction = {
        "Time": time,
        **features,
        "Amount": amount
    }

    try:

        with st.spinner("Analyzing transaction..."):

            response = requests.post(
                API_URL,
                json=transaction,
                timeout=10
            )

            response.raise_for_status()

            result = response.json()

        probability = result["fraud_probability"]
        threshold = result["threshold"]
        prediction = result["prediction"]
        label = result["prediction_label"]

        st.divider()

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3, result_col4 = st.columns(4)

        with result_col1:
            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

        with result_col2:
            st.metric(
                "Decision Threshold",
                f"{threshold:.4f}"
            )

        with result_col3:
            st.metric(
                "Prediction",
                str(prediction)
            )

        with result_col4:
            st.metric(
                "Classification",
                label
            )

        # --------------------------------------------------
        # Visual Result
        # --------------------------------------------------

        if prediction == 1:

            st.error(
                "🚨 FRAUD DETECTED — This transaction crossed "
                "the configured fraud decision threshold."
            )

        else:

            st.success(
                "✅ LEGITIMATE TRANSACTION — The fraud probability "
                "is below the configured decision threshold."
            )

        # Probability bar
        st.progress(
            min(float(probability), 1.0),
            text=f"Fraud probability: {probability:.2%}"
        )

    except requests.exceptions.RequestException as e:

        st.error(
            "❌ Could not connect to the Fraud Detection API."
        )

        st.info(
            "Make sure FastAPI is running on "
            "http://127.0.0.1:8000"
        )

        st.code(str(e))

    except Exception as e:

        st.error("❌ Prediction failed.")
        st.code(str(e))

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Credit Card Fraud Detection • XGBoost • FastAPI • Streamlit"
)