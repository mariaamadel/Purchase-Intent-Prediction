import joblib
import pandas as pd
import streamlit as st

# Keep this threshold identical to the one selected in the notebook.
THRESHOLD = 0.327
MODEL_PATH = "purchase_intent_pipeline.joblib"

st.set_page_config(page_title="Purchase Intent Predictor", page_icon="🛍️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🛍️ Will This Visitor Buy?")
st.write(
    "Enter a website session's browsing information to estimate whether "
    "the visitor is likely to make a purchase."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}`. Save the trained pipeline from your notebook "
        "in the same folder as this app, using the filename shown above."
    )
    st.stop()

with st.form("session_form"):
    st.subheader("Browsing activity")

    col1, col2 = st.columns(2)
    with col1:
        administrative = st.number_input("Administrative pages", min_value=0, max_value=30, value=0, step=1)
        informational = st.number_input("Informational pages", min_value=0, max_value=30, value=0, step=1)
        product_related = st.number_input("Product-related pages", min_value=0, max_value=200, value=10, step=1)
        page_values = st.number_input("Page values", min_value=0.0, max_value=5000.0, value=0.0, step=1.0)
    with col2:
        administrative_duration = st.number_input("Administrative duration (seconds)", min_value=0.0, max_value=100000.0, value=0.0, step=10.0)
        informational_duration = st.number_input("Informational duration (seconds)", min_value=0.0, max_value=100000.0, value=0.0, step=10.0)
        product_related_duration = st.number_input("Product-related duration (seconds)", min_value=0.0, max_value=200000.0, value=300.0, step=10.0)

    st.subheader("Session behavior")
    col3, col4 = st.columns(2)
    with col3:
        bounce_rates = st.number_input("Bounce rate", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.4f")
        exit_rates = st.number_input("Exit rate", min_value=0.0, max_value=1.0, value=0.04, step=0.01, format="%.4f")
        special_day = st.number_input("Special day", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    with col4:
        visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
        weekend = st.checkbox("Weekend session", value=False)

    st.subheader("Technical/session details")
    col5, col6 = st.columns(2)
    with col5:
        operating_systems = st.selectbox("Operating system code", list(range(1, 9)), index=1)
        browser = st.selectbox("Browser code", list(range(1, 14)), index=1)
        region = st.selectbox("Region code", list(range(1, 10)), index=0)
    with col6:
        traffic_type = st.selectbox("Traffic type code", list(range(1, 21)), index=1)

    submitted = st.form_submit_button("Predict purchase intent", type="primary", use_container_width=True)

if submitted:
    # Column names and order must match the dataset used to train the pipeline.
    input_data = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_systems,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend,
    }])

    try:
        probability = float(model.predict_proba(input_data)[0, 1])
    except Exception as exc:
        st.error(
            "Prediction failed. Check that the saved pipeline was trained with "
            "the same 17 input columns and preprocessing used in this app."
        )
        st.exception(exc)
        st.stop()

    prediction = int(probability >= THRESHOLD)
    st.divider()
    st.subheader("Prediction result")
    st.metric("Estimated purchase probability", f"{probability:.1%}")
    st.caption(f"Decision threshold: {THRESHOLD:.3f}")

    if prediction == 1:
        st.success("Purchase intent detected — this session is classified as likely to purchase.")
        st.write("Suggested action: prioritize helpful support or a relevant offer; avoid unnecessary discounts.")
    else:
        st.info("No purchase intent detected at the selected threshold.")
        st.write("Suggested action: consider a gentle engagement message or product recommendations.")

    st.caption(
        "This is a model prediction, not a guarantee that the visitor will or will not purchase."
    )
