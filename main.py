
import streamlit as st
import joblib
import pandas as pd

# Load saved model and encoder
data = joblib.load("mobile_price_model.pkl")

model = data["model"]
encoder = data["encoder"]
feature_columns = data["feature_columns"]

st.title("📱 Mobile Price Predictor")
st.write("Enter mobile specifications to estimate its price.")

with st.form("mobile_form"):
    brand = st.selectbox(
        "Mobile Brand",
        encoder.classes_.tolist()
    )

    storage = st.selectbox(
    "Storage (GB)",
    options=[32, 64, 128, 256, 512, 1024, 2048],
    index=2
)
    ram = st.selectbox(
        "RAM",
        options=[4,6,8,12,16,24,64],
        index=2
    )
    
    screen_size = st.number_input(
        "Screen Size (inches)", min_value=3.0,
        max_value=15.0, value=6.5, step=0.5
    )

    camera = st.number_input(
        "Camera (MP)", min_value=1, max_value=300, value=50
    )

    battery = st.number_input(
        "Battery Capacity (mAh)", min_value=500,
        max_value=20000, value=5000 , step = 100
    )

    submitted = st.form_submit_button("Predict Price")

if submitted:
    brand_encoded = encoder.transform([brand])[0]

    input_data = pd.DataFrame([{
        "Brand": brand_encoded,
        "Storage": storage,
        "RAM": ram,
        "Screen Size (inches)": screen_size,
        "Camera (MP)": camera,
        "Battery Capacity (mAh)": battery
    }])

    # Match the exact feature order used during training
    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Mobile Price: ${prediction:,.2f}")
