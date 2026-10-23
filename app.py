from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from datetime import date

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "car_price_model.joblib"
DATA_PATH = ROOT / "dataset" / "car_data.csv"

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Used Car Price Prediction")
st.write("Enter the car details below to estimate its resale price.")
st.info("Demo project: the included dataset is synthetic sample data. The prediction is for demonstration only and is not a real market valuation.")

if not MODEL_PATH.exists():
    st.warning("The model file is missing. Run `python train_model.py` first, then restart this website.")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

brands = sorted(df["brand"].dropna().unique().tolist())
fuels = sorted(df["fuel"].dropna().unique().tolist())
transmissions = sorted(df["transmission"].dropna().unique().tolist())

with st.form("car_details"):
    st.subheader("Car Details")
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Car brand", brands)
        year = st.number_input("Manufacturing year", min_value=1990, max_value=date.today().year, value=2021, step=1)
        km_driven = st.number_input("Kilometres driven", min_value=0, max_value=1_000_000, value=35000, step=1000)
    with col2:
        fuel = st.selectbox("Fuel type", fuels)
        transmission = st.selectbox("Transmission", transmissions)
        owner = st.selectbox("Previous owner number", [1, 2, 3], index=0)
    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "brand": brand,
        "year": int(year),
        "km_driven": int(km_driven),
        "fuel": fuel,
        "transmission": transmission,
        "owner": int(owner)
    }])
    predicted_lakh = float(model.predict(input_data)[0])
    predicted_rupees = max(0, predicted_lakh * 100000)
    st.success("Estimated price (demo)")
    st.metric("Predicted resale price", f"₹{predicted_rupees:,.0f}")
    st.caption("This is a model estimate from synthetic demonstration data. Actual prices depend on condition, location, service history, variant and current market demand.")

with st.expander("About this project"):
    st.write("This mini project uses a Random Forest Regression model. Categorical fields are encoded, the model is trained using a train/test split, and performance is evaluated using MAE and R².")
    st.write("Replace the demo CSV with a real dataset before making real-world claims.")
