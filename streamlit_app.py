# streamlit_app.py
# Streamlit app for Tehran house price estimation
# Assumes there's a saved joblib model pipeline named like 'best_model_*.joblib'

import streamlit as st
import joblib
import pandas as pd
import glob
import os
import numpy as np

st.set_page_config(page_title="Tehran House Price Estimator", layout="centered")
st.title("Tehran House Price لEstimator")
st.write("This app uses a saved (joblib) model to predict the estimated price of a house.")

# --- Find model file ---
model_files = glob.glob("best_model_*.joblib")
if len(model_files) == 0:
    st.error("The model file was not found. Please place the file **best_model_<ModelName>.joblib** in the folder.")
    st.stop()

model_file = model_files[0]
st.sidebar.write(f"Loading model: `{os.path.basename(model_file)}`")

# Load model
try:
    model = joblib.load(model_file)
except Exception as e:
    st.error(f"Moddel loading error: {e}")
    st.stop()

# Try to read CSV to populate Area choices
csv_path = "tehran_housing.csv"
area_options = None
if os.path.exists(csv_path):
    try:
        df_full = pd.read_csv(csv_path)
        if 'Area' in df_full.columns:
            area_options = sorted(df_full['Area'].dropna().unique())
    except Exception:
        area_options = None

st.sidebar.header("Input settings")

# Inputs
st.write("House details")
if area_options:
    area = st.selectbox(" (Area)", options=area_options)
else:
    area = st.text_input(" (Area)")

area_numeric = st.number_input(" (m²)", min_value=1.0, max_value=10000.0, value=50.0, step=1.0)
room = st.number_input("Number of rooms", min_value=0, max_value=10, value=2)
parking = st.checkbox("Parking")
warehouse = st.checkbox("Storage room")
elevator = st.checkbox("Elevator")

# District category (if you used District_cat during training)
district_choices = ['missing','1-5','6-10','11-20','21+']
district_cat = st.selectbox("District category (Optional)", options=district_choices, index=0)

# If model was trained on log(Price), allow inverse transform
log_target = st.checkbox("The model was trained on log(price) — apply expm1 to the output.", value=False)

if st.button("Price estimate"):
    # Construct input dataframe matching expected training columns
    input_df = pd.DataFrame([{ 
        'Room': int(room),
        'Area_numeric': float(area_numeric),
        'Parking': int(parking),
        'Warehouse': int(warehouse),
        'Elevator': int(elevator),
        'Area': area if area is not None else 'missing',
        'District_cat': district_cat
    }])

    # Predict
    try:
        pred = model.predict(input_df)
        # If pipeline returns array
        if isinstance(pred, (list, tuple, np.ndarray)):
            pred_val = float(pred[0])
        else:
            pred_val = float(pred)

        if log_target:
            pred_val = np.expm1(pred_val)

        st.success(f"Estimated price : {pred_val:,.0f}")

    except Exception as e:
        st.error(f" Prediction error : {e}")

st.markdown("---")
st.write("The model file must be named best_model_<ModelName>.joblib and placed in the same folder.")
st.write("If you used a log transform on the target during training, check the log(price) option so the output is converted back to its real value.")
