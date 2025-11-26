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
st.title("تخمین قیمت خانه — تهران")
st.write("این اپ از یک مدل ذخیره‌شده (joblib) استفاده می‌کند تا قیمت تخمینی خانه را پیش‌بینی کند.")

# --- Find model file ---
model_files = glob.glob("best_model_*.joblib")
if len(model_files) == 0:
    st.error("فایل مدل پیدا نشد. لطفاً فایل best_model_<ModelName>.joblib را در پوشه قرار بدهید.")
    st.stop()

model_file = model_files[0]
st.sidebar.write(f"بارگذاری مدل: `{os.path.basename(model_file)}`")

# Load model
try:
    model = joblib.load(model_file)
except Exception as e:
    st.error(f"خطا در بارگذاری مدل: {e}")
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

st.sidebar.header("تنظیمات ورودی")

# Inputs
st.write("### مشخصات ملک")
if area_options:
    area = st.selectbox("منطقه (Area)", options=area_options)
else:
    area = st.text_input("منطقه (Area)")

area_numeric = st.number_input("متراژ (m²)", min_value=1.0, max_value=10000.0, value=50.0, step=1.0)
room = st.number_input("تعداد اتاق", min_value=0, max_value=10, value=2)
parking = st.checkbox("پارکینگ")
warehouse = st.checkbox("انباری")
elevator = st.checkbox("آسانسور")

# District category (if you used District_cat during training)
district_choices = ['missing','1-5','6-10','11-20','21+']
district_cat = st.selectbox("District category (اختیاری)", options=district_choices, index=0)

# If model was trained on log(Price), allow inverse transform
log_target = st.checkbox("مدل روی log(price) آموزش دیده است — خروجی را expm1 کن", value=False)

if st.button("تخمین قیمت"):
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

        st.success(f"قیمت تخمینی: {pred_val:,.0f}")
        st.write("(واحد را متناسب با دیتاست خودتان بررسی کنید؛ ممکن است تومان یا دلار باشد)")

    except Exception as e:
        st.error(f"خطا در پیش‌بینی: {e}")

st.markdown("---")
st.write("راهنما:\n- فایل مدل باید با نام `best_model_<ModelName>.joblib` در همان پوشه باشد.\n- اگر در زمان آموزش هدف را لاگ ترنسفورم کردید، تیک گزینه `log(price)` را بزنید تا خروجی به حالت واقعی برگردد.")
