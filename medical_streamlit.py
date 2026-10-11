import joblib
import pandas as pd
import streamlit as st

# 1. Load the trained pipeline once at startup
model = joblib.load('insurance_model.pkl')

st.title("Health Insurance Price Estimator")
st.write("Enter the details of the patient to get a real-time prediction of an premium insurance price.")

name = st.text_input("Enter your name:")
if name:
    st.success(f"Hello, {name}! Glad you are here.")

# 2. Input controls in two side-by-side columns
left, right = st.columns(2)

with left:

    age = st.number_input("age", min_value=0, max_value=120, value=40)

    bmi = st.number_input("bmi", min_value=0.0, max_value=70.0, value=25.0)

    children = st.number_input("Children",min_value=0, max_value=10, value=0)

    sex = st.selectbox("sex", ["Male", "Female"])
    region = st.selectbox("region", ["northeast", "northwest", "southeast", "southwest"])
    smoker = st.selectbox("smoker", ["yes", "no"])


if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "age": int(age),
        "bmi": float(bmi),
        "sex": sex,
        "children": int(children),
        "smoker": smoker,
        "region": region
    }])


import joblib
prediction = model.predict(input_df)[0] #cite 6,7
st.success(f"Estimated yearly premium: ${prediction:,.2f}")