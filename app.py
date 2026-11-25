import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Liver Disease Demo", page_icon="🧪")
artifact = joblib.load("liver_rf_model.joblib")
model = artifact["model"]
features = artifact["features"]

st.title("Liver Disease Classification Demo")
st.warning(
    "শিক্ষামূলক demonstration মাত্র—এটি medical diagnosis বা চিকিৎসকের পরামর্শ নয়।"
)

defaults = {
    "Alkphos Alkaline Phosphotase": 208.0,
    "Sgpt Alamine Aminotransferase": 35.0,
    "Sgot Aspartate Aminotransferase": 42.0,
    "Total Bilirubin": 1.0,
    "ALB Albumin": 3.1,
    "Total Protiens": 6.6,
    "A/G Ratio Albumin and Globulin Ratio": 0.93,
    "Direct Bilirubin": 0.3,
}

with st.form("patient_values"):
    values = {}
    for feature in features:
        values[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=float(defaults[feature]),
            step=0.1
        )
    submitted = st.form_submit_button("Run model")

if submitted:
    row = pd.DataFrame([[values[f] for f in features]], columns=features)
    prediction = int(model.predict(row)[0])
    disease_probability = float(model.predict_proba(row)[0, 1])

    if prediction == 1:
        st.error("Model prediction: Liver disease (class 1)")
    else:
        st.success("Model prediction: No liver disease (class 0)")

    st.metric("Model-estimated disease probability", f"{disease_probability:.1%}")
    st.caption("এই prediction বাস্তব রোগ নির্ণয়ের বিকল্প নয়।")
