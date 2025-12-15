import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Prediction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("🚢 Titanic Survival Prediction (RF Model)")

st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 80, 30)
sibsp = st.sidebar.slider("Siblings/Spouses", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0.0, 600.0, 50.0)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

sex = 1 if sex == "Male" else 0
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][pred]

    if pred == 1:
        st.success(f"🟢 Survived (Confidence: {prob:.2f})")
    else:
        st.error(f"🔴 Not Survived (Confidence: {prob:.2f})")

    st.write(f"Passenger Class: {pclass}")
    st.write(f"Sex: {sex}")
    st.write(f"Age: {age}")
    st.write(f"Siblings/Spouses: {sibsp}")
    st.write(f"Parents/Children: {parch}")
    st.write(f"Fare: {fare}")
    st.write(f"Embarked: {embarked}")