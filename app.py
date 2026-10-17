import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Prediction System", page_icon="❤️")


CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
MAX_HEART_RATE = ["Low", "Medium", "High"]
ST_DEPRESSION = ["Zero", "Low", "High"]
EXERCISE_ANGINA = ["No", "Yes"]


def categorize_max_heart_rate(value):
    if value < 130:
        return "Low"
    elif value < 160:
        return "Medium"
    return "High"


def categorize_st_depression(value):
    if value <= 0:
        return "Zero"
    elif value <= 1.5:
        return "Low"
    return "High"


def make_encoder(categories):
    encoder = LabelEncoder()
    encoder.fit(np.ravel(categories))
    return encoder


chest_pain_encoder = make_encoder(CHEST_PAIN)
max_heart_rate_encoder = make_encoder(MAX_HEART_RATE)
st_depression_encoder = make_encoder(ST_DEPRESSION)
exercise_angina_encoder = make_encoder(EXERCISE_ANGINA)


@st.cache_resource
def load_model():
    with open("svc_trained_model.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()

st.title(" Heart Disease Prediction System")

chest_pain = st.selectbox("Chest pain type", CHEST_PAIN)
max_heart_rate = st.number_input("Maximum heart rate achieved (bpm)", min_value=60, max_value=220, value=150, step=1)
st_depression = st.number_input("ST depression induced by exercise (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
exercise_angina = st.radio("Exercise-induced angina?", EXERCISE_ANGINA, horizontal=True)

if st.button("Predict", type="primary"):

    user_input = pd.DataFrame({
        "ChestPain": [chest_pain],
        "MaxHeartRate": [categorize_max_heart_rate(max_heart_rate)],
        "STDepression": [categorize_st_depression(st_depression)],
        "ExerciseAngina": [exercise_angina],
    })

    features = pd.DataFrame({
        "ChestPain": chest_pain_encoder.transform(user_input["ChestPain"]),
        "MaxHeartRate": max_heart_rate_encoder.transform(user_input["MaxHeartRate"]),
        "STDepression": st_depression_encoder.transform(user_input["STDepression"]),
        "ExerciseAngina": exercise_angina_encoder.transform(user_input["ExerciseAngina"]),
    })

    result = model.predict(features)[0]
    st.write("Grouped input:", user_input.iloc[0].to_dict())
    if result == 1:
        st.error("HEART DISEASE LIKELY. Please consult a doctor.")
    else:
        st.success("NO HEART DISEASE predicted.")
