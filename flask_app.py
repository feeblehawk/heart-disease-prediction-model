
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# PythonAnywhere does not run from this folder, so build the model path from this file's location
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svc_trained_model.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ---- same fixed categories as Step 5.1 of the notebook ----
CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
MAX_HEART_RATE = ["Low", "Medium", "High"]
ST_DEPRESSION = ["Zero", "Low", "High"]
EXERCISE_ANGINA = ["No", "Yes"]


def make_encoder(categories):
    encoder = LabelEncoder()
    encoder.fit(np.ravel(categories))
    return encoder


chest_pain_encoder = make_encoder(CHEST_PAIN)
max_heart_rate_encoder = make_encoder(MAX_HEART_RATE)
st_depression_encoder = make_encoder(ST_DEPRESSION)
exercise_angina_encoder = make_encoder(EXERCISE_ANGINA)


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


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Heart Disease Prediction System</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f4f6f8; margin:0; padding:24px; }
    .card { max-width:480px; margin:auto; background:#fff; padding:24px; border-radius:10px;
            box-shadow:0 2px 8px rgba(0,0,0,.1); }
    h2 { margin-top:0; }
    label { display:block; margin-top:14px; font-weight:bold; }
    select, input { width:100%; padding:8px; margin-top:6px; box-sizing:border-box; }
    button { margin-top:20px; width:100%; padding:10px; background:#c0392b; color:#fff;
             border:0; border-radius:6px; font-size:16px; cursor:pointer; }
    .result { margin-top:20px; padding:14px; border-radius:6px; font-weight:bold; }
    .high { background:#fdecea; color:#a61e11; }
    .low { background:#e8f6ec; color:#1e7a3a; }
    .error { background:#fff4e5; color:#8a5300; }
    small { color:#666; }
  </style>
</head>
<body>
  <div class="card">
    <h2>&#10084; Heart Disease Prediction System</h2>
    <small>AIC354 Assignment 2. A learning demo, not medical advice.</small>
    <form method="post">
      <label>Chest pain type</label>
      <select name="chest_pain">
        {% for o in chest_pain %}<option {{ 'selected' if o == form.chest_pain }}>{{ o }}</option>{% endfor %}
      </select>

      <label>Maximum heart rate achieved (bpm)</label>
      <input name="max_heart_rate" type="number" min="60" max="220" step="1" value="{{ form.max_heart_rate }}" required>

      <label>ST depression induced by exercise (oldpeak)</label>
      <input name="st_depression" type="number" min="0" max="7" step="0.1" value="{{ form.st_depression }}" required>

      <label>Exercise-induced angina?</label>
      <select name="exercise_angina">
        {% for o in exercise_angina %}<option {{ 'selected' if o == form.exercise_angina }}>{{ o }}</option>{% endfor %}
      </select>

      <button type="submit">Predict</button>
    </form>

    {% if error %}<div class="result error">{{ error }}</div>{% endif %}
    {% if result %}
      <div class="result {{ css }}">{{ result }}</div>
      <small>Grouped input: {{ grouped }}</small>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    form = {"chest_pain": "Asymptomatic", "max_heart_rate": 150,
            "st_depression": 1.0, "exercise_angina": "No"}
    result = css = grouped = error = None

    if request.method == "POST":
        form.update({k: request.form.get(k, form[k]) for k in form})
        try:
            max_hr = float(form["max_heart_rate"])
            st_dep = float(form["st_depression"])
            if form["chest_pain"] not in CHEST_PAIN or form["exercise_angina"] not in EXERCISE_ANGINA:
                raise ValueError
            if not (60 <= max_hr <= 220 and 0 <= st_dep <= 7):
                raise ValueError

            # Step 8.2: convert input into the same feature vector as the sample data
            user_input = pd.DataFrame({
                "ChestPain": [form["chest_pain"]],
                "MaxHeartRate": [categorize_max_heart_rate(max_hr)],
                "STDepression": [categorize_st_depression(st_dep)],
                "ExerciseAngina": [form["exercise_angina"]],
            })
            # Step 8.3: label encode the feature vector
            features = pd.DataFrame({
                "ChestPain": chest_pain_encoder.transform(user_input["ChestPain"]),
                "MaxHeartRate": max_heart_rate_encoder.transform(user_input["MaxHeartRate"]),
                "STDepression": st_depression_encoder.transform(user_input["STDepression"]),
                "ExerciseAngina": exercise_angina_encoder.transform(user_input["ExerciseAngina"]),
            })
            # Step 8.5: predict
            if model.predict(features)[0] == 1:
                result, css = "HEART DISEASE LIKELY. Please consult a doctor.", "high"
            else:
                result, css = "NO HEART DISEASE predicted.", "low"
            grouped = ", ".join(f"{k}: {v[0]}" for k, v in user_input.items())
        except ValueError:
            error = "Please enter valid values (heart rate 60-220, ST depression 0-7)."

    return render_template_string(PAGE, chest_pain=CHEST_PAIN, exercise_angina=EXERCISE_ANGINA,
                                  form=form, result=result, css=css, grouped=grouped, error=error)


# NOTE: do NOT call app.run() here. PythonAnywhere serves `app` through its WSGI file.