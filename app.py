from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# -------------------------------
# Load saved files safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "threshold.pkl"))

# -------------------------------
# Feature order
# -------------------------------
FEATURES = [
    "BMI", "Smoking", "AlcoholDrinking", "Stroke",
    "PhysicalHealth", "MentalHealth", "DiffWalking",
    "Sex", "AgeCategory", "Race", "Diabetic",
    "PhysicalActivity", "GenHealth", "SleepTime",
    "Asthma", "KidneyDisease", "SkinCancer"
]

# -------------------------------
# Prediction route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        for col in encoders:
            if col in data:
                data[col] = encoders[col].transform([data[col]])[0]

        input_data = [data[f] for f in FEATURES]

        prob = model.predict_proba([input_data])[0][1]
        pred = int(prob >= threshold)

        return jsonify({
            "prediction": pred,
            "probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Local run
# -------------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)