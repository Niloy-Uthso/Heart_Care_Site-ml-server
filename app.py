from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# -------------------------------
# Load saved files
# -------------------------------
model = joblib.load("heart_model.pkl")
encoders = joblib.load("encoders.pkl")
threshold = joblib.load("threshold.pkl")

# -------------------------------
# Feature order (VERY IMPORTANT)
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

        # Encode categorical fields
        for col in encoders:
            if col in data:
                data[col] = encoders[col].transform([data[col]])[0]

        # Maintain correct order
        input_data = [data[f] for f in FEATURES]

        # Prediction
        prob = model.predict_proba([input_data])[0][1]
        pred = int(prob >= threshold)

        return jsonify({
            "prediction": pred,
            "probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)