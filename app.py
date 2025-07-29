import os
import pandas as pd
import joblib
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Load models and encoders
vectorizer = joblib.load("vectorizer.joblib")
fault_model = joblib.load("fault_model.joblib")
subfault_model = joblib.load("subfault_model.joblib")
fault_encoder = joblib.load("fault_encoder.joblib")
subfault_encoder = joblib.load("subfault_encoder.joblib")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    if request.method == "POST":
        review = request.form["review"]
        if review.strip():
            vec = vectorizer.transform([review])
            fault_pred = fault_model.predict(vec)
            subfault_pred = subfault_model.predict(vec)

            fault_label = fault_encoder.inverse_transform(fault_pred)[0]
            subfault_label = subfault_encoder.inverse_transform(subfault_pred)[0]

            predictions = {
                "fault": fault_label,
                "subfault": subfault_label
            }

    return render_template("index.html", predictions=predictions)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # For Render compatibility
    app.run(host="0.0.0.0", port=port, debug=True)
