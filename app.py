from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained models and encoders
vectorizer = joblib.load("vectorizer.joblib")
fault_model = joblib.load("fault_model.joblib")
subfault_model = joblib.load("subfault_model.joblib")
fault_encoder = joblib.load("fault_encoder.joblib")
subfault_encoder = joblib.load("subfault_encoder.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = []
    fault_counts = {}
    subfault_counts = {}

    if request.method == "POST":
        # Check if a file was uploaded
        file = request.files.get("csv_file")
        if file:
            df = pd.read_csv(file)
            if "DESCRIPTION" not in df.columns:
                return render_template("index.html", error="CSV must contain a 'DESCRIPTION' column.")

            texts = df["DESCRIPTION"].astype(str)
            vectors = vectorizer.transform(texts)

            fault_preds = fault_model.predict(vectors)
            subfault_preds = subfault_model.predict(vectors)

            decoded_faults = fault_encoder.inverse_transform(fault_preds)
            decoded_subfaults = subfault_encoder.inverse_transform(subfault_preds)

            for review, fault, subfault in zip(texts, decoded_faults, decoded_subfaults):
                predictions.append({
                    "review": review,
                    "fault": fault,
                    "subfault": subfault
                })

            fault_counts = pd.Series(decoded_faults).value_counts().to_dict()
            subfault_counts = pd.Series(decoded_subfaults).value_counts().to_dict()

    return render_template("index.html", predictions=predictions,
                           fault_counts=fault_counts,
                           subfault_counts=subfault_counts)

if __name__ == "__main__":
    app.run(debug=True)
