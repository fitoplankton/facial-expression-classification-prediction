# app.py
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from utils import extract_features
import pickle
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Mapping ekspresi berdasarkan urutan label model
emotion_labels = ['NEUTRAL', 'HAPPY', 'ANGRY', 'DISGUST', 'FEAR', 'SADNESS', 'SURPRISE']

# Load semua model
models = {
    "hog8x8": pickle.load(open("new_models/HOG_8x8_linear.pkl", "rb")),
    "hog16x16": pickle.load(open("new_models/HOG_16x16_linear.pkl", "rb")),
    "hog8x8_pca": pickle.load(open("new_models/HOG_8x8_PCA_linear.pkl", "rb")),
    "hog16x16_pca": pickle.load(open("new_models/HOG_16x16_PCA_linear.pkl", "rb")),
    "odlbp": pickle.load(open("new_models/OD-LBP_linear.pkl", "rb")),
    "odlbp_pca": pickle.load(open("new_models/OD-LBP_PCA_linear.pkl", "rb")),
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    method = None
    image_path = None

    if request.method == "POST":
        method = request.form["method"]
        img = request.files["image"]
        filename = secure_filename(img.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        img.save(filepath)
        image_path = filepath

        # Ekstrak fitur
        features = extract_features(filepath, method)
        features = np.array([features])

        # Prediksi
        model = models[method]
        label_index = model.predict(features)[0]
        prediction = emotion_labels[label_index]

        # Confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = np.max(proba) * 100
        else:
            confidence = 100.0

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        method=method,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
