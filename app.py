from flask import Flask, render_template, request, jsonify
import joblib
import json
import os
from pathlib import Path
from preprocessing import preprocess

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ===============================
# Load Model & Evaluasi (sekali saat startup)
# ===============================

tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
model_ros = joblib.load(MODEL_DIR / "xgboost_ros_model.pkl")

with (BASE_DIR / "metrics.json").open(encoding="utf-8") as f:
    metrics = json.load(f)
    # Konversi list -> dictionary terstruktur
    metrics_dict = {
        "baseline": metrics[0],
        "ros": metrics[1]
    }

with (BASE_DIR / "roc_data.json").open(encoding="utf-8") as f:
    roc_data = json.load(f)

with (BASE_DIR / "pr_data.json").open(encoding="utf-8") as f:
    pr_data = json.load(f)

with (BASE_DIR / "conf_matrix.json").open(encoding="utf-8") as f:
    cm_data = json.load(f)

# ===============================
# Landing Page (Evaluasi Model)
# ===============================
@app.route("/")
def home():
    return render_template(
        "index.html",
        metrics=metrics_dict,
        roc_data=roc_data,
        pr_data=pr_data,
        cm_data=cm_data
    )


# ===============================
# Halaman Prediksi
# ===============================
@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Body harus berupa objek JSON yang valid"}), 400

    if "text" not in data:
        return jsonify({"error": "Field text wajib diisi"}), 400
    text = data["text"]
    if not isinstance(text, str):
        return jsonify({"error": "Teks harus berupa string"}), 400

    text = text.strip()
    if not text:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    # Gunakan preprocessing yang sama dengan pipeline evaluasi.
    text_clean = preprocess(text)
    if not text_clean:
        return jsonify({"error": "Teks tidak mengandung kata yang dapat dianalisis setelah dibersihkan"}), 400
    vector = tfidf.transform([text_clean])

    # Gunakan model ROS sebagai model final penelitian
    prediction = model_ros.predict(vector)[0]
    probability = model_ros.predict_proba(vector)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
# if __name__ == "__main__":
#     app.run(debug=True)
