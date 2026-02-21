from flask import Flask, render_template, request, jsonify
import joblib
import json
import os

app = Flask(__name__)

# ===============================
# Load Model & Evaluasi (sekali saat startup)
# ===============================

tfidf = joblib.load("tfidf_vectorizer.pkl")
model_baseline = joblib.load("xgboost_baseline_model.pkl")
model_ros = joblib.load("xgboost_ros_model.pkl")

with open("metrics.json") as f:
    metrics = json.load(f)
    # Konversi list -> dictionary terstruktur
    metrics_dict = {
        "baseline": metrics[0],
        "ros": metrics[1]
    }

with open("roc_data.json") as f:
    roc_data = json.load(f)

with open("pr_data.json") as f:
    pr_data = json.load(f)

with open("conf_matrix.json") as f:
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
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    # Transform TF-IDF
    vector = tfidf.transform([text])

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