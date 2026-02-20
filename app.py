from flask import Flask, render_template, request, jsonify
import pandas as pd
from model_pipeline import run_pipeline, predict_text

app = Flask(__name__)

# ===============================
# Landing Page (Halaman Informasi)
# ===============================
@app.route('/') 
def home():
    return render_template('index.html')


# ===============================
# Halaman Pengujian Model
# ===============================
@app.route('/test-page')
def test_page():
    return render_template('test.html')


@app.route('/train', methods=['POST'])
def train():
    file = request.files['file']
    df = pd.read_csv(file)
    results = run_pipeline(df)
    return jsonify(results)


# ===============================
# Halaman Prediksi Teks
# ===============================
@app.route('/predict-page')
def predict_page():
    return render_template("predict.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    result = predict_text(text)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)