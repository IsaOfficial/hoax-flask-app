import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)

from preprocessing import preprocess


# Load artifacts sekali saat server start
BASE_DIR = Path(__file__).resolve().parent
tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
model_baseline = joblib.load(BASE_DIR / "xgboost_baseline_model.pkl")
model_ros = joblib.load(BASE_DIR / "xgboost_ros_model.pkl")


def evaluate(model, X_tfidf, y_true):
    y_pred = model.predict(X_tfidf)
    y_proba = model.predict_proba(X_tfidf)[:, 1]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Metrik dasar
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # F1 tambahan
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # False Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    # PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        },
        "pr_curve": {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist()
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            }
        }
    }

def predict_text(text):

    text_clean = preprocess(text)
    X = tfidf.transform([text_clean])

    y_pred = model_ros.predict(X)[0]
    y_proba = model_ros.predict_proba(X)[0][1]

    return {
        "prediction": int(y_pred),
        "probability": float(y_proba)
    }

def run_pipeline(df):

    if 'content_id' not in df.columns or 'label' not in df.columns:
        return {"error": "Dataset harus memiliki kolom content_id dan label"}

    df['text_clean'] = df['content_id'].astype(str).apply(preprocess)

    X = tfidf.transform(df['text_clean'])
    y = df['label']

    results = {
        "baseline": evaluate(model_baseline, X, y),
        "ros": evaluate(model_ros, X, y)
    }

    return results
