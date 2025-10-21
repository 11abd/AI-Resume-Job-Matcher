"""
explainability.py
-----------------
Explain hybrid resume–JD similarity scores using SHAP.
Generates:
  - shap_summary.png
  - shap_force_plot.html
  - MLflow artifacts
"""

import os
import pandas as pd
import numpy as np
import mlflow
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from joblib import dump
import joblib

DATA_PATH = "data/features/hybrid_scores.csv"
PLOTS_DIR = "plots"
SUMMARY_PNG = os.path.join(PLOTS_DIR, "shap_summary.png")
FORCE_HTML = os.path.join(PLOTS_DIR, "shap_force_plot.html")

def build_shap_explainability():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ hybrid_scores.csv not found. Run hybrid_matcher.py first.")
    
    df = pd.read_csv(DATA_PATH)
    print(f"🔍 Loaded {len(df)} rows from {DATA_PATH}")

    # Combine resume & job texts for vectorization
    print("🧠 Combining resume and job texts for TF-IDF features...")
    resumes = joblib.load("data/features/resume_embeddings.pkl")
    jobs = joblib.load("data/features/job_embeddings.pkl")

    # Map text back into the hybrid dataframe
    resume_texts = dict(zip(resumes["file"], resumes["cleaned_text"]))
    job_texts = dict(zip(jobs["file"], jobs["cleaned_text"]))

    df["resume_text"] = df["resume_file"].map(resume_texts)
    df["job_text"] = df["job_file"].map(job_texts)
    df["combined_text"] = df["job_text"].fillna("") + " " + df["resume_text"].fillna("")

    print("🧩 Vectorizing combined text...")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vectorizer.fit_transform(df["combined_text"])
    y = df["hybrid_score"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print("🚀 Training linear regression surrogate model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"✅ Model trained — example predictions range: {preds.min():.3f} to {preds.max():.3f}")

    # Explain with SHAP
    print("🔦 Generating SHAP explanations...")
    explainer = shap.Explainer(model, X_train, feature_names=vectorizer.get_feature_names_out())
    shap_values = explainer(X_test[:200])  # sample subset

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Summary plot (global importance)
    shap.summary_plot(shap_values, X_test[:200], show=False)
    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved global SHAP summary plot → {SUMMARY_PNG}")

    # Force plot (example local explanation)
    sample_idx = min(0, len(shap_values) - 1)
    shap_html = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx].values,
        feature_names=vectorizer.get_feature_names_out(),
        matplotlib=False
    )
    shap.save_html(FORCE_HTML, shap_html)
    print(f"✅ Saved example force plot → {FORCE_HTML}")

    # Save model + vectorizer (for reuse in API)
    os.makedirs("models", exist_ok=True)
    dump(model, "models/shap_surrogate_model.pkl")
    dump(vectorizer, "models/tfidf_vectorizer.pkl")

    # Log to MLflow
    mlflow.set_experiment("resume_match_explainability")
    with mlflow.start_run(run_name="shap_explainer"):
        mlflow.log_artifact(SUMMARY_PNG)
        mlflow.log_artifact(FORCE_HTML)
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("mean_pred", float(np.mean(preds)))
        mlflow.log_metric("std_pred", float(np.std(preds)))
    print("📊 Logged SHAP artifacts to MLflow.")

if __name__ == "__main__":
    build_shap_explainability()
