"""
FastAPI backend for AI Resume Matcher
-------------------------------------
Endpoints:
  /health  -> quick status
  /match   -> POST JD text, return ranked resumes + explainability
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import shap 

app = FastAPI(title="AI Resume Matcher", version="1.0")

# ----- Load required components -----
resume_df = joblib.load("data/features/resume_embeddings.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
shap_model = joblib.load("models/shap_surrogate_model.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-compute resume text and embeddings
resume_embeddings = np.stack(resume_df["embedding"].values)
resume_texts = resume_df["cleaned_text"].fillna("").tolist()
resume_files = resume_df["file"].tolist()

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# ----- Pydantic schema -----
class JDRequest(BaseModel):
    job_text: str
    top_k: int = 5

# ----- Helper functions -----
def compute_keyword_overlap(job_text, resumes):
    """Compute keyword overlap using TF-IDF."""
    corpus = [job_text] + resumes
    tfidf = vectorizer.transform(corpus)
    job_vec = tfidf[0]
    sim = (tfidf[1:] @ job_vec.T).toarray().ravel()
    return sim / sim.max() if sim.max() != 0 else sim


def explain_keywords(job_text):
    """Use SHAP surrogate model to get top influencing keywords."""
    X = vectorizer.transform([job_text])
    X_dense = X.toarray()

    # Define callable model
    def model_predict(X_local):
        return shap_model.predict(X_local)

    # ✅ Define masker using the dense data shape
    masker = shap.maskers.Independent(X_dense, max_samples=100)

    # ✅ Create the SHAP explainer with the masker and model
    explainer = shap.Explainer(model_predict, masker, feature_names=vectorizer.get_feature_names_out())

    # Compute SHAP values
    shap_values = explainer(X_dense)
    vals = shap_values[0].values
    features = vectorizer.get_feature_names_out()

    # Top 5 positive contributors
    top_idx = np.argsort(vals)[::-1][:5]
    top_pos = [features[i] for i in top_idx][:5]
    return top_pos



# ----- Routes -----
@app.get("/health")
def health():
    return {"status": "API running ✅"}

@app.post("/match")
def match(request: JDRequest):
    job_text = request.job_text
    top_k = request.top_k

    # Compute JD embedding
    jd_emb = embedder.encode([job_text])
    semantic_sim = cosine_similarity(jd_emb, resume_embeddings).ravel()

    # Compute keyword similarity
    keyword_sim = compute_keyword_overlap(job_text, resume_texts)

    # Hybrid score
    hybrid_score = SEMANTIC_WEIGHT * semantic_sim + KEYWORD_WEIGHT * keyword_sim

    # Rank resumes
    top_idx = np.argsort(hybrid_score)[::-1][:top_k]
    top_results = []
    for i in top_idx:
        top_results.append({
            "resume_file": resume_files[i],
            "semantic_score": float(semantic_sim[i]),
            "keyword_score": float(keyword_sim[i]),
            "hybrid_score": float(hybrid_score[i])
        })

    # Explain JD features
    top_keywords = explain_keywords(job_text)

    return {
        "job_summary": job_text[:150] + "...",
        "top_resumes": top_results,
        "influential_keywords": top_keywords
    }
