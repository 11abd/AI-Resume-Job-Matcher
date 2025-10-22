"""
FastAPI backend for AI Resume Matcher
-------------------------------------
Endpoints:
  /health  -> quick status
  /match   -> POST JD text, return ranked resumes (semantic + keyword + hybrid)
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="AI Resume Matcher", version="1.0")

# ----- Load required components -----
# resume_df expected to be a DataFrame-like object saved with joblib that contains:
# columns: "embedding" (list/np.array), "cleaned_text" (str), "file" (filename)
resume_df = joblib.load("data/features/resume_embeddings.pkl")

# If embed vectors are stored as lists/arrays in a column "embedding"
resume_embeddings = np.stack(resume_df["embedding"].values)  # shape: (n_resumes, dim)
resume_texts = resume_df["cleaned_text"].fillna("").tolist()
resume_files = resume_df["file"].tolist()

# Use a stable sentence-transformer name
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Weights for hybrid score
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# ----- Pydantic schema -----
class JDRequest(BaseModel):
    job_text: str
    top_k: int = 5

# ----- Helper functions -----
def compute_keyword_overlap(job_text: str, resumes: list) -> np.ndarray:
    """
    Compute keyword similarity between job_text and each resume text.
    We fit a TF-IDF on [job_text] + resumes (so job tokens appear in vocab),
    then compute cosine similarity between the job vector and each resume vector.
    Returns an array of shape (n_resumes,) with values in [0,1] (normalized).
    """
    # If there are no resumes, return empty array
    if not resumes:
        return np.array([])

    corpus = [job_text] + resumes
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)  # shape: (1 + n_resumes, n_features)

    job_vec = tfidf_matrix[0]             # sparse row
    resume_matrix = tfidf_matrix[1:]      # sparse matrix (n_resumes x n_features)

    # cosine between job_vec (1 x n_features) and resume_matrix (n_resumes x n_features)
    sims = cosine_similarity(resume_matrix, job_vec).ravel()  # shape: (n_resumes,)

    # cosine_similarity returns values in [-1,1], but TF-IDF cosine should be [0,1]. Clip/normalize for safety.
    sims = np.clip(sims, 0.0, 1.0)

    # If all zeros, avoid division by zero when normalizing: just return sims
    if sims.max() > 0:
        sims = sims / sims.max()  # normalize to [0,1] relative scale (keeps comparability)
    return sims

# ----- Routes -----
@app.get("/health")
def health():
    return {"status": "API running ✅"}

@app.post("/match")
def match(request: JDRequest):
    job_text = request.job_text or ""
    top_k = max(1, int(request.top_k or 5))

    # Semantic similarity
    # jd_emb: shape (1, dim)
    jd_emb = embedder.encode([job_text])
    semantic_sim = cosine_similarity(jd_emb, resume_embeddings).ravel()  # shape: (n_resumes,)

    # Keyword similarity (TF-IDF on-the-fly)
    keyword_sim = compute_keyword_overlap(job_text, resume_texts)
    # If keyword_sim empty (no resumes), fill zeros to match shapes
    if keyword_sim.size == 0:
        keyword_sim = np.zeros_like(semantic_sim)

    # Hybrid score
    hybrid_score = SEMANTIC_WEIGHT * semantic_sim + KEYWORD_WEIGHT * keyword_sim

    # Rank resumes
    top_idx = np.argsort(hybrid_score)[::-1][:top_k]
    top_results = []
    for i in top_idx:
        top_results.append({
            "resume_file": resume_files[i],
            "semantic_score": float(np.round(semantic_sim[i], 4)),
            "keyword_score": float(np.round(keyword_sim[i], 4)),
            "hybrid_score": float(np.round(hybrid_score[i], 4))
        })

    return {
        "job_summary": (job_text[:150] + "...") if len(job_text) > 150 else job_text,
        "top_resumes": top_results,
    }
