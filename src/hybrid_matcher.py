# src/hybrid_matcher.py
"""
Hybrid Scoring: semantic (cosine) + keyword (TF-IDF overlap) fusion
Outputs:
 - data/features/hybrid_scores.csv
 - plots/hybrid_distribution.png
 - MLflow artifacts & metrics

Usage:
    python src\hybrid_matcher.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

RESUME_PKL = "data/features/resume_embeddings.pkl"
JOB_PKL = "data/features/job_embeddings.pkl"
OUT_CSV = "data/features/hybrid_scores.csv"
PLOTS_DIR = "plots"
HIST_PATH = os.path.join(PLOTS_DIR, "hybrid_distribution.png")

# Tunable weights
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# number of top keywords to extract per doc
TOP_K_KEYWORDS = 10


def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    df = joblib.load(path)
    if "embedding" not in df.columns or "file" not in df.columns:
        raise ValueError(f"Embeddings dataframe must contain 'file' and 'embedding' columns: {path}")
    return df


def compute_semantic_matrix(jobs_df, resumes_df):
    job_vecs = np.stack(jobs_df["embedding"].values)
    resume_vecs = np.stack(resumes_df["embedding"].values)
    sim_mat = cosine_similarity(job_vecs, resume_vecs)  # shape: [n_jobs, n_resumes]
    return sim_mat


def extract_top_tfidf_keywords(corpus_texts, top_k=TOP_K_KEYWORDS):
    """
    Fit TF-IDF on corpus_texts (list of strings).
    Return list of top_k keywords for each document in the same order.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    tfidf = vectorizer.fit_transform(corpus_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_keywords_per_doc = []
    for i in range(tfidf.shape[0]):
        row = tfidf[i].toarray().ravel()
        if row.sum() == 0:
            top = []
        else:
            top_idx = row.argsort()[::-1][:top_k]
            top = feature_names[top_idx][row[top_idx] > 0].tolist()
        top_keywords_per_doc.append(top)
    return top_keywords_per_doc, vectorizer


def keyword_overlap_score(job_keywords, resume_keywords):
    """
    Compute ratio of common keywords over number of job keywords.
    If job_keywords is empty, returns 0.
    """
    job_set = set(job_keywords)
    if not job_set:
        return 0.0
    resume_set = set(resume_keywords)
    common = job_set.intersection(resume_set)
    return float(len(common)) / float(len(job_set))


def build_hybrid_scores(jobs_df, resumes_df, semantic_matrix, job_keywords_list, resume_keywords_list, top_n=5):
    """
    Return dataframe with rows for each job-resume pair containing:
    job_file, resume_file, semantic_score, keyword_score, hybrid_score
    """
    rows = []
    n_jobs, n_resumes = semantic_matrix.shape
    for i in range(n_jobs):
        for j in range(n_resumes):
            semantic_score = float(semantic_matrix[i, j])
            keyword_score = keyword_overlap_score(job_keywords_list[i], resume_keywords_list[j])
            hybrid_score = SEMANTIC_WEIGHT * semantic_score + KEYWORD_WEIGHT * keyword_score
            rows.append({
                "job_file": jobs_df.iloc[i]["file"],
                "resume_file": resumes_df.iloc[j]["file"],
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "hybrid_score": hybrid_score
            })
    df = pd.DataFrame(rows)
    # sort by job_file and descending hybrid_score
    df.sort_values(["job_file", "hybrid_score"], ascending=[True, False], inplace=True)
    return df


def save_and_log(df, hist_path=HIST_PATH):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Hybrid scores saved to {OUT_CSV}")

    # histogram of hybrid scores
    plt.figure(figsize=(8, 4))
    sns.histplot(df["hybrid_score"], bins=40, kde=True)
    plt.title("Distribution of Hybrid Scores")
    plt.xlabel("Hybrid score")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"✅ Saved hybrid score distribution plot to {hist_path}")

    # MLflow logging
    mlflow.set_experiment("resume_match_hybrid")
    with mlflow.start_run(run_name="hybrid_scoring"):
        mlflow.log_param("semantic_weight", SEMANTIC_WEIGHT)
        mlflow.log_param("keyword_weight", KEYWORD_WEIGHT)
        mlflow.log_metric("avg_semantic", float(df["semantic_score"].mean()))
        mlflow.log_metric("avg_keyword", float(df["keyword_score"].mean()))
        mlflow.log_metric("avg_hybrid", float(df["hybrid_score"].mean()))
        mlflow.log_artifact(OUT_CSV)
        mlflow.log_artifact(hist_path)
    print("📊 Logged artifacts & metrics to MLflow.")


def main():
    print("🔄 Loading embeddings...")
    resumes_df = load_embeddings(RESUME_PKL)
    jobs_df = load_embeddings(JOB_PKL)

    print(f" - resumes: {len(resumes_df)}  jobs: {len(jobs_df)}")

    print("🔢 Computing semantic similarity matrix...")
    sem_mat = compute_semantic_matrix(jobs_df, resumes_df)

    # create combined text corpus so TF-IDF vectorizer learns consistent feature space
    print("🔎 Extracting TF-IDF keywords (top K per doc)...")
    combined_texts = list(jobs_df["cleaned_text"].fillna("")) + list(resumes_df["cleaned_text"].fillna(""))
    all_keywords, vectorizer = extract_top_tfidf_keywords(combined_texts, top_k=TOP_K_KEYWORDS)

    job_keywords_list = all_keywords[:len(jobs_df)]
    resume_keywords_list = all_keywords[len(jobs_df):]

    print("⚖️ Building hybrid scores (semantic + keyword overlap)...")
    hybrid_df = build_hybrid_scores(jobs_df, resumes_df, sem_mat, job_keywords_list, resume_keywords_list)

    # Save and log
    save_and_log(hybrid_df)

    # Print top 5 matches for first few jobs as quick check
    for job in hybrid_df["job_file"].unique()[:3]:
        topk = hybrid_df[hybrid_df["job_file"] == job].head(5)
        print(f"\nTop matches for job: {job}")
        print(topk[["resume_file", "hybrid_score", "semantic_score", "keyword_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
