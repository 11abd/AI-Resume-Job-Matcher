"""
model_train.py (v2)
-------------------
Computes cosine similarity between resume and job embeddings.
Stores a ranked list of top matching resumes per job.
Logs results to MLflow.
"""

import os
import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(resume_pkl, job_pkl, output_csv="data/features/match_results.csv"):
    resumes = joblib.load(resume_pkl)
    jobs = joblib.load(job_pkl)

    resume_vecs = np.stack(resumes["embedding"].values)
    job_vecs = np.stack(jobs["embedding"].values)

    sim_matrix = cosine_similarity(job_vecs, resume_vecs)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results = []
    for i, job_row in enumerate(jobs.itertuples()):
        scores = sim_matrix[i]
        ranked_idx = np.argsort(scores)[::-1]
        top_resumes = [(resumes.iloc[j].file, float(scores[j])) for j in ranked_idx[:5]]
        for res_file, score in top_resumes:
            results.append({
                "job_file": job_row.file,
                "resume_file": res_file,
                "similarity_score": score
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved similarity results → {output_csv}")

    # Log to MLflow
    mlflow.set_experiment("resume_match_similarity")
    with mlflow.start_run(run_name="cosine_similarity_scores"):
        mlflow.log_artifact(output_csv)
        mlflow.log_param("num_jobs", len(jobs))
        mlflow.log_param("num_resumes", len(resumes))
        mlflow.log_metric("avg_similarity", df["similarity_score"].mean())

if __name__ == "__main__":
    compute_similarity(
        "data/features/resume_embeddings.pkl",
        "data/features/job_embeddings.pkl"
    )
