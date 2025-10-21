"""
evaluate_model.py (v2)
----------------------
Visualizes similarity matrix and top matches from match_results.csv.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import os

def evaluate_similarity(result_csv="data/features/match_results.csv"):
    df = pd.read_csv(result_csv)
    print("🔍 Top 10 matches:")
    print(df.head(10))

    pivot = df.pivot(index="job_file", columns="resume_file", values="similarity_score")

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="Blues", annot=False)
    plt.title("Resume–Job Similarity Heatmap")
    plt.tight_layout()
    plt.savefig("plots/similarity_heatmap.png")

    mlflow.set_experiment("resume_match_similarity")
    with mlflow.start_run(run_name="evaluation_similarity"):
        mlflow.log_artifact(result_csv)
        mlflow.log_artifact("plots/similarity_heatmap.png")
    print("✅ Logged similarity heatmap to MLflow.")

if __name__ == "__main__":
    evaluate_similarity()
