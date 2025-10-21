"""
feature_engineering.py (v2)
---------------------------
Generate embeddings separately for resumes and job descriptions.
"""

import pandas as pd
import joblib
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def generate_embeddings(input_csv: str, output_path: str):
    df = pd.read_csv(input_csv)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"🔄 Generating embeddings for {len(df)} documents from {input_csv}")

    embeddings = []
    for text in tqdm(df["cleaned_text"], desc=f"Encoding {os.path.basename(input_csv)}"):
        emb = model.encode(text)
        embeddings.append(emb)

    df["embedding"] = embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(df, output_path)
    print(f"✅ Saved embeddings to {output_path}")

if __name__ == "__main__":
    generate_embeddings("data/processed/resumes_cleaned.csv", "data/features/resume_embeddings.pkl")
    generate_embeddings("data/processed/jobs_cleaned.csv", "data/features/job_embeddings.pkl")
