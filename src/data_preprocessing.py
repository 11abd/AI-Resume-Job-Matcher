"""
data_preprocessing.py (v2)
--------------------------
Processes resumes and job descriptions separately.
Extracts raw text, cleans it, and saves processed CSVs.
"""

import os
import re
import pandas as pd
import fitz  # PyMuPDF
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def process_folder(input_folder: str, output_csv: str):
    data = []
    for fname in os.listdir(input_folder):
        path = os.path.join(input_folder, fname)
        if not os.path.isfile(path):
            continue
        if fname.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(path)
        elif fname.lower().endswith(".txt"):
            raw = extract_text_from_txt(path)
        else:
            continue
        cleaned = clean_text(raw)
        data.append({"file": fname, "cleaned_text": cleaned})

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Processed {len(df)} files from {input_folder} → {output_csv}")

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    process_folder("data/raw/resumes", "data/processed/resumes_cleaned.csv")
    process_folder("data/raw/job_descriptions", "data/processed/jobs_cleaned.csv")
