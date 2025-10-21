# src/api/upload_and_match.py
"""
Upgraded dynamic Upload-and-Match API with skill-aware TF-IDF + spaCy tokenizer,
skill coverage metric, and improved hybrid scoring.

Endpoint:
  POST /upload_and_match (multipart/form-data)
    - files: up to 10 resume files (pdf/txt)
    - job_text: job description text (form field)

Output (minimal):
  {
    "results": [
      {"resume_file": "r1.pdf", "hybrid_score": 0.8234},
      ...
    ],
    "explainability": {
      "r1.pdf": ["python","aws","docker"],
      ...
    }
  }
"""

import os
import time
import json
import re
import traceback
from typing import List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import fitz  # PyMuPDF for PDF extraction
import spacy

# -----------------------
# CONFIG
# -----------------------
MAX_FILES = 10
MAX_SIZE_MB = 5
SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.3
SKILL_COVERAGE_WEIGHT = 0.1
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")  # change via env if needed
TOP_EXPLAIN = 5
MAX_TFIDF_FEATURES = 5000

# Load spaCy once
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception as e:
    # If spaCy model missing, raise a helpful error
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm") from e

# Load embedder globally
EMBEDDER = SentenceTransformer(MODEL_NAME)

app = FastAPI(title="Skill-aware Dynamic Resume Matcher")

# -----------------------
# Utilities
# -----------------------

def extract_text_from_pdf(path: str) -> str:
    try:
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                text.append(page.get_text("text"))
        return " ".join(text).strip()
    except Exception:
        return ""

def basic_clean(text: str) -> str:
    t = text.lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"\S+@\S+", " ", t)
    t = re.sub(r"[^a-zA-Z0-9\s\+\#\-\_\.]", " ", t)  # keep hashtags, dots etc for tools like C++ or c#
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def spacy_tokenizer(text: str) -> List[str]:
    """
    Return tokens focusing on nouns/proper nouns/compound nouns and skill-like tokens
    """
    doc = nlp(text)
    tokens = []
    for tok in doc:
        # take nouns, proper nouns, and compounds/adjectives that often form skill phrases
        if tok.pos_ in {"NOUN", "PROPN", "ADJ"}:
            tok_text = tok.lemma_.lower().strip()
            if len(tok_text) > 1 and not tok_text.isdigit():
                tokens.append(tok_text)
    return tokens

# Skill lexicon: if skills.txt exists in repo root, load it (one skill per line),
# otherwise use a compact curated fallback list.
def load_skill_lexicon(path="skills.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            skills = [line.strip().lower() for line in f if line.strip()]
            return set(skills)
    # fallback compact list (extend as you like)
    fallback = [
        "python","java","c++","c#","javascript","aws","azure","gcp","docker","kubernetes",
        "sql","nosql","postgres","mysql","react","angular","node","tensorflow","pytorch",
        "git","jira","confluence","excel","powerbi","tableau","linux","bash","rest","api",
        "adobe","figma","photoshop","illustrator","ux","ui","scrum","agile","itil","helpdesk",
        "active directory","m365","salesforce","sap","devops","ci/cd"
    ]
    return set(fallback)

SKILL_LEXICON = load_skill_lexicon()

def extract_skills_from_text(text: str):
    """
    Heuristic skill extraction: intersection with skill lexicon + noun/proper tokens
    """
    cleaned = basic_clean(text)
    tokens = spacy_tokenizer(cleaned)
    found = set()
    for s in SKILL_LEXICON:
        # simple substring match (skills can be multi-word)
        if s in cleaned:
            found.add(s)
    # also include tokens that are in the lexicon
    for tok in tokens:
        if tok in SKILL_LEXICON:
            found.add(tok)
    return sorted(found)

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBEDDER.get_sentence_embedding_dimension()))
    embs = EMBEDDER.encode(texts, show_progress_bar=False)
    return np.array(embs)

def compute_skill_coverage(jd_skills: set, resume_skills: set) -> float:
    if not jd_skills:
        return 0.0
    matched = jd_skills.intersection(resume_skills)
    return float(len(matched)) / float(len(jd_skills))

# Custom tokenizer wrapper for TfidfVectorizer
def tfidf_tokenizer_for_vectorizer(text: str):
    # returns list of tokens (strings) suitable for TF-IDF
    cleaned = basic_clean(text)
    toks = spacy_tokenizer(cleaned)
    # also include multi-word skills from lexicon if present
    for skill in SKILL_LEXICON:
        if " " in skill and skill in cleaned:
            toks.append(skill)
    return toks

# -----------------------
# Main Endpoint
# -----------------------

@app.post("/upload_and_match")
async def upload_and_match(files: List[UploadFile] = File(...), job_text: str = Form(...), top_k: int = Form(5)):
    # Validation
    if not files:
        raise HTTPException(status_code=400, detail="No resume files uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} files allowed.")
    # prepare temp folder per request
    ts = int(time.time())
    tmp = os.path.join("data", "tmp_dynamic", f"run_{ts}")
    raw_dir = os.path.join(tmp, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    filenames = []
    texts = []
    try:
        # Save uploads & extract text
        for upload in files:
            content = await upload.read()
            size_mb = len(content) / (1024 * 1024)
            if size_mb > MAX_SIZE_MB:
                raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds size limit ({MAX_SIZE_MB} MB)")
            dest = os.path.join(raw_dir, upload.filename)
            with open(dest, "wb") as fh:
                fh.write(content)
            # extract text
            if upload.filename.lower().endswith(".pdf"):
                txt = extract_text_from_pdf(dest)
            else:
                try:
                    txt = content.decode("utf-8", errors="ignore")
                except Exception:
                    txt = ""
            txt_clean = basic_clean(txt)
            texts.append(txt_clean)
            filenames.append(upload.filename)

        # Clean job text and extract JD skills
        job_clean = basic_clean(job_text)
        jd_skills = set(extract_skills_from_text(job_clean))

        # Embeddings
        jd_emb = embed_texts([job_clean])
        resume_embs = embed_texts(texts)

        # semantic scores
        semantic = cosine_similarity(jd_emb, resume_embs).ravel() if resume_embs.size else np.array([])

        # TF-IDF with spaCy tokenizer
        vectorizer = TfidfVectorizer(
            tokenizer=tfidf_tokenizer_for_vectorizer,
            token_pattern=None,
            max_features=MAX_TFIDF_FEATURES,
            stop_words="english"
        )
        corpus = [job_clean] + texts
        tfidf = vectorizer.fit_transform(corpus)
        job_vec = tfidf[0]
        keyword_sims = cosine_similarity(job_vec, tfidf[1:]).ravel()

        # skill coverage per resume
        resume_skills_list = [set(extract_skills_from_text(t)) for t in texts]
        skill_coverages = np.array([compute_skill_coverage(jd_skills, rset) for rset in resume_skills_list])

        # hybrid score (adaptive)
        hybrid = SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * keyword_sims + SKILL_COVERAGE_WEIGHT * skill_coverages

        # sort results
        order = np.argsort(hybrid)[::-1]
        top_k = max(1, int(min(top_k, len(order))))

        # explainability: top overlapping terms (skill-aware)
        explainability = {}
        feature_names = vectorizer.get_feature_names_out()
        for idx in order[:min(5, len(order))]:
            # get TF-IDF vector for resume idx+1 (since 0 is job)
            res_vec = tfidf[idx+1].toarray().ravel()
            # compute product with job vector to get term-level contribution (simple proxy)
            job_arr = job_vec.toarray().ravel()
            term_scores = job_arr * res_vec  # element-wise product: high where both have high tfidf
            top_indices = np.argsort(term_scores)[::-1][:TOP_EXPLAIN]
            keywords = []
            for ti in top_indices:
                feat = feature_names[ti]
                # filter small tokens & non-alpha unless in skill lexicon
                if (feat.isalpha() and len(feat) > 1) or feat in SKILL_LEXICON:
                    keywords.append(feat)
            # fallback: if none, pick top TF-IDF terms from the resume vector
            if not keywords:
                top_idx2 = np.argsort(res_vec)[::-1][:TOP_EXPLAIN]
                for ti in top_idx2:
                    feat = feature_names[ti]
                    if (feat.isalpha() and len(feat) > 1) or feat in SKILL_LEXICON:
                        keywords.append(feat)
            explainability[filenames[idx]] = keywords[:TOP_EXPLAIN]

        # build minimal results list
        results = [{"resume_file": filenames[i], "hybrid_score": round(float(hybrid[i]), 4)} for i in order]

        response = {"results": results, "explainability": explainability}

        # OPTIONAL: persist audit (append to JSON)
        os.makedirs("logs", exist_ok=True)
        audit_path = "logs/updates.json"
        try:
            audit = {"timestamp": datetime.utcnow().isoformat()+"Z", "files": filenames, "top_result": results[0] if results else None}
            if os.path.exists(audit_path):
                with open(audit_path, "r", encoding="utf-8") as f:
                    arr = json.load(f)
            else:
                arr = []
            arr.append(audit)
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(arr, f, indent=2)
        except Exception:
            pass

        # optionally cleanup tmp to save disk; keep for debugging now
        # import shutil; shutil.rmtree(tmp)

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
