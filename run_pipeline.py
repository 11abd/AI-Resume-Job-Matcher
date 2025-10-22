#!/usr/bin/env python3
"""
run_pipeline.py

Master pipeline for AI-Resume-Matcher.
It runs available scripts in the repo in a sensible order and optionally starts FastAPI.

Usage:
  python run_pipeline.py          # run pipeline stages only
  python run_pipeline.py --start-api   # run pipeline then start uvicorn for the API
"""

import os
import subprocess
import sys
import argparse
from datetime import datetime

# Force UTF-8 encoding globally (fix for Windows)
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LOG_FILE = "logs/pipeline_log.txt"
os.makedirs("logs", exist_ok=True)


# canonical order we prefer (only run if file exists)
PREFERRED_ORDER = [
    ("Data Preprocessing", "src/data_preprocessing.py"),
    ("Feature Engineering (embeddings)", "src/feature_engineering.py"),
    ("Hybrid Scoring", "src/hybrid_matcher.py"),
    ("Model Training (if present)", "src/model_train.py"),
    ("Evaluation", "src/evaluate_model.py")
]

def run_cmd(cmd, capture=True):
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=capture,
            encoding="utf-8",  # ✅ force UTF-8 decoding
            errors="replace"   # ✅ replace undecodable chars
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except Exception as e:
        print(f"❌ Error running command {cmd}: {e}")
        return 1, "", str(e)


def run_stage(name, path):
    print("\n" + "="*60)
    print(f" Stage: {name}")
    print(f" Running: {path}")
    print("="*60)

    if not os.path.exists(path):
        print(f" Skipping — script not found: {path}")
        return True

    try:
        rc, out, err = run_cmd([sys.executable, path])
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now()}] Stage: {name} ({path})\n")
            f.write(out + "\n")
            if err:
                f.write("STDERR:\n" + err + "\n")

        if rc != 0:
            print(f" Stage FAILED: {name} (rc={rc})")
            print(err)
            return False

        print(f" Stage completed: {name}")
        return True
    except Exception as e:
        print(f" Exception running stage {name}: {e}")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now()}] Exception running {name}: {e}\n")
        return False

def build_stage_list():
    # pick scripts that exist in preferred order
    stages = [s for s in PREFERRED_ORDER if os.path.exists(s[1])]
    # if none found, warn
    if not stages:
        print(" No pipeline scripts found in expected locations. Check src/ folder.")
    return stages

def start_api():
    print("\n Starting FastAPI (uvicorn) ...")
    # uses same python executable to launch uvicorn as a child process
    cmd = [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    # start as a blocking subprocess (Ctrl+C to stop)
    subprocess.run(cmd)

def main(args):
    print("\n" + "#"*60)
    print("AI-Resume-Matcher — Master Pipeline")
    print("#"*60 + "\n")

    stages = build_stage_list()
    for name, path in stages:
        ok = run_stage(name, path)
        if not ok:
            print(f"\n Pipeline stopped at stage: {name}")
            print(f"Check logs: {LOG_FILE}")
            sys.exit(1)

    print("\n Pipeline finished successfully. Check logs:", LOG_FILE)
    if args.start_api:
        start_api()
    else:
        print("\n To run the API manually:")
        print("   uvicorn src.api.main:app --reload")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-api", action="store_true", help="Start FastAPI (uvicorn) after pipeline completes")
    parsed = parser.parse_args()
    main(parsed)
