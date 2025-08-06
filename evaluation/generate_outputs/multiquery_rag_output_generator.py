import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from multiquery_rag import run_multiquery_rag

# Import Multi-Query RAG pipeline
PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "pipelines")
sys.path.append(os.path.abspath(PIPELINES_DIR))


# File paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(BASE_DIR, "Questions_and_Reference_Answers_100.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation", "predictions")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "multiquery_rag_predictions.jsonl")

# Output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the 100 reference queries
ref_df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")
print(f" Loaded {len(ref_df)} queries")

# Run each query
results = []

for idx, row in tqdm(ref_df.iterrows(), total=len(ref_df), desc="Running Multi-Query RAG"):
    query = row["query"]
    reference = row["reference_answer"]

    try:
        result = run_multiquery_rag(query)
        results.append({
            "question": query,
            "ground_truth": reference,
            "answer": result["generated_answer"],
            "contexts": result["contexts"]
        })
    except Exception as e:
        print(f" Error at index {idx}: {e}")

# Save results to a JSONL file for RAGAS
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n Saved {len(results)} results to {OUTPUT_PATH}")
