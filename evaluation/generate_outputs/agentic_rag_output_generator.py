import os
import sys
import json
import pandas as pd
from tqdm import tqdm



# project directories
PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "pipelines")
sys.path.append(os.path.abspath(PIPELINES_DIR))

from agentic_rag import run_query

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(BASE_DIR, "Questions_and_Reference_Answers_100.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation", "predictions")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "agentic_rag_predictions.jsonl")


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the benchmark questions 
ref_df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")
print(f"Loaded {len(ref_df)} queries")

# Run pipeline on each query
results = []

for idx, row in tqdm(ref_df.iterrows(), total=len(ref_df), desc="⚙️ Running Agentic RAG"):
    query = row["query"]
    reference = row["reference_answer"]

    try:
        result = run_query(query)
        results.append({
            "question": query,
            "ground_truth": reference,
            "answer": result["generated_answer"]["output"] if isinstance(result["generated_answer"], dict) else result["generated_answer"],
            "contexts": result["contexts"]
        })
    except Exception as e:
        print(f"Error at index {idx}: {e}")

# Save to .jsonl file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")
