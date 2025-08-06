import os
import pandas as pd

# Absolute root path of the directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Basic Rag 
#EVAL_PATH = os.path.join(PROJECT_ROOT, "evaluation", "evaluation_results", "basic_rag_evaluation.csv")

# Multiquery Rag
#EVAL_PATH = os.path.join(PROJECT_ROOT, "evaluation", "evaluation_results", "multiquery_rag_evaluation.csv")

# Agentic Rag
EVAL_PATH = os.path.join(PROJECT_ROOT, "evaluation", "evaluation_results", "agentic_rag_evaluation.csv")

# Load CSV
df = pd.read_csv(EVAL_PATH)

# Define expected columns
expected_columns = [
    "question", "ground_truth", "answer", "contexts",
    "faithfulness", "answer_relevancy", "context_recall", "context_precision"
]

# Check for missing columns
missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    print(f" Missing expected columns: {missing_cols}")
else:
    print(" All expected columns are present.")

# Check for null values
null_counts = df[expected_columns].isnull().sum()
print("\n Null value counts:")
print(null_counts)

# Show total and incomplete rows
total_rows = len(df)
incomplete_rows_mask = df[expected_columns].isnull().any(axis=1)
num_incomplete = incomplete_rows_mask.sum()

print(f"\n Total rows: {total_rows}")
print(f" Rows with missing values: {num_incomplete}")

if num_incomplete > 0:
    print("\n Incomplete rows (with missing values):")
    print(df[incomplete_rows_mask])
else:
    print("\n No incomplete rows — file is ready for analysis!")
