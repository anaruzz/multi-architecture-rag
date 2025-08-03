import pandas as pd
import json


df = pd.read_csv("basic_rag_output.csv")

'''
required_columns = [
    "query", "reference_answer",
    "basic_rag_answer", "basic_rag_grounding",
    "multiquery_rag_answer", "multiquery_rag_grounding",
    "agentic_rag_answer", "agentic_rag_grounding"
]

# Check for missing values in any required column
missing_rows = df[df[required_columns].isnull().any(axis=1)]


if missing_rows.empty:
    print("✅ All rows are complete. No missing values found.")
else:
    print(f"❌ Found {len(missing_rows)} rows with missing values:")
    print(missing_rows[["query"] + [col for col in required_columns if col != "query"]])

'''

example = df["basic_rag_grounding"].iloc[0]
print(type(example))  # string

parsed = json.loads(example)
print(type(parsed))   # list
print(len(parsed))    

for i, chunk in enumerate(parsed, 1):
    print(f"--- Chunk {i} ---\n{chunk}\n")