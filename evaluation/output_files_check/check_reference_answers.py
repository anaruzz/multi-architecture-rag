import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "Questions_and_Reference_Answers_100.csv")

# Load the file
ref_df = pd.read_csv(CSV_PATH, encoding='ISO-8859-1')
print(ref_df.head())

# === Basic file checks ===
print("\n Columns:", ref_df.columns.tolist())
print(" Total rows:", len(ref_df))
print(" Null values per column:\n", ref_df.isnull().sum())
print(" Duplicate rows:", ref_df.duplicated().sum())


# === Check column name consistency ===
expected_columns = {"query", "reference_answer"}
missing_columns = expected_columns - set(ref_df.columns)

if missing_columns:
    print(f"\n Missing expected columns: {missing_columns}")
else:
    print("\n File format looks correct and ready for evaluation.")