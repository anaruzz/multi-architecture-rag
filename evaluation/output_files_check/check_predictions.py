import os
import json

# Root and predictions folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "evaluation", "predictions")


#INPUT_FILE = os.path.join(PREDICTIONS_DIR, "basic_rag_predictions.jsonl")
#INPUT_FILE = os.path.join(PREDICTIONS_DIR, "multiquery_rag_predictions.jsonl")
INPUT_FILE = os.path.join(PREDICTIONS_DIR, "agentic_rag_predictions.jsonl")

REQUIRED_KEYS = {"question", "ground_truth", "answer", "contexts"}


if not os.path.exists(INPUT_FILE):
    print(f" File not found:\n{INPUT_FILE}")
    exit()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"\n Checking: {os.path.basename(INPUT_FILE)}")
print(f" Total entries: {len(lines)}")

missing_keys_count = 0
malformed_rows = 0

for i, line in enumerate(lines):
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        print(f" Malformed JSON at line {i+1}")
        malformed_rows += 1
        continue

    record_keys = set(record.keys())
    missing = REQUIRED_KEYS - record_keys
    if missing:
        print(f"  Line {i+1} missing keys: {missing}")
        missing_keys_count += 1


if malformed_rows == 0 and missing_keys_count == 0:
    print("\n All records are valid and contain expected keys.")
else:
    print(f"\n Issues found:")
    print(f" - Rows with missing keys: {missing_keys_count}")
    print(f" - Malformed JSON rows: {malformed_rows}")
