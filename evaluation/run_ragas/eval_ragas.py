import os
import json
import pandas as pd
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings


# Basic RAG
# input_file = "basic_rag_predictions.jsonl"
# output_file = "basic_rag_evaluation.csv"

# Multi-Query RAG
#input_file = "multiquery_rag_predictions.jsonl"
#output_file = "multiquery_rag_evaluation.csv"

# Agentic RAG
input_file = "agentic_rag_predictions.jsonl"
output_file = "agentic_rag_evaluation.csv"


# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_PATH = os.path.join(BASE_DIR, "evaluation", "predictions", input_file)
OUTPUT_PATH = os.path.join(BASE_DIR, "evaluation", "evaluation_results", output_file)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load JSONL
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    records = [json.loads(line.strip()) for line in f]
df = pd.DataFrame(records)

# Setup LLM + Embeddings
llm = ChatOllama(model="openchat")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



results = []

for i in range(len(df)):
    print(f" Evaluating query {i+1}/{len(df)}")

    try:
        single_ds = Dataset.from_pandas(df.iloc[[i]][["question", "answer", "contexts", "ground_truth"]])
        result = evaluate(
            single_ds,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(timeout=300, max_retries=4)
        )
        result_row = result.to_pandas().iloc[0].to_dict()
    except Exception as e:
        print(f" Query {i+1} failed: {e}")
        result_row = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_recall": None,
            "context_precision": None,
        }

    results.append(result_row)
    time.sleep(0.5)

# Save output
scores_df = pd.DataFrame(results)
final_df = pd.concat([df, scores_df], axis=1)
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n Evaluation complete. Results saved to:\n{OUTPUT_PATH}")
