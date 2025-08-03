import pandas as pd
import json
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load the CSV
df = pd.read_csv("../output/basic_rag_output.csv")
df["contexts"] = df["basic_rag_grounding"].apply(json.loads)

# Create RAGAS-compatible Dataset
ragas_input = pd.DataFrame({
    "question": df["query"],
    "answer": df["basic_rag_answer"],
    "contexts": df["contexts"],
    "ground_truth": df["reference_answer"]
})
dataset = Dataset.from_pandas(ragas_input)

# Setup Ollama LLM & Embeddings
llm = ChatOllama(model="openchat")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Evaluation loop with progress and error handling
results = []
for i in range(len(dataset)):
    print(f"⏳ Evaluating query {i+1}/{len(dataset)}...")
    try:
        result = evaluate(
            dataset.select([i]),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(timeout=1200, max_retries=3),
        )
        score_dict = result.to_pandas().iloc[0].to_dict()
    except Exception as e:
        print(f"❌ Evaluation error on query {i+1}: {e}")
        score_dict = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }
    results.append(score_dict)
    time.sleep(1)

# Merge scores back to original data
final_df = pd.concat([df, pd.DataFrame(results)], axis=1)

# Save final results
final_df.to_csv("basic_rag_evaluation.csv", index=False)
print("✅ Evaluation done and saved to basic_rag_evaluation.csv")
