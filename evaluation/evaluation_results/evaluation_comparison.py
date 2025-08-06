import pandas as pd
import os


base_dir = os.path.dirname(__file__)

# paths
files = {
    "Basic RAG": os.path.join(base_dir, "basic_rag_evaluation.csv"),
    "Multi-Query RAG": os.path.join(base_dir, "multiquery_rag_evaluation.csv"),
    "Agentic RAG": os.path.join(base_dir, "agentic_rag_evaluation.csv")
}

# metrics
metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

# average metrics for each RAG pipeline
results = {}

for model_name, file_path in files.items():
    df = pd.read_csv(file_path)
    averages = df[metrics].mean().to_dict()
    results[model_name] = averages

comparison_df = pd.DataFrame(results).T.round(4)
comparison_df.index.name = "RAG Pipeline"

print("\n🔍 RAGAS Evaluation Comparison:\n")
print(comparison_df)
