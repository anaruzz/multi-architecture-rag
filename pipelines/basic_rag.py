import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langsmith import traceable
import pandas as pd
import json

FAISS_DIR = "../faiss_index"

# Load .env variables
load_dotenv(dotenv_path="../.env") 


# Load FAISS 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Prompt Template
prompt = PromptTemplate.from_template("""
Use the following context to answer the question concisely and accurately.

Context:
{context}

Question:
{question}
""")


# llm initialization
llm = OllamaLLM(model="openchat")



@traceable(name="Basic RAG Query Trace")
def run_query(query: str) -> dict:
    # Step 1: retrieve docs
    docs = retriever.invoke(query)
    
    # Step 2: build context
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Step 3: generate answer
    filled_prompt = prompt.format(context=context, question=query)
    answer = llm.invoke(filled_prompt)

    # Step 4: return structured result
    return {
        "question": query,
        "generated_answer": answer,
        "contexts": [doc.page_content for doc in docs]
    }




if __name__ == "__main__":
    input_path = "../Questions_and_Reference_Answers_100.csv"
    output_path = "../output/basic_rag_output.csv"

    df = pd.read_csv(input_path)
    df["basic_rag_answer"] = ""
    df["basic_rag_grounding"] = ""

    total = len(df)

    for row_number, (idx, row) in enumerate(df.iterrows(), start=1):
        query = row["query"]
        result = run_query(query)

        df.at[idx, "basic_rag_answer"] = result["generated_answer"]
        df.at[idx, "basic_rag_grounding"] = json.dumps(result["contexts"])

        print(f"Progress: {row_number}/{total} queries completed.\n")

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")