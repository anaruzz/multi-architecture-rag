import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langsmith import traceable

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
    query = "Whow to configure a Vlan on a cisco switch?"
    result = run_query(query)
    print("Answer:", result)

    print("\nGenerated Answer:\n", result["generated_answer"])
    print("\nRetrieved Contexts:")
    for i, ctx in enumerate(result["contexts"], 1):
        print(f"\nContext {i}:\n{ctx}")
    