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
retriever = vectorstore.as_retriever()


# Prompt Template
prompt = PromptTemplate.from_template("""
Use the following context to answer the question concisely and accurately.

Context:
{context}

Question:
{question}
""")


# llm initialization
llm = OllamaLLM(model="mistral")


# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff", 
    chain_type_kwargs={"prompt": prompt}
)


@traceable(name="Basic RAG Query Trace")
def run_query(query: str) -> str:
    return qa.invoke({"query": query})





if __name__ == "__main__":
    query = "What are the configuration steps for VLAN on a Cisco switch?"
    result = run_query(query)
    print("Answer:", result)
