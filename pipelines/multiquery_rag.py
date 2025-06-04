import os
from dotenv import load_dotenv
from langsmith import traceable
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document


FAISS_DIR = "../faiss_index"


# Load .env variables
load_dotenv(dotenv_path="../.env") 


# Load Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = OllamaLLM(model="openchat")


subquery_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Rewrite the given question into multiple diverse sub-questions that cover different aspects of the original question.

Original Question: {question}
Sub-questions:
""")


# MultiQueryRetriever
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
    prompt=subquery_prompt
)

final_prompt = PromptTemplate.from_template("""
Use the following context to answer the question. Be concise and accurate.

Context:
{context}

Question:
{question}
""")

@traceable(name="MultiQuery RAG Trace")
def run_multiquery_rag(query: str) -> dict:
    docs = docs = multi_retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents from multi-query retrieval.\n")

    context = "\n\n".join(doc.page_content for doc in docs)
    filled_prompt = final_prompt.format(context=context, question=query)

    answer = llm.invoke(filled_prompt)
    return {
        "question": query,
        "generated_answer": answer,
        "contexts": [doc.page_content for doc in docs]
    }


if __name__ == "__main__":
    query = "How do I configure a VLAN on a Cisco switch?"
    result = run_multiquery_rag(query)

    print("\nFinal Answer:\n", result["generated_answer"])
    print("\nRetrieved Contexts:")
    for i, ctx in enumerate(result["contexts"], 1):
        print(f"\nContext {i}:\n{ctx}")
