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
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM initialization
llm = OllamaLLM(model="mistral")


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
def generate_answer(query: str) -> str:
    docs = docs = multi_retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents from multi-query retrieval.\n")

    context = "\n\n".join(doc.page_content for doc in docs)
    filled_prompt = final_prompt.format(context=context, question=query)

    return llm.invoke(filled_prompt)


if __name__ == "__main__":
    query = "How do I configure VLANs on a Cisco switch?"
    answer = generate_answer(query)

    print("Final Answer:\n")
    print(answer)
