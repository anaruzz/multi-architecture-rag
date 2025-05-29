import os
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


DATA_DIR = "data"
CHUNK_CACHE = "chunks.pkl"
FAISS_DIR = "faiss_index"


# Load PDFs
def load_all_pdfs(folder):
    docs = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
    return docs

# Chunk
def chunk_docs(docs, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

# Embed and Save in FAISS ---
def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_DIR)
    print(f"FAISS Vector DB saved to {FAISS_DIR}")

if __name__ == "__main__":
    raw_docs = load_all_pdfs(DATA_DIR)
    print("Loading the PDFs Done")

    chunks = chunk_docs(raw_docs)
    print("Chunking Done")

    print(f"Saving {len(chunks)} chunks to {CHUNK_CACHE}")
    with open(CHUNK_CACHE, "wb") as f:
        pickle.dump(chunks, f)

    embed_and_store(chunks)
    print("Embedding and storing in FAISS...")
