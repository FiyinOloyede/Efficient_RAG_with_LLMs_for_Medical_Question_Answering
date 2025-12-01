# Traditional BM25 retrieval setup (Traditional retrieval with BM25)
from langchain_community.retrievers import BM25Retriever
import pickle

# Load pre-chunked documents from pickle (external knowledge base)
with open("chunks.pkl", "rb") as f:
    docs = pickle.load(f)

# Load pre-chunked documents from pickle (knowledge base with dataset context)
with open("chunks2.pkl", "rb") as f:
    docs_dcontext = pickle.load(f)

# Initialize BM25 retrievers
bm25_retriever = BM25Retriever.from_documents(docs, k=5) # from external knowledge base
bm25_retriever2 = BM25Retriever.from_documents(docs_dcontext, k=5) # from external knowledge base with dataset context

