import random
import pickle

# Load pre-chunked documents
with open("chunks.pkl", "rb") as f:
    docs = pickle.load(f)

def random_retriever(query: str, k: int = 5):
    return random.sample(docs, k)
