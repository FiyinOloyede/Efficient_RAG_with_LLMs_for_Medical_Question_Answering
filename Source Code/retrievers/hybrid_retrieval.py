from langchain.retrievers import EnsembleRetriever
from dense_retrieval import dense_retriever_gtr, dense_retriever_gtr2, dense_retriever_sbb, dense_retriever_sbb2 # returns Dense retrievers
from bm25_retrieval import bm25_retriever,  bm25_retriever2 # returns BM25Retrievers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Create ensembles combining BM25 Retriever and different variant of dense retrievers with equal weights
ensemble_gtr = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever_gtr],
    weights=[0.5, 0.5] # balance influence from each
)

ensemble_gtr2 = EnsembleRetriever(
    retrievers=[bm25_retriever2, dense_retriever_gtr2], # retrievers + dataset contexts
    weights=[0.5, 0.5] # balance influence from each
)

ensemble_sbb = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever_sbb],
    weights=[0.5, 0.5] # balance influence from each
)

ensemble_sbb2 = EnsembleRetriever(
    retrievers=[bm25_retriever2, dense_retriever_sbb2],  # retrievers + dataset contexts
    weights=[0.5, 0.5] # balance influence from each
)


########################################
def hybrid_retrieve_gtr(query: str, k=10):
    return ensemble_gtr.invoke(query) # returns combined top docs or Weighted combination results

def hybrid_retrieve_gtr2(query: str, k=10):
    return ensemble_gtr2.invoke(query) # returns combined top docs or Weighted combination results

def hybrid_retrieve_sbb(query: str, k=10):
    return ensemble_sbb.invoke(query) # returns combined top docs or Weighted combination results

def hybrid_retrieve_sbb2(query: str, k=10):
    return ensemble_sbb2.invoke(query) # returns combined top docs or Weighted combination results


# Load scoring model
tokenizer_r = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model_r = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2",
                                                              torch_dtype=torch.float16,
                                                              device_map="auto")

# Re-ranker function
def rerank(query: str, docs, top_n=5):
    inputs = tokenizer_r([query] * len(docs), [d.page_content for d in docs], truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(model_r.device) for k, v in inputs.items()}
    with torch.no_grad():
        scores = model_r(**inputs).logits.squeeze(-1)
    ranked = sorted(zip(docs, scores.cpu().numpy()), key=lambda x: x[1], reverse=True)
    return [d for d, s in ranked[:top_n]]

# Hybrids with Rerank
def hybrid_with_rerank_gtr(query: str, k_retrieve=10, k_final=5):
    # Stage 1: retrieve
    candidates = hybrid_retrieve_gtr(query, k=k_retrieve)
    # Stage 2: rerank top N
    return rerank(query, candidates, top_n=k_final) # final top-5 after neural refinement


def hybrid_with_rerank_gtr2(query: str, k_retrieve=10, k_final=5):
    # Stage 1: retrieve
    candidates = hybrid_retrieve_gtr2(query, k=k_retrieve) 
    # Stage 2: rerank top N
    return rerank(query, candidates, top_n=k_final) # final top-5 after neural refinement


def hybrid_with_rerank_sbb(query: str, k_retrieve=10, k_final=5):
    # Stage 1: retrieve
    candidates = hybrid_retrieve_sbb(query, k=k_retrieve)
    # Stage 2: rerank top N
    return rerank(query, candidates, top_n=k_final) # final top-5 after neural refinement


def hybrid_with_rerank_sbb2(query: str, k_retrieve=10, k_final=5):
    # Stage 1: retrieve
    candidates = hybrid_retrieve_sbb2(query, k=k_retrieve)
    # Stage 2: rerank top N
    return rerank(query, candidates, top_n=k_final) # final top-5 after neural refinement