# Dense retrieval pipelines (Advanced retrieval with Dense Embeddings + FAISS)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle, os

# Load pre-chunked documents from pickle (external knowledge base)
with open("chunks.pkl", "rb") as f:
    docs = pickle.load(f)

# Load pre-chunked documents from pickle (external knowledge base with dataset context)
with open("chunks2.pkl", "rb") as f:
    docs_dcontext = pickle.load(f)

# Initialize embeddings
embedding_model_gtr = HuggingFaceEmbeddings(
    model_name='sentence-transformers/gtr-t5-large',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={"normalize_embeddings": True}
)


embedding_model_sbb = HuggingFaceEmbeddings(
    model_name='pritamdeka/S-BioBert-snli-multinli-stsb',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={"normalize_embeddings": True}
)


# Load (or build) FAISS vector stores
vector_db_gtr = 'vector_store_gtr-t5-large' # external Knowledge base 
if os.path.exists(vector_db_gtr):
    # Load existing FAISS index
    index = FAISS.load_local(vector_db_gtr, 
                             embedding_model_gtr,
                             allow_dangerous_deserialization=True
                             )
else:
    # Create new FAISS index
    index = FAISS.from_documents(docs, embedding_model_gtr)
    index.save_local(vector_db_gtr)

# Create a retriever for dense embeddings (external Knowledge base)
dense_retriever_gtr = index.as_retriever(search_kwargs={"k": 5}) 


vector_db_gtr2 = 'vector_store_gtr-t5-large2' # external Knowledge base (with dataset contexts)
if os.path.exists(vector_db_gtr2):
    # Load existing FAISS index
    index = FAISS.load_local(vector_db_gtr2, 
                             embedding_model_gtr,
                             allow_dangerous_deserialization=True
                             )
else:
    # Create new FAISS index
    index = FAISS.from_documents(docs_dcontext, embedding_model_gtr)
    index.save_local(vector_db_gtr2)

# Create a retriever for dense embeddings (external Knowledge base with dataset contexts)
dense_retriever_gtr2 = index.as_retriever(search_kwargs={"k": 5}) 


vector_db_sbb = 'vector_store_pritamdeka' # external Knowledge base
if os.path.exists(vector_db_sbb):
    # Load existing FAISS index
    index = FAISS.load_local(vector_db_sbb, 
                             embedding_model_sbb,
                             allow_dangerous_deserialization=True
                             )
else:
    # Create new FAISS index
    index = FAISS.from_documents(docs, embedding_model_sbb)
    index.save_local(vector_db_sbb)

# Create a retriever for dense embeddings (external Knowledge base)
dense_retriever_sbb = index.as_retriever(search_kwargs={"k": 5})


vector_db_sbb2 = 'vector_store_pritamdeka2'  # external Knowledge base with dataset contexts
if os.path.exists(vector_db_sbb2):
    # Load existing FAISS index
    index = FAISS.load_local(vector_db_sbb2, 
                             embedding_model_sbb,
                             allow_dangerous_deserialization=True
                             )
else:
    # Create new FAISS index
    index = FAISS.from_documents(docs_dcontext, embedding_model_sbb)
    index.save_local(vector_db_sbb2)

# Create a retriever for dense embeddings (external Knowledge base with dataset contexts)
dense_retriever_sbb2 = index.as_retriever(search_kwargs={"k": 5})
