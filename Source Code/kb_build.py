from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pytesseract, pickle


# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Student\Tesseract-OCR\tesseract.exe"

# Set up the document loader for PDFs and text files for Knowledge base build
loader = DirectoryLoader(
    "Knowledge base build",
    glob=["**/*.txt", "**/*.pdf"],
    loader_cls=UnstructuredFileLoader,
    loader_kwargs={}, 
    recursive=True,
    show_progress=True
)

# Split longer documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Load + split
documents = loader.load()
chunks = text_splitter.split_documents(documents)

print(f"Loaded {len(documents)} documents → {len(chunks)} chunks")


# Save chunks to a file
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print(f"Saved {len(chunks)} chunks to disk.")


# Load chunks from the file
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"Loaded {len(chunks)} chunks from cache.")


# Set up the document loader for PDFs and text files for Knowledge base build (with dataset contexts)
loader2 = DirectoryLoader(
    "Knowledge base build (with dataset contexts)",
    glob=["**/*.txt", "**/*.pdf"],
    loader_cls=UnstructuredFileLoader,
    loader_kwargs={}, 
    recursive=True,
    show_progress=True
)

# Split longer documents into manageable chunks
text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Load + split
documents2 = loader2.load()
chunks2 = text_splitter2.split_documents(documents2)

print(f"Loaded {len(documents2)} documents → {len(chunks2)} chunks")


# Save chunks to a file
with open("chunks2.pkl", "wb") as f:
    pickle.dump(chunks2, f)
print(f"Saved {len(chunks2)} chunks to disk.")


# Load chunks from the file
with open("chunks2.pkl", "rb") as f:
    chunks2 = pickle.load(f)
print(f"Loaded {len(chunks2)} chunks (dataset contexts) from cache.")


# Create embeddings using pre-trained HuggingFace model (GTR-T5-Large)
embedding_model_gtr = HuggingFaceEmbeddings(
    model_name='sentence-transformers/gtr-t5-large',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={"normalize_embeddings": True}
)

# Create FAISS vector stores from the document chunks and their embeddings
vector_db_gtr = FAISS.from_documents(chunks, embedding_model_gtr)
vector_db_gtr2 = FAISS.from_documents(chunks2, embedding_model_gtr) # dataset contexts

# Save the FIASS vetor stores
vector_db_gtr.save_local('vector_store_gtr-t5-large')
vector_db_gtr2.save_local('vector_store_gtr-t5-large2') # dataset contexts


# Create embeddings using pre-trained HuggingFace model (pritamdeka/S-BioBert-snli-multinli-stsb)
embedding_model_sbb = HuggingFaceEmbeddings(
    model_name="pritamdeka/S-BioBert-snli-multinli-stsb",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# Create FAISS vector stores from the document chunks and their embeddings
vector_db_sbb = FAISS.from_documents(chunks, embedding_model_sbb)
vector_db_sbb2 = FAISS.from_documents(chunks2, embedding_model_sbb) # dataset contexts

# Save the FAISS vetor stores
vector_db_sbb.save_local('vector_store_pritamdeka')
vector_db_sbb2.save_local('vector_store_pritamdeka2') # dataset contexts
