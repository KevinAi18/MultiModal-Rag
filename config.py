import os
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

# Model Config
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Storage Config
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pdf_rag_collection")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_docs")

# RAG Hyperparameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "5000"))

# HuggingFace Credentials
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    # Set HF_TOKEN environment variable for transformers and huggingface_hub libraries
    os.environ["HF_TOKEN"] = HF_TOKEN
