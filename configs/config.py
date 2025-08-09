import os
from pathlib import Path
from dotenv import load_dotenv

# load the environment variables from .env file 
load_dotenv()

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# LLM Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# Vector Database Configuration
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(BASE_DIR /"data"/"vector_db"))

# Application Configuration
HOST  = os.getenv("HOST", "0.0.0.0")
PORT  = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# PDF Configuration
PDF_DIR = BASE_DIR/"data"/"raw"

# Chunking Configuration
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100

# RAG Configration
TOP_K_RESULTS = 3


