✨ Simple FAQ Chatbot (RAG)

Local or hosted LLMs • PDF ingestion • Modern UI • Theming • Sources & reranking

A local-first FAQ chatbot that lets you upload PDFs, builds a vector index, and answers questions with citations. Run fully offline via Ollama, or switch to OpenAI/Anthropic with API keys. Skinnable UI, clean API, and practical defaults.

Features

✅ Local or hosted LLMs: Ollama (local), or OpenAI / Anthropic (cloud).

✅ RAG pipeline: HuggingFace embeddings + FAISS + optional reranker.

✅ PDF ingestion: upload control; incremental and full rebuild indexing.

✅ Token-aware chunking: smarter splits with configurable overlap.

✅ Citations: expandable source snippets with page hints.

✅ Polished UI: React + FastAPI; themeable (emerald/indigo/slate/rose/amber/teal).

✅ Health/ready endpoints for container orchestration.

✅ Safer uploads: PDF-only (extension + magic bytes), size limits, path-traversal safe.

Quickstart

Python ≥ 3.10 recommended.
# 1) Clone & enter
git clone <your-repo-url>
cd simple-FAQ-chatbot

# 2) Create venv & install
python -m venv .venv
# Windows:
. .venv/Scripts/activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt

# 3) Configure environment
cp .env.example .env  # or create .env using the template below

# 4) (Optional) Local LLM with Ollama
# https://ollama.ai
ollama pull gemma2:9b

# 5) Run the app
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
# Open: http://localhost:8000

Configuration

Create .env at the project root. Full template:

# ========== LLM Configuration ==========
# Choose one: ollama | openai | anthropic
LLM_PROVIDER=ollama

# When LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma2:9b

# When LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-xxxxxxxx
# OPENAI_MODEL=gpt-4o-mini

# When LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
# ANTHROPIC_MODEL=claude-3-haiku-20240307

# ========== Embeddings ==========
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# ========== Vector DB ==========
VECTOR_DB_PATH=./data/vector_db
# Only set True if you trust the FAISS docstore pickle files you created yourself
ALLOW_DANGEROUS_DESERIALIZATION=False

# ========== Application ==========
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Raw PDF folder (uploads also saved here)
PDF_DIR=./data/raw_pdfs

# ========== Chunking & Retrieval ==========
# Token-aware by default; set to 'chars' to use character-based
CHUNK_STRATEGY=tokens
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K_RESULTS=3

# Reranker model (optional but recommended)
RERANKER_MODEL=BAAI/bge-reranker-base

# ========== File Acceptance ==========
# PDF only, server-side validated (extension + magic bytes)
MAX_UPLOAD_BYTES=26214400  # ~25MB

# ========== UI Config ==========
APP_NAME=Simple FAQ Chatbot
ASSISTANT_NAME=Assistant
THEME_CHOICES=emerald,indigo,slate,rose,amber,teal
DEFAULT_THEME=emerald


Run the App
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Ingestion & Reindexing

Uploads save to PDF_DIR and trigger indexing.

Chunking: token-based (CHUNK_STRATEGY=tokens) for better semantic continuity; overlap helps cross-chunk context.

Embeddings: BAAI/bge-base-en-v1.5 (fast and solid for English).

Vector DB: FAISS on disk at VECTOR_DB_PATH.

Reranking: BAAI/bge-reranker-base (optional) to reorder top-K hits before generation.


Testing

Smoke tests with pytest:

pip install pytest reportlab python-multipart
pytest -q

Covers:

status before/after indexing

upload PDF (reject non-PDF)

incremental vs full rebuild behavior

basic query flow

License

MIT — see LICENSE.

Credits

Built with ❤️ using FastAPI, React, FAISS, LangChain Community, HuggingFace models, and optionally Ollama / OpenAI / Anthropic for generation.
