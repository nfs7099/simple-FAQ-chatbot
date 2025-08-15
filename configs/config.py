import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------- Base Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = Path(os.getenv("PDF_DIR", str(DATA_DIR / "raw")))
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", str(DATA_DIR / "vector_db")))

# ---------- Application ----------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").strip().lower() == "true"

# ---------- PDF Chunking / RAG ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))

# ---------- LLM Providers ----------
# Supported: ollama | openai | anthropic
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# Ollama (local)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or ""
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

# ---------- embeddings ----------
# Supported: huggingface | openai
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# ---------- Reranker ----------
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# ---------- token-based chunking config ----------
def _to_bool(s: str | None, default: bool = False) -> bool:
    if s is None:
        return default
    return s.strip().lower() in ("1", "true", "yes", "y", "on")

ALLOW_DANGEROUS_DESERIALIZATION = _to_bool(
    os.getenv("ALLOW_DANGEROUS_DESERIALIZATION"), True  # default True for local dev
)

# Manifest + ingest policies
MANIFEST_PATH = VECTOR_DB_PATH / "manifest.json"
# FAISS cannot cleanly delete; safest is to rebuild when files were modified or deleted
REBUILD_ON_MODIFICATION = _to_bool(os.getenv("REBUILD_ON_MODIFICATION"), True)
REBUILD_ON_DELETE = _to_bool(os.getenv("REBUILD_ON_DELETE"), True)

# Already added earlier, but ensure it’s present:
ALLOW_DANGEROUS_DESERIALIZATION = _to_bool(os.getenv("ALLOW_DANGEROUS_DESERIALIZATION"), True)


USE_TOKEN_SPLITTER = _to_bool(os.getenv("USE_TOKEN_SPLITTER"), False)
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", EMBEDDING_MODEL)  # default: same as embedder
# if empty / unset -> pass None,  VectorStore will fall back to CHUNK_SIZE/OVERLAP
_TOKEN_CHUNK_SIZE = os.getenv("TOKEN_CHUNK_SIZE")
_TOKEN_CHUNK_OVERLAP = os.getenv("TOKEN_CHUNK_OVERLAP")
TOKEN_CHUNK_SIZE = int(_TOKEN_CHUNK_SIZE) if _TOKEN_CHUNK_SIZE else None
TOKEN_CHUNK_OVERLAP = int(_TOKEN_CHUNK_OVERLAP) if _TOKEN_CHUNK_OVERLAP else None

# --- File acceptance policy ---
# comma-separated list, case-insensitive, strictly enforced
ALLOWED_EXTENSIONS = {ext.strip().lower() for ext in os.getenv("ALLOWED_EXTENSIONS", ".pdf").split(",")}

# hard cap on upload size (in MB) to avoid memory abuse via /api/upload
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


# ---------- UI config ----------
def _csv_lower(s: str) -> list[str]:
    return [t.strip().lower() for t in (s or "").split(",") if t.strip()]

APP_NAME = os.getenv("APP_NAME", "Simple FAQ Chatbot").strip()
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Assistant").strip()

THEME_CHOICES = _csv_lower(os.getenv("THEME_CHOICES", "emerald,indigo,slate,rose,amber,teal"))
if not THEME_CHOICES:
    THEME_CHOICES = ["emerald"]

_default_theme_env = os.getenv("DEFAULT_THEME", "emerald").strip().lower()
DEFAULT_THEME = _default_theme_env if _default_theme_env in THEME_CHOICES else THEME_CHOICES[0]


def ensure_directories() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

def validate_configuration() -> list[str]:
    """
    Validate the current environment configuration.
    Returns:
        config_errors (list[str]): problems found (empty if valid).
    """
    config_errors: list[str] = []

    # LLM provider checks
    if LLM_PROVIDER == "ollama":
        pass
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            config_errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
    elif LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            config_errors.append("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
    else:
        config_errors.append(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    # embedding checks
    if EMBEDDING_PROVIDER == "openai" and not OPENAI_API_KEY:
        config_errors.append("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")

    # chunking sanity checks
    if CHUNK_SIZE <= 0:
        config_errors.append("CHUNK_SIZE must be > 0.")
    if CHUNK_OVERLAP < 0:
        config_errors.append("CHUNK_OVERLAP must be >= 0.")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        config_errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE.")
    if TOP_K_RESULTS <= 0:
        config_errors.append("TOP_K_RESULTS must be > 0.")

    # token chunking sanity checks (if set)
    if TOKEN_CHUNK_SIZE is not None and TOKEN_CHUNK_SIZE <= 0:
        config_errors.append("TOKEN_CHUNK_SIZE must be > 0.")
    if TOKEN_CHUNK_OVERLAP is not None and TOKEN_CHUNK_OVERLAP < 0:
        config_errors.append("TOKEN_CHUNK_OVERLAP must be >= 0.")
    if (TOKEN_CHUNK_SIZE is not None and TOKEN_CHUNK_OVERLAP is not None
        and TOKEN_CHUNK_OVERLAP >= TOKEN_CHUNK_SIZE):
        config_errors.append("TOKEN_CHUNK_OVERLAP must be less than TOKEN_CHUNK_SIZE.")
    
    if _default_theme_env not in THEME_CHOICES:
        # informational (not an error) auto-fall back to first theme
        pass

    return config_errors

ensure_directories()


