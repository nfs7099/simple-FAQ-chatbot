import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import File, UploadFile
import threading
from contextlib import asynccontextmanager
import ollama 


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.rag_pipeline.rag_chain import RAGPipeline
from src.ingestion.ingest_service import IngestService
from configs.config import (
    HOST, PORT, DEBUG, PDF_DIR, VECTOR_DB_PATH,
    OLLAMA_MODEL, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES,
    APP_NAME, ASSISTANT_NAME, DEFAULT_THEME, THEME_CHOICES
)
from configs.config import validate_configuration

logger = logging.getLogger(__name__)


_ingest_lock = threading.Lock()

# ---------- Pydantic models ----------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class StatusResponse(BaseModel):
    status: str
    pdf_count: int
    vector_db_initialized: bool
    llm_initialized: bool
    pdfs: List[str]

class UIConfigResponse(BaseModel):
    appName: str
    assistantName: str
    defaultTheme: str
    themes: List[str]

# defining lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.setLevel(logging.INFO)
    errors = validate_configuration()
    if errors:
        for err in errors:
            logger.error(f"Config error: {err}")
    yield

app = FastAPI(
    title="Simple FAQ Chatbot",
    description="A simple FAQ chatbot using a local/hosted LLM with RAG.",
    version="1.1.0",
    lifespan=lifespan,     
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates_dir = Path(__file__).resolve().parent.parent.parent / "templates"
templates_available = templates_dir.exists()

# Global RAG instance
rag_pipeline: Optional[RAGPipeline] = None

def _check_llm_provider_ready() -> bool:
    provider = (LLM_PROVIDER or "").lower()
    try:
        if provider == "ollama":
            try:
                ollama.list()
                return True
            except Exception as e:
                logger.error(f"Ollama check failed: {e}")
                return False
        if provider == "openai":
            return bool(OPENAI_API_KEY)
        if provider == "anthropic":
            return bool(ANTHROPIC_API_KEY)
        logger.error(f"Unsupported LLM_PROVIDER: {provider}")
        return False
    except Exception as e:
        logger.error(f"LLM provider readiness check error: {e}", exc_info=True)
        return False
    
def _faiss_index_present(db_path: Path) -> bool:
    # langChain FAISS typically writes "index.faiss" (plus a .pkl for docstore)
    return (db_path / "index.faiss").exists()


def get_rag_pipeline() -> Optional[RAGPipeline]:
    global rag_pipeline
    if rag_pipeline is not None:
        return rag_pipeline

    try:
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(
            llm_model_name=OLLAMA_MODEL,
            embedding_model_name=EMBEDDING_MODEL,
            vector_db_path=str(VECTOR_DB_PATH),
            chunk_size=CHUNK_SIZE,          
            chunk_overlap=CHUNK_OVERLAP,
            top_k=TOP_K_RESULTS,
        )

        if not rag_pipeline.vector_store.is_initialized():
            logger.warning("Vector store not initialized yet. Upload PDFs & run processing.")
        else:
            logger.info("Vector store initialized successfully.")

        return rag_pipeline
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}", exc_info=True)
        return None
    
def _safe_filename(name: str) -> str:
    return Path(name).name

def _looks_like_pdf(raw: bytes) -> bool:
    # check: PDF files begin with "%PDF-"
    return len(raw) >= 5 and raw[:5] == b"%PDF-"

@app.get("/api/ui-config", response_model=UIConfigResponse)
async def get_ui_config():
    return UIConfigResponse(
        appName=APP_NAME,
        assistantName=ASSISTANT_NAME,
        defaultTheme=DEFAULT_THEME,
        themes=THEME_CHOICES,
    )


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if templates_available:
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory=str(templates_dir))
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse(
        "<html><body><h3>Simple FAQ Chatbot</h3><p>Try /api/status and /api/query</p></body></html>"
    )

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/readyz")
async def readyz():
    rag = get_rag_pipeline()
    vector_ready = bool(rag and rag.vector_store.is_initialized())
    llm_ready = _check_llm_provider_ready()
    return {
        "status": "ready" if (vector_ready and llm_ready) else "not ready",
        "vector_db_initialized": vector_ready,
        "llm_initialized": llm_ready,
    }

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    pdf_dir_path = Path(PDF_DIR)
    pdf_files = list(pdf_dir_path.glob("*.pdf")) if pdf_dir_path.exists() else []
    pdf_count = len(pdf_files)

    vector_db_path = Path(VECTOR_DB_PATH)
    vector_db_initialized = _faiss_index_present(vector_db_path)

    llm_initialized = _check_llm_provider_ready()

    return StatusResponse(
        status="ready" if vector_db_initialized and llm_initialized else "not ready",
        pdf_count=pdf_count,
        vector_db_initialized=vector_db_initialized,
        llm_initialized=llm_initialized,
        pdfs=[f.name for f in pdf_files],
    )

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag: Optional[RAGPipeline] = Depends(get_rag_pipeline)):
    if rag is None:
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline not initialized. Ensure model is reachable and the vector store is initialized.",
        )
    try:
        return rag.query(request.query)
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": f"An unexpected error occurred: {str(exc)}"})

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a single PDF and make it queryable immediately.
    Validation:
      - extension must be in ALLOWED_EXTENSIONS (default: .pdf)
      - content-type should be application/pdf (some clients send octet-stream; we allow it)
      - magic bytes must start with %PDF-
      - size <= MAX_UPLOAD_BYTES
    """
    filename = _safe_filename(file.filename or "")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} files are accepted.")

    # read into memory once
    payload = await file.read()
    await file.close()

    if len(payload) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)} MB allowed.")

    ct = (getattr(file, "content_type", "") or "").lower()
    if "pdf" not in ct and ct not in ("", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Content-Type must be application/pdf.")

    if not _looks_like_pdf(payload):
        raise HTTPException(status_code=400, detail="File content is not a valid PDF (missing %PDF- header).")

   
    dest = Path(PDF_DIR) / filename
    dest.write_bytes(payload)

    # ingest
    rag = get_rag_pipeline()
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized.")
    svc = IngestService(rag.vector_store)
    with _ingest_lock:
        stats = svc.scan_and_sync(Path(PDF_DIR))

    return {"status": "ok", "saved_as": str(dest.name), "ingest": stats}

@app.post("/api/reindex")
async def reindex(mode: str = "incremental"):
    """
    Re-scan the raw folder and:
      - mode=incremental (default): add new files only
      - mode=full: rebuild the entire index
    """
    rag = get_rag_pipeline()
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized.")

    svc = IngestService(rag.vector_store)
    with _ingest_lock:
        if mode == "full":
            ok = svc._full_rebuild(Path(PDF_DIR))
            if not ok:
                raise HTTPException(status_code=500, detail="Full rebuild failed.")
            stats = {"mode": "full-rebuild"}
        else:
            stats = svc.scan_and_sync(Path(PDF_DIR))

    return {"status": "ok", "result": stats}

def start():
    uvicorn.run("src.api.app:app", host=HOST, port=PORT, reload=DEBUG)

if __name__ == "__main__":
    start()
