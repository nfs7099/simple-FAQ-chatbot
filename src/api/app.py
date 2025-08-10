import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import ollama
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.rag_pipeline.rag_chain import RAGPipeline
from configs.config import (
    OLLAMA_MODEL, EMBEDDING_MODEL, CHUNK_OVERLAP, CHUNK_SIZE,
    PDF_DIR, VECTOR_DB_PATH, TOP_K_RESULTS, HOST, PORT
)

logger = logging.getLogger(__name__)

# request and response models
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
    

# Intialize FastAPI app
app = FastAPI(
    title="Simple FAQ Chatbot",
    description="A simple FAQ chatbot using RAG pipeline.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static files
static_dir = Path(__file__).resolve().parent.parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# templates
templates_dir = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


# Global RAG pipeline instance
rag_pipline = None

def get_rag_pipeline() -> RAGPipeline:

    global rag_pipline

    if rag_pipline is not None:
        return rag_pipline
    
    try:
        if rag_pipline is None:
            rag_pipline = RAGPipeline(
                llm_model_name=OLLAMA_MODEL,
                embedding_model_name=EMBEDDING_MODEL,
                vector_db_path=VECTOR_DB_PATH,
                chuck_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                top_k=TOP_K_RESULTS
            )

        # Check if vector store is initialized
        if not rag_pipline.is_vector_store_initialized():
            logger.warning("Vector store not initialized.")
        else:
            logger.info("Vector store initialized successfully.")

        return rag_pipline
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        return None
    

@app.get("/", response_class=HTMLResponse)
async def root(request: Request): 
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    # Check pdf directory
    pdf_dir_path = Path(PDF_DIR)
    pdf_exists = pdf_dir_path.exists() and pdf_dir_path.is_dir()
    pdf_files = list(pdf_dir_path.glob("*.pdf")) if pdf_exists else []
    pdf_count = len(pdf_files)

    # Check vector database
    vector_db_path = Path(VECTOR_DB_PATH)
    vector_db_initialized = vector_db_path.exists() and any(vector_db_path.iterdir())

    # Check ollam model
    llm_initialized = True
    try:
        ollama.list()
    except Exception as e:
        logger.error(f"Error connecting to Ollama model: {e}")
        llm_initialized = False


    return StatusResponse(
        status="ready" if vector_db_initialized and llm_initialized else "not ready",
        pdf_count=pdf_count,
        vector_db_initialized=vector_db_initialized,
        llm_initialized=llm_initialized,
        pdfs=[file.name for file in pdf_files]
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag: Optional[RAGPipeline] = Depends(get_rag_pipeline)):
    if rag is None:
        raise HTTPException(
            status_code=500, 
            detail="RAG pipeline not initialized. Please check if the llama model is running and vector store is initialized."
        )

    try:
        result = rag.query(request.query)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}", exec_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )
    

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

def start():
    uvicorn.run("src.api.app:app", host=HOST, port=PORT, reload=True)


if __name__ == "__main__":
    start() 