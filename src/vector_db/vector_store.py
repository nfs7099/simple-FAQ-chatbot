import logging
import shutil
from pathlib import Path
from typing import List, Optional

import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitters import TokenTextSplitter  # fallback if HF tokenizer path fails
    _TOKEN_SPLITTER_AVAILABLE = True
except Exception:
    _TOKEN_SPLITTER_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    _HF_TOKENIZER_AVAILABLE = True
except Exception:
    _HF_TOKENIZER_AVAILABLE = False

from configs.config import (
    EMBEDDING_PROVIDER,          
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,    
    ALLOW_DANGEROUS_DESERIALIZATION,
    ALLOWED_EXTENSIONS
)

logger = logging.getLogger(__name__)

def _has_pdf_magic(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(5)
        return head == b"%PDF-"
    except Exception:
        return False
    

def _faiss_index_present(path: Path) -> bool:
    return (path / "index.faiss").exists()


class VectorStore:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5", 
        vector_db_path: str = "./data/vector_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        use_token_splitter: bool = False,
        tokenizer_name: Optional[str] = None,     
        token_chunk_size: Optional[int] = None,
        token_chunk_overlap: Optional[int] = None,
    ):
        """
        Local FAISS-based vector store with provider-aware embeddings.

        Args:
            embedding_model_name: HF embedding model id (HF path only).
            vector_db_path: folder where FAISS index is stored.
            chunk_size / chunk_overlap: character-based splitter defaults.
            top_k: default retriever k.
            use_token_splitter: enable token-aware chunking (best with HF where tokenizer matches embeddings).
            tokenizer_name: HF tokenizer id for splitting (default: embedding_model_name).
            token_chunk_size / token_chunk_overlap: token-based params; fall back to char params if None.
        """
        self.embedding_model_name = embedding_model_name
        self.vector_db_path = Path(vector_db_path)
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.top_k = int(top_k)

        self.use_token_splitter = bool(use_token_splitter)
        self.tokenizer_name = tokenizer_name or embedding_model_name
        self.token_chunk_size = int(token_chunk_size or self.chunk_size)
        self.token_chunk_overlap = int(token_chunk_overlap or self.chunk_overlap)

        self.vector_db_path.mkdir(parents=True, exist_ok=True)

        self._init_embedding_model()
        self._init_text_splitter()

        self.vector_store: Optional[FAISS] = None
        self.retriever = None

    
    def _init_embedding_model(self):
        provider = (EMBEDDING_PROVIDER or "huggingface").lower()
        logger.info(f"Initializing embeddings provider={provider}")

        if provider == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")
            try:
                self.embedding_model = OpenAIEmbeddings(
                    model=OPENAI_EMBEDDING_MODEL, 
                )
                logger.info(f"OpenAI embeddings initialized: model={OPENAI_EMBEDDING_MODEL}")
                return
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}", exc_info=True)
                raise

        # Default: HuggingFace
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"HuggingFace embeddings initialized: model={self.embedding_model_name}, device={device}")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {e}", exc_info=True)
            raise

    def _init_text_splitter(self):
        """
        Prefer token-aware splitting via the SAME tokenizer as the embeddings (best with HF).
        If EMBEDDING_PROVIDER=openai, we don't have a matching HF tokenizer; we:
          - try TokenTextSplitter if available (generic tokenization),
          - else fall back to character-based splitting.
        """
        if self.use_token_splitter:
            if EMBEDDING_PROVIDER == "huggingface" and _HF_TOKENIZER_AVAILABLE:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
                    self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                        tokenizer=tokenizer,
                        chunk_size=self.token_chunk_size,
                        chunk_overlap=self.token_chunk_overlap,
                        add_start_index=True,
                    )
                    logger.info(
                        f"Using HF-tokenizer-based splitter "
                        f"(model={self.tokenizer_name}, size={self.token_chunk_size}, overlap={self.token_chunk_overlap})"
                    )
                    return
                except Exception as e:
                    logger.warning(
                        f"HF tokenizer splitter failed ({self.tokenizer_name}): {e}. "
                        "Will try TokenTextSplitter fallback."
                    )

            if _TOKEN_SPLITTER_AVAILABLE:
                try:
                    self.text_splitter = TokenTextSplitter(
                        chunk_size=self.token_chunk_size,
                        chunk_overlap=self.token_chunk_overlap,
                    )
                    logger.info(
                        f"Using TokenTextSplitter fallback "
                        f"(size={self.token_chunk_size}, overlap={self.token_chunk_overlap})"
                    )
                    return
                except Exception as e:
                    logger.warning(f"TokenTextSplitter fallback failed: {e}. Falling back to char-based splitter.")

        # fallback: character-based splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len
        )
        logger.info(
            f"Using character-based splitter (size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
    
    
    def is_initialized(self) -> bool:
        """True if the FAISS store has been loaded or built."""
        if self.vector_store is None:
            try:
                self._init_retriever()
                return self.vector_store is not None
            except Exception:
                return False
        return True

    def _init_retriever(self):
        if not self.vector_store:
            if not _faiss_index_present(self.vector_db_path):
                logger.info("No FAISS index on disk yet; retriever will be ready after first ingest.")
                return
            logger.warning("Vector store not initialized. Loading from disk...")
            try:
                logger.info(f"FAISS load_local dangerous_deserialization={ALLOW_DANGEROUS_DESERIALIZATION}")
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.vector_db_path),
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=ALLOW_DANGEROUS_DESERIALIZATION,
                )
                logger.info(f"Loaded vector store from {self.vector_db_path}")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}", exc_info=True)
                return

        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": self.top_k}
            )
            logger.info("Retriever initialized successfully.")


    # build or ingest
    def load_documents(self, pdf_dir: str) -> bool:
        """
        Build a fresh FAISS index from all valid PDFs in `pdf_dir`.

        Safety / robustness:
        - Only accepts files whose extensions are in ALLOWED_EXTENSIONS (default: .pdf)
        - Verifies PDF magic bytes ("%PDF-") before attempting to parse
        - Skips unreadable/corrupted files without failing the whole run
        - Rebuilds the FAISS index cleanly (deletes old folder)
        - Persists to disk and (re)initializes the retriever
        """
        pdf_dir_path = Path(pdf_dir)
        if not pdf_dir_path.exists() or not pdf_dir_path.is_dir():
            logger.error(f"PDF directory {pdf_dir_path} does not exist or is not a directory.")
            return False

        pdf_files = [
            p for p in pdf_dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir_path}.")
            return False

        try:
            all_docs: List[Document] = []

            for pdf_file in pdf_files:
                # to void renamed/non-PDF files crashing the loader
                if not _has_pdf_magic(pdf_file):
                    logger.warning(f"Skipping non-PDF or corrupted file: {pdf_file.name}")
                    continue

                logger.info(f"Loading PDF: {pdf_file.name}")
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    documents = loader.load()
                except Exception as e:
                    logger.warning(f"Skipping unreadable PDF {pdf_file.name}: {e}")
                    continue

                for doc in documents:
                    doc.metadata["source"] = pdf_file.name

                split_docs = self.text_splitter.split_documents(documents)
                all_docs.extend(split_docs)
                logger.info(f"  -> {len(split_docs)} chunks")

            if not all_docs:
                logger.warning("No chunks produced from PDFs (all files skipped or unreadable).")
                return False

            # rebuild index
            if self.vector_db_path.exists():
                logger.info(f"Deleting existing vector store at {self.vector_db_path}")
                shutil.rmtree(self.vector_db_path)
                self.vector_db_path.mkdir(parents=True, exist_ok=True)

            self.vector_store = FAISS.from_documents(
                documents=all_docs,
                embedding=self.embedding_model,
            )
            self.vector_store.save_local(str(self.vector_db_path))
            logger.info(f"Vector store saved at {self.vector_db_path}")

            self._init_retriever()
            return True

        except Exception as e:
            logger.error(f"Failed to load documents: {e}", exc_info=True)
            return False

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve default top_k documents using the retriever interface."""
        try:
            if not self.retriever:
                self._init_retriever()
            if not self.retriever:
                logger.error("Retriever is not initialized. Please load documents first.")
                return []
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} documents.")
            return docs
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}", exc_info=True)
            return []

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Direct FAISS similarity search for power users (e.g., wide recall before reranking).
        Falls back to retriever if FAISS store isn’t loaded.
        """
        try:
            if self.vector_store is None:
                self._init_retriever()
            if self.vector_store is not None:
                return self.vector_store.similarity_search(query, k or self.top_k)
            return self.get_relevant_documents(query)[: (k or self.top_k)]
        except Exception as e:
            logger.error(f"similarity_search failed: {e}", exc_info=True)
            return []
    
    def load_and_split_pdf(self, pdf_path: Path) -> List[Document]:
        """Load and split a single PDF into chunks (with source/page metadata)."""
        if pdf_path.suffix.lower() not in ALLOWED_EXTENSIONS or not _has_pdf_magic(pdf_path):
            logger.warning(f"Rejected non-PDF or corrupted file: {pdf_path.name}")
            return []
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
        except Exception as e:
            logger.warning(f"Skipping unreadable PDF {pdf_path.name}: {e}")
            return []
        for d in documents:
            d.metadata["source"] = pdf_path.name
        return self.text_splitter.split_documents(documents)

    def add_documents(self, docs: List[Document]) -> bool:
        """Incrementally add new documents to FAISS and persist."""
        try:
            if self.vector_store is None:
                self._init_retriever()
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(docs, self.embedding_model)
            else:
                self.vector_store.add_documents(docs)
            self.vector_store.save_local(str(self.vector_db_path))
            self._init_retriever()
            logger.info(f"Added {len(docs)} chunks and saved vector store.")
            return True
        except Exception as e:
            logger.error(f"add_documents failed: {e}", exc_info=True)
            return False