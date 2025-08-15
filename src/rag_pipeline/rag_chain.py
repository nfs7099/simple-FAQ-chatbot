import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.vector_db.vector_store import VectorStore
from configs.config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    RERANKER_MODEL,
    USE_TOKEN_SPLITTER,
    TOKENIZER_NAME,
    TOKEN_CHUNK_SIZE,
    TOKEN_CHUNK_OVERLAP,
    PDF_DIR,
    TOP_K_RESULTS,
)
from configs.config import validate_configuration

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline with:
      - Provider switch: ollama | openai | anthropic
      - Optional CrossEncoder reranking
      - Token-aware chunking support (configured via .env)
    """

    def __init__(
        self,
        llm_model_name: str = "gemma2:9b",
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        vector_db_path: str = "./data/vector_db",
        temperature: float = 0.1,
        top_k: int = 3,
        chunk_size: int = 500,      
        chunk_overlap: int = 50,
        enable_reranker: bool = True,
        reranker_model_name: Optional[str] = None,
    ):
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.top_k = top_k
        self.initial_k = max(self.top_k * 4, 20) # wider recall for reranking

        self.llm = self._build_llm(provider=LLM_PROVIDER, temperature=self.temperature)

        self.vector_store = VectorStore(
            embedding_model_name=embedding_model_name,
            vector_db_path=vector_db_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            use_token_splitter=USE_TOKEN_SPLITTER,
            tokenizer_name=TOKENIZER_NAME,
            token_chunk_size=TOKEN_CHUNK_SIZE,
            token_chunk_overlap=TOKEN_CHUNK_OVERLAP,
        )

        self.reranker = None
        self.reranker_model_name = reranker_model_name or RERANKER_MODEL
        if enable_reranker:
            self._init_reranker()

    # LLM factory
    def _build_llm(self, provider: str, temperature: float):
        provider = (provider or "").lower()
        logger.info(f"Initializing LLM provider={provider}")

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=self.llm_model_name or OLLAMA_MODEL, temperature=temperature)

        if provider == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai.")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)

        if provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic.")
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model_name=ANTHROPIC_MODEL, temperature=temperature)

        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    # reranker
    def _init_reranker(self):
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Initializing reranker: {self.reranker_model_name}")
            self.reranker = CrossEncoder(self.reranker_model_name)
            logger.info("Reranker initialized successfully.")
        except Exception as e:
            logger.warning(
                f"Reranker disabled (could not initialize '{self.reranker_model_name}'): {e}. "
                "Proceeding without reranking."
            )
            self.reranker = None

    # public API 
    def load_documents(self, documents_dir: Union[str, Path]) -> bool:
        """Load documents (PDF directory path) into the vector store."""
        logger.info("Loading documents into the vector store.")
        return self.vector_store.load_documents(str(documents_dir))

    def query(self, question: str) -> Dict[str, Any]:
        """Retrieve context, optionally rerank, and answer via the selected LLM."""
        logger.info(f"Processing query: {question}")
        try:
            docs = self._retrieve_candidates(question)

            if not docs:
                logger.info("No relevant documents found.")
                return {
                    "answer": "No relevant documents found. Please make sure the vector store is populated with documents.",
                    "sources": [],
                }

            if self.reranker is not None and len(docs) > self.top_k:
                docs = self._rerank_select(question, docs, top_k=self.top_k)

            context = "\n\n".join([d.page_content for d in docs])
            prompt = (
                "You are a precise assistant. Answer ONLY using the context. "
                "If the answer is not fully contained in the context, say: "
                "\"I don’t know from the provided files.\""
                "\n\nContext:\n"
                f"{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            sources = [
                {
                    "content": d.page_content,
                    "source": d.metadata.get("source", "Unknown"),
                    "page": d.metadata.get("page", 0),
                }
                for d in docs
            ]
            return {"answer": answer, "sources": sources}

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "answer": f"I'm sorry, but an error occurred while processing your question: {str(e)}",
                "sources": [],
            }

    def _retrieve_candidates(self, question: str):
        """Pull wider set (initial_k) for reranking, else retriever default."""
        try:
            if self.reranker is not None:
                return self.vector_store.similarity_search(question, k=self.initial_k)
            return self.vector_store.get_relevant_documents(question)
        except Exception as e:
            logger.error(f"Candidate retrieval failed: {e}", exc_info=True)
            return []

    def _rerank_select(self, question: str, docs: List[Any], top_k: int) -> List[Any]:
        """CrossEncoder reranking; returns top_k docs."""
        try:
            pairs = [(question, d.page_content) for d in docs]
            scores = self.reranker.predict(pairs)  # numpy array
            ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
            selected = [d for d, _ in ranked[:top_k]]
            logger.debug(f"Reranker selected top {len(selected)} / {len(docs)} docs.")
            return selected
        except Exception as e:
            logger.warning(f"Reranking failed, falling back to vector similarity order: {e}")
            return docs[:top_k]



if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Smoke test for RAGPipeline")
    parser.add_argument("--ask", type=str, default="What is e-waste and why is it important to recycle it?")
    parser.add_argument("--pdf-dir", type=str, default=str(PDF_DIR))
    parser.add_argument("--top-k", type=int, default=TOP_K_RESULTS)
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()

    errors = validate_configuration()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f" PDF dir: {args.pdf_dir}")
    print(f"Provider: {LLM_PROVIDER} | Model: {OLLAMA_MODEL if LLM_PROVIDER=='ollama' else OPENAI_MODEL if LLM_PROVIDER=='openai' else ANTHROPIC_MODEL}")
    print(f"Reranker: {'disabled' if args.no_reranker else RERANKER_MODEL} | top_k={args.top_k}")

    rag = RAGPipeline(
        llm_model_name=OLLAMA_MODEL,
        embedding_model_name="BAAI/bge-base-en-v1.5",
        vector_db_path=str(Path(PDF_DIR).parent / "vector_db"),
        chunk_size=800,                
        chunk_overlap=120,
        top_k=args.top_k,
        enable_reranker=not args.no_reranker,
    )

    loaded = rag.load_documents(args.pdf_dir)
    if not loaded:
        print("No PDFs loaded. Make sure there are .pdf files in the directory above.")
        sys.exit(1)

    result = rag.query(args.ask)
    print("\n================= ANSWER =================")
    print(result["answer"])
    print("=============== SOURCES =================")
    if result["sources"]:
        for i, s in enumerate(result["sources"], 1):
            print(f"{i}. {s.get('source')}  (page {s.get('page')})")
    else:
        print("(no sources)")
