import os
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.config import (
    OLLAMA_MODEL, EMBEDDING_MODEL, CHUNK_OVERLAP, CHUNK_SIZE,
    PDF_DIR, VECTOR_DB_PATH, TOP_K_RESULTS,
)
from configs.config import validate_configuration
from src.rag_pipeline.rag_chain import RAGPipeline

logger = logging.getLogger(__name__)


def process_pdfs(
    pdf_dir: Path = Path(PDF_DIR),
    vector_db_path: Path = Path(VECTOR_DB_PATH),
    smoke_test: bool = False,
) -> bool:
    try:
        errors = validate_configuration()
        if errors:
            for err in errors:
                logger.error(f"Config error: {err}")
            logger.error("Configuration invalid. Aborting PDF processing.")
            return False

        logger.info("Starting PDF processing and vector database initialization...")
        logger.info(f"PDF directory: {pdf_dir}")
        logger.info(f"Vector DB path: {vector_db_path}")

        if not pdf_dir.exists() or not pdf_dir.is_dir():
            logger.info(f"Creating PDF directory at: {pdf_dir}")
            pdf_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"No PDFs found. Please add PDF files to {pdf_dir} and run this again.")
            return False

        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}. Please add PDF files to this directory.")
            return False
        logger.info(f"Found {len(pdf_files)} PDF(s): {[file.name for file in pdf_files]}")

        rag = RAGPipeline(
            llm_model_name=OLLAMA_MODEL,
            embedding_model_name=EMBEDDING_MODEL,
            vector_db_path=str(vector_db_path),
            chunk_size=CHUNK_SIZE,            
            chunk_overlap=CHUNK_OVERLAP,
            top_k=TOP_K_RESULTS,
        )

        logger.info("Loading documents into the vector store...")
        success = rag.load_documents(str(pdf_dir))
        if not success:
            logger.error("Failed to load documents into the vector store. See logs above for details.")
            return False

        logger.info("Successfully processed PDF files and initialized vector database.")
        logger.info(f"Vector database stored at: {vector_db_path}")

        if smoke_test:
            test_question = "What is e-waste and why is it important to recycle it?"
            logger.info(f"Smoke test — asking: {test_question}")
            response = rag.query(test_question)
            logger.info("Smoke test answer:\n" + response.get("answer", ""))
            logger.info(f"Smoke test sources: {len(response.get('sources', []))} document(s).")

        return True

    except Exception as e:
        logger.error(f"An error occurred during PDF processing or vector database initialization: {e}", exc_info=True)
        return False


def main() -> None:
    smoke = os.getenv("PROCESS_SMOKE_TEST", "false").strip().lower() == "true"
    ok = process_pdfs(smoke_test=smoke)
    if ok:
        logger.info("PDF processing completed successfully.")
    else:
        logger.warning("PDF processing did not complete successfully.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Process PDFs and initialize the vector store.")
    parser.add_argument("--pdf-dir", type=str, default=str(PDF_DIR), help="Directory containing PDF files")
    parser.add_argument("--vector-db", type=str, default=str(VECTOR_DB_PATH), help="Directory for the FAISS index")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick Q&A after indexing")
    args = parser.parse_args()

    process_pdfs(pdf_dir=Path(args.pdf_dir), vector_db_path=Path(args.vector_db), smoke_test=args.smoke_test)
