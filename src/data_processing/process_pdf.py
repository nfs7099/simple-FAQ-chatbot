import os
import sys
import logging
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.config import (
    OLLAMA_MODEL, EMBEDDING_MODEL, CHUNK_OVERLAP, CHUNK_SIZE,
    PDF_DIR, VECTOR_DB_PATH, TOP_K_RESULTS
)
from src.rag_pipeline.rag_chain import RAGPipeline

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting PDF processing and vector database initialization...")
    

    # Check if pdf directory exists
    pdf_dir_path = Path(PDF_DIR)
    if not pdf_dir_path.exists() or not pdf_dir_path.is_dir():
        logger.info(f"Creating PDF directory at: {pdf_dir_path}")
        pdf_dir_path.mkdir(parents=True, exist_ok=True)
        logger.warning(f"No PDFs found. Please add PDF files to {pdf_dir_path} and this script again.")
        return
    
    # Check if there are any PDF files in the directory
    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir_path}. Please add PDF files to this directory.")
        return
    logger.info(f"Found {len(pdf_files)} PDF files: {[file.name for file in pdf_files]}")

    try:
        # Inititalize the RAG pipeline
        rag = RAGPipeline(
            llm_model_name=OLLAMA_MODEL,
            embedding_model_name=EMBEDDING_MODEL,
            vector_db_path=str(VECTOR_DB_PATH),
            chuck_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            top_k=TOP_K_RESULTS
        )

        # Load documents into the vector store
        logger.info("Loading documents into the vector store...")
        success = rag.load_documents(PDF_DIR)

        if success:
            logger.info("Successfully processed PDF files and intitialized vector database.")
            logger.info(f"Vecotor database stored at: {VECTOR_DB_PATH}")

            test_question = "What is e-waste and why is it important to recycle it?"
            logger.info(f"Testing with question: {test_question}")
            response = rag.query(test_question)

            logger.info(f"\nQuestion: {test_question}")
            logger.info(f"Answer: {response['answer']}")
            logger.info(f"Sources: {len(response['sources'])} documents retrieved.")

            logger.info("PDF processing and vector database initialization completed successfully.")

        else:
            logger.error("Failed to load documents into the vector store. Please check the logs for details.")

    except Exception as e:
        logger.error(f"An error occurred during PDF processing or vector database initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()