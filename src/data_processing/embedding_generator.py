import os
from pathlib import Path
import logging
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")
        try:
            self.model_name = SentenceTransformer(model_name)
            self.embedding_dim = self.model_name.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded successfully with dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def generate_embedding(self, texts: str) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts or not texts.strip():
            logger.warning("No texts provided for embedding generation.")
            return np.zeros(self.embedding_dim)

        try:
            embedding = self.model_name.encode(texts, show_progress_bar=False)
            logger.info(f"Generated embeddings for {len(texts)} texts.")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.zeros(self.embedding_dim)
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for a list of texts."""
        if not texts:
            logger.warning("No texts provided for embedding generation.")
            return []

        try:
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model_name.encode(texts, batch_size=batch_size, show_progress_bar=True)
            logger.info(f"Successfully Generated {len(embeddings)} embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [np.zeros(self.embedding_dim) for _ in texts]
        
    def process_document(self, documents: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Process a list of documents to generate embeddings."""
        if not documents:
            logger.warning("No documents provided for embedding generation.")
            return []

        texts = [doc['text'] for doc in documents]
        embeddings = self.generate_embeddings(texts, batch_size)

        for i, doc in enumerate(documents):
            doc['embedding'] = embeddings[i] if i < len(embeddings) else np.zeros(self.embedding_dim)
        
        logger.info(f"Processed {len(documents)} documents with embeddings.")
        return documents
    
if __name__ == "__main__":
    # Example usage
    from configs.config import EMBEDDING_MODEL, PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from .pdf_processor import PDFProcessor
    
    processor = PDFProcessor(PDF_DIR)
    documents = processor.process_all_pdfs(CHUNK_SIZE, CHUNK_OVERLAP)

    if documents:
        generator = EmbeddingGenerator(model_name=EMBEDDING_MODEL)
        document_with_embeddings = generator.process_document(documents)
        logger.info(f"Generated embeddings for {len(document_with_embeddings)} documents.")
        if document_with_embeddings:
            logger.info(f"Sample document with embedding: {document_with_embeddings[0]}")
            logger.info(f"Sample embedding shape: {document_with_embeddings[0]['embedding'].shape}")
    
    else:
        logger.warning("No documents found to process.")