import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Generator

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
        
    def process_documents_batch(self, documents_iterator, batch_size: int = 32) -> Generator[Dict[str, Any], None, None]:
        """Process documents in batches to generate embeddings."""
        current_batch = []
        
        for doc in documents_iterator:
            current_batch.append(doc)
            
            if len(current_batch) >= batch_size:
                # Process the current batch
                texts = [doc['text'] for doc in current_batch]
                embeddings = self.generate_embeddings(texts, batch_size)
                
                # Yield processed documents one by one
                for i, doc in enumerate(current_batch):
                    doc['embedding'] = embeddings[i] if i < len(embeddings) else np.zeros(self.embedding_dim)
                    yield doc
                
                # Clear the batch
                current_batch = []
                
        # Process any remaining documents
        if current_batch:
            texts = [doc['text'] for doc in current_batch]
            embeddings = self.generate_embeddings(texts, batch_size)
            
            for i, doc in enumerate(current_batch):
                doc['embedding'] = embeddings[i] if i < len(embeddings) else np.zeros(self.embedding_dim)
                yield doc
        
    
if __name__ == "__main__":
    # Example usage
    from configs.config import EMBEDDING_MODEL, PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from .pdf_processor import PDFProcessor
    
    processor = PDFProcessor(PDF_DIR)
    generator = EmbeddingGenerator(model_name=EMBEDDING_MODEL)
    
    processed_count = 0
    batch_size = 32
    
    try:
        # Process documents in batches
        for doc_with_embedding in generator.process_documents_batch(
            processor.process_all_pdfs(CHUNK_SIZE, CHUNK_OVERLAP), 
            batch_size
        ):
            processed_count += 1
            
            # Print sample for first few documents
            if processed_count <= 3:
                logger.info(f"Sample document {processed_count}:")
                logger.info(f"Text: {doc_with_embedding['text'][:100]}...")
                logger.info(f"Embedding shape: {doc_with_embedding['embedding'].shape}")
            
            # Print progress every 100 documents
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} documents so far...")
                
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    finally:
        logger.info(f"\nTotal documents processed: {processed_count}")