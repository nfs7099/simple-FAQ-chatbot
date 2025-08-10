import unittest
import os
import sys
from pathlib import Path
import shutil

from langchain.embeddings import HuggingFaceEmbeddings

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline.rag_chain import RAGPipeline
from src.vector_db.vector_store import VectorStore
from configs.config import EMBEDDING_MODEL


class TestRAGPipeline(unittest.TestCase):
    
    def setUp(self):
        self.test_model_name = os.getenv("LLM_MODEL_NAME", "gemma2:9b")
        self.test_vector_db_path = "tests/test_vector_db"


    def test_embedding_model_initialization(self):

        try:
            # Use a smaller faster model for testing
            test_model = "all-MiniLM-L6-v2"

            embedding_model = HuggingFaceEmbeddings(
                model_name=test_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

            # Test embedding generation
            test_text = "This is a test sentence."
            embedding = embedding_model.embed_query(test_text)

            # Check that embedding is a list of floats with expected dimensions
            self.assertIsInstance(embedding, list)
            self.assertTrue(all(isinstance(x, float) for x in embedding))
            self.assertGreater(len(embedding), 0)  # MiniLM has 384 dimensions

        except Exception as e:
            self.fail(f"Embedding model initialization failed: {e}")


    def test_vector_store_initialization(self):
        try:
            vector_store = VectorStore(
                embedding_model_name=EMBEDDING_MODEL,
                vector_db_path=self.test_vector_db_path,
                chunk_size=100,
                chunk_overlap=10,
                top_k=1
            )
            
            self.assertIsNotNone(vector_store.embedding_model)
            self.assertIsNotNone(vector_store.text_splitter)
        except Exception as e:
            self.fail(f"Vector store initialization failed: {e}")
    
    def test_rag_pipeline_initialization(self):
        try:
            rag_pipeline = RAGPipeline(
                llm_model_name=self.test_model_name,
                embedding_model_name=EMBEDDING_MODEL,
                vector_db_path=self.test_vector_db_path,
                temperature=0.1,
                top_k=1,
                chuck_size=100,
                chunk_overlap=10
            )
            
            self.assertIsNotNone(rag_pipeline.llm)
            self.assertIsNotNone(rag_pipeline.vectore_store)
            self.assertIsNotNone(rag_pipeline.vectore_store.embedding_model)
            self.assertIsNotNone(rag_pipeline.vectore_store.text_splitter)

        except Exception as e:
            self.fail(f"RAG pipeline initialization failed: {e}")

    def tearDown(self):
        if Path(self.test_vector_db_path).exists():
            shutil.rmtree(self.test_vector_db_path)

if __name__ == "__main__":
    unittest.main()