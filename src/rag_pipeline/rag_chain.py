import os
import sys
import logging
from typing import List, Dict, Any
from pathlib import Path
import logging

from langchain_community.llms import ollama

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.vector_db.vector_store import VectorStore


logger = logging.getLogger(__name__)

class RAGPipeline:

    def __init__(
        self, 
        llm_model_name: str = "gemma2:9b",
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        vector_db_path: str = "./data/vector_db",
        temperature: float = 0.1,
        top_k: int = 3,
        chuck_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.llm_model_name = llm_model_name 
        self.temperature = temperature

        self._init_llm()

        self.vectore_store = VectorStore(
            embedding_model_name=embedding_model_name,
            vector_db_path=vector_db_path,
            chunk_size=chuck_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k
        )


    def _init_llm(self):
        """Initialize the LLM model."""
        logger.info(f"Initializing LLM model: {self.llm_model_name}")
        try:
            self.llm = ollama.Ollama(
                model=self.llm_model_name,
                temperature=self.temperature
            )
            logger.info("LLM model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            sys.exit(1)
    
    def load_documents(self, documents: List[str]) -> bool:
        """Load documents into the vector store."""
        logger.info("Loading documents into the vector store.")
        return self.vectore_store.load_documents(documents)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the vector store and generate a response using the LLM."""
        logger.info(f"Processing query: {question}")
        try:
            docs = self.vectore_store.get_relevant_documents(question)

            if not docs:
                logger.info("No relevant documents found.")
                return {"answer": "No relevant documents found. Please make sure the vector store is populated with documents.",
                        "sources": []
                }
            
            # Create context from docuements
            context = "\n\n".join([doc.page_content for doc in docs])

            # Create a prompt for the LLM
            prompt = f"""
            You are a helpful assistant. Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            Keep the answer concise and relevant to the question.
            
            IMportant: Provide a direct answer without referring to the context as "already provided" or similar phrases.
            Just give the answer as if you're answering the question directly.
            
            Very Important: Preserve the exact formatting style from the source material. If the source materia uses a specific format for lists
            (such as "a. item1; b. item2; c. item3" or any other format), make sure to use the same format in your answer.
            For example, if the source uses lettered lists like "a. First Item; b. Second Item", use the same format rather than the numbered lists
            
            
            Context:
            {context}
            
            
            Question: {question}

            Answer:
            """

            answer = self.llm.invoke(prompt)

            sources = []
            for doc in docs:
                sources.append({
                    "contetnt": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 0)
                })

            return {
                "answer": answer,
                "sources": sources
            }
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")    
            return {
                "answer": f"I'm sorry, but an error occurred while processing your question: {str(e)}",
                "sources": []
            }
        
if __name__ == "__main__":

    from configs.config import OLLAMA_MODEL, EMBEDDING_MODEL, VECTOR_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, PDF_DIR
    # Example usage
    rag = RAGPipeline(
        llm_model_name=OLLAMA_MODEL,
        embedding_model_name=EMBEDDING_MODEL,
        vector_db_path=VECTOR_DB_PATH,
        chuck_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=TOP_K_RESULTS
    )

    success = rag.load_documents(PDF_DIR)

    if success:
        test_question = "What is e-waste and why is it important to recycle it?"
        response = rag.query(test_question)

        logger.info(f"\nQuestion: {test_question}")
        logger.info(f"Answer: {response['answer']}")
        logger.info(f"Sources:")
        for i, source in enumerate(response['sources']):
            logger.info(f"{i + 1}. Source: {source['source']}, Page: {source['page']}, Content: {source['contetnt'][:100]}...")
        logger.info("Documents loaded successfully into the vector store.")

    else:
        print("Failed to load documents into the vector store. Check the logs for more details.")