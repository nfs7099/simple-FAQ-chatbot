import os, sys
import logging
import shutil
from pathlib import Path
from typing import List
import logging

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

class VectorStore:

    def __init__(
        self, 
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        vector_db_path: str = "./data/vector_db", 
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3

    ):
        self.embedding_model = embedding_model_name
        self.vector_db_path = Path(vector_db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Ensure the vector database directory exists
        if not self.vector_db_path.exists():
            logger.info(f"Creating vector database directory at {self.vector_db_path}")
            self.vector_db_path.mkdir(parents=True, exist_ok=True)

        # Intialize the embedding model
        self._init_embedding_model()
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        

        # Vector store and retriever will be initialized later
        self.vector_store = None
        self.retriever = None

    def _init_embedding_model(self):
        """Initialize the embedding model if not already done."""
        logger.info(f"Initializing embedding model: {self.embedding_model}")
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embedding model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise


    def is_initialized(self) -> bool:
        """Check if the vector store is initialized."""
        if self.vector_store is None:
            try:
                self._init_retriever()
                return self.vector_store is not None
            except:
                return False
        else:
            return True
        
    def _init_retriever(self):
        if not self.vector_store:
            logger.warning("Vector store is not initialized. Initializing now...")
            try:
                self.vectore_store = FAISS.load_local(
                    folder_path=str(self.vector_db_path),
                    embeddings= self.embedding_model
                )
                logger.info(f"Loaded existing vectore stroe from {self.vector_db_path}")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                return

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        logger.info("Retriever initialized successfully.")


    def load_documents(self, pdf_dr: str)-> bool:
        """Load documents into the vector store."""
        pdf_dir_path = Path(pdf_dr)
        if not pdf_dir_path.exists():
            logger.error(f"PDF directory {pdf_dir_path} does not exist.")
            return False
        
        pdf_files = list(pdf_dir_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir_path}.")
            return False
        
        try:
            all_docs = []
            for pdf_file in pdf_files:
                logger.info(f"Loading PDF file: {pdf_file}")
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                for doc in documents:
                    doc.metadata['source'] = pdf_file.name

                split_docs = self.text_splitter.split_documents(documents)
                all_docs.extend(split_docs)
                logger.info(f"Loaded {len(split_docs)} documents from {pdf_file}")
            
            logger.info(f"Total documents loaded: {len(all_docs)}")

            # Delete existing vector store if it exists
            if self.vector_db_path.exists():
                logger.info(f"Deleting existing vector store at {self.vector_db_path}")
                shutil.rmtree(self.vector_db_path)
                logger.info("Existing vector store deleted.")

            # Create a new vector store directory
            self.vector_db_path.mkdir(parents=True, exist_ok=True)

            # create the vector store
            self.vector_store = FAISS.from_documents(
                documents=all_docs,
                embedding=self.embedding_model
            )

            # save the vector store to disk
            self.vector_store.save_local(str(self.vector_db_path))
            logger.info(f"Vector store created and saved at {self.vector_db_path}")

            # Initialize the retriever
            self._init_retriever()

            return True
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return False


    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents based on the query."""
        if not hasattr(self, 'retriever') or self.retriever is None:
            logger.error("Retriever is not initialized. Please load documents first.")
            return []
        
        try:
            logger.info(f"Retrieving relevant documents for query: {query}")
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} relevant documents.")
            return docs
        except Exception as e:
            logger.error(f"Failed to retrieve relevant documents: {e}")
            return []

