import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from typing import List, Dict, Any

from config.config import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    RETRIEVAL_CHUNK_SIZE,
    RETRIEVAL_CHUNK_OVERLAP,
    TOP_K_RESULTS,
)

class VectorStore:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        # Create vector store directory if it doesn't exist
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        
        # Initialize or load the vector store
        self.db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embeddings
        )
        
    def index_documents(self, directory_path: str) -> None:
        """Index documents from a directory into the vector store."""
        # Load documents
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RETRIEVAL_CHUNK_SIZE,
            chunk_overlap=RETRIEVAL_CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add documents to vector store
        self.db.add_documents(chunks)
        self.db.persist()
        
        return len(chunks)
    
    def similarity_search(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Perform similarity search for the query."""
        results = self.db.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
            
        return formatted_results 