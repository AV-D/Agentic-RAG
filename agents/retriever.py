from agents.base_agent import BaseAgent
from utils.vector_store import VectorStore
from config.config import RETRIEVER_MODEL, OLLAMA_BASE_URL, TOP_K_RESULTS
from typing import List, Dict, Any

class RetrieverAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are a retrieval agent responsible for finding relevant information from a knowledge base.
        Your job is to:
        1. Process user queries and identify key information needs
        2. Generate appropriate search queries for the vector database
        3. Return the most relevant retrieved passages for the query
        """
        
        super().__init__(
            model_name=RETRIEVER_MODEL,
            base_url=OLLAMA_BASE_URL,
            system_prompt=system_prompt,
            temperature=0.2  # Lower temperature for more focused retrieval queries
        )
        
        self.vector_store = VectorStore()
    
    def generate_search_query(self, user_query: str) -> str:
        """
        Generate an optimized search query based on the user query.
        
        Args:
            user_query: The original user question
            
        Returns:
            Optimized search query for the vector database
        """
        prompt = f"""
        Given the user question: "{user_query}"
        
        Generate an optimized search query for a vector database that:
        1. Extracts the key concepts and entities
        2. Removes conversational filler words
        3. Focuses on the core information need
        4. Uses synonyms where appropriate to expand the search
        
        Return only the optimized search query text, nothing else.
        """
        
        return self.call(prompt)
    
    def retrieve(self, user_query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            user_query: The user's question
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with content and metadata
        """
        search_query = self.generate_search_query(user_query)
        return self.vector_store.similarity_search(search_query, k=k) 