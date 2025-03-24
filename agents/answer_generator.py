from agents.base_agent import BaseAgent
from config.config import ANSWER_GENERATION_MODEL, OLLAMA_BASE_URL
from typing import List, Dict, Any, Optional

class AnswerGeneratorAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are an answer generation agent responsible for creating comprehensive and accurate responses.
        Your job is to:
        1. Use the provided context to answer the user's question accurately
        2. State only facts that are supported by the given context
        3. Indicate when information is incomplete or uncertain
        4. Provide a coherent, well-structured response
        
        Never make up information that is not provided in the context.
        """
        
        super().__init__(
            model_name=ANSWER_GENERATION_MODEL,
            base_url=OLLAMA_BASE_URL,
            system_prompt=system_prompt,
            temperature=0.5
        )
    
    def generate_answer(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]], 
        web_results: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate an answer based on the query and context.
        
        Args:
            query: The user's question
            context_docs: Retrieved documents from the knowledge base
            web_results: Optional web search results
            
        Returns:
            Generated answer as a string
        """
        # Format context documents
        formatted_context = ""
        for i, doc in enumerate(context_docs):
            formatted_context += f"Document {i+1}:\n{doc['content']}\n\n"
        
        # Format web results if available
        formatted_web = ""
        if web_results and len(web_results) > 0:
            formatted_web = "Web Search Results:\n"
            for i, result in enumerate(web_results):
                formatted_web += f"Web Result {i+1}:\nTitle: {result['title']}\nContent: {result['content']}\nURL: {result['url']}\n\n"
        
        prompt = f"""
        User Query: "{query}"
        
        Context Information:
        {formatted_context}
        
        {formatted_web}
        
        Based on the provided context, please generate a comprehensive answer to the user's query.
        - Only use information from the provided context
        - If the context doesn't contain enough information to fully answer the question, state that clearly
        - Structure your answer logically with paragraphs for different aspects of the answer
        - Include relevant details but be concise
        - Do not introduce information that isn't in the context
        
        Answer:
        """
        
        return self.call(prompt) 