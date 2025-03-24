from agents.base_agent import BaseAgent
from config.config import DIALOGUE_MANAGER_MODEL, OLLAMA_BASE_URL

class DialogueManagerAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are a helpful assistant that manages the flow of a conversation. 
        Your job is to:
        1. Understand the user's query
        2. Determine if the query requires retrieving information or can be answered directly
        3. Coordinate with other agents to retrieve information and generate an answer
        4. Ensure the final response is coherent and helpful
        
        Always be polite and professional in your responses.
        """
        
        super().__init__(
            model_name=DIALOGUE_MANAGER_MODEL,
            base_url=OLLAMA_BASE_URL,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for more consistent responses
        )
    
    def analyze_query(self, query: str) -> dict:
        """
        Analyze the user's query to determine the appropriate flow.
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary with analysis results
        """
        prompt = f"""
        Analyze the following user query:
        
        "{query}"
        
        Please determine:
        1. If this requires knowledge retrieval or can be answered directly
        2. The main topic(s) of the query
        3. If the query is ambiguous and needs clarification
        
        Respond in JSON format with the following fields:
        - needs_retrieval: boolean
        - topics: list of strings
        - is_ambiguous: boolean
        - suggested_clarification: string (only if ambiguous)
        """
        
        return self.structured_call(prompt) 