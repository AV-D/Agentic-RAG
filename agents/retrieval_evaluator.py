from agents.base_agent import BaseAgent
from config.config import EVALUATION_MODEL, OLLAMA_BASE_URL, RETRIEVAL_QUALITY_THRESHOLD
from typing import List, Dict, Any

class RetrievalEvaluatorAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are an evaluation agent responsible for assessing the quality and relevance of retrieved documents.
        Your job is to:
        1. Analyze the retrieved documents in relation to the user's query
        2. Determine if the retrieved documents contain the information needed to answer the query
        3. Provide a numerical score and reasoning for your evaluation
        
        Be critical but fair in your assessments.
        """
        
        super().__init__(
            model_name=EVALUATION_MODEL,
            base_url=OLLAMA_BASE_URL,
            system_prompt=system_prompt,
            temperature=0.3
        )
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the quality and relevance of retrieved documents.
        
        Args:
            query: The user's question
            retrieved_docs: List of retrieved documents
            
        Returns:
            Evaluation results with score and reasoning
        """
        # Format retrieved documents for the prompt
        formatted_docs = ""
        for i, doc in enumerate(retrieved_docs):
            formatted_docs += f"Document {i+1}:\n{doc['content']}\n\n"
        
        prompt = f"""
        User Query: "{query}"
        
        Retrieved Documents:
        {formatted_docs}
        
        Please evaluate the quality and relevance of these retrieved documents for answering the user's query:
        
        1. Assess how well the retrieved documents address the key information needs in the query
        2. Identify any missing information that would be needed to fully answer the query
        3. Determine if a web search would be beneficial to supplement the retrieved information
        
        Provide your evaluation in JSON format with the following fields:
        - relevance_score: float between 0 and 1, where 1 is perfectly relevant
        - completeness_score: float between 0 and 1, where 1 is completely sufficient
        - overall_score: float between 0 and 1, average of relevance and completeness
        - missing_information: list of strings describing key missing information
        - needs_web_search: boolean indicating if web search is needed
        - reasoning: string explaining your evaluation
        """
        
        return self.structured_call(prompt)
    
    def is_retrieval_sufficient(self, evaluation_result: Dict[str, Any]) -> bool:
        """
        Determine if the retrieval quality is sufficient based on the evaluation.
        
        Args:
            evaluation_result: The evaluation result from evaluate_retrieval
            
        Returns:
            Boolean indicating if retrieval is sufficient
        """
        # Extract the overall score from the evaluation result
        try:
            overall_score = evaluation_result.get("response", {}).get("overall_score", 0)
            # If the response is a string, we need to handle it differently
            if isinstance(overall_score, str):
                try:
                    overall_score = float(overall_score)
                except ValueError:
                    overall_score = 0
        except:
            overall_score = 0
        
        return overall_score >= RETRIEVAL_QUALITY_THRESHOLD 