from agents.base_agent import BaseAgent
from config.config import HALLUCINATION_CHECK_MODEL, OLLAMA_BASE_URL, HALLUCINATION_THRESHOLD
from typing import List, Dict, Any

class HallucinationCheckerAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are a hallucination detection agent responsible for identifying factual inaccuracies in generated answers.
        Your job is to:
        1. Compare the generated answer with the provided context
        2. Identify any claims in the answer that are not supported by the context
        3. Rate the severity of any hallucinations found
        4. Suggest corrections for hallucinations
        
        Be very strict about factual accuracy and evidence.
        """
        
        super().__init__(
            model_name=HALLUCINATION_CHECK_MODEL,
            base_url=OLLAMA_BASE_URL,
            system_prompt=system_prompt,
            temperature=0.2  # Lower temperature for more consistent evaluation
        )
    
    def check_hallucinations(
        self, 
        query: str, 
        answer: str, 
        context_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check the generated answer for hallucinations.
        
        Args:
            query: The user's question
            answer: The generated answer
            context_docs: The context documents used to generate the answer
            
        Returns:
            Hallucination check results with identified issues
        """
        # Format context documents
        formatted_context = ""
        for i, doc in enumerate(context_docs):
            formatted_context += f"Document {i+1}:\n{doc['content']}\n\n"
        
        prompt = f"""
        User Query: "{query}"
        
        Generated Answer:
        {answer}
        
        Context Used:
        {formatted_context}
        
        Please carefully check the generated answer for hallucinations or factual inaccuracies. A hallucination is any claim in the answer that:
        1. Is not supported by the provided context
        2. Contradicts information in the provided context
        3. Goes beyond what can be reasonably inferred from the context
        
        Extract each claim from the answer and verify if it is supported by the context.
        
        Provide your analysis in JSON format with the following fields:
        - hallucination_score: float between 0 and 1, where 0 means no hallucinations and 1 means completely hallucinated
        - hallucinated_claims: list of strings, each describing a claim that is not supported by the context
        - severity: string, one of ["none", "minor", "moderate", "major"] based on how problematic the hallucinations are
        - corrected_answer: string, a version of the answer with hallucinations removed or corrected
        """
        
        return self.structured_call(prompt)
    
    def is_answer_factual(self, hallucination_result: Dict[str, Any]) -> bool:
        """
        Determine if the answer has an acceptable level of factuality based on the hallucination check.
        
        Args:
            hallucination_result: The result from check_hallucinations
            
        Returns:
            Boolean indicating if answer is factual enough
        """
        try:
            hallucination_score = hallucination_result.get("response", {}).get("hallucination_score", 1.0)
            # If the response is a string, we need to handle it differently
            if isinstance(hallucination_score, str):
                try:
                    hallucination_score = float(hallucination_score)
                except ValueError:
                    hallucination_score = 1.0
        except:
            hallucination_score = 1.0
        
        return hallucination_score <= HALLUCINATION_THRESHOLD 