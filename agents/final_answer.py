from agents.base_agent import BaseAgent
from config.config import ANSWER_GENERATION_MODEL, OLLAMA_BASE_URL
from typing import Dict, Any, Optional

class FinalAnswerAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are a final answer agent responsible for refining and polishing answers.
        Your job is to:
        1. Consider the original answer and all evaluation feedback
        2. Correct any issues identified in the evaluations
        3. Enhance the clarity, coherence, and completeness of the answer
        4. Ensure the final answer is factually accurate and well-structured
        
        Your goal is to deliver the best possible response to the user.
        """
        
        super().__init__(
            model_name=ANSWER_GENERATION_MODEL,
            base_url=OLLAMA_BASE_URL,
            system_prompt=system_prompt,
            temperature=0.4
        )
    
    def refine_answer(
        self,
        query: str,
        original_answer: str,
        answer_evaluation: Optional[Dict[str, Any]] = None,
        hallucination_check: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Refine the answer based on evaluation feedback.
        
        Args:
            query: The user's question
            original_answer: The original generated answer
            answer_evaluation: Optional evaluation results for the answer
            hallucination_check: Optional hallucination check results
            
        Returns:
            Refined answer as a string
        """
        prompt = f"""
        User Query: "{query}"
        
        Original Answer:
        {original_answer}
        """
        
        # Add answer evaluation feedback if available
        if answer_evaluation:
            prompt += f"""
            Answer Evaluation:
            {answer_evaluation}
            """
        
        # Add hallucination check feedback if available
        if hallucination_check:
            prompt += f"""
            Hallucination Check:
            {hallucination_check}
            """
        
        prompt += """
        Please refine the original answer to address the issues identified in the evaluations:
        
        1. Correct any factual inaccuracies or hallucinations
        2. Improve clarity and coherence
        3. Add any missing information that's important for a complete answer
        4. Ensure the tone is helpful and professional
        5. Format the answer for readability
        
        Provide a refined, polished answer that best addresses the user's query.
        """
        
        return self.call(prompt) 