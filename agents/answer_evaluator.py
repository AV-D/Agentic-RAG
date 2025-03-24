from agents.base_agent import BaseAgent
from config.config import EVALUATION_MODEL, OLLAMA_BASE_URL, ANSWER_QUALITY_THRESHOLD
from typing import List, Dict, Any

class AnswerEvaluatorAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are an evaluation agent responsible for assessing the quality of generated answers.
        Your job is to:
        1. Analyze if the answer correctly addresses the user's query
        2. Check if the answer is consistent with the provided context
        3. Evaluate the answer for completeness, clarity, and coherence
        4. Identify any issues or improvements needed
        
        Be thorough and critical in your evaluation.
        """
        
        super().__init__(
            model_name=EVALUATION_MODEL,
            base_url=OLLAMA_BASE_URL,
            system_prompt=system_prompt,
            temperature=0.3
        )
    
    def evaluate_answer(
        self, 
        query: str, 
        answer: str, 
        context_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated answer.
        
        Args:
            query: The user's question
            answer: The generated answer
            context_docs: The context documents used to generate the answer
            
        Returns:
            Evaluation results with scores and feedback
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
        
        Please evaluate the quality of the generated answer based on the following criteria:
        
        1. Relevance: Does the answer address the user's query?
        2. Accuracy: Is the answer consistent with the provided context?
        3. Completeness: Does the answer cover all aspects of the query?
        4. Clarity: Is the answer easy to understand?
        5. Coherence: Is the answer well-structured and logically organized?
        
        Provide your evaluation in JSON format with the following fields:
        - relevance_score: float between 0 and 1
        - accuracy_score: float between 0 and 1
        - completeness_score: float between 0 and 1
        - clarity_score: float between 0 and 1
        - coherence_score: float between 0 and 1
        - overall_score: float between 0 and 1 (average of all scores)
        - strengths: list of strings
        - weaknesses: list of strings
        - improvement_suggestions: list of strings
        """
        
        return self.structured_call(prompt)
    
    def is_answer_sufficient(self, evaluation_result: Dict[str, Any]) -> bool:
        """
        Determine if the answer quality is sufficient based on the evaluation.
        
        Args:
            evaluation_result: The evaluation result from evaluate_answer
            
        Returns:
            Boolean indicating if answer is sufficient
        """
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
        
        return overall_score >= ANSWER_QUALITY_THRESHOLD 