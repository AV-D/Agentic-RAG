from agents.dialogue_manager import DialogueManagerAgent
from agents.retriever import RetrieverAgent
from agents.retrieval_evaluator import RetrievalEvaluatorAgent
from agents.answer_generator import AnswerGeneratorAgent
from agents.answer_evaluator import AnswerEvaluatorAgent
from agents.hallucination_checker import HallucinationCheckerAgent
from agents.final_answer import FinalAnswerAgent
from utils.web_search import WebSearch
from config.config import WEB_SEARCH_ENABLED
from typing import Dict, Any, List, Optional
import json

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot with all its components."""
        # Initialize all agent components
        self.dialogue_manager = DialogueManagerAgent()
        self.retriever = RetrieverAgent()
        self.retrieval_evaluator = RetrievalEvaluatorAgent()
        self.answer_generator = AnswerGeneratorAgent()
        self.answer_evaluator = AnswerEvaluatorAgent()
        self.hallucination_checker = HallucinationCheckerAgent()
        self.final_answer_agent = FinalAnswerAgent()
        
        # Initialize web search if enabled
        self.web_search = WebSearch() if WEB_SEARCH_ENABLED else None
    
    def process_query(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: The user's question
            debug: Whether to include debug information in the response
            
        Returns:
            Dictionary with the final answer and optionally debug information
        """
        result = {"query": query}
        debug_info = {}
        
        # Step 1: Analyze the query with the dialogue manager
        print("Analyzing query...")
        query_analysis = self.dialogue_manager.analyze_query(query)
        if debug:
            debug_info["query_analysis"] = query_analysis
        
        # Step 2: Retrieve information from the knowledge base
        print("Retrieving from knowledge base...")
        retrieved_docs = self.retriever.retrieve(query)
        if debug:
            debug_info["retrieved_docs"] = retrieved_docs
        
        # Step 3: Evaluate retrieval quality
        print("Evaluating retrieval quality...")
        retrieval_evaluation = self.retrieval_evaluator.evaluate_retrieval(query, retrieved_docs)
        if debug:
            debug_info["retrieval_evaluation"] = retrieval_evaluation
        
        # Step 4: Decide if web search is needed
        web_results = None
        retrieval_sufficient = self.retrieval_evaluator.is_retrieval_sufficient(retrieval_evaluation)
        
        if not retrieval_sufficient and WEB_SEARCH_ENABLED:
            print("Retrieval insufficient, performing web search...")
            web_results = self.web_search.search(query)
            if debug:
                debug_info["web_results"] = web_results
        
        # Step 5: Generate answer
        print("Generating answer...")
        answer = self.answer_generator.generate_answer(query, retrieved_docs, web_results)
        if debug:
            debug_info["initial_answer"] = answer
        
        # Step 6: Evaluate answer quality
        print("Evaluating answer quality...")
        answer_evaluation = self.answer_evaluator.evaluate_answer(query, answer, retrieved_docs)
        if debug:
            debug_info["answer_evaluation"] = answer_evaluation
        
        # Step 7: Check for hallucinations
        print("Checking for hallucinations...")
        hallucination_check = self.hallucination_checker.check_hallucinations(query, answer, retrieved_docs)
        if debug:
            debug_info["hallucination_check"] = hallucination_check
        
        # Step 8: Produce final answer
        print("Refining final answer...")
        final_answer = self.final_answer_agent.refine_answer(
            query, 
            answer,
            answer_evaluation.get("response", None) if isinstance(answer_evaluation, dict) else None,
            hallucination_check.get("response", None) if isinstance(hallucination_check, dict) else None
        )
        
        # Prepare the result
        result["answer"] = final_answer
        if debug:
            result["debug"] = debug_info
        
        return result
    
    def chat(self, query: str) -> str:
        """
        Simple chat interface that returns just the answer.
        
        Args:
            query: The user's question
            
        Returns:
            The final answer as a string
        """
        result = self.process_query(query)
        return result["answer"]


# Example usage
if __name__ == "__main__":
    chatbot = RAGChatbot()
    
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        result = chatbot.process_query(user_input, debug=True)
        print("\nAnswer:")
        print(result["answer"])
        
        # Optional: Print debug info
        print("\nDebug info:")
        print(json.dumps(result.get("debug", {}), indent=2, default=str)) 