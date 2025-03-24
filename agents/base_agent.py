from langchain_ollama import OllamaLLM
from typing import Dict, Any, List, Optional

class BaseAgent:
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize the base agent with Ollama LLM.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: URL of the Ollama server
            system_prompt: Optional system prompt to guide the model's behavior
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter (0-1)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = system_prompt
        
        # Initialize Ollama client
        self.llm = OllamaLLM(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            system=system_prompt
        )
    
    def call(self, prompt: str, **kwargs) -> str:
        """
        Call the Ollama model with the given prompt.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            The model's response as a string
        """
        try:
            response = self.llm.invoke(prompt, **kwargs)
            return response
        except Exception as e:
            print(f"Error calling Ollama model: {e}")
            return f"Error: {str(e)}"
            
    def structured_call(self, prompt: str, json_mode: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Call the Ollama model and attempt to get a structured JSON response.
        
        Args:
            prompt: Input prompt for the model
            json_mode: Whether to instruct the model to return JSON
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            The model's response as a dictionary if possible, otherwise returns
            a dictionary with the full text response
        """
        if json_mode:
            formatted_prompt = f"{prompt}\n\nPlease provide your response in valid JSON format."
        else:
            formatted_prompt = prompt
            
        try:
            response = self.call(formatted_prompt, **kwargs)
            
            # For simplicity, we're just returning the text as a dict here
            # In a real-world implementation, you would add code to parse JSON from the response
            return {"response": response}
        except Exception as e:
            print(f"Error in structured call: {e}")
            return {"error": str(e)} 