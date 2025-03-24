from typing import List, Dict, Any
from duckduckgo_search import DDGS
from config.config import MAX_WEB_RESULTS

class WebSearch:
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query: str, max_results: int = MAX_WEB_RESULTS) -> List[Dict[str, Any]]:
        """
        Perform a web search using DuckDuckGo.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, body, href
        """
        try:
            results = self.ddgs.text(query, max_results=max_results)
            formatted_results = []
            
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "content": result.get("body", ""),
                    "url": result.get("href", ""),
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error during web search: {e}")
            return [] 