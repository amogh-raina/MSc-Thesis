from __future__ import annotations
from typing import List, Dict, Any, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
import os


class WebSearchTool:
    """
    Web search tool using Tavily API for legal information retrieval.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API key is required")
        
        self.search_tool = TavilySearchResults(
            api_key=self.api_key,
            max_results=5,
            search_depth="advanced"
        )
    
    async def search_case_details(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for case details using Tavily.
        
        Args:
            search_term: Case name, CELEX ID, or legal concept
            
        Returns:
            List of search results with content and metadata
        """
        try:
            # Enhance search term for legal context
            enhanced_query = f"EU law case {search_term} EUR-Lex European Court Justice"
            
            results = await self.search_tool.ainvoke({"query": enhanced_query})
            
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "content": result.get("content", ""),
                        "url": result.get("url", ""),
                        "source": "Tavily"
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in web search for '{search_term}': {e}")
            return []
    
    async def search_legal_concept(self, concept: str) -> List[Dict[str, Any]]:
        """Search for general legal concepts and principles"""
        try:
            query = f"EU law {concept} principle doctrine European Union Treaty"
            results = await self.search_tool.ainvoke({"query": query})
            
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "content": result.get("content", ""),
                        "url": result.get("url", ""),
                        "source": "Tavily"
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching legal concept '{concept}': {e}")
            return []
