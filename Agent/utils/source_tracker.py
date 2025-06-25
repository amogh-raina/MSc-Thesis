from __future__ import annotations
from typing import Dict, Any, List


class SourceTracker:
    """
    Tracks and formats sources from different retrieval stages.
    """
    
    def compile_sources(
        self, 
        sources: List[Dict[str, Any]], 
        stages_used: List[str]
    ) -> Dict[str, Any]:
        """
        Compile and format sources from all retrieval stages.
        
        Returns:
            Dict with organized source information
        """
        compiled = {
            "total_sources": len(sources),
            "stages_used": stages_used,
            "sources_by_type": self._group_by_type(sources),
            "formatted_sources": self._format_sources(sources)
        }
        
        return compiled
    
    def _group_by_type(self, sources: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group sources by their type"""
        type_counts = {}
        for source in sources:
            source_type = source.get("type", "unknown")
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
        
        return type_counts
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Format sources for display"""
        formatted = []
        
        for i, source in enumerate(sources, 1):
            if source.get("type") == "vector_store":
                formatted.append(
                    f"{i}. {source.get('case_title', 'Unknown Case')} "
                    f"({source.get('celex', 'No CELEX')}:{source.get('para_no', 'N/A')})"
                )
            elif source.get("type") == "dataset":
                formatted.append(
                    f"{i}. {source.get('case_title', 'Unknown Case')} "
                    f"({source.get('celex', 'No CELEX')}:{source.get('para_no', 'N/A')}) [Dataset]"
                )
            elif source.get("type") == "web_search":
                formatted.append(
                    f"{i}. {source.get('title', 'Unknown')} "
                    f"({source.get('url', 'No URL')}) [Web]"
                )
            else:
                formatted.append(f"{i}. Unknown source")
        
        return formatted
