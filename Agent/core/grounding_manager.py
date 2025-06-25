from __future__ import annotations
from typing import Dict, Any


class GroundingManager:
    """
    Assesses grounding level based on context quality and confidence.
    """
    
    def __init__(self):
        self.grounding_levels = {
            "strict": {"min_confidence": 0.8, "description": "High confidence, vector store only"},
            "flexible": {"min_confidence": 0.6, "description": "Medium confidence, dataset enhanced"},
            "hybrid": {"min_confidence": 0.4, "description": "Lower confidence, web enhanced"},
            "enhanced": {"min_confidence": 0.0, "description": "Low confidence, parametric knowledge"}
        }
    
    def assess_grounding_level(
        self, 
        question: str, 
        context: str, 
        confidence_score: float
    ) -> str:
        """
        Determine the appropriate grounding level based on context and confidence.
        
        Args:
            question: The user's question
            context: Retrieved context
            confidence_score: Confidence from retrieval
            
        Returns:
            Grounding level: "strict", "flexible", "hybrid", or "enhanced"
        """
        if confidence_score >= 0.8 and len(context) > 500:
            return "strict"
        elif confidence_score >= 0.6 and len(context) > 200:
            return "flexible"
        elif confidence_score >= 0.4:
            return "hybrid"
        else:
            return "enhanced"
    
    def get_grounding_info(self, level: str) -> Dict[str, Any]:
        """Get information about a grounding level"""
        return self.grounding_levels.get(level, self.grounding_levels["enhanced"])
