from __future__ import annotations
from typing import Dict, Any, List


class ConfidenceScorer:
    """
    Calculates confidence scores based on retrieval results and context quality.
    """
    
    def __init__(self):
        self.weights = {
            "retrieval_confidence": 0.4,
            "context_quality": 0.3,
            "grounding_level": 0.3
        }
    
    def calculate_final_confidence(
        self,
        retrieval_confidence: float,
        grounding_level: str,
        context_quality: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate final confidence score based on multiple factors.
        
        Returns:
            Dict with overall confidence and component scores
        """
        # Map grounding level to confidence
        grounding_confidence_map = {
            "strict": 0.9,
            "flexible": 0.7,
            "hybrid": 0.5,
            "enhanced": 0.3
        }
        
        grounding_confidence = grounding_confidence_map.get(grounding_level, 0.3)
        quality_score = context_quality.get("score", 0.0)
        
        # Calculate weighted score
        final_score = (
            self.weights["retrieval_confidence"] * retrieval_confidence +
            self.weights["context_quality"] * quality_score +
            self.weights["grounding_level"] * grounding_confidence
        )
        
        return {
            "overall_confidence": min(final_score, 1.0),
            "components": {
                "retrieval_confidence": retrieval_confidence,
                "context_quality": quality_score,
                "grounding_confidence": grounding_confidence
            },
            "weights": self.weights
        }
