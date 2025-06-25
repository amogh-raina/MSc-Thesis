from __future__ import annotations
from typing import Dict, Any


class AgentConfig:
    """Configuration settings for the legal QA agent"""
    
    def __init__(self):
        self.retrieval_config = {
            "vector_threshold": 0.6,
            "dataset_threshold": 0.7,
            "max_vector_results": 7,
            "max_dataset_results": 5,
            "max_web_results": 3
        }
        
        self.confidence_weights = {
            "retrieval_confidence": 0.4,
            "context_quality": 0.3,
            "grounding_level": 0.3
        }
        
        self.grounding_thresholds = {
            "strict": 0.8,
            "flexible": 0.6,
            "hybrid": 0.4,
            "enhanced": 0.0
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return {
            "retrieval": self.retrieval_config,
            "confidence": self.confidence_weights,
            "grounding": self.grounding_thresholds
        }
