from __future__ import annotations
from typing import Dict, Any
import re


class ContextAnalyzer:
    """
    Analyzes context sufficiency and quality.
    """
    
    def __init__(self):
        self.min_context_length = 200
        self.min_sources = 2
    
    def check_sufficiency(self, question: str, context: str) -> Dict[str, Any]:
        """Check if context is sufficient to answer the question"""
        if not context or not context.strip():
            return {
                "score": 0.0,
                "reasoning": "No context available",
                "sufficient": False
            }
        
        score = 0.0
        reasons = []
        
        # Length check
        if len(context) >= self.min_context_length:
            score += 0.3
            reasons.append("Adequate context length")
        else:
            reasons.append("Context too short")
        
        # Keyword overlap
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        overlap = len(question_words.intersection(context_words))
        overlap_ratio = overlap / len(question_words) if question_words else 0
        
        if overlap_ratio > 0.3:
            score += 0.4
            reasons.append(f"Good keyword overlap ({overlap_ratio:.2f})")
        else:
            reasons.append(f"Low keyword overlap ({overlap_ratio:.2f})")
        
        # Legal entities check
        if self._has_legal_entities(context):
            score += 0.3
            reasons.append("Contains legal entities")
        else:
            reasons.append("Missing legal entities")
        
        return {
            "score": min(score, 1.0),
            "reasoning": "; ".join(reasons),
            "sufficient": score >= 0.6
        }
    
    def assess_quality(self, context: str) -> Dict[str, Any]:
        """Assess context quality"""
        if not context:
            return {"score": 0.0, "issues": ["No context"]}
        
        score = 0.5
        issues = []
        
        if re.search(r'\b\d{6,}CJ\d{4,}\b', context):
            score += 0.2
        else:
            issues.append("No CELEX IDs found")
        
        if re.search(r'\[.*?\]', context):
            score += 0.2
        else:
            issues.append("No case titles found")
        
        if len(context) > 500:
            score += 0.1
        else:
            issues.append("Context too short")
        
        return {
            "score": min(score, 1.0),
            "issues": issues
        }
    
    def _has_legal_entities(self, text: str) -> bool:
        """Check for legal entities in text"""
        patterns = [
            r'\b\d{6,}CJ\d{4,}\b',  # CELEX
            r'\[.*?\]',  # Case titles
            r'\bArticle\s+\d+',  # Article references
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
