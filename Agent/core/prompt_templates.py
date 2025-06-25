from __future__ import annotations
from typing import Dict


class PromptTemplateManager:
    """
    Manages different prompt templates for different grounding levels.
    """
    
    def __init__(self):
        self.templates = {
            "strict": self._strict_template(),
            "flexible": self._flexible_template(),
            "hybrid": self._hybrid_template(),
            "enhanced": self._enhanced_template()
        }
    
    def get_template(self, grounding_level: str) -> str:
        """Get prompt template for the specified grounding level"""
        return self.templates.get(grounding_level, self.templates["enhanced"])
    
    def _strict_template(self) -> str:
        """Template for strict grounding - only use provided context"""
        return """You are an EU law expert. Answer the question using ONLY the information provided in the context below. Do not add any information not explicitly stated in the context.

Context:
{context}

Question: {question}

Answer: Based on the provided context, """
    
    def _flexible_template(self) -> str:
        """Template for flexible grounding - can paraphrase and connect ideas"""
        return """You are an EU law expert. Answer the question using the provided context as your primary source. You may paraphrase and connect related ideas from the context.

Context:
{context}

Question: {question}

Answer: """
    
    def _hybrid_template(self) -> str:
        """Template for hybrid grounding - can use some general knowledge"""
        return """You are an EU law expert. Answer the question using the provided context and your general knowledge of EU law. Clearly distinguish between information from the context and general legal principles.

Context:
{context}

Question: {question}

Answer: """
    
    def _enhanced_template(self) -> str:
        """Template for enhanced grounding - can use parametric knowledge"""
        return """You are an EU law expert. Answer the question using both the provided context (if any) and your knowledge of EU law. If the context is limited, rely on your legal expertise while noting the limitations.

Context:
{context}

Question: {question}

Answer: """
