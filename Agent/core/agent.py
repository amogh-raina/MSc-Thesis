from __future__ import annotations
from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict
import asyncio
from datetime import datetime

from Agent.core.retrieval_stages import MultiStageRetriever
from Agent.core.grounding_manager import GroundingManager
from Agent.core.prompt_templates import PromptTemplateManager
from Agent.tools.context_analyzer import ContextAnalyzer
from Agent.utils.confidence_scorer import ConfidenceScorer
from Agent.utils.source_tracker import SourceTracker


class LegalQAAgent:
    """
    Main orchestrator for the agentic RAG system.
    Coordinates multi-stage retrieval, grounding assessment, and answer generation.
    """
    
    def __init__(
        self,
        vector_store,
        dataset_df,
        llm,
        web_search_tool=None,
        title_index=None,
        config: Dict[str, Any] = None
    ):
        self.vector_store = vector_store
        self.dataset_df = dataset_df
        self.llm = llm
        self.web_search_tool = web_search_tool
        self.title_index = title_index
        self.config = config or {}
        
        # Initialize components
        self.retriever = MultiStageRetriever(
            vector_store=vector_store,
            dataset_df=dataset_df,
            web_search_tool=web_search_tool,
            title_index=title_index
        )
        
        self.grounding_manager = GroundingManager()
        self.prompt_manager = PromptTemplateManager()
        self.context_analyzer = ContextAnalyzer()
        self.confidence_scorer = ConfidenceScorer()
        self.source_tracker = SourceTracker()
    
    async def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Main entry point for answering questions using the agentic RAG system.
        
        Args:
            question: The user's legal question
            
        Returns:
            Dict containing answer, metadata, sources, and confidence scores
        """
        start_time = datetime.now()
        
        # Stage 1: Multi-stage retrieval
        retrieval_result = await self.retriever.retrieve_context(question)
        
        # Stage 2: Assess grounding level
        grounding_level = self.grounding_manager.assess_grounding_level(
            question=question,
            context=retrieval_result["context"],
            confidence_score=retrieval_result["confidence"]
        )
        
        # Stage 3: Generate answer with appropriate grounding
        answer_result = await self._generate_contextual_answer(
            question=question,
            context=retrieval_result["context"],
            grounding_level=grounding_level
        )
        
        # Stage 4: Calculate final confidence and compile sources
        final_confidence = self.confidence_scorer.calculate_final_confidence(
            retrieval_confidence=retrieval_result["confidence"],
            grounding_level=grounding_level,
            context_quality=retrieval_result["context_quality"]
        )
        
        sources = self.source_tracker.compile_sources(
            retrieval_result["sources"],
            retrieval_result["stages_used"]
        )
        
        end_time = datetime.now()
        
        return {
            "answer": answer_result["answer"],
            "grounding_level": grounding_level,
            "confidence": final_confidence,
            "sources": sources,
            "retrieval_stages": retrieval_result["stages_used"],
            "context_quality": retrieval_result["context_quality"],
            "processing_time": (end_time - start_time).total_seconds(),
            "metadata": {
                "question": question,
                "timestamp": start_time.isoformat(),
                "context_length": len(retrieval_result["context"]),
                "num_sources": len(sources["formatted_sources"]) if sources.get("formatted_sources") else 0
            }
        }
    
    async def _generate_contextual_answer(
        self, 
        question: str, 
        context: str, 
        grounding_level: str
    ) -> Dict[str, Any]:
        """Generate answer using appropriate prompt template based on grounding level."""
        
        prompt_template = self.prompt_manager.get_template(grounding_level)
        prompt = prompt_template.format(question=question, context=context)
        
        try:
            response = await self.llm.ainvoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "answer": answer,
                "prompt_used": grounding_level,
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "prompt_used": grounding_level,
                "success": False,
                "error": str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and configuration."""
        return {
            "vector_store_ready": self.vector_store is not None,
            "dataset_size": len(self.dataset_df) if self.dataset_df is not None else 0,
            "web_search_enabled": self.web_search_tool is not None,
            "title_index_ready": self.title_index is not None,
            "llm_model": str(self.llm) if self.llm else "Not configured",
            "config": self.config
        }