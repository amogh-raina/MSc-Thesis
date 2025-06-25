from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from Agent.tools.dataset_tool import DatasetTool
from Agent.tools.context_analyzer import ContextAnalyzer


class MultiStageRetriever:
    """
    Handles multi-stage retrieval: Vector Store -> Dataset -> Web Search
    """
    
    def __init__(self, vector_store, dataset_df, web_search_tool=None, title_index=None):
        self.vector_store = vector_store
        self.dataset_tool = DatasetTool(dataset_df, title_index)
        self.web_search_tool = web_search_tool
        self.context_analyzer = ContextAnalyzer()
        
        # Thresholds for moving to next stage
        self.vector_threshold = 0.6
        self.dataset_threshold = 0.7
    
    async def retrieve_context(self, question: str) -> Dict[str, Any]:
        """
        Perform multi-stage retrieval based on context sufficiency.
        
        Returns:
            Dict with context, confidence, sources, and stages used
        """
        all_context = []
        all_sources = []
        stages_used = []
        confidence_scores = []
        
        # Stage 1: Vector Store Retrieval
        vector_result = await self._vector_retrieval(question)
        all_context.extend(vector_result["context"])
        all_sources.extend(vector_result["sources"])
        stages_used.append("vector_store")
        confidence_scores.append(vector_result["confidence"])
        
        # Check if vector retrieval is sufficient
        current_context = "\n\n".join(all_context)
        sufficiency = self.context_analyzer.check_sufficiency(question, current_context)
        
        if sufficiency["score"] < self.vector_threshold:
            # Stage 2: Dataset Query
            dataset_result = await self._dataset_retrieval(question, vector_result["entities"])
            if dataset_result["context"]:
                all_context.extend(dataset_result["context"])
                all_sources.extend(dataset_result["sources"])
                stages_used.append("dataset_query")
                confidence_scores.append(dataset_result["confidence"])
                
                # Re-check sufficiency
                current_context = "\n\n".join(all_context)
                sufficiency = self.context_analyzer.check_sufficiency(question, current_context)
        
        if sufficiency["score"] < self.dataset_threshold and self.web_search_tool:
            # Stage 3: Web Search
            web_result = await self._web_search_retrieval(question, all_sources)
            if web_result["context"]:
                all_context.extend(web_result["context"])
                all_sources.extend(web_result["sources"])
                stages_used.append("web_search")
                confidence_scores.append(web_result["confidence"])
        
        # Calculate overall confidence and context quality
        final_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        context_quality = self.context_analyzer.assess_quality("\n\n".join(all_context))
        
        return {
            "context": "\n\n".join(all_context),
            "sources": all_sources,
            "stages_used": stages_used,
            "confidence": final_confidence,
            "context_quality": context_quality,
            "sufficiency_score": sufficiency["score"]
        }
    
    async def _vector_retrieval(self, question: str) -> Dict[str, Any]:
        """Stage 1: Vector store retrieval"""
        try:
            docs = self.vector_store.similarity_search(question, k=7)
            context = [doc.page_content for doc in docs]
            sources = [self._format_source(doc.metadata) for doc in docs]
            
            # Extract entities (case names, CELEX IDs) for next stage
            entities = self._extract_entities(docs)
            
            confidence = min(1.0, len(docs) / 7.0)  # Simple confidence based on retrieval count
            
            return {
                "context": context,
                "sources": sources,
                "entities": entities,
                "confidence": confidence
            }
        except Exception as e:
            return {
                "context": [],
                "sources": [],
                "entities": {},
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _dataset_retrieval(self, question: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Stage 2: Direct dataset querying"""
        try:
            results = []
            
            # Query by CELEX IDs found in vector results
            for celex_id in entities.get("celex_ids", []):
                celex_results = self.dataset_tool.search_by_celex(celex_id)
                results.extend(celex_results)
            
            # Query by case titles
            for case_title in entities.get("case_titles", []):
                title_results = self.dataset_tool.search_by_case_title(case_title)
                results.extend(title_results)
            
            # Remove duplicates
            unique_results = self._deduplicate_results(results)
            
            context = [self._format_dataset_context(result) for result in unique_results[:5]]
            sources = [self._format_dataset_source(result) for result in unique_results[:5]]
            
            confidence = min(1.0, len(unique_results) / 5.0)
            
            return {
                "context": context,
                "sources": sources,
                "confidence": confidence
            }
        except Exception as e:
            return {
                "context": [],
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _web_search_retrieval(self, question: str, existing_sources: List[Dict]) -> Dict[str, Any]:
        """Stage 3: Web search for additional context"""
        if not self.web_search_tool:
            return {"context": [], "sources": [], "confidence": 0.0}
        
        try:
            # Extract case names and CELEX IDs from existing sources for targeted search
            search_terms = self._extract_search_terms(existing_sources)
            
            web_results = []
            for term in search_terms[:3]:  # Limit to top 3 terms
                results = await self.web_search_tool.search_case_details(term)
                web_results.extend(results)
            
            context = [result["content"] for result in web_results[:3]]
            sources = [self._format_web_source(result) for result in web_results[:3]]
            
            confidence = min(1.0, len(web_results) / 3.0)
            
            return {
                "context": context,
                "sources": sources,
                "confidence": confidence
            }
        except Exception as e:
            return {
                "context": [],
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _extract_entities(self, docs) -> Dict[str, List[str]]:
        """Extract CELEX IDs and case titles from retrieved documents"""
        celex_ids = []
        case_titles = []
        
        for doc in docs:
            metadata = doc.metadata
            if "celex" in metadata and metadata["celex"]:
                celex_ids.append(metadata["celex"])
            if "case_title" in metadata and metadata["case_title"]:
                case_titles.append(metadata["case_title"])
        
        return {
            "celex_ids": list(set(celex_ids)),
            "case_titles": list(set(case_titles))
        }
    
    def _format_source(self, metadata: Dict) -> Dict[str, Any]:
        """Format vector store source metadata"""
        return {
            "type": "vector_store",
            "celex": metadata.get("celex", ""),
            "case_title": metadata.get("case_title", ""),
            "para_no": metadata.get("para_no", ""),
            "date": metadata.get("date", "")
        }
    
    def _format_dataset_context(self, result: pd.Series) -> str:
        """Format dataset result as context"""
        title = result.get("TITLE_FROM", "Unknown Case")
        text = result.get("TEXT_FROM", "")
        celex = result.get("CELEX_FROM", "")
        para = result.get("NUMBER_FROM", "")
        
        return f"[{title}] (#{celex}:{para})\n{text}"
    
    def _format_dataset_source(self, result: pd.Series) -> Dict[str, Any]:
        """Format dataset source metadata"""
        return {
            "type": "dataset",
            "celex": result.get("CELEX_FROM", ""),
            "case_title": result.get("TITLE_FROM", ""),
            "para_no": result.get("NUMBER_FROM", ""),
            "date": result.get("DATE_FROM", "")
        }
    
    def _format_web_source(self, result: Dict) -> Dict[str, Any]:
        """Format web search source metadata"""
        return {
            "type": "web_search",
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "source": result.get("source", "")
        }
    
    def _deduplicate_results(self, results: List[pd.Series]) -> List[pd.Series]:
        """Remove duplicate results based on CELEX ID and paragraph number"""
        seen = set()
        unique_results = []
        
        for result in results:
            key = (result.get("CELEX_FROM", ""), result.get("NUMBER_FROM", ""))
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results
    
    def _extract_search_terms(self, sources: List[Dict]) -> List[str]:
        """Extract search terms from existing sources for web search"""
        terms = []
        for source in sources:
            if source.get("case_title"):
                terms.append(source["case_title"])
            if source.get("celex"):
                terms.append(f"CELEX {source['celex']}")
        
        return list(set(terms)) 