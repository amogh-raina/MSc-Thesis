"""
Variant 1: Baseline GraphRAG Retrieval

Implementation: Query → Vector Search → Graph Expansion → Combined Context → LLM

This is the baseline variant for testing multi-relationship GraphRAG:
- Multiple relationships: CITES, CITES_CASE, CONTAINS
- No authority scoring (simple baseline)
- Simple ranking: vector results first, then graph results
- Simple concatenation of results
"""

from __future__ import annotations
from typing import List, Dict, Any
import logging
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from pydantic import Field

logger = logging.getLogger(__name__)


class BaselineGraphRAGRetriever(BaseRetriever):
    """
    Variant 1: Baseline GraphRAG Retriever
    
    This implements the simplest form of GraphRAG search:
    1. Vector search for semantic relevance
    2. Graph expansion using multiple relationships (CITES, CITES_CASE, CONTAINS)
    3. Simple concatenation (vector results first, then graph results)
    4. No authority scoring or complex ranking
    """
    
    vector_store: Any = Field(description="Neo4j vector store instance")
    k: int = Field(default=10, description="Number of initial vector search results")
    expansion_k: int = Field(default=20, description="Maximum additional results from graph expansion")
    max_total_results: int = Field(default=30, description="Maximum total results to return")
    variant_name: str = Field(default="Variant 1 (Baseline)", description="Variant identifier")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Internal method required by LangChain BaseRetriever
        
        Parameters
        ----------
        query : str
            Search query
            
        Returns
        -------
        List[Document]
            Retrieved documents with baseline GraphRAG
        """
        return self._variant_1_search(
            query=query,
            k=self.k,
            expansion_k=self.expansion_k,
            max_total_results=self.max_total_results
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents using baseline GraphRAG approach
        
        Parameters
        ----------
        query : str
            Search query
            
        Returns
        -------
        List[Document]
            Retrieved documents with baseline GraphRAG
        """
        # Use the internal method directly to avoid recursion
        return self._variant_1_search(
            query=query,
            k=self.k,
            expansion_k=self.expansion_k,
            max_total_results=self.max_total_results
        )
    
    def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version (falls back to sync for now)"""
        return self.get_relevant_documents(query)
    
    def _variant_1_search(
        self,
        query: str,
        k: int = 10,
        expansion_k: int = 20,
        max_total_results: int = 30
    ) -> List[Document]:
        """
        Variant 1: Baseline GraphRAG Search Implementation
        
        Implementation: Query → Vector Search → Graph Expansion → Combined Context → LLM
        
        This is the baseline variant for testing multi-relationship GraphRAG:
        - Multiple relationships: CITES, CITES_CASE, CONTAINS
        - No authority scoring (simple baseline)
        - Simple ranking: vector results first, then graph results
        - Simple concatenation of results
        
        Parameters
        ----------
        query : str
            Search query
        k : int, default 10
            Number of initial vector search results
        expansion_k : int, default 20
            Maximum additional results from graph expansion
        max_total_results : int, default 30
            Maximum total results to return
            
        Returns
        -------
        List[Document]
            Combined search results (vector + graph expansion)
        """
        if not self.vector_store.vector_store:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        logger.info(f"🔍 Variant 1 Search: query='{query[:50]}...', k={k}, expansion_k={expansion_k}")
        
        # Step 1: Initial vector search for semantic relevance
        logger.info("   🎯 Step 1: Vector search for semantic relevance...")
        initial_results = self.vector_store.custom_similarity_search(query, k=k, with_graph_context=False)
        
        if not initial_results:
            logger.warning("   ⚠️ No initial vector results found")
            return []
        
        logger.info(f"   ✅ Found {len(initial_results)} initial results")
        
        # Step 2: Graph expansion using multiple relationships
        logger.info("   🔗 Step 2: Graph expansion using multiple relationships...")
        
        # Extract paragraph IDs from initial results
        initial_para_ids = []
        for doc in initial_results:
            para_id = doc.metadata.get('paragraph_id')
            if para_id:
                initial_para_ids.append(para_id)
        
        if not initial_para_ids:
            logger.warning("   ⚠️ No valid paragraph IDs found in initial results")
            return initial_results
        
        # Multi-relationship graph expansion query - CITES, CITES_CASE, CONTAINS
        expansion_query = """
        // Get initial paragraphs and their cases
        MATCH (initial_para:Paragraph)
        WHERE initial_para.id IN $initial_para_ids
        OPTIONAL MATCH (initial_case:Case)-[:CONTAINS]->(initial_para)
        
        // Find related paragraphs through multiple relationships
        WITH initial_para, initial_case
        CALL (initial_para, initial_case) {
            WITH initial_para, initial_case
            // CITES relationship - direct paragraph citations
            MATCH (initial_para)-[:CITES]->(cited_para:Paragraph)
            WHERE cited_para.embedding IS NOT NULL
            RETURN cited_para, 'CITES' as relationship_type
            UNION
            WITH initial_para, initial_case
            // CITES_CASE relationship - case-level citations
            MATCH (initial_case)-[:CITES_CASE]->(cited_case:Case)-[:CONTAINS]->(cited_para:Paragraph)
            WHERE cited_para.embedding IS NOT NULL
            RETURN cited_para, 'CITES_CASE' as relationship_type
            UNION  
            WITH initial_para, initial_case
            // CONTAINS relationship - same case context
            MATCH (initial_case)-[:CONTAINS]->(cited_para:Paragraph)
            WHERE cited_para.embedding IS NOT NULL AND cited_para.id <> initial_para.id
            RETURN cited_para, 'CONTAINS' as relationship_type
        }
        
        // Format content and return results (with proper variable scoping)
        RETURN DISTINCT
            CASE 
                WHEN cited_para.celex IS NOT NULL AND cited_para.number IS NOT NULL AND cited_para.title IS NOT NULL
                THEN "(#" + cited_para.celex + ":" + toString(cited_para.number) + ") [" + cited_para.title + "] " + cited_para.text
                WHEN cited_para.celex IS NOT NULL AND cited_para.number IS NOT NULL  
                THEN "(#" + cited_para.celex + ":" + toString(cited_para.number) + ") " + cited_para.text
                ELSE cited_para.text
            END AS formatted_content,
            {
                celex: cited_para.celex,
                para_no: toString(cited_para.number),
                case_title: cited_para.title,
                paragraph_id: cited_para.id,
                node_type: "paragraph",
                expansion_source: "graph_multi_relationship",
                relationship_type: relationship_type,
                
                // Basic metadata
                word_count: size(split(cited_para.text, ' ')),
                char_count: size(cited_para.text)
            } AS metadata
        LIMIT $expansion_limit
        """
        
        try:
            expansion_results = self.vector_store.graph.query(
                expansion_query, 
                params={
                    "initial_para_ids": initial_para_ids,
                    "expansion_limit": expansion_k
                }
            )
            
            logger.info(f"   📈 Found {len(expansion_results)} expansion results")
            
        except Exception as e:
            logger.error(f"   ❌ Graph expansion failed: {e}")
            expansion_results = []
        
        # Step 3: Convert expansion results to Documents
        expansion_docs = []
        for result in expansion_results:
            doc = Document(
                page_content=result['formatted_content'],
                metadata=result['metadata']
            )
            expansion_docs.append(doc)
        
        # Step 4: Simple concatenation - vector results first, then graph results
        logger.info("   🔄 Step 3: Simple concatenation of results...")
        
        # Mark initial results with source
        for doc in initial_results:
            doc.metadata['expansion_source'] = 'vector_search'
        
        # Simple deduplication - avoid same paragraph appearing twice
        seen_paragraphs = set()
        final_results = []
        
        # Add vector results first (priority)
        for doc in initial_results:
            para_id = doc.metadata.get('paragraph_id')
            if para_id and para_id not in seen_paragraphs:
                final_results.append(doc)
                seen_paragraphs.add(para_id)
        
        # Add graph expansion results
        for doc in expansion_docs:
            para_id = doc.metadata.get('paragraph_id')
            if para_id and para_id not in seen_paragraphs:
                final_results.append(doc)
                seen_paragraphs.add(para_id)
        
        # Limit total results
        final_results = final_results[:max_total_results]
        
        logger.info(f"   ✅ Variant 1 search complete: {len(final_results)} final results")
        logger.info(f"   📊 Breakdown: {len(initial_results)} vector + {len(expansion_docs)} expansion → {len(final_results)} final")
        
        return final_results


def create_baseline_retriever(
    vector_store,
    k: int = 10,
    expansion_k: int = 20,
    max_total_results: int = 30
) -> BaselineGraphRAGRetriever:
    """
    Factory function to create Baseline GraphRAG Retriever (Variant 1)
    
    Parameters
    ----------
    vector_store : Neo4jNativeVectorStore
        Initialized vector store instance
    k : int, default 10
        Number of initial vector search results
    expansion_k : int, default 20
        Maximum additional results from graph expansion
    max_total_results : int, default 30
        Maximum total results to return
        
    Returns
    -------
    BaselineGraphRAGRetriever
        Configured baseline retriever
    """
    if not vector_store.vector_store:
        raise ValueError("Vector store not initialized. Call vector_store.initialize() first.")
    
    retriever = BaselineGraphRAGRetriever(
        vector_store=vector_store,
        k=k,
        expansion_k=expansion_k,
        max_total_results=max_total_results
    )
    
    logger.info("✅ Baseline GraphRAG Retriever (Variant 1) created:")
    logger.info(f"   🎯 Vector search: k={k}")
    logger.info(f"   🔗 Graph expansion: expansion_k={expansion_k} (CITES, CITES_CASE, CONTAINS)")
    logger.info(f"   🔢 Max results: {max_total_results}")
    logger.info(f"   📊 Scoring: None (simple concatenation)")
    logger.info(f"   🎨 Variant: Baseline for comparison")
    
    return retriever
