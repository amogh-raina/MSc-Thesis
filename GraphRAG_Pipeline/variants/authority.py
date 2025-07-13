"""
Variant 3: Authority-weighted GraphRAG Retrieval

Implementation: Query → Vector Search → Graph Expansion → Enhanced Authority Scoring → LLM

This variant implements enhanced authority scoring with:
- Logarithmic scaling to prevent outlier dominance
- Dynamic divisor based on actual dataset statistics  
- Component weighting refinement (cited_by > cites)
- Improved normalization strategy
"""

from __future__ import annotations
from typing import List, Dict, Any
import logging
import math
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from pydantic import Field

logger = logging.getLogger(__name__)


class AuthorityGraphRAGRetriever(BaseRetriever):
    """
    Variant 3: Authority-weighted GraphRAG Retriever
    
    This implements enhanced authority scoring for legal precedence:
    1. Vector search for semantic relevance
    2. Graph expansion using multiple relationships (CITES, CITES_CASE, CONTAINS)
    3. Enhanced authority calculation with logarithmic scaling
    4. Combined relevance + authority scoring with configurable weighting
    """
    
    vector_store: Any = Field(description="Neo4j vector store instance")
    k: int = Field(default=10, description="Number of initial vector search results")
    expansion_k: int = Field(default=20, description="Maximum additional results from graph expansion")
    max_total_results: int = Field(default=50, description="Maximum total results to return")
    authority_weight: float = Field(default=0.3, description="Weight for authority in combined scoring (0.0-1.0)")
    relevance_threshold: float = Field(default=0.1, description="Minimum combined score for inclusion")
    cited_by_weight: float = Field(default=1.0, description="Weight for being cited (precedential authority)")
    cites_weight: float = Field(default=0.8, description="Weight for citing others (research thoroughness)")
    case_authority_factor: float = Field(default=0.1, description="Factor for case-level authority relative to paragraph-level")
    use_logarithmic_scaling: bool = Field(default=True, description="Whether to use logarithmic scaling to prevent outlier dominance")
    variant_name: str = Field(default="Variant 3 (Authority-weighted)", description="Variant identifier")
    avg_paragraphs_per_case: float = Field(default=10.0, description="Average paragraphs per case (calculated)")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize Authority GraphRAG Retriever"""
        super().__init__(**kwargs)
        
        # Calculate dynamic divisor based on actual dataset
        self.avg_paragraphs_per_case = self._calculate_avg_paragraphs_per_case()
        
        logger.info(f"✅ Authority GraphRAG Retriever initialized:")
        logger.info(f"   📊 Authority weight: {self.authority_weight} (vs {1-self.authority_weight} relevance)")
        logger.info(f"   ⚖️ Component weights: cited_by={self.cited_by_weight}, cites={self.cites_weight}")
        logger.info(f"   📈 Scaling: {'Logarithmic' if self.use_logarithmic_scaling else 'Linear'}")
        logger.info(f"   🔢 Avg paragraphs/case: {self.avg_paragraphs_per_case:.1f}")
    
    def _calculate_avg_paragraphs_per_case(self) -> float:
        """
        Calculate average paragraphs per case from actual dataset.
        This replaces the arbitrary "/10" divisor with data-driven value.
        """
        try:
            query = """
            MATCH (c:Case)-[:CONTAINS]->(p:Paragraph)
            WITH c, count(p) as para_count
            RETURN avg(para_count) as average_paragraphs
            """
            result = self.vector_store.graph.query(query)
            avg_paras = result[0]['average_paragraphs'] if result else 10.0
            
            # Ensure reasonable minimum (avoid division by very small numbers)
            return max(avg_paras, 5.0)
            
        except Exception as e:
            logger.warning(f"Could not calculate avg paragraphs per case: {e}")
            return 10.0  # Fallback to original assumption
    
    def _calculate_enhanced_authority_score(
        self, 
        para_cited_by: int, 
        para_cites: int, 
        case_cited_by: int, 
        case_cites: int
    ) -> float:
        """
        Calculate enhanced authority score with logarithmic scaling and component weighting.
        
        This implements O3's suggested improvements:
        1. Logarithmic scaling to prevent outlier dominance
        2. Component weighting (being cited > citing others)
        3. Dynamic case authority factor based on actual dataset
        
        Parameters
        ----------
        para_cited_by : int
            Number of times this paragraph is cited (precedential authority)
        para_cites : int  
            Number of citations this paragraph makes (research thoroughness)
        case_cited_by : int
            Number of times the containing case is cited
        case_cites : int
            Number of citations the containing case makes
            
        Returns
        -------
        float
            Enhanced authority score
        """
        if self.use_logarithmic_scaling:
            # O3's logarithmic approach with refined component weights
            authority_score = (
                self.cited_by_weight * math.log(1 + para_cited_by) +           # Primary: precedential weight
                self.cites_weight * math.log(1 + para_cites) +                 # Secondary: research depth
                self.case_authority_factor * math.log(1 + case_cited_by / self.avg_paragraphs_per_case) +  # Case authority
                (self.case_authority_factor * 0.5) * math.log(1 + case_cites / self.avg_paragraphs_per_case)   # Case grounding
            )
        else:
            # Original linear approach (for comparison)
            authority_score = (
                self.cited_by_weight * para_cited_by +
                self.cites_weight * para_cites +
                self.case_authority_factor * (case_cited_by / self.avg_paragraphs_per_case) +
                (self.case_authority_factor * 0.5) * (case_cites / self.avg_paragraphs_per_case)
            )
        
        return authority_score
    
    def _calculate_combined_score(
        self, 
        relevance_score: float, 
        authority_score: float
    ) -> float:
        """
        Calculate combined relevance + authority score.
        
        Uses improved normalization that scales authority relative to the 
        expected range for logarithmic scores.
        
        Parameters
        ----------
        relevance_score : float
            Base relevance score (1.0 for vector results, 0.7 for expansion)
        authority_score : float
            Raw authority score from enhanced calculation
            
        Returns
        -------
        float
            Combined score for ranking
        """
        if self.use_logarithmic_scaling:
            # For logarithmic scores, normalize using expected max range
            # log(1+300) ≈ 5.7, so we use max_expected ≈ 8.0 for safety
            max_expected_log_score = 8.0
            normalized_authority = min(authority_score / max_expected_log_score, 1.0)
        else:
            # Original normalization for linear scores
            normalized_authority = min(authority_score / 10.0, 1.0)
        
        combined_score = (
            (1.0 - self.authority_weight) * relevance_score + 
            self.authority_weight * normalized_authority
        )
        
        return combined_score
    
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
            Retrieved documents with authority-weighted ranking
        """
        return self._authority_weighted_search(
            query=query,
            k=self.k,
            expansion_k=self.expansion_k,
            max_total_results=self.max_total_results,
            authority_weight=self.authority_weight,
            relevance_threshold=self.relevance_threshold
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents using authority-weighted GraphRAG approach
        
        Parameters
        ----------
        query : str
            Search query
            
        Returns
        -------
        List[Document]
            Retrieved documents with authority-weighted ranking
        """
        # Use the internal method directly to avoid recursion
        return self._authority_weighted_search(
            query=query,
            k=self.k,
            expansion_k=self.expansion_k,
            max_total_results=self.max_total_results,
            authority_weight=self.authority_weight,
            relevance_threshold=self.relevance_threshold
        )
    
    def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version (falls back to sync for now)"""
        return self.get_relevant_documents(query)
    
    def _authority_weighted_search(
        self,
        query: str,
        k: int = 10,
        expansion_k: int = 20,
        max_total_results: int = 50,
        authority_weight: float = 0.3,
        relevance_threshold: float = 0.1
    ) -> List[Document]:
        """
        Authority-weighted GraphRAG Search Implementation
        
        This implements enhanced authority scoring with logarithmic scaling:
        1. Vector search for semantic relevance
        2. Graph expansion using multiple relationships  
        3. Enhanced authority calculation (logarithmic + component weighting)
        4. Combined scoring with improved normalization
        5. Relevance threshold filtering and deduplication
        
        Parameters
        ----------
        query : str
            Search query
        k : int, default 10
            Number of initial vector search results
        expansion_k : int, default 20
            Maximum additional results from graph expansion
        max_total_results : int, default 50
            Maximum total results to return
        authority_weight : float, default 0.3
            Weight for authority scoring (0.0-1.0)
        relevance_threshold : float, default 0.1
            Minimum combined score for inclusion
            
        Returns
        -------
        List[Document]
            Authority-weighted search results
        """
        if not self.vector_store.vector_store:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        logger.info(f"🔍 Authority-weighted search: query='{query[:50]}...', k={k}, expansion_k={expansion_k}")
        logger.info(f"   ⚖️ Authority config: weight={authority_weight}, scaling={'log' if self.use_logarithmic_scaling else 'linear'}")
        
        # Step 1: Initial vector search for semantic relevance
        logger.info("   🎯 Step 1: Vector search for semantic relevance...")
        initial_results = self.vector_store.custom_similarity_search(query, k=k, with_graph_context=True)
        
        if not initial_results:
            logger.warning("   ⚠️ No initial vector results found")
            return []
        
        logger.info(f"   ✅ Found {len(initial_results)} initial results")
        
        # Step 2: Graph expansion using multiple relationships with enhanced authority calculation
        logger.info("   🔗 Step 2: Graph expansion with enhanced authority scoring...")
        
        # Extract paragraph IDs from initial results
        initial_para_ids = []
        for doc in initial_results:
            para_id = doc.metadata.get('paragraph_id')
            if para_id:
                initial_para_ids.append(para_id)
        
        if not initial_para_ids:
            logger.warning("   ⚠️ No valid paragraph IDs found in initial results")
            return initial_results
        
        # Enhanced expansion query with logarithmic authority calculation
        if self.use_logarithmic_scaling:
            authority_calculation = f"""
            // Enhanced authority calculation with logarithmic scaling
            WITH related_para, related_case, relationship_type,
                 COUNT {{ (related_para)-[:CITES]->() }} AS cites_count,
                 COUNT {{ ()-[:CITES]->(related_para) }} AS cited_by_count,
                 COUNT {{ (related_case)-[:CITES_CASE]->() }} AS case_cites_count,
                 COUNT {{ ()-[:CITES_CASE]->(related_case) }} AS case_cited_by_count
            
            // Calculate enhanced authority score with logarithmic scaling and component weighting
            WITH related_para, related_case, relationship_type, cites_count, cited_by_count, case_cites_count, case_cited_by_count,
                 {self.cited_by_weight} * log(1 + cited_by_count) +
                 {self.cites_weight} * log(1 + cites_count) +
                 {self.case_authority_factor} * log(1 + case_cited_by_count / {self.avg_paragraphs_per_case}) +
                 {self.case_authority_factor * 0.5} * log(1 + case_cites_count / {self.avg_paragraphs_per_case})
                 AS authority_score
            """
        else:
            authority_calculation = f"""
            // Original linear authority calculation  
            WITH related_para, related_case, relationship_type,
                 COUNT {{ (related_para)-[:CITES]->() }} AS cites_count,
                 COUNT {{ ()-[:CITES]->(related_para) }} AS cited_by_count,
                 COUNT {{ (related_case)-[:CITES_CASE]->() }} AS case_cites_count,
                 COUNT {{ ()-[:CITES_CASE]->(related_case) }} AS case_cited_by_count
            
            // Calculate linear authority score with component weighting  
            WITH related_para, related_case, relationship_type, cites_count, cited_by_count, case_cites_count, case_cited_by_count,
                 {self.cited_by_weight} * cited_by_count +
                 {self.cites_weight} * cites_count +
                 {self.case_authority_factor} * (case_cited_by_count / {self.avg_paragraphs_per_case}) +
                 {self.case_authority_factor * 0.5} * (case_cites_count / {self.avg_paragraphs_per_case})
                 AS authority_score
            """
        
        expansion_query = f"""
        // Get initial paragraphs and their cases
        MATCH (initial_para:Paragraph)
        WHERE initial_para.id IN $initial_para_ids
        OPTIONAL MATCH (initial_case:Case)-[:CONTAINS]->(initial_para)
        
        // Find related paragraphs through multiple relationships
        WITH initial_para, initial_case
        CALL (initial_para, initial_case) {{
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
        }}
        
        // Rename for clarity and get case information
        WITH cited_para AS related_para, relationship_type
        OPTIONAL MATCH (related_case:Case)-[:CONTAINS]->(related_para)
        
        {authority_calculation}
        
        // Format content and return results
        RETURN
            CASE 
                WHEN related_para.celex IS NOT NULL AND related_para.number IS NOT NULL AND related_para.title IS NOT NULL
                THEN "(#" + related_para.celex + ":" + toString(related_para.number) + ") [" + related_para.title + "] " + related_para.text
                WHEN related_para.celex IS NOT NULL AND related_para.number IS NOT NULL  
                THEN "(#" + related_para.celex + ":" + toString(related_para.number) + ") " + related_para.text
                ELSE related_para.text
            END AS formatted_content,
            {{
                celex: related_para.celex,
                para_no: toString(related_para.number),
                case_title: related_para.title,
                paragraph_id: related_para.id,
                node_type: "paragraph",
                
                // Enhanced authority information
                authority_score: authority_score,
                cites_count: cites_count,
                cited_by_count: cited_by_count,
                case_cites_count: case_cites_count,
                case_cited_by_count: case_cited_by_count,
                
                // Relationship context
                expansion_source: "graph_authority_weighted",
                relationship_type: relationship_type,
                case_year: COALESCE(related_case.year, date(related_case.date).year),
                
                // Content metadata
                word_count: size(split(related_para.text, ' ')),
                char_count: size(related_para.text)
            }} AS metadata,
            authority_score
        ORDER BY authority_score DESC
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
        
        # Step 4: Enhanced authority weighting and combined scoring
        logger.info("   ⚖️ Step 3: Enhanced authority scoring and deduplication...")
        
        all_docs = []
        seen_paragraphs = set()
        
        # Process initial results (higher base relevance)
        for doc in initial_results:
            para_id = doc.metadata.get('paragraph_id')
            if para_id and para_id not in seen_paragraphs:
                # Extract citation data from metadata (if available from vector search)
                para_cited_by = doc.metadata.get('cited_by_count', 0)
                para_cites = doc.metadata.get('cites_count', 0)
                case_cited_by = doc.metadata.get('case_cited_by_count', 0)
                case_cites = doc.metadata.get('case_cites_count', 0)
                
                # If metadata doesn't have citation data, fetch it from graph
                if para_cited_by == 0 and para_cites == 0 and case_cited_by == 0 and case_cites == 0:
                    try:
                        citation_query = f"""
                        MATCH (p:Paragraph {{id: $para_id}})
                        OPTIONAL MATCH (c:Case)-[:CONTAINS]->(p)
                        
                        WITH p, c,
                             COUNT {{ (p)-[:CITES]->() }} AS para_cites,
                             COUNT {{ ()-[:CITES]->(p) }} AS para_cited_by,
                             COUNT {{ (c)-[:CITES_CASE]->() }} AS case_cites,
                             COUNT {{ ()-[:CITES_CASE]->(c) }} AS case_cited_by
                        
                        RETURN para_cites, para_cited_by, case_cites, case_cited_by
                        """
                        
                        citation_result = self.vector_store.graph.query(
                            citation_query,
                            params={"para_id": para_id}
                        )
                        
                        if citation_result:
                            result = citation_result[0]
                            para_cites = result.get('para_cites', 0)
                            para_cited_by = result.get('para_cited_by', 0)
                            case_cites = result.get('case_cites', 0)
                            case_cited_by = result.get('case_cited_by', 0)
                            
                    except Exception as e:
                        logger.warning(f"Failed to fetch citation data for {para_id}: {e}")
                        # Use default values (0,0,0,0)
                
                authority_score = self._calculate_enhanced_authority_score(
                    para_cited_by, para_cites, case_cited_by, case_cites
                )
                
                # Calculate combined score
                combined_score = self._calculate_combined_score(1.0, authority_score)
                
                # Update metadata
                enhanced_metadata = doc.metadata.copy()
                enhanced_metadata.update({
                    'expansion_source': 'vector_search',
                    'base_relevance_score': 1.0,
                    'authority_score': authority_score,
                    'combined_score': combined_score,
                    'scaling_method': 'logarithmic' if self.use_logarithmic_scaling else 'linear',
                    # Store actual citation counts for debugging
                    'para_cited_by': para_cited_by,
                    'para_cites': para_cites,
                    'case_cited_by': case_cited_by,
                    'case_cites': case_cites
                })
                
                doc.metadata = enhanced_metadata
                all_docs.append(doc)
                seen_paragraphs.add(para_id)
        
        # Process expansion results (lower base relevance, authority-focused)
        for doc in expansion_docs:
            para_id = doc.metadata.get('paragraph_id')
            if para_id and para_id not in seen_paragraphs:
                authority_score = doc.metadata.get('authority_score', 0.0)
                base_relevance = 0.7  # Lower base relevance for expanded results
                
                # Calculate combined score using enhanced authority
                combined_score = self._calculate_combined_score(base_relevance, authority_score)
                
                # Apply relevance threshold
                if combined_score >= relevance_threshold:
                    doc.metadata.update({
                        'base_relevance_score': base_relevance,
                        'combined_score': combined_score,
                        'scaling_method': 'logarithmic' if self.use_logarithmic_scaling else 'linear'
                    })
                    all_docs.append(doc)
                    seen_paragraphs.add(para_id)
        
        # Step 5: Final sorting and limiting by enhanced combined score
        logger.info("   📊 Step 4: Final ranking by enhanced combined score...")
        
        # Sort by combined score (descending)
        all_docs.sort(key=lambda x: x.metadata.get('combined_score', 0.0), reverse=True)
        
        # Limit total results
        final_results = all_docs[:max_total_results]
        
        logger.info(f"   ✅ Authority-weighted search complete: {len(final_results)} final results")
        logger.info(f"   📊 Breakdown: {len(initial_results)} vector + {len(expansion_docs)} expansion → {len(final_results)} final")
        logger.info(f"   ⚖️ Scoring: {self.use_logarithmic_scaling and 'Logarithmic' or 'Linear'} authority + {authority_weight:.1%} weighting")
        
        return final_results


def create_authority_retriever(
    vector_store,
    k: int = 10,
    expansion_k: int = 20,
    max_total_results: int = 50,
    authority_weight: float = 0.3,
    relevance_threshold: float = 0.1,
    # Enhanced authority parameters
    cited_by_weight: float = 1.0,
    cites_weight: float = 0.8,
    case_authority_factor: float = 0.1,
    use_logarithmic_scaling: bool = True
) -> AuthorityGraphRAGRetriever:
    """
    Factory function to create Authority-weighted GraphRAG Retriever (Variant 3)
    
    Parameters
    ----------
    vector_store : Neo4jNativeVectorStore
        Initialized vector store instance
    k : int, default 10
        Number of initial vector search results
    expansion_k : int, default 20
        Maximum additional results from graph expansion
    max_total_results : int, default 50
        Maximum total results to return
    authority_weight : float, default 0.3
        Weight for authority scoring (0.0-1.0, higher values favor highly cited content)
    relevance_threshold : float, default 0.1
        Minimum combined score for inclusion
    cited_by_weight : float, default 1.0
        Weight for paragraph being cited (precedential authority)
    cites_weight : float, default 0.8
        Weight for paragraph citing others (research thoroughness)
    case_authority_factor : float, default 0.1
        Factor for case-level authority relative to paragraph-level
    use_logarithmic_scaling : bool, default True
        Whether to use logarithmic scaling to prevent outlier dominance
        
    Returns
    -------
    AuthorityGraphRAGRetriever
        Configured authority-weighted retriever
    """
    if not vector_store.vector_store:
        raise ValueError("Vector store not initialized. Call vector_store.initialize() first.")
    
    retriever = AuthorityGraphRAGRetriever(
        vector_store=vector_store,
        k=k,
        expansion_k=expansion_k,
        max_total_results=max_total_results,
        authority_weight=authority_weight,
        relevance_threshold=relevance_threshold,
        cited_by_weight=cited_by_weight,
        cites_weight=cites_weight,
        case_authority_factor=case_authority_factor,
        use_logarithmic_scaling=use_logarithmic_scaling
    )
    
    logger.info("✅ Authority GraphRAG Retriever (Variant 3) created:")
    logger.info(f"   🎯 Vector search: k={k}")
    logger.info(f"   🔗 Graph expansion: expansion_k={expansion_k} (CITES, CITES_CASE, CONTAINS)")
    logger.info(f"   🔢 Max results: {max_total_results}")
    logger.info(f"   📊 Authority scoring: {'Logarithmic' if use_logarithmic_scaling else 'Linear'} scaling")
    logger.info(f"   ⚖️ Component weights: cited_by={cited_by_weight}, cites={cites_weight}")
    logger.info(f"   🎨 Variant: Authority-weighted for legal precedence")
    
    return retriever
