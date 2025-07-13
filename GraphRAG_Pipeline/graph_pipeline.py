"""
GraphRAG Pipeline - Main orchestrator for graph and vector store setup

This module coordinates:
1. Graph population (using graph_populator.py) 
2. Document building (using graph_doc_builder.py)
3. Vector store creation (using graph_vector_store.py)

Similar structure to RAG_Pipeline for consistency and modularity.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import pandas as pd
from langchain.embeddings.base import Embeddings

# Local imports
from .Graph_populator import GraphPopulator
from .graph_vector_store import Neo4jNativeVectorStore, create_neo4j_vector_store

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """
    Main GraphRAG pipeline that coordinates graph creation and vector indexing.
    
    This class provides a unified interface for:
    - Populating Neo4j graph with legal citation data (including formatted content)
    - Creating vector embeddings directly from graph content (no separate documents needed)  
    - Creating retrievers for hybrid GraphRAG queries
    """
    
    def __init__(
        self,
        dataset_path: str | Path,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        embedding: Embeddings,
        force_rebuild_graph: bool = False,
        force_rebuild_vectors: bool = False,
        vector_index_name: str = "paragraph_vector_index"
    ):
        """
        Initialize GraphRAG Pipeline
        
        Parameters
        ----------
        dataset_path : str | Path
            Path to the paragraph-to-paragraph citation dataset (CSV)
        neo4j_uri : str
            Neo4j database URI
        neo4j_user : str
            Neo4j username
        neo4j_password : str
            Neo4j password
        embedding : Embeddings
            LangChain embeddings instance
        force_rebuild_graph : bool, default False
            Whether to rebuild the graph database from scratch
        force_rebuild_vectors : bool, default False
            Whether to rebuild vector indices
        vector_index_name : str, default "paragraph_vector_index"
            Name for the vector index in Neo4j
        """
        self.dataset_path = Path(dataset_path)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.embedding = embedding
        self.force_rebuild_graph = force_rebuild_graph
        self.force_rebuild_vectors = force_rebuild_vectors
        self.vector_index_name = vector_index_name
        
        # Pipeline components
        self.graph_populator: Optional[GraphPopulator] = None
        self.vector_store: Optional[Neo4jNativeVectorStore] = None
        
        # Pipeline state
        self.graph_ready = False
        self.vectors_ready = False
    
    def build_graph(self) -> None:
        """
        Step 1: Build the citation graph in Neo4j with formatted content
        
        This creates the graph structure AND stores the content in the same 
        format as doc_builder.py for vector indexing.
        """
        logger.info("🔧 Building citation graph with formatted content...")
        
        try:
            # Initialize graph populator
            self.graph_populator = GraphPopulator()
            
            # Process the dataset and populate graph
            # The updated populator now stores formatted content on nodes
            self.graph_populator.process_csv(str(self.dataset_path))
            
            self.graph_ready = True
            logger.info("✅ Citation graph built successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to build graph: {e}")
            raise
        finally:
            if self.graph_populator:
                self.graph_populator.close()
    
    def build_vector_store(self) -> None:
        """
        Step 2: Build vector store directly from graph content
        
        This uses Neo4j's native vector capabilities to create embeddings
        from the formatted content stored on graph nodes.
        """
        if not self.graph_ready:
            raise ValueError("Graph not ready. Call build_graph() first.")
        
        logger.info("🧠 Building vector store from graph content...")
        
        try:
            # Create Neo4j native vector store
            self.vector_store = Neo4jNativeVectorStore(
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password,
                embedding=self.embedding,
                index_name=self.vector_index_name,
                force_rebuild=self.force_rebuild_vectors
            )
            
            # Initialize vector store (creates embeddings + index)
            self.vector_store.initialize()
            
            self.vectors_ready = True
            logger.info("✅ Vector store built successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to build vector store: {e}")
            raise
    
    def initialize(self) -> None:
        """
        Initialize the complete GraphRAG pipeline
        
        This method runs all pipeline steps in sequence:
        1. Build citation graph with formatted content  
        2. Build vector store directly from graph
        """
        logger.info("🚀 Initializing GraphRAG Pipeline...")
        
        # Step 1: Build graph (if needed)
        if self.force_rebuild_graph:
            logger.info("🔄 Force rebuilding graph as requested")
            self.build_graph()
        elif not self._check_graph_exists():
            logger.info("🔧 Building graph (none found or validation failed)")
            self.build_graph()
        else:
            logger.info("📊 Using existing citation graph")
            self.graph_ready = True
        
        # Step 2: Build vector store directly from graph
        self.build_vector_store()
        
        logger.info("✅ GraphRAG Pipeline initialization complete!")
        self._print_pipeline_stats()
    
    def _check_graph_exists(self) -> bool:
        """
        Comprehensive check if graph is properly populated with complete dataset
        
        This performs multiple validation levels:
        1. Dataset completeness (CSV rows vs graph nodes)
        2. Data integrity (expected relationships exist)
        3. Schema validation (constraints and indexes)
        """
        try:
            from langchain_neo4j import Neo4jGraph
            import pandas as pd
            
            graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
                refresh_schema=False
            )
            
            logger.info("🔍 Performing comprehensive graph validation...")
            
            # Level 1: Basic existence check
            basic_check = graph.query("""
                MATCH (p:Paragraph) 
                RETURN count(p) as paragraph_count
            """)
            graph_paragraph_count = basic_check[0]["paragraph_count"] if basic_check else 0
            
            if graph_paragraph_count == 0:
                logger.info("❌ No paragraphs found in graph")
                graph._driver.close()
                return False
            
            # Level 2: Dataset completeness check
            if self.dataset_path.exists():
                try:
                    # Count expected rows in CSV (multiply by 2 since each row creates 2 paragraphs)
                    df = pd.read_csv(self.dataset_path)
                    expected_paragraphs = len(df) * 2  # citing + cited paragraphs
                    
                    completeness_ratio = graph_paragraph_count / expected_paragraphs
                    logger.info(f"📊 Graph completeness: {graph_paragraph_count:,} / {expected_paragraphs:,} paragraphs ({completeness_ratio:.1%})")
                    
                    if completeness_ratio < 0.40:  # Allow much more tolerance for duplicates and data quality issues
                        logger.warning(f"⚠️ Graph appears severely incomplete ({completeness_ratio:.1%} of expected data)")
                        graph._driver.close()
                        return False
                    elif completeness_ratio < 0.95:
                        logger.info(f"ℹ️ Graph has moderate completeness ({completeness_ratio:.1%}) - likely due to duplicate filtering")
                    else:
                        logger.info(f"✅ Graph appears complete ({completeness_ratio:.1%})")
                        
                except Exception as e:
                    logger.warning(f"Could not validate dataset completeness: {e}")
            
            # Level 3: Data integrity check
            integrity_check = graph.query("""
                MATCH (p:Paragraph)
                OPTIONAL MATCH (c:Case)-[:CONTAINS]->(p)
                OPTIONAL MATCH (p)-[:CITES]->(cited:Paragraph)
                
                WITH count(p) as total_paragraphs,
                     count(c) as paragraphs_with_cases,
                     count(cited) as citation_relationships
                     
                RETURN total_paragraphs, paragraphs_with_cases, citation_relationships
            """)
            
            if integrity_check:
                stats = integrity_check[0]
                total_p = stats["total_paragraphs"]
                with_cases = stats["paragraphs_with_cases"] 
                citations = stats["citation_relationships"]
                
                logger.info(f"🔗 Data integrity: {with_cases:,}/{total_p:,} paragraphs linked to cases, {citations:,} citations")
                
                # Check if most paragraphs are properly linked
                if with_cases / total_p < 0.9:  # 90% should be linked to cases
                    logger.warning("⚠️ Many paragraphs not properly linked to cases")
                    graph._driver.close()
                    return False
                
                if citations == 0:
                    logger.warning("⚠️ No citation relationships found")
                    graph._driver.close()
                    return False
            
            # Level 4: Schema validation
            logger.info("🔍 Checking database schema...")
            try:
                # Check for essential indexes/constraints
                constraints_check = graph.query("SHOW CONSTRAINTS")
                indexes_check = graph.query("SHOW INDEXES")
                logger.info(f"   Found {len(constraints_check)} constraints, {len(indexes_check)} indexes")
                
                essential_constraints = ["paragraph_id_unique", "case_celex_unique"]
                existing_constraints = [c.get("name", "") for c in constraints_check]
                
                missing_constraints = [name for name in essential_constraints 
                                     if not any(name in existing for existing in existing_constraints)]
                
                if missing_constraints:
                    logger.warning(f"⚠️ Missing essential constraints: {missing_constraints}")
                else:
                    logger.info("✅ Essential constraints found")
                    # Don't fail on this - constraints might have different names
                
            except Exception as e:
                logger.warning(f"Schema validation failed (non-critical): {e}")
            
            # Level 5: Check graph structure completeness (efficient separate queries)
            logger.info("📊 Checking graph structure...")
            
            # Count each node type separately (much more efficient)
            case_count = graph.query("MATCH (c:Case) RETURN count(c) as count")[0]["count"]
            paragraph_count = graph.query("MATCH (p:Paragraph) RETURN count(p) as count")[0]["count"]  
            year_count = graph.query("MATCH (y:Year) RETURN count(y) as count")[0]["count"]
            
            structure_check = [{
                "cases": case_count,
                "paragraphs": paragraph_count, 
                "years": year_count
            }]
            
            if structure_check:
                stats = structure_check[0]
                logger.info(f"📈 Graph structure: {stats['cases']:,} cases, {stats['paragraphs']:,} paragraphs, {stats['years']:,} years")
                
                # Basic sanity checks
                if stats['cases'] == 0 or stats['years'] == 0:
                    logger.warning("⚠️ Missing essential node types (Cases or Years)")
                    graph._driver.close()
                    return False
            
            graph._driver.close()
            logger.info("✅ Graph validation passed - using existing graph")
            return True
            
        except Exception as e:
            logger.warning(f"Graph validation failed: {e}")
            return False
    
    def _print_pipeline_stats(self) -> None:
        """Print statistics about the built pipeline"""
        logger.info("📊 GraphRAG Pipeline Statistics:")
        
        # Graph statistics
        if self.graph_ready:
            logger.info("   📈 Graph Database: Ready with formatted content")
        
        # Vector store statistics
        if self.vector_store and self.vectors_ready:
            stats = self.vector_store.get_stats()
            logger.info(f"   🧠 Vector Store: {stats}")
    
    # REMOVED: get_retriever() - use pipeline.vector_store.get_retriever() directly
    
    def get_baseline_retriever(
        self,
        k: int = 10,
        expansion_k: int = 20,
        max_total_results: int = 30
    ):
        """
        Get Variant 1: Baseline GraphRAG Retriever
        
        This returns a retriever that implements:
        Query → Vector Search → Graph Expansion → Combined Context → LLM
        
        Parameters
        ----------
        k : int, default 10
            Number of initial vector search results
        expansion_k : int, default 20
            Maximum additional results from graph expansion
        max_total_results : int, default 30
            Maximum total results to return
            
        Returns
        -------
        BaselineGraphRAGRetriever
            Configured baseline retriever for Variant 1
        """
        if not self.vector_store or not self.vectors_ready:
            raise ValueError("Vector store not ready. Call initialize() first.")
        
        from .variants import create_baseline_retriever
        
        return create_baseline_retriever(
            vector_store=self.vector_store,
            k=k,
            expansion_k=expansion_k,
            max_total_results=max_total_results
        )
    
    def get_langchain_reranker_retriever(
        self,
        reranker_type: str = "cross_encoder",
        k: int = 10,
        expansion_k: int = 20,
        max_total_results: int = 30,
        rerank_top_n: int = 15,
        reranker_config: Optional[Dict[str, Any]] = None
    ):
        """
        Get Variant 2: LangChain Reranker GraphRAG Retriever
        
        This returns a retriever that implements:
        Query → Vector Search → Graph Expansion → LangChain Reranker → LLM
        
        Parameters
        ----------
        reranker_type : str, default "cross_encoder"
            Type of reranker: "cross_encoder", "flashrank", "jina"
        k : int, default 10
            Number of initial vector search results
        expansion_k : int, default 20
            Maximum additional results from graph expansion
        max_total_results : int, default 30
            Maximum total results before reranking
        rerank_top_n : int, default 15
            Number of results to return after reranking
        reranker_config : Dict[str, Any], optional
            Configuration for the specific reranker
            
        Returns
        -------
        LangChainRerankerGraphRAGRetriever
            Configured LangChain reranker retriever for Variant 2
            
        Examples
        --------
        # Cross-encoder with BGE large model (local, recommended)
        >>> retriever = pipeline.get_langchain_reranker_retriever(
        ...     reranker_type="cross_encoder",
        ...     reranker_config={"model_name": "BAAI/bge-reranker-large"}
        ... )
        
        # FlashRank for fast reranking
        >>> retriever = pipeline.get_langchain_reranker_retriever(
        ...     reranker_type="flashrank"
        ... )
        
        # Jina reranker (requires JINA_API_KEY environment variable)
        >>> retriever = pipeline.get_langchain_reranker_retriever(
        ...     reranker_type="jina"
        ... )
        """
        if not self.vector_store or not self.vectors_ready:
            raise ValueError("Vector store not ready. Call initialize() first.")
        
        from .variants import create_langchain_reranker_retriever
        
        return create_langchain_reranker_retriever(
            vector_store=self.vector_store,
            reranker_type=reranker_type,
            k=k,
            expansion_k=expansion_k,
            max_total_results=max_total_results,
            rerank_top_n=rerank_top_n,
            reranker_config=reranker_config
        )
    
    def get_authority_retriever(
        self,
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
    ):
        """
        Get Variant 3: Authority-weighted GraphRAG Retriever
        
        This returns a retriever that implements enhanced authority scoring:
        Query → Vector Search → Graph Expansion → Enhanced Authority Scoring → LLM
        
        Features:
        - Logarithmic scaling to prevent outlier dominance
        - Dynamic divisor based on actual dataset statistics
        - Component weighting refinement (cited_by > cites)
        - Improved normalization strategy
        
        Parameters
        ----------
        k : int, default 10
            Number of initial vector search results
        expansion_k : int, default 20
            Maximum additional results from graph expansion
        max_total_results : int, default 50
            Maximum total results to return
        authority_weight : float, default 0.3
            Weight for authority in combined scoring (0.0-1.0)
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
            Configured authority-weighted retriever for Variant 3
            
        Examples
        --------
        # Standard authority-weighted retriever (30% authority, 70% relevance)
        >>> retriever = pipeline.get_authority_retriever()
        
        # High authority weighting for precedent research (60% authority, 40% relevance)
        >>> retriever = pipeline.get_authority_retriever(
        ...     authority_weight=0.6,
        ...     cited_by_weight=1.0,
        ...     cites_weight=0.5
        ... )
        
        # Linear scaling for comparison with logarithmic approach
        >>> retriever = pipeline.get_authority_retriever(
        ...     use_logarithmic_scaling=False
        ... )
        """
        if not self.vector_store or not self.vectors_ready:
            raise ValueError("Vector store not ready. Call initialize() first.")
        
        from .variants import create_authority_retriever
        
        return create_authority_retriever(
            vector_store=self.vector_store,
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

    # REMOVED: similarity_search() - use pipeline.vector_store.custom_similarity_search() directly
    
    # REMOVED: hybrid_search() - use pipeline.vector_store.hybrid_search() directly
    
    def get_graph_connection(self):
        """
        Get a Neo4j graph connection for direct queries
        
        Returns
        -------
        Neo4jGraph instance
        """
        from langchain_neo4j import Neo4jGraph
        
        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            refresh_schema=False
        )
    
    def get_pipeline_status(self) -> dict:
        """
        Get the status of all pipeline components
        
        Returns
        -------
        dict
            Status information for each component
        """
        status = {
            "dataset_path": str(self.dataset_path),
            "dataset_exists": self.dataset_path.exists(),
            "graph_ready": self.graph_ready,
            "vectors_ready": self.vectors_ready,
            "vector_index_name": self.vector_index_name,
            "uses_native_vectors": True  # Flag to indicate we're using Neo4j native vectors
        }
        
        # Add vector store stats if available
        if self.vector_store and self.vectors_ready:
            status.update(self.vector_store.get_stats())
        
        return status
    
    def close(self):
        """Close all connections"""
        if self.vector_store:
            self.vector_store.close()


# Factory function for easy initialization
def create_graphrag_pipeline(
    dataset_path: str | Path,
    embedding: Embeddings,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    force_rebuild_graph: bool = False,
    force_rebuild_vectors: bool = False,
    auto_initialize: bool = True
) -> GraphRAGPipeline:
    """
    Factory function to create and optionally initialize a GraphRAG pipeline
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to the citation dataset
    embedding : Embeddings
        Embedding model
    neo4j_uri : str, optional
        Neo4j URI (uses environment variable if not provided)
    neo4j_user : str, optional
        Neo4j username (uses environment variable if not provided)
    neo4j_password : str, optional
        Neo4j password (uses environment variable if not provided)
    force_rebuild_graph : bool, default False
        Whether to rebuild the graph from scratch
    force_rebuild_vectors : bool, default False
        Whether to rebuild vector indices
    auto_initialize : bool, default True
        Whether to automatically run the full pipeline
        
    Returns
    -------
    GraphRAGPipeline
        Configured (and optionally initialized) pipeline
    """
    import os
    
    # Get Neo4j credentials from environment if not provided
    neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
    neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME")
    neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        raise ValueError("Neo4j credentials must be provided or set in environment variables")
    
    # Create pipeline
    pipeline = GraphRAGPipeline(
        dataset_path=dataset_path,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        embedding=embedding,
        force_rebuild_graph=force_rebuild_graph,
        force_rebuild_vectors=force_rebuild_vectors
    )
    
    # Auto-initialize if requested
    if auto_initialize:
        pipeline.initialize()
    
    return pipeline