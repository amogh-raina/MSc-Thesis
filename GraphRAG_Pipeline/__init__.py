"""
GraphRAG Pipeline Package

A modular GraphRAG system for legal research combining:
- Neo4j graph database for citation relationships
- Vector embeddings for semantic search  
- Hybrid query interface for comprehensive analysis
"""

from .graph_pipeline import GraphRAGPipeline, create_graphrag_pipeline
from .graph_vector_store import Neo4jNativeVectorStore, create_neo4j_vector_store
from .Graph_populator import GraphPopulator

# Optional import for hybrid QA (may not exist in all setups)
try:
    from .hybrid_graphrag_qa import HybridGraphRAGQA, create_hybrid_graphrag_qa, QueryType, QueryResult
    _HYBRID_QA_AVAILABLE = True
except ImportError:
    _HYBRID_QA_AVAILABLE = False

__all__ = [
    # Main pipeline
    "GraphRAGPipeline",
    "create_graphrag_pipeline",
    
    # Vector store management
    "Neo4jNativeVectorStore", 
    "create_neo4j_vector_store",
    
    # Graph population
    "GraphPopulator",
]

# Add hybrid QA exports only if available
if _HYBRID_QA_AVAILABLE:
    __all__.extend([
        "HybridGraphRAGQA",
        "create_hybrid_graphrag_qa", 
        "QueryType",
        "QueryResult"
    ])
