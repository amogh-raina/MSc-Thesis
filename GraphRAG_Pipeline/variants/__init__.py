"""
GraphRAG Variants Package

This package contains different retrieval variants for the GraphRAG system:
- Variant 1: Baseline GraphRAG (vector search + graph expansion)
- Variant 2: LangChain Reranker GraphRAG (baseline + LangChain reranking)
- Variant 3: Authority Score Reranker GraphRAG (baseline + authority-based reranking)
"""

# Variant 1: Baseline GraphRAG
from .baseline import (
    BaselineGraphRAGRetriever,
    create_baseline_retriever
)

# Variant 2: LangChain Reranker GraphRAG
from .langchain_reranker import (
    LangChainRerankerGraphRAGRetriever,
    create_langchain_reranker_retriever,
    get_available_rerankers,
    print_reranker_options,
    RerankerType
)

# Variant 3: Authority-weighted GraphRAG
from .authority import (
    AuthorityGraphRAGRetriever,
    create_authority_retriever
)

__all__ = [
    # Variant 1: Baseline
    "BaselineGraphRAGRetriever",
    "create_baseline_retriever",
    
    # Variant 2: LangChain Reranker
    "LangChainRerankerGraphRAGRetriever", 
    "create_langchain_reranker_retriever",
    "get_available_rerankers",
    "print_reranker_options",
    "RerankerType",
    
    # Variant 3: Authority-weighted
    "AuthorityGraphRAGRetriever", 
    "create_authority_retriever",
]
