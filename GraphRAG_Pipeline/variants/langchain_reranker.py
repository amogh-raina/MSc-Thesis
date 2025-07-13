"""
Variant 2: LangChain Reranker GraphRAG Retrieval

Implementation: Query → Vector Search → Graph Expansion → LangChain Reranker → LLM

This variant adds LangChain-based reranking to the baseline GraphRAG approach:
- Same as Variant 1: Vector search + multi-relationship graph expansion
- Added: LangChain reranker for improved result quality
- Multiple reranker options: Cohere, BGE Cross-Encoder, FlashRank
- Configurable reranking parameters
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
import logging
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from pydantic import Field
from .baseline import BaselineGraphRAGRetriever

logger = logging.getLogger(__name__)

# Reranker type definitions
RerankerType = Literal["cross_encoder", "flashrank", "jina"]


class LangChainRerankerGraphRAGRetriever(BaseRetriever):
    """
    Variant 2: LangChain Reranker GraphRAG Retriever
    
    This implements GraphRAG search with LangChain reranking:
    1. Vector search for semantic relevance
    2. Graph expansion using multiple relationships (CITES, CITES_CASE, CONTAINS)
    3. LangChain reranker for improved result ranking
    4. Configurable reranker types and parameters
    """
    
    vector_store: Any = Field(description="Neo4j vector store instance")
    reranker_type: str = Field(description="Type of reranker to use")
    k: int = Field(default=10, description="Number of initial vector search results")
    expansion_k: int = Field(default=20, description="Maximum additional results from graph expansion")
    max_total_results: int = Field(default=30, description="Maximum total results before reranking")
    rerank_top_n: int = Field(default=15, description="Number of results to return after reranking")
    reranker_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the specific reranker")
    variant_name: str = Field(default="Variant 2 (LangChain Reranker)", description="Variant identifier")
    
    def __init__(
        self,
        vector_store,
        reranker_type: RerankerType = "cross_encoder",
        k: int = 10,
        expansion_k: int = 20,
        max_total_results: int = 30,
        rerank_top_n: int = 15,
        reranker_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize LangChain Reranker GraphRAG Retriever
        
        Parameters
        ----------
        vector_store : Neo4jNativeVectorStore
            The vector store instance for similarity search
        reranker_type : RerankerType, default "cross_encoder"
            Type of reranker to use: "cohere", "cross_encoder", "flashrank", "jina"
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
        """

        # First, initialise the underlying Pydantic/BaseModel fields properly
        super().__init__(
            vector_store=vector_store,
            reranker_type=reranker_type,
            k=k,
            expansion_k=expansion_k,
            max_total_results=max_total_results,
            rerank_top_n=rerank_top_n,
            reranker_config=reranker_config or {},
            **kwargs
        )

        # Derived / runtime-only attributes (allowed via Config.extra = "allow")
        self.variant_name = f"Variant 2 (LangChain {reranker_type.title()} Reranker)"

        # Initialize the reranker and supporting retrievers
        self.reranker = self._initialize_reranker()
        self.baseline_retriever = self._create_baseline_retriever()
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.baseline_retriever
        )
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # allow setting runtime-only attributes like `reranker`
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Internal method required by LangChain BaseRetriever"""
        return self.get_relevant_documents(query)
    
    def _initialize_reranker(self):
        """Initialize the specified reranker"""
        try:
            if self.reranker_type == "cross_encoder":
                return self._create_cross_encoder_reranker()
            elif self.reranker_type == "flashrank":
                return self._create_flashrank_reranker()
            elif self.reranker_type == "jina":
                return self._create_jina_reranker()
            else:
                raise ValueError(f"Unknown reranker type: {self.reranker_type}")
                
        except ImportError as e:
            logger.error(f"Failed to import {self.reranker_type} reranker: {e}")
            logger.warning(f"Please install required packages for {self.reranker_type} reranker")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize {self.reranker_type} reranker: {e}")
            raise
    

    
    def _create_cross_encoder_reranker(self):
        """Create Cross-Encoder reranker using BGE models"""
        try:
            from langchain.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            
            # Default config for Cross-Encoder
            config = {
                "model_name": "BAAI/bge-reranker-base",
                "top_n": self.rerank_top_n,
                **self.reranker_config
            }
            
            logger.info(f"Initializing Cross-Encoder reranker with model: {config['model_name']}")
            
            model = HuggingFaceCrossEncoder(model_name=config["model_name"])
            return CrossEncoderReranker(model=model, top_n=config["top_n"])
            
        except ImportError:
            raise ImportError("Please install required packages: pip install sentence-transformers")
    
    def _create_flashrank_reranker(self):
        """Create FlashRank reranker"""
        try:
            from langchain_community.document_compressors import FlashrankRerank
            
            # Default config for FlashRank
            config = {
                "top_n": self.rerank_top_n,
                **self.reranker_config
            }
            
            logger.info("Initializing FlashRank reranker")
            return FlashrankRerank(**config)
            
        except ImportError:
            raise ImportError("Please install flashrank: pip install flashrank")
    
    def _create_jina_reranker(self):
        """Create Jina reranker"""
        try:
            from langchain_community.document_compressors import JinaRerank
            
            # Default config for Jina
            config = {
                "top_n": self.rerank_top_n,
                **self.reranker_config
            }
            
            logger.info("Initializing Jina reranker")
            return JinaRerank(**config)
            
        except ImportError:
            raise ImportError("Please install jina: pip install jina")
    
    
    def _create_baseline_retriever(self):
        """Create baseline retriever (same as Variant 1)"""
        return BaselineGraphRAGRetriever(
            vector_store=self.vector_store,
            k=self.k,
            expansion_k=self.expansion_k,
            max_total_results=self.max_total_results
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents using LangChain reranker GraphRAG approach
        
        Parameters
        ----------
        query : str
            Search query
            
        Returns
        -------
        List[Document]
            Retrieved documents with LangChain reranking
        """
        logger.info(f"🔍 {self.variant_name} Search: query='{query[:50]}...', reranker={self.reranker_type}")
        
        try:
            # Use compression retriever which combines baseline + reranking
            results = self.compression_retriever.invoke(query)
            
            # Add metadata about the reranking
            for doc in results:
                doc.metadata['expansion_source'] = 'langchain_reranked'
                doc.metadata['reranker_type'] = self.reranker_type
                doc.metadata['variant'] = 'variant_2'
            
            logger.info(f"   ✅ {self.variant_name} search complete: {len(results)} results after reranking")
            
            return results
            
        except Exception as e:
            logger.error(f"   ❌ {self.variant_name} search failed: {e}")
            logger.warning("   🔄 Falling back to baseline retriever...")
            
            # Fallback to baseline retriever using invoke method
            return self.baseline_retriever.invoke(query)[:self.rerank_top_n]
    
    def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version (falls back to sync for now)"""
        return self.get_relevant_documents(query)


def create_langchain_reranker_retriever(
    vector_store,
    reranker_type: RerankerType = "cross_encoder",
    k: int = 10,
    expansion_k: int = 20,
    max_total_results: int = 30,
    rerank_top_n: int = 15,
    reranker_config: Optional[Dict[str, Any]] = None
) -> LangChainRerankerGraphRAGRetriever:
    """
    Factory function to create LangChain Reranker GraphRAG Retriever (Variant 2)

    Parameters
    ----------
    vector_store : Neo4jNativeVectorStore
        Initialized vector store instance
    reranker_type : RerankerType, default "cross_encoder"
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
        Configured LangChain reranker retriever
        
    Examples
    --------
    # Cross-encoder with BGE model (local, recommended)
    >>> retriever = create_langchain_reranker_retriever(
    ...     vector_store=vector_store,
    ...     reranker_type="cross_encoder",
    ...     reranker_config={"model_name": "BAAI/bge-reranker-large"}
    ... )
    
    # FlashRank (fast, lightweight)
    >>> retriever = create_langchain_reranker_retriever(
    ...     vector_store=vector_store,
    ...     reranker_type="flashrank"
    ... )
    
    # Jina reranker (requires JINA_API_KEY)
    >>> retriever = create_langchain_reranker_retriever(
    ...     vector_store=vector_store,
    ...     reranker_type="jina"
    ... )
    """
    if not vector_store.vector_store:
        raise ValueError("Vector store not initialized. Call vector_store.initialize() first.")
    
    retriever = LangChainRerankerGraphRAGRetriever(
        vector_store=vector_store,
        reranker_type=reranker_type,
        k=k,
        expansion_k=expansion_k,
        max_total_results=max_total_results,
        rerank_top_n=rerank_top_n,
        reranker_config=reranker_config
    )
    
    logger.info("✅ LangChain Reranker GraphRAG Retriever (Variant 2) created:")
    logger.info(f"   🎯 Vector search: k={k}")
    logger.info(f"   🔗 Graph expansion: expansion_k={expansion_k} (CITES, CITES_CASE, CONTAINS)")
    logger.info(f"   🔄 Reranker: {reranker_type}")
    logger.info(f"   🔢 Final results: {rerank_top_n}")
    logger.info(f"   🎨 Variant: LangChain Reranker (Variant 2)")
    
    return retriever


def get_available_rerankers() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available rerankers
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Information about available rerankers including requirements and configs
    """
    return {
        "cross_encoder": {
            "description": "Hugging Face Cross-Encoder models (BGE, etc.) - Recommended",
            "requirements": ["sentence-transformers"],
            "default_model": "BAAI/bge-reranker-base",
            "pros": ["Local/self-hosted", "Good performance", "Free to use", "Multiple model sizes"],
            "cons": ["Requires GPU for good performance", "Larger memory footprint"],
            "available_models": [
                "BAAI/bge-reranker-base",
                "BAAI/bge-reranker-large", 
                "BAAI/bge-reranker-v2-m3",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ]
        },
        "flashrank": {
            "description": "Ultra-fast lightweight reranking",
            "requirements": ["flashrank"],
            "default_model": "Built-in model",
            "pros": ["Very fast", "Lightweight", "Good for production", "No GPU required"],
            "cons": ["May have lower accuracy than neural rerankers", "Limited customization"]
        },
        "jina": {
            "description": "Jina AI rerankers",
            "requirements": ["jina", "jina API key"],
            "default_model": "jina-reranker-m0",
            "pros": ["Good performance", "Optimized for search", "API-based"],
            "cons": ["Requires API key", "External dependency", "API costs"]
        }
    }


def print_reranker_options():
    """Print available reranker options with details"""
    rerankers = get_available_rerankers()
    
    print("🔄 Available LangChain Rerankers for Variant 2:")
    print("=" * 60)
    
    for name, info in rerankers.items():
        print(f"\n📌 {name.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Requirements: {', '.join(info['requirements'])}")
        print(f"   Default Model: {info['default_model']}")
        print(f"   ✅ Pros: {', '.join(info['pros'])}")
        print(f"   ❌ Cons: {', '.join(info['cons'])}")
    
    print("\n💡 Recommendations:")
    print("   • For best performance + API: cohere or voyage")
    print("   • For local/self-hosted: cross_encoder with BGE models")
    print("   • For speed/production: flashrank")
    print("   • For balanced approach: cross_encoder (bge-reranker-base)")


if __name__ == "__main__":
    # Print available options when run directly
    print_reranker_options()
