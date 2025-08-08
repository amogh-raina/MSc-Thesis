"""
Hybrid RAG Pipeline - Combines dense (vector) and sparse (BM25) retrieval
for improved legal Q&A performance, especially on definitional questions
and exact citation matching.

REQUIRED DEPENDENCIES:
- BM25 retrieval: pip install rank-bm25

OPTIONAL DEPENDENCIES for reranking:
- BGE: pip install sentence-transformers torch
- Jina: pip install jina (requires API key)
- Cohere: pip install cohere (requires API key)

Recommended reranker models:
- BAAI/bge-reranker-base: Free, good performance, runs locally
- BAAI/bge-reranker-large: Free, better performance, more compute
- BAAI/bge-reranker-v2-m3: Latest model, best performance

INSTALLATION:
pip install rank-bm25 sentence-transformers torch
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import logging

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors import JinaRerank

from ..rag_pipeline import RAGPipeline
from ..data_loader import load_df
from ..doc_builder import build_optimized_legal_docs
from ..vector_store import VectorStoreManager

logger = logging.getLogger("RAG_Pipeline.variants.hybrid")

class HybridRAGPipeline(RAGPipeline):
    """
    Enhanced RAG Pipeline with hybrid dense + sparse retrieval.
    
    Combines:
    - Dense retrieval (existing Chroma vector store) for semantic similarity
    - Sparse retrieval (BM25) for exact keyword/citation matching  
    - Optional reranking for final result refinement
    
    Especially beneficial for:
    - Legal citation matching (Article X TFEU, case names, CELEX IDs)
    - Definitional questions requiring exact terminology
    - Technical legal phrase matching
    """
    
    def __init__(
        self,
        dataset_path: Path,
        persist_dir: Path,
        embedding,
        k: int = 10,
        # Hybrid-specific parameters
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        hybrid_k: int = 20,  # Over-fetch for reranking
        use_reranking: bool = False,
        rerank_type: str = "bge",  # "bge", "jina", or "cohere"
        rerank_model: str = "BAAI/bge-reranker-base",
        rerank_top_n: int = 15,
        **kwargs
    ):
        """
        Initialize Hybrid RAG Pipeline
        
        Args:
            bm25_weight: Weight for BM25/sparse retrieval (0.0-1.0)
            vector_weight: Weight for vector/dense retrieval (0.0-1.0) 
            hybrid_k: Number of docs to retrieve before reranking
            use_reranking: Whether to apply reranking after hybrid retrieval
            rerank_type: Type of reranker ("bge", "jina", or "cohere")
            rerank_model: Model name for the reranker
            rerank_top_n: Final number of documents after reranking
        """
        # Initialize base RAG pipeline first
        super().__init__(
            dataset_path=dataset_path,
            persist_dir=persist_dir, 
            embedding=embedding,
            k=k,
            **kwargs
        )
        
        # Store hybrid parameters
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.hybrid_k = hybrid_k
        self.use_reranking = use_reranking
        self.rerank_type = rerank_type
        self.rerank_model = rerank_model
        self.rerank_top_n = rerank_top_n
        
        # Will be initialized in setup_hybrid_retrieval
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.vector_store = None
        
    def initialise(self):
        """Initialize the hybrid RAG system (standard method)"""
        # First initialize the base vector store
        super().initialise()
        
        # Store the vector store reference for hybrid setup
        if self.retriever and hasattr(self.retriever, 'vectorstore'):
            self.vector_store = self.retriever.vectorstore
        
        # Then set up hybrid retrieval components
        self.setup_hybrid_retrieval()
        
        logger.info(f"Hybrid RAG initialized - Vector: {self.vector_weight}, BM25: {self.bm25_weight}")
        if self.use_reranking:
            logger.info(f"Reranking enabled: {self.rerank_type} using {self.rerank_model} (top {self.rerank_top_n})")
    
    def initialise_streaming(self):
        """
        Initialize the hybrid RAG system with streaming for large datasets.
        This method processes data in chunks and builds BM25 index incrementally.
        """
        # First, call the base class streaming initialization to set up vector store
        super().initialise_streaming()
        
        # Verify base initialization worked
        if not self.retriever:
            logger.error("Base streaming initialization failed - no retriever created")
            return
        
        # Store the vector store reference for hybrid setup
        if hasattr(self.retriever, 'vectorstore'):
            self.vector_store = self.retriever.vectorstore
        else:
            logger.error("Base retriever doesn't have vectorstore attribute")
            return
        
        # Now setup hybrid retrieval (this will handle BM25 for large datasets)
        self.setup_hybrid_retrieval_streaming()
        
        if self.use_reranking:
            logger.info(f"Reranking enabled: {self.rerank_type} using {self.rerank_model}")
    
    def setup_hybrid_retrieval(self):
        """Set up BM25 retriever and ensemble combination (standard method)"""
        try:
            # Get documents from Chroma collection
            collection = self.vector_store._collection
            all_data = collection.get(include=['documents', 'metadatas'])
            
            total_docs = len(all_data['documents'])
            if total_docs > 100000:
                logger.warning(f"Large dataset detected ({total_docs} docs). BM25 indexing may take several minutes.")
            
            # Convert to Document objects
            all_docs = []
            for doc_text, metadata in zip(all_data['documents'], all_data['metadatas']):
                doc = Document(page_content=doc_text, metadata=metadata)
                all_docs.append(doc)
            
            # Create BM25 retriever with dependency checking
            try:
                import rank_bm25
                self.bm25_retriever = BM25Retriever.from_documents(all_docs)
                self.bm25_retriever.k = self.hybrid_k
                logger.info(f"BM25 retriever created with k={self.hybrid_k}")
                
            except ImportError:
                logger.error("Missing dependency: rank_bm25. Install with: pip install rank-bm25")
                raise Exception("BM25 requires rank_bm25 package. Install with: pip install rank-bm25")
            except Exception as bm25_error:
                logger.error(f"Failed to create BM25 retriever: {bm25_error}")
                raise Exception(f"BM25 indexing failed: {bm25_error}")
            
            # Create ensemble retriever
            self._create_ensemble_retriever()
            
        except Exception as e:
            logger.error(f"Failed to setup hybrid retrieval: {e}")
            logger.warning("Falling back to vector-only retrieval")
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
    
    def setup_hybrid_retrieval_streaming(self):
        """
        Set up BM25 retriever and ensemble combination for streaming/large datasets.
        This method handles BM25 indexing more efficiently for large datasets.
        """
        try:
            all_docs = []
            collection = self.vector_store._collection
            total_docs = collection.count()
            
            # For extremely large datasets, skip BM25 entirely
            if total_docs > 300000:
                logger.warning(f"Extremely large dataset ({total_docs} docs). Using vector-only retrieval.")
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
                return
            elif total_docs > 200000:
                logger.warning(f"Large dataset ({total_docs} docs). Using memory-efficient BM25 indexing...")
                
                try:
                    all_data = collection.get(include=['documents', 'metadatas'])
                    
                    # Process in batches to avoid memory issues
                    batch_size = 5000
                    total_loaded = len(all_data['documents'])
                    
                    for i in range(0, total_loaded, batch_size):
                        batch_docs = all_data['documents'][i:i+batch_size]
                        batch_metas = all_data['metadatas'][i:i+batch_size]
                        
                        for doc_text, metadata in zip(batch_docs, batch_metas):
                            doc = Document(page_content=doc_text, metadata=metadata)
                            all_docs.append(doc)
                            
                except Exception as large_dataset_error:
                    logger.error(f"Failed to load large dataset for BM25: {large_dataset_error}")
                    logger.warning("Falling back to vector-only retrieval")
                    self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
                    return
            else:
                # For smaller datasets, load all at once
                all_data = collection.get(include=['documents', 'metadatas'])
                
                for doc_text, metadata in zip(all_data['documents'], all_data['metadatas']):
                    doc = Document(page_content=doc_text, metadata=metadata)
                    all_docs.append(doc)
            
            # Create BM25 retriever with dependency checking
            try:
                import rank_bm25
                self.bm25_retriever = BM25Retriever.from_documents(all_docs)
                self.bm25_retriever.k = self.hybrid_k
                logger.info(f"BM25 retriever created with k={self.hybrid_k}")
                
            except ImportError:
                logger.error("Missing dependency: rank_bm25. Install with: pip install rank-bm25")
                logger.warning("Falling back to vector-only retrieval")
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
                return
            except Exception as bm25_error:
                logger.error(f"Failed to create BM25 retriever: {bm25_error}")
                logger.warning("BM25 indexing failed. Falling back to vector-only retrieval.")
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
                return
            
            # Create ensemble retriever
            self._create_ensemble_retriever()
            
        except Exception as e:
            logger.error(f"Failed to setup streaming hybrid retrieval: {e}")
            logger.warning("Falling back to vector-only retrieval")
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
    
    def _create_ensemble_retriever(self):
        """Create the ensemble retriever from vector and BM25 retrievers"""
        # Create vector retriever to use hybrid_k
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.hybrid_k}
        )
        
        # Create ensemble retriever (hybrid)
        try:
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, self.bm25_retriever],
                weights=[self.vector_weight, self.bm25_weight]
            )
            logger.info("Ensemble retriever created successfully")
        except Exception as ensemble_error:
            logger.error(f"Failed to create ensemble retriever: {ensemble_error}")
            raise Exception(f"Ensemble retriever creation failed: {ensemble_error}")
        
        # Add reranking if enabled
        if self.use_reranking:
            try:
                reranker = self._create_reranker()
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=self.hybrid_retriever
                )
                logger.info(f"Reranking enabled using {self.rerank_type}")
            except Exception as e:
                logger.warning(f"Reranking setup failed: {e}. Using hybrid without reranking.")
                self.retriever = self.hybrid_retriever
        else:
            self.retriever = self.hybrid_retriever
    
    def _create_reranker(self):
        """Create the appropriate reranker based on rerank_type"""
        if self.rerank_type.lower() == "bge":
            # HuggingFace BGE Cross Encoder
            cross_encoder = HuggingFaceCrossEncoder(model_name=self.rerank_model)
            return CrossEncoderReranker(model=cross_encoder, top_n=self.rerank_top_n)
        
        elif self.rerank_type.lower() == "jina":
            # Jina Rerank
            return JinaRerank(top_n=self.rerank_top_n)
        
        elif self.rerank_type.lower() == "cohere":
            # Cohere Rerank (fallback option)
            try:
                from langchain.retrievers.document_compressors import CohereRerank
                return CohereRerank(model=self.rerank_model, top_n=self.rerank_top_n)
            except ImportError:
                logger.error("CohereRerank not available. Install with: pip install cohere")
                raise
        
        else:
            raise ValueError(f"Unsupported rerank_type: {self.rerank_type}. "
                           f"Supported types: 'bge', 'jina', 'cohere'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid RAG system"""
        base_stats = super().get_stats()
        
        hybrid_stats = {
            **base_stats,
            "retrieval_type": "hybrid",
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "hybrid_k": self.hybrid_k,
            "reranking_enabled": self.use_reranking,
            "rerank_type": self.rerank_type if self.use_reranking else None,
            "rerank_model": self.rerank_model if self.use_reranking else None,
            "final_k": self.rerank_top_n if self.use_reranking else self.hybrid_k
        }
        
        if hasattr(self, 'bm25_retriever') and self.bm25_retriever:
            try:
                hybrid_stats["bm25_docs_indexed"] = len(self.bm25_retriever.docs)
            except:
                hybrid_stats["bm25_docs_indexed"] = "unknown"
        
        return hybrid_stats

# Convenience function for creating large dataset hybrid pipeline
def create_large_hybrid_dataset_pipeline(
    dataset_path: Path,
    persist_dir: Path, 
    embedding,
    batch_size: int = 2000,
    chunk_size: int = 10000,
    force_rebuild: bool = False,
    col_map: Optional[Dict] = None,
    doc_format_style: str = "legal_standard",
    include_celex_in_content: bool = True,
    k: int = 15,
    # Hybrid parameters
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
    use_reranking: bool = False,
    rerank_type: str = "bge",
    rerank_model: str = "BAAI/bge-reranker-base",
    **kwargs
) -> HybridRAGPipeline:
    """
    Create hybrid RAG pipeline optimized for large datasets with streaming processing
    
    This combines the streaming/batch processing capabilities with hybrid retrieval.
    """
    
    hybrid_pipeline = HybridRAGPipeline(
        dataset_path=dataset_path,
        persist_dir=persist_dir,
        embedding=embedding,
        k=k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        use_reranking=use_reranking,
        rerank_type=rerank_type,
        rerank_model=rerank_model,
        force_rebuild=force_rebuild,
        col_map=col_map,
        doc_format_style=doc_format_style,
        include_celex_in_content=include_celex_in_content,
        batch_size=batch_size,
        chunk_size=chunk_size,
        **kwargs
    )
    
    # Use streaming initialization for large datasets
    try:
        hybrid_pipeline.initialise_streaming()
        
        # Check if retriever was properly initialized
        if hybrid_pipeline.retriever is None:
            logger.error("Retriever initialization failed")
            
            # Try to create a fallback retriever
            if hybrid_pipeline.vector_store is not None:
                logger.warning("Creating fallback vector-only retriever")
                hybrid_pipeline.retriever = hybrid_pipeline.vector_store.as_retriever(search_kwargs={"k": k})
            else:
                raise Exception("No vector store available for fallback")
            
    except Exception as init_error:
        logger.error(f"Pipeline initialization failed: {init_error}")
        raise Exception(f"Large hybrid dataset pipeline initialization failed: {init_error}")
    
    return hybrid_pipeline

