from __future__ import annotations
from pathlib import Path
from typing import Iterator, List
import logging
import pandas as pd
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

from RAG_Pipeline.data_loader import load_documents, load_df
from RAG_Pipeline.doc_builder import build_optimized_legal_docs
from RAG_Pipeline.title_index import TitleIndex
from RAG_Pipeline.vector_store import VectorStoreManager

# Set up logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(
        self,
        dataset_path: str | Path,
        persist_dir:  str | Path,
        embedding:    Embeddings,     
        k:            int  = 10,
        force_rebuild: bool = False,
        col_map:      dict | None = None,
        doc_format_style: str = "legal_standard",
        include_celex_in_content: bool = True,
        batch_size: int = 1000,  # Batch size for processing large datasets
        chunk_size: int = 5000,  # Chunk size for reading large datasets
    ):
        self.dataset_path = Path(dataset_path)
        self.persist_dir  = Path(persist_dir)
        self.embedding    = embedding
        self.k            = k
        self.force_rebuild = force_rebuild
        self.col_map       = col_map or {}
        self.doc_format_style = doc_format_style
        self.include_celex_in_content = include_celex_in_content
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        self.title_index: TitleIndex | None = None
        self.retriever  : Chroma     | None = None

    # ------------------------------------------------------------------ #
    def _stream_documents_from_dataframe(self, df: pd.DataFrame) -> Iterator[List[Document]]:
        """
        Stream documents from a DataFrame in chunks to avoid memory issues.
        
        Yields batches of documents for processing.
        """
        total_rows = len(df)
        logger.info(f"Streaming documents from {total_rows} rows in chunks of {self.chunk_size}")
        
        for i in range(0, total_rows, self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size]
            logger.info(f"Processing chunk {i//self.chunk_size + 1} (rows {i+1}-{min(i+self.chunk_size, total_rows)})")
            
            # Build documents from this chunk
            docs = build_optimized_legal_docs(
                chunk,
                citing_cols=self.col_map.get("citing"),
                cited_cols=self.col_map.get("cited"),
                filter_empty=True,
                include_celex_in_content=self.include_celex_in_content,
                format_style=self.doc_format_style,
            )
            
            # Yield documents in batches
            for j in range(0, len(docs), self.batch_size):
                batch = docs[j:j + self.batch_size]
                yield batch

    def initialise(self):
        """Initialize the RAG pipeline (original method - loads all data at once)."""
        df = load_documents(self.dataset_path)
        docs = build_optimized_legal_docs(
            df,
            citing_cols=self.col_map.get("citing"),
            cited_cols=self.col_map.get("cited"),
            filter_empty=True,
            include_celex_in_content=self.include_celex_in_content,
            format_style=self.doc_format_style,
        )
        vs_manager = VectorStoreManager(
            persist_dir   = self.persist_dir,
            embedding     = self.embedding,
            force_rebuild = self.force_rebuild,
            batch_size    = self.batch_size,
        )
        self.retriever = vs_manager.build(docs).as_retriever(
            search_kwargs={"k": self.k}
        )

    def initialise_streaming(self):
        """
        Initialize the RAG pipeline with streaming for large datasets.
        This method is memory-efficient and handles large datasets by processing them in chunks.
        """
        logger.info(f"Initializing RAG pipeline with streaming from {self.dataset_path}")
        
        # Load DataFrame using the efficient parquet caching
        df = load_df(self.dataset_path)
        logger.info(f"Loaded dataset with {len(df)} rows")
        
        # Create vector store manager
        vs_manager = VectorStoreManager(
            persist_dir   = self.persist_dir,
            embedding     = self.embedding,
            force_rebuild = self.force_rebuild,
            batch_size    = self.batch_size,
        )
        
        # Stream documents and build vector store
        doc_generator = self._stream_documents_from_dataframe(df)
        vector_store = vs_manager.build_streaming(doc_generator)
        
        if vector_store:
            self.retriever = vector_store.as_retriever(
                search_kwargs={"k": self.k}
            )
            logger.info("RAG pipeline initialization complete")
        else:
            logger.error("Failed to create vector store")
            
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        if self.retriever and hasattr(self.retriever, 'vectorstore'):
            vs = self.retriever.vectorstore
            if hasattr(vs, '_collection'):
                return {
                    "total_documents": vs._collection.count(),
                    "persist_directory": str(self.persist_dir),
                    "batch_size": self.batch_size,
                    "chunk_size": self.chunk_size
                }
        return {"error": "Vector store not initialized"}

# ------------------------------------------------------------------ #
# Convenience function for large datasets
# ------------------------------------------------------------------ #
def create_large_dataset_pipeline(
    dataset_path: str | Path,
    persist_dir: str | Path,
    embedding: Embeddings,
    batch_size: int = 1500,  # High-throughput batch size for premium accounts
    chunk_size: int = 10000,  # Large chunks for maximum efficiency
    **kwargs
) -> RAGPipeline:
    """
    Create a RAG pipeline optimized for large datasets with API rate limits.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to the large dataset
    persist_dir : str | Path  
        Directory to store the vector database
    embedding : Embeddings
        Embedding model instance
    batch_size : int, default 1500
        Number of documents to embed in each API call  
    chunk_size : int, default 10000
        Number of rows to process from dataset at once
    **kwargs
        Additional arguments passed to RAGPipeline
        
    Returns
    -------
    RAGPipeline
        Initialized pipeline ready for querying
    """
    pipeline = RAGPipeline(
        dataset_path=dataset_path,
        persist_dir=persist_dir,
        embedding=embedding,
        batch_size=batch_size,
        chunk_size=chunk_size,
        **kwargs
    )
    
    # Use streaming initialization for large datasets
    pipeline.initialise_streaming()
    return pipeline