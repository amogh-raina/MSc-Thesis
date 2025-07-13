from __future__ import annotations
from pathlib import Path
from typing import List, Iterator
import logging

from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(
        self,
        persist_dir: str | Path,
        embedding: Embeddings,
        force_rebuild: bool = False,
        batch_size: int = 1000,  # Default batch size for processing
    ):
        """
        Parameters
        ----------
        persist_dir : str | Path
            Directory where Chroma will save its collection.
        embedding : Embeddings
            *Initialised* LangChain embeddings instance – you pass in the same
            object you already use elsewhere (OpenAIEmbeddings, NomicEmbeddings,
            JinaEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings…).
        force_rebuild : bool, default False
            If True, always discard the on-disk collection and rebuild it.
        batch_size : int, default 1000
            Number of documents to process in each batch.
        """
        self._path = Path(persist_dir)
        self._embed = embedding
        self._vs = None
        self.batch_size = batch_size

        if not force_rebuild and self._path.exists():
            # Re-use an existing collection – fast start-up path
            try:
                self._vs = Chroma(
                    persist_directory=str(self._path),
                    embedding_function=self._embed,
                )
                logger.info(f"Loaded existing vector store from {self._path}")
                logger.info(f"Existing collection size: {self._vs._collection.count()}")
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")
                logger.info("Will rebuild vector store from scratch")
                self._vs = None

    # ------------------------------------------------------------------ #
    # Helper methods for batch processing
    # ------------------------------------------------------------------ #
    def _chunk_documents(self, docs: List[Document]) -> Iterator[List[Document]]:
        """Yield successive chunks of documents."""
        for i in range(0, len(docs), self.batch_size):
            yield docs[i:i + self.batch_size]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self, docs: List[Document]):
        """
        Build the Chroma store (only the first time or when forced).
        Now supports batch processing for large document collections.

        Returns
        -------
        langchain.vectorstores.Chroma
        """
        if self._vs is not None:
            # Vector store already exists (loaded from disk)
            logger.info("Using existing vector store (skipping rebuild)")
            return self._vs
            
        if self._vs is None:
            # Fresh build with batch processing
            logger.info(f"Building vector store with {len(docs)} documents in batches of {self.batch_size}")
            
            # Process first batch to initialize the vector store
            doc_chunks = list(self._chunk_documents(docs))
            if not doc_chunks:
                logger.warning("No documents to process")
                return None
            
            # Initialize with first batch
            first_batch = doc_chunks[0]
            logger.info(f"Initializing vector store with first batch of {len(first_batch)} documents")
            self._vs = Chroma.from_documents(
                first_batch,
                self._embed,
                persist_directory=str(self._path),
            )
            
            # Add remaining batches
            for i, batch in enumerate(doc_chunks[1:], 1):
                logger.info(f"Processing batch {i+1}/{len(doc_chunks)} ({len(batch)} documents)")
                try:
                    self._vs.add_documents(batch)
                    logger.info(f"Successfully added batch {i+1}")
                except Exception as e:
                    logger.error(f"Error processing batch {i+1}: {str(e)}")
                    # Continue with next batch instead of failing completely
                    continue
            
            logger.info(f"Vector store build complete. Total collection size: {self._vs._collection.count()}")
            
        return self._vs

    def build_streaming(self, doc_generator: Iterator[List[Document]]):
        """
        Build the Chroma store from a generator/iterator of document batches.
        This is memory-efficient for very large datasets.

        Parameters
        ----------
        doc_generator : Iterator[List[Document]]
            Generator that yields batches of documents
            
        Returns
        -------
        langchain.vectorstores.Chroma
        """
        if self._vs is not None:
            # Vector store already exists (loaded from disk)
            logger.info("Using existing vector store (skipping streaming rebuild)")
            return self._vs
            
        if self._vs is None:
            logger.info("Building vector store from streaming document generator")
            
            # Initialize with first batch
            try:
                first_batch = next(doc_generator)
                logger.info(f"Initializing vector store with first batch of {len(first_batch)} documents")
                self._vs = Chroma.from_documents(
                    first_batch,
                    self._embed,
                    persist_directory=str(self._path),
                )
            except StopIteration:
                logger.warning("No documents in generator")
                return None
            
            # Process remaining batches
            batch_count = 1
            for batch in doc_generator:
                batch_count += 1
                logger.info(f"Processing batch {batch_count} ({len(batch)} documents)")
                try:
                    self._vs.add_documents(batch)
                    logger.info(f"Successfully added batch {batch_count}")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_count}: {str(e)}")
                    continue
            
            logger.info(f"Streaming build complete. Processed {batch_count} batches. Total collection size: {self._vs._collection.count()}")
            
        return self._vs

    def get(self):
        """Return the live Chroma instance (after .build() has been called)."""
        return self._vs