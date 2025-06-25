from __future__ import annotations
from pathlib import Path
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings

from RAG_Pipeline.data_loader import load_documents
from RAG_Pipeline.doc_builder import build_case_paragraph_docs
from RAG_Pipeline.title_index import TitleIndex
from RAG_Pipeline.vector_store import VectorStoreManager


class RAGPipeline:
    def __init__(
        self,
        dataset_path: str | Path,
        persist_dir:  str | Path,
        embedding:    Embeddings,     
        k:            int  = 5,
        force_rebuild: bool = False,
        col_map:      dict | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.persist_dir  = Path(persist_dir)
        self.embedding    = embedding
        self.k            = k
        self.force_rebuild = force_rebuild
        self.col_map       = col_map or {}

        self.title_index: TitleIndex | None = None
        self.retriever  : Chroma     | None = None

    # ------------------------------------------------------------------ #
    def initialise(self):
        df = load_documents(self.dataset_path)
        docs = build_case_paragraph_docs(
            df,
            citing_cols=self.col_map.get("citing"),
            cited_cols=self.col_map.get("cited"),
            filter_empty=True,
        )
        vs_manager = VectorStoreManager(
            persist_dir   = self.persist_dir,
            embedding     = self.embedding,
            force_rebuild = self.force_rebuild,
        )
        self.retriever = vs_manager.build(docs).as_retriever(
            search_kwargs={"k": self.k}
        )