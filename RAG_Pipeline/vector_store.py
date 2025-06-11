from __future__ import annotations
from pathlib import Path
from typing import List

from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings


class VectorStoreManager:
    def __init__(
        self,
        persist_dir: str | Path,
        embedding: Embeddings,
        force_rebuild: bool = False,
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
        """
        self._path = Path(persist_dir)
        self._embed = embedding
        self._vs = None

        if not force_rebuild and self._path.exists():
            # Re-use an existing collection – fast start-up path
            self._vs = Chroma(
                persist_directory=str(self._path),
                embedding_function=self._embed,
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self, docs: List[Document]):
        """
        Build the Chroma store (only the first time or when forced).

        Returns
        -------
        langchain.vectorstores.Chroma
        """
        if self._vs is None:
            # Fresh build
            self._vs = Chroma.from_documents(
                docs,
                self._embed,
                persist_directory=str(self._path),
            )
            self._vs.persist()
        return self._vs

    def get(self):
        """Return the live Chroma instance (after .build() has been called)."""
        return self._vs