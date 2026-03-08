"""
vector_store.py
---------------
FAISS-backed vector store for dense document retrieval.

Design decisions:
- IndexFlatIP (flat inner-product index): exact nearest-neighbour search on
  L2-normalised vectors (IP == cosine for unit vectors). Exact search is
  preferred here because 20 Newsgroups has ~18 000 documents — well within
  the range where brute-force is faster than IVF due to small index size.
  Swap to IndexIVFFlat if scaling to millions of documents.
- Metadata (texts, categories, cluster probs) is stored in parallel Python
  lists rather than inside FAISS, because FAISS only handles float vectors.
- Persisting to disk (joblib + faiss.write_index) means subsequent server
  restarts skip the ~2-min indexing step.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.utils import logger

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.pkl"


class VectorStore:
    """
    FAISS-based vector store supporting indexed search and disk persistence.

    Attributes:
        dim:      Embedding dimensionality (must match the encoder output).
        index:    FAISS IndexFlatIP instance.
        doc_ids:  Parallel list mapping FAISS internal integer → doc_id.
        texts:    Cleaned document texts.
        metadata: Per-document metadata dicts (target_name, etc.).
        cluster_probs: Soft cluster membership arrays (set after clustering).
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        # Parallel lists — FAISS index position i corresponds to index i here.
        self.doc_ids: list[int] = []
        self.texts: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        # Filled after clustering via `set_cluster_probs`
        self.cluster_probs: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(
        self,
        doc_ids: list[int],
        embeddings: np.ndarray,
        texts: list[str],
        metadata: list[dict[str, Any]],
    ) -> None:
        """
        Add a batch of documents to the index.

        Args:
            doc_ids:    Unique integer identifier for each document.
            embeddings: Float32 array shape (n, dim) — must be L2-normalised.
            texts:      Cleaned text corresponding to each document.
            metadata:   List of metadata dicts (e.g. {"target_name": "sci.space"}).

        Raises:
            ValueError: If lengths are inconsistent or embedding dim doesn't match.
        """
        n = len(doc_ids)
        if not (n == len(texts) == len(metadata) == embeddings.shape[0]):
            raise ValueError("All inputs must have the same length.")
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.dim}, got {embeddings.shape[1]}"
            )

        logger.info("Adding %d documents to FAISS index.", n)
        self.index.add(embeddings.astype(np.float32))
        self.doc_ids.extend(doc_ids)
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        logger.info("FAISS index now contains %d vectors.", self.index.ntotal)

    def set_cluster_probs(self, probs: np.ndarray) -> None:
        """
        Store soft cluster membership probabilities alongside document metadata.

        Called once after `FuzzyClusterer.fit()`. The array is expected to be
        ordered identically to the documents in this store.

        Args:
            probs: Float array of shape (n_docs, n_clusters).
        """
        if probs.shape[0] != len(self.doc_ids):
            raise ValueError(
                f"Expected {len(self.doc_ids)} rows in probs, got {probs.shape[0]}."
            )
        self.cluster_probs = [probs[i] for i in range(probs.shape[0])]
        logger.info(
            "Cluster probabilities stored for %d documents (%d clusters).",
            len(self.cluster_probs),
            probs.shape[1],
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k most similar documents to the query embedding.

        Args:
            query_embedding: 1-D float32 array of shape (dim,).
            k:               Number of results to return.

        Returns:
            List of dicts, each containing:
            - doc_id (int)
            - text (str)
            - score (float, cosine similarity 0–1)
            - metadata (dict)
            - cluster_probs (dict[int, float] or {})
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty index.")
            return []

        k = min(k, self.index.ntotal)
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                # FAISS returns -1 for padded results when k > ntotal
                continue
            cp: dict[int, float] = {}
            if self.cluster_probs:
                raw = self.cluster_probs[idx]
                cp = {int(c): float(raw[c]) for c in range(len(raw))}

            results.append(
                {
                    "doc_id": self.doc_ids[idx],
                    "text": self.texts[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx],
                    "cluster_probs": cp,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        """
        Persist the FAISS index and all parallel metadata to disk.

        Args:
            index_path:    Path for the binary FAISS index file.
            metadata_path: Path for the pickled metadata file.
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "texts": self.texts,
                    "metadata": self.metadata,
                    "cluster_probs": self.cluster_probs,
                    "dim": self.dim,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("VectorStore saved → %s, %s", index_path, metadata_path)

    @classmethod
    def load(
        cls,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> "VectorStore":
        """
        Load a previously saved VectorStore from disk.

        Args:
            index_path:    Path to binary FAISS index file.
            metadata_path: Path to pickled metadata file.

        Returns:
            Fully restored VectorStore instance.

        Raises:
            FileNotFoundError: If either file is missing.
        """
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Saved index not found at {index_path} / {metadata_path}. "
                "Run indexing first."
            )
        logger.info("Loading VectorStore from disk...")
        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)

        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(str(index_path))
        store.doc_ids = meta["doc_ids"]
        store.texts = meta["texts"]
        store.metadata = meta["metadata"]
        store.cluster_probs = meta.get("cluster_probs", [])
        logger.info(
            "VectorStore loaded: %d documents, %d vectors.",
            len(store.doc_ids),
            store.index.ntotal,
        )
        return store

    @property
    def is_empty(self) -> bool:
        """True if no documents have been indexed yet."""
        return self.index.ntotal == 0
