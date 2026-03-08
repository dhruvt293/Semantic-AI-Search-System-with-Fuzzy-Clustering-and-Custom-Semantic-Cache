"""
embeddings.py
-------------
Singleton wrapper around the SentenceTransformer model.

Design decisions:
- Singleton via module-level instance: the model is large (~80 MB) so we
  load it once at startup and reuse it for all requests.
- L2 normalisation is applied before returning embeddings so that inner
  product (dot product) == cosine similarity. This lets us use FAISS
  IndexFlatIP (inner-product) as a cosine index without an extra division
  step at query time.
- `batch_size` is tunable; default 64 works well on CPU for MiniLM.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.utils import logger

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384          # Fixed output dimension for all-MiniLM-L6-v2
DEFAULT_BATCH_SIZE = 64


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer producing L2-normalised embeddings.

    Attributes:
        model_name:  HuggingFace model identifier.
        _model:      Underlying SentenceTransformer instance (lazy-loaded).
        batch_size:  Number of sentences per GPU/CPU mini-batch.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> SentenceTransformer:
        """
        Lazily load the SentenceTransformer model.

        Deferred loading allows importing this module without triggering the
        ~2-second model download/load at import time — useful for tests.
        """
        if self._model is None:
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded. Embedding dimension: %d", EMBEDDING_DIM)
        return self._model

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalise a 2-D array of vectors row-wise.

        Args:
            vectors: Array of shape (n, dim).

        Returns:
            Row-normalised array of the same shape. Vectors with zero norm
            are left as-is to avoid NaN.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division-by-zero for zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings into dense embedding vectors.

        Args:
            texts:             List of input strings.
            normalize:         Whether to L2-normalise the output (default True).
            show_progress_bar: Show tqdm progress bar (useful for large corpora).

        Returns:
            Float32 ndarray of shape (len(texts), EMBEDDING_DIM).
        """
        model = self._load()
        logger.debug("Encoding %d text(s)", len(texts))
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        vectors = vectors.astype(np.float32)
        if normalize:
            vectors = self._l2_normalize(vectors)
        return vectors

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Convenience wrapper: encode exactly one string.

        Args:
            text:      Input string.
            normalize: Whether to L2-normalise (default True).

        Returns:
            Float32 ndarray of shape (EMBEDDING_DIM,).
        """
        return self.encode([text], normalize=normalize)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — import and use this everywhere.
# ─────────────────────────────────────────────────────────────────────────────

embedding_model = EmbeddingModel()
