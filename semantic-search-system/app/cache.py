"""
cache.py
--------
In-memory semantic cache built entirely from scratch.

NO Redis, Memcached, or external caching libraries are used.

Design overview:
───────────────
1. Every query embedding is stored alongside its computed result.
2. On a new query, we first identify candidate entries by dominant cluster
   (a fast O(bucket_size) filter instead of O(total_entries) linear scan).
3. We then compute the cosine similarity between the new query embedding
   and each candidate's stored embedding.
4. If max similarity ≥ threshold, we serve the cached result (cache HIT).
5. Otherwise we return None (cache MISS) and the caller stores the new entry.

Why cluster-based candidate filtering?
───────────────────────────────────────
With many cached entries, checking every single entry is O(n). By grouping
entries by dominant cluster first, we only compare against the (likely
small) subset of entries that share the same semantic region. As the cache
grows this keeps lookup fast without any tree or hash structure.

Thread safety:
──────────────
A threading.Lock guards every mutation and lookup so the cache is safe for
the multi-threaded context that uvicorn creates.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.utils import logger

# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD: float = 0.85   # Minimum cosine similarity for a cache hit
MAX_CACHE_ENTRIES: int = 10_000   # Soft cap — oldest entries evicted first


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """
    A single entry stored in the semantic cache.

    Attributes:
        query_text:       Original query string.
        query_embedding:  L2-normalised embedding of the query (shape: (dim,)).
        result:           Serialisable result payload returned to the client.
        dominant_cluster: Cluster index with the highest membership probability.
        timestamp:        Unix timestamp when the entry was created.
    """

    query_text: str
    query_embedding: np.ndarray
    result: Any
    dominant_cluster: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    """Running statistics for the semantic cache."""

    total_entries: int = 0
    hit_count: int = 0
    miss_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of total requests that were cache hits."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Cache
# ─────────────────────────────────────────────────────────────────────────────


class SemanticCache:
    """
    A thread-safe, in-memory semantic cache with cluster-based candidate
    filtering and cosine similarity lookup.

    Usage::

        cache = SemanticCache(threshold=0.88)

        # On a new request:
        hit = cache.lookup(query_emb, dominant_cluster=2)
        if hit:
            return hit["result"]
        result = run_search(...)
        cache.store(query_text, query_emb, result, dominant_cluster=2)

    Args:
        threshold:  Minimum cosine similarity (0–1) required for a cache hit.
                    Higher values are stricter; 0.85 is a sensible default.
        max_entries: Maximum number of entries before LRU-style eviction.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        max_entries: int = MAX_CACHE_ENTRIES,
    ) -> None:
        self.threshold = threshold
        self.max_entries = max_entries

        # Primary store: list of CacheEntry objects (insertion order = recency)
        self._entries: list[CacheEntry] = []

        # Secondary index: cluster_id → list of indices into _entries
        # Enables O(bucket) lookup instead of O(n) full scan.
        self._cluster_index: dict[int, list[int]] = {}

        self._stats = CacheStats()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the cosine similarity between two L2-normalised vectors.

        Because both vectors are already unit-length (normalised by the
        embedding model) the cosine similarity reduces to the dot product,
        which is both numerically stable and very fast.

        Args:
            a, b: 1-D float32 arrays with the same shape.

        Returns:
            Scalar float in [-1, 1] (practically [0, 1] for text embeddings).
        """
        return float(np.dot(a, b))

    def _evict_oldest(self) -> None:
        """
        Remove the oldest entry to stay below max_entries.

        This simple FIFO eviction keeps memory bounded. For a production
        system you could swap this for LRU (using an OrderedDict) or
        time-based expiry.
        """
        if not self._entries:
            return
        oldest_entry = self._entries.pop(0)

        # Rebuild cluster index (rebuild is cheap; cache is small)
        self._rebuild_cluster_index()
        logger.debug("Cache evicted oldest entry: '%s'", oldest_entry.query_text[:50])

    def _rebuild_cluster_index(self) -> None:
        """Reconstruct the cluster→entry-positions secondary index."""
        self._cluster_index = {}
        for i, entry in enumerate(self._entries):
            c = entry.dominant_cluster
            self._cluster_index.setdefault(c, []).append(i)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        query_embedding: np.ndarray,
        dominant_cluster: int,
    ) -> Optional[dict[str, Any]]:
        """
        Look up the cache for a semantically similar prior query.

        Algorithm:
        1. Retrieve candidate indices from the cluster bucket.
        2. Also check a few entries from adjacent clusters (±1) to catch
           cross-cluster boundary queries.
        3. Compute cosine similarity for each candidate.
        4. Return the best match if it meets the threshold.

        Args:
            query_embedding:  L2-normalised query vector.
            dominant_cluster: Dominant cluster for the query (narrows search).

        Returns:
            Dict with keys {query_text, result, dominant_cluster, similarity}
            if a match is found, else None.
        """
        with self._lock:
            # Gather candidate index sets from the dominant cluster
            candidate_indices: list[int] = []
            for c_offset in (0, 1, -1):
                c = dominant_cluster + c_offset
                candidate_indices.extend(self._cluster_index.get(c, []))

            if not candidate_indices:
                self._stats.miss_count += 1
                logger.debug("Cache MISS (empty cluster bucket %d).", dominant_cluster)
                return None

            # Find best cosine similarity among candidates
            best_sim: float = -1.0
            best_entry: Optional[CacheEntry] = None

            for idx in candidate_indices:
                entry = self._entries[idx]
                sim = self._cosine_similarity(query_embedding, entry.query_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry is not None and best_sim >= self.threshold:
                self._stats.hit_count += 1
                logger.info(
                    "Cache HIT (sim=%.4f ≥ %.4f): '%s'",
                    best_sim,
                    self.threshold,
                    best_entry.query_text[:60],
                )
                return {
                    "query_text": best_entry.query_text,
                    "result": best_entry.result,
                    "dominant_cluster": best_entry.dominant_cluster,
                    "similarity": best_sim,
                }

            self._stats.miss_count += 1
            logger.info(
                "Cache MISS (best_sim=%.4f < %.4f).",
                best_sim,
                self.threshold,
            )
            return None

    def store(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        result: Any,
        dominant_cluster: int,
    ) -> None:
        """
        Store a new entry in the cache.

        Evicts the oldest entry if the cache is at capacity before inserting.

        Args:
            query_text:       Original query string.
            query_embedding:  L2-normalised embedding vector.
            result:           The search result to cache.
            dominant_cluster: Cluster index for fast future retrieval.
        """
        with self._lock:
            if len(self._entries) >= self.max_entries:
                self._evict_oldest()

            entry = CacheEntry(
                query_text=query_text,
                query_embedding=query_embedding.copy(),  # Defensive copy
                result=result,
                dominant_cluster=dominant_cluster,
            )
            idx = len(self._entries)
            self._entries.append(entry)
            self._cluster_index.setdefault(dominant_cluster, []).append(idx)
            self._stats.total_entries = len(self._entries)

            logger.debug(
                "Cache STORE: '%s' → cluster %d (total entries: %d)",
                query_text[:60],
                dominant_cluster,
                len(self._entries),
            )

    def stats(self) -> dict[str, Any]:
        """
        Return a snapshot of current cache statistics.

        Returns:
            Dict with total_entries, hit_count, miss_count, hit_rate.
        """
        with self._lock:
            return {
                "total_entries": len(self._entries),
                "hit_count": self._stats.hit_count,
                "miss_count": self._stats.miss_count,
                "hit_rate": self._stats.hit_rate,
            }

    def clear(self) -> None:
        """
        Reset the cache: remove all entries and zero all statistics.

        Thread-safe — acquires the lock before clearing.
        """
        with self._lock:
            self._entries.clear()
            self._cluster_index.clear()
            self._stats = CacheStats()
            logger.info("Semantic cache cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — shared across all API requests
# ─────────────────────────────────────────────────────────────────────────────

semantic_cache = SemanticCache()
