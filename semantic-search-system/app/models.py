"""
models.py
---------
Pydantic v2 models for all API request and response contracts.

Using Pydantic ensures:
- Runtime type validation + clear error messages for bad payloads.
- Auto-generated OpenAPI schema (visible at /docs).
- Clean separation of transport-layer types from internal domain objects.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """
    Request payload for POST /query.

    Attributes:
        query: Natural language search query. Must be non-empty after stripping.
        top_k: Number of documents to return (default 5, max 20).
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language search query.",
        examples=["space shuttle NASA mission"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top documents to retrieve (1–20).",
    )

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, v: Any) -> Any:
        """Strip leading/trailing whitespace before length validation."""
        if isinstance(v, str):
            return v.strip()
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Inner response types
# ─────────────────────────────────────────────────────────────────────────────


class Document(BaseModel):
    """
    A single retrieved document returned inside a query response.

    Attributes:
        doc_id:        Internal integer document ID.
        text:          Cleaned document text (truncated for readability).
        score:         Cosine similarity score between query and document (0–1).
        cluster_probs: Soft cluster membership probabilities (sums to ~1.0).
        target_name:   Newsgroup category label (e.g. 'sci.space').
    """

    doc_id: int = Field(..., description="Internal document ID.")
    text: str = Field(..., description="Cleaned document text snippet.")
    score: float = Field(..., description="Cosine similarity score (0–1).")
    cluster_probs: dict[int, float] = Field(
        default_factory=dict,
        description="Fuzzy cluster membership {cluster_id: probability}.",
    )
    target_name: str = Field(
        default="",
        description="Original 20-Newsgroups category label.",
    )


class SearchResult(BaseModel):
    """
    Wrapper around the list of retrieved documents.

    Attributes:
        documents: Ranked list of matching documents.
    """

    documents: list[Document] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Primary response models
# ─────────────────────────────────────────────────────────────────────────────


class QueryResponse(BaseModel):
    """
    Full response for POST /query.

    Attributes:
        query:            The original query string.
        cache_hit:        Whether this result was served from the cache.
        matched_query:    The cached query that triggered a cache hit (None if miss).
        similarity_score: Similarity between the current query and cached query embedding.
        result:           The search results.
        dominant_cluster: Index of the cluster with the highest average probability
                          across returned documents.
    """

    query: str
    cache_hit: bool = False
    matched_query: str | None = None
    similarity_score: float | None = None
    result: SearchResult
    dominant_cluster: int | None = None


class CacheStatsResponse(BaseModel):
    """
    Response for GET /cache/stats.

    Attributes:
        total_entries: Number of entries currently in the cache.
        hit_count:     Cumulative cache hits since last reset.
        miss_count:    Cumulative cache misses since last reset.
        hit_rate:      hit_count / (hit_count + miss_count), or 0.0 if no queries.
    """

    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class MessageResponse(BaseModel):
    """Generic message response (used by DELETE /cache)."""

    message: str
