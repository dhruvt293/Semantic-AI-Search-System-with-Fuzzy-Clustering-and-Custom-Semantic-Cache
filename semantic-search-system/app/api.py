"""
api.py
------
FastAPI router exposing the semantic search endpoints.

Endpoints:
    POST   /query         — embed → cache check → FAISS search → cache store
    GET    /cache/stats   — return cache statistics
    DELETE /cache         — clear cache and reset statistics

Design decisions:
- The router is defined in a separate module (api.py) and mounted in
  main.py. This keeps routing logic decoupled from app startup/lifespan.
- App-level state (vector_store, clusterer, embedding_model, cache) is
  accessed via `request.app.state` rather than global variables. This is
  the FastAPI-idiomatic way to share initialised objects across requests
  without circular imports.
- All heavy computation (embedding + FAISS search) is I/O-free and CPU-
  bound, so we keep it synchronous. For true async benefits, these would
  be offloaded with `run_in_executor`, but that adds complexity not worth
  it here on CPU-bound NumPy code.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Request, status

from app.models import (
    CacheStatsResponse,
    Document,
    MessageResponse,
    QueryRequest,
    QueryResponse,
    SearchResult,
)
from app.utils import logger

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _compute_dominant_cluster(documents: list[Document]) -> int | None:
    """
    Determine the dominant cluster across a list of result documents.

    Strategy: average the cluster probability vectors across all returned
    documents; the cluster with the highest average is declared dominant.
    This is more robust than taking the cluster of the top-1 document alone.

    Args:
        documents: Retrieved documents, each carrying cluster_probs.

    Returns:
        Integer cluster index, or None if no cluster info is available.
    """
    if not documents or not documents[0].cluster_probs:
        return None

    n_clusters = len(documents[0].cluster_probs)
    avg = np.zeros(n_clusters, dtype=np.float64)
    for doc in documents:
        for c, p in doc.cluster_probs.items():
            avg[c] += p
    avg /= len(documents)
    return int(np.argmax(avg))


# ─────────────────────────────────────────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic search with caching",
    description=(
        "Embed the query, check the semantic cache, and return top-k documents "
        "from the FAISS index. Results are cached for future similar queries."
    ),
)
def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """
    Main search endpoint.

    Workflow:
    1. Embed the incoming query with the SentenceTransformer model.
    2. Run cluster inference to determine the dominant cluster (used for
       cache candidate filtering).
    3. Check the semantic cache.
       - HIT  → return the cached result immediately.
       - MISS → run FAISS nearest-neighbour search.
    4. Build the response from FAISS results.
    5. Store the result in the cache.
    6. Return the response.
    """
    state = request.app.state

    # ── 1. Embed query ──────────────────────────────────────────────────
    try:
        query_embedding: np.ndarray = state.embedding_model.encode_single(payload.query)
    except Exception as exc:
        logger.exception("Embedding failed for query: %s", payload.query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding error: {exc}",
        ) from exc

    # ── 2. Cluster inference ───────────────────────────────────────────
    try:
        probs: np.ndarray = state.clusterer.predict_proba(
            query_embedding.reshape(1, -1)
        )[0]
        dominant_cluster: int = state.clusterer.dominant_cluster(probs)
    except Exception as exc:
        logger.exception("Cluster inference failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clustering error: {exc}",
        ) from exc

    # ── 3. Cache lookup ────────────────────────────────────────────────
    cache_hit = state.cache.lookup(query_embedding, dominant_cluster)
    if cache_hit is not None:
        return QueryResponse(
            query=payload.query,
            cache_hit=True,
            matched_query=cache_hit["query_text"],
            similarity_score=round(cache_hit["similarity"], 6),
            result=cache_hit["result"],
            dominant_cluster=cache_hit["dominant_cluster"],
        )

    # ── 4. FAISS search ────────────────────────────────────────────────
    try:
        raw_results = state.vector_store.search(query_embedding, k=payload.top_k)
    except Exception as exc:
        logger.exception("FAISS search failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search error: {exc}",
        ) from exc

    if not raw_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents found. Ensure the index has been populated.",
        )

    # ── 5. Build response documents ────────────────────────────────────
    documents: list[Document] = [
        Document(
            doc_id=r["doc_id"],
            text=r["text"][:500],          # Truncate for response readability
            score=round(r["score"], 6),
            cluster_probs=r["cluster_probs"],
            target_name=r["metadata"].get("target_name", ""),
        )
        for r in raw_results
    ]
    search_result = SearchResult(documents=documents)
    result_dominant = _compute_dominant_cluster(documents) or dominant_cluster

    # ── 6. Store in cache ──────────────────────────────────────────────
    state.cache.store(
        query_text=payload.query,
        query_embedding=query_embedding,
        result=search_result,
        dominant_cluster=result_dominant,
    )

    return QueryResponse(
        query=payload.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=search_result,
        dominant_cluster=result_dominant,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /cache/stats
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache statistics",
    description="Return cumulative hit/miss statistics for the semantic cache.",
)
def cache_stats(request: Request) -> CacheStatsResponse:
    """Return live cache statistics."""
    s = request.app.state.cache.stats()
    return CacheStatsResponse(**s)


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /cache
# ─────────────────────────────────────────────────────────────────────────────


@router.delete(
    "/cache",
    response_model=MessageResponse,
    summary="Clear cache",
    description="Purge all cache entries and reset hit/miss counters.",
)
def clear_cache(request: Request) -> MessageResponse:
    """Clear the semantic cache and reset all statistics."""
    request.app.state.cache.clear()
    logger.info("Cache cleared via DELETE /cache endpoint.")
    return MessageResponse(message="Cache cleared and statistics reset.")
