"""
main.py
-------
Application entry point and lifespan manager.

Startup logic:
1. Try to load a pre-built FAISS index + GMM model from disk (fast path).
2. If not found (first run), build from scratch:
   a. Load + clean 20 Newsgroups dataset.
   b. Encode all documents with SentenceTransformer.
   c. Add to FAISS index.
   d. Fit GMM fuzzy clusterer.
   e. Attach cluster probabilities to each indexed document.
   f. Save everything to disk.
3. Attach shared objects to app.state (vector_store, clusterer,
   embedding_model, cache) so API handlers can access them.

First run: ~2–4 minutes (embedding 18 000 docs + GMM fitting).
Subsequent runs: ~10 seconds (load from disk).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI
from sklearn.datasets import fetch_20newsgroups

from app.api import router
from app.cache import SemanticCache
from app.clustering import FuzzyClusterer
from app.embeddings import EmbeddingModel, embedding_model
from app.utils import clean_text, logger
from app.vector_store import VectorStore

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading + cleaning
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset() -> tuple[list[str], list[str], list[int]]:
    """
    Load and clean the 20 Newsgroups dataset.

    We use the 'all' subset (train + test) with headers/footers/quotes
    stripped by sklearn before our own cleaner runs. This gives ~18 800 docs.

    Returns:
        Tuple of (cleaned_texts, target_names, targets) — parallel lists.
    """
    logger.info("Loading 20 Newsgroups dataset…")
    dataset = fetch_20newsgroups(
        subset="all",
        # sklearn can strip headers/footers/quotes at load time
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    raw_texts: list[str] = dataset.data
    target_names: list[str] = [dataset.target_names[t] for t in dataset.target]
    targets: list[int] = dataset.target.tolist()

    logger.info("Raw documents loaded: %d", len(raw_texts))

    cleaned: list[str] = []
    valid_names: list[str] = []
    valid_targets: list[int] = []

    for text, name, target in zip(raw_texts, target_names, targets):
        c = clean_text(text)
        # Skip documents that are essentially empty after cleaning
        if len(c) < 20:
            continue
        cleaned.append(c)
        valid_names.append(name)
        valid_targets.append(target)

    logger.info("Documents after cleaning: %d", len(cleaned))
    return cleaned, valid_names, valid_targets


# ─────────────────────────────────────────────────────────────────────────────
# Index building
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    emb_model: EmbeddingModel,
) -> tuple[VectorStore, FuzzyClusterer]:
    """
    Build FAISS index and fit GMM clusterer from scratch.

    This runs once on first startup (~2–4 min on CPU) and saves artefacts
    to the `data/` directory so subsequent restarts load in ~10 seconds.

    Args:
        emb_model: The singleton EmbeddingModel instance.

    Returns:
        Tuple (vector_store, clusterer).
    """
    texts, target_names, _ = load_dataset()

    # ── Embeddings ──────────────────────────────────────────────────────
    logger.info("Encoding %d documents (this may take several minutes)…", len(texts))
    t0 = time.time()
    embeddings: np.ndarray = emb_model.encode(
        texts, normalize=True, show_progress_bar=True
    )
    logger.info("Encoding complete in %.1f seconds.", time.time() - t0)

    # ── FAISS indexing ──────────────────────────────────────────────────
    store = VectorStore(dim=embeddings.shape[1])
    store.add(
        doc_ids=list(range(len(texts))),
        embeddings=embeddings,
        texts=texts,
        metadata=[{"target_name": n} for n in target_names],
    )

    # ── Fuzzy clustering ────────────────────────────────────────────────
    logger.info("Fitting fuzzy clusterer…")
    t0 = time.time()
    clusterer = FuzzyClusterer()
    clusterer.fit(embeddings)
    logger.info("Clustering complete in %.1f seconds.", time.time() - t0)

    # ── Attach cluster probs to FAISS store ─────────────────────────────
    logger.info("Computing cluster probabilities for all documents…")
    cluster_probs = clusterer.predict_proba(embeddings)
    store.set_cluster_probs(cluster_probs)

    # ── Persist to disk ─────────────────────────────────────────────────
    store.save()
    clusterer.save()
    logger.info("Index and clusterer saved to disk.")

    return store, clusterer


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.

    Runs startup logic before the server accepts requests, and teardown
    after it stops. Using the lifespan pattern instead of deprecated
    @app.on_event("startup") is the modern FastAPI approach.
    """
    logger.info("═══════════════════════════════════════════")
    logger.info("  Semantic Search System — Starting up")
    logger.info("═══════════════════════════════════════════")

    # Try loading pre-built artefacts first (fast path on restart)
    try:
        vector_store = VectorStore.load()
        clusterer = FuzzyClusterer.load()
        logger.info("Loaded pre-built index and clusterer from disk.")
    except FileNotFoundError:
        logger.info("No pre-built index found. Building from scratch…")
        vector_store, clusterer = build_index(embedding_model)

    # Attach shared state — accessible in routes via request.app.state
    app.state.vector_store = vector_store
    app.state.clusterer = clusterer
    app.state.embedding_model = embedding_model
    app.state.cache = SemanticCache(threshold=0.85)

    logger.info("Server ready. Index size: %d documents.", vector_store.index.ntotal)
    logger.info("═══════════════════════════════════════════")

    yield  # ← server runs here

    # Shutdown (nothing to clean up for in-memory structures)
    logger.info("Semantic Search System — Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Semantic Search System",
        description=(
            "Lightweight semantic search over the 20 Newsgroups corpus. "
            "Features: dense vector retrieval (FAISS), fuzzy GMM clustering, "
            "and a from-scratch semantic cache."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(router)

    @app.get("/health", tags=["Health"])
    def health() -> dict:
        """Health check endpoint."""
        return {"status": "ok"}

    return app


app = create_app()


# ─────────────────────────────────────────────────────────────────────────────
# Dev runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # Set True during development
        log_level="info",
    )
