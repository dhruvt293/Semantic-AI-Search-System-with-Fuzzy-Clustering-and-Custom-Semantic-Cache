"""
clustering.py
-------------
Gaussian Mixture Model (GMM) fuzzy clustering for document embeddings.

Why GMM over hard clustering (k-means)?
- GMM is a probabilistic model: it assigns every document a *probability
  distribution* across n_clusters, not a single label. This is "fuzzy"
  (soft) clustering by definition.
- Boundary documents naturally have spread-out distributions; cohesive
  documents have one dominant cluster probability.
- BIC (Bayesian Information Criterion) lets us automatically select the
  number of components without manual tuning: lower BIC = better model
  (penalises complexity).

Design decisions:
- PCA to 64 dims before GMM: full 384-dim embeddings are high-dimensional
  for Gaussian estimation (curse of dimensionality). PCA retains the bulk
  of variance and makes covariance estimation numerically stable.
- `covariance_type="diag"` scales better than "full" for high-dim data
  while still capturing per-dimension variances.
- Model is persisted with joblib to disk so it survives server restarts.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from app.utils import logger

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
CLUSTER_MODEL_PATH = DATA_DIR / "gmm_model.pkl"

# PCA dimensionality before fitting GMM
PCA_DIM = 64

# Range of cluster counts to evaluate via BIC
N_COMPONENTS_RANGE = range(5, 21)   # 5 to 20 inclusive

# GMM random seed for reproducibility
RANDOM_STATE = 42


class FuzzyClusterer:
    """
    Probabilistic (fuzzy) document clusterer using a Gaussian Mixture Model.

    Pipeline:
        raw embeddings (384-d)
          → PCA to PCA_DIM dimensions
          → GaussianMixture.fit / predict_proba

    Attributes:
        n_components:  Number of Gaussian components (clusters).
        pca:           Fitted PCA transformer.
        gmm:           Fitted GaussianMixture model.
    """

    def __init__(self) -> None:
        self.n_components: Optional[int] = None
        self.pca: Optional[PCA] = None
        self.gmm: Optional[GaussianMixture] = None

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def _select_n_components(self, embeddings_pca: np.ndarray) -> int:
        """
        Select the optimal number of clusters by minimising BIC.

        BIC rewards good fit while penalising model complexity (number of
        parameters), preventing over-fitting by picking too many clusters.

        Args:
            embeddings_pca: PCA-reduced embeddings, shape (n, PCA_DIM).

        Returns:
            Integer n_components with the lowest BIC score.
        """
        logger.info(
            "Selecting n_components via BIC over range %s…",
            list(N_COMPONENTS_RANGE),
        )
        best_bic = np.inf
        best_n = N_COMPONENTS_RANGE.start

        for n in N_COMPONENTS_RANGE:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="diag",
                random_state=RANDOM_STATE,
                max_iter=150,
                n_init=1,
            )
            gmm.fit(embeddings_pca)
            bic = gmm.bic(embeddings_pca)
            logger.debug("  n=%d  BIC=%.2f", n, bic)
            if bic < best_bic:
                best_bic = bic
                best_n = n

        logger.info("Optimal n_components = %d (BIC=%.2f)", best_n, best_bic)
        return best_n

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray) -> "FuzzyClusterer":
        """
        Fit PCA + GMM on a corpus of document embeddings.

        Steps:
        1. Reduce dimensionality with PCA.
        2. Select n_components via BIC.
        3. Fit final GMM with the chosen n_components.

        Args:
            embeddings: Float32 array shape (n_docs, EMBEDDING_DIM).

        Returns:
            self, for chaining.
        """
        logger.info("Fitting PCA (n_components=%d) on %d documents…", PCA_DIM, len(embeddings))
        self.pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
        embeddings_pca = self.pca.fit_transform(embeddings)

        self.n_components = self._select_n_components(embeddings_pca)

        logger.info("Fitting final GMM (n_components=%d)…", self.n_components)
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            random_state=RANDOM_STATE,
            max_iter=200,
            n_init=3,            # Multiple initialisations → avoid bad local optima
        )
        self.gmm.fit(embeddings_pca)
        logger.info("GMM fitting complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _to_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings through the fitted PCA."""
        if self.pca is None:
            raise RuntimeError("FuzzyClusterer is not fitted. Call .fit() first.")
        return self.pca.transform(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return soft cluster membership probabilities for a batch of embeddings.

        Args:
            embeddings: Float32 array shape (n, EMBEDDING_DIM).

        Returns:
            Float64 array shape (n, n_components). Rows sum to 1.0.
        """
        if self.gmm is None:
            raise RuntimeError("FuzzyClusterer is not fitted. Call .fit() first.")
        return self.gmm.predict_proba(self._to_pca(embeddings))

    def dominant_cluster(self, probs: np.ndarray) -> int:
        """
        Return the cluster index with the highest probability.

        Args:
            probs: 1-D array of cluster probabilities (length n_components).

        Returns:
            Integer cluster index.
        """
        return int(np.argmax(probs))

    # ------------------------------------------------------------------
    # Inspection utilities
    # ------------------------------------------------------------------

    def get_top_docs_per_cluster(
        self,
        cluster_probs: np.ndarray,
        texts: list[str],
        k: int = 5,
    ) -> dict[int, list[dict]]:
        """
        For each cluster, return the k documents with the highest membership.

        Args:
            cluster_probs: Array shape (n_docs, n_clusters).
            texts:         Corresponding document texts.
            k:             Documents to return per cluster.

        Returns:
            Dict mapping cluster_id → list of {doc_idx, text, probability}.
        """
        result: dict[int, list[dict]] = {}
        for c in range(self.n_components):
            col = cluster_probs[:, c]
            top_indices = np.argsort(col)[::-1][:k]
            result[c] = [
                {"doc_idx": int(i), "text": texts[i][:200], "probability": float(col[i])}
                for i in top_indices
            ]
        return result

    def get_boundary_docs(
        self,
        cluster_probs: np.ndarray,
        texts: list[str],
        max_dominant_prob: float = 0.40,
    ) -> list[dict]:
        """
        Identify boundary documents: those where no cluster is clearly dominant.

        A document is a boundary doc if its maximum cluster probability is
        below *max_dominant_prob* (uniform distribution = 1/n_components).

        Args:
            cluster_probs:      Array shape (n_docs, n_clusters).
            texts:              Corresponding texts.
            max_dominant_prob:  Upper bound on max cluster probability.

        Returns:
            List of dicts with doc_idx, text, max_prob, cluster_probs.
        """
        boundaries = []
        for i, probs in enumerate(cluster_probs):
            max_p = float(np.max(probs))
            if max_p < max_dominant_prob:
                boundaries.append(
                    {
                        "doc_idx": i,
                        "text": texts[i][:200],
                        "max_prob": max_p,
                        "cluster_probs": {c: float(probs[c]) for c in range(len(probs))},
                    }
                )
        return boundaries

    def get_cluster_summary(
        self,
        cluster_probs: np.ndarray,
        texts: list[str],
        top_docs: int = 20,
        top_words: int = 10,
    ) -> dict[int, list[str]]:
        """
        Summarise each cluster by the most frequent words in its top documents.

        Uses a simple TF approach (token frequency) over the top-*top_docs*
        members of each cluster.

        Args:
            cluster_probs: Array shape (n_docs, n_clusters).
            texts:         Corresponding texts.
            top_docs:      Number of top docs per cluster to analyse.
            top_words:     Number of top words to return per cluster.

        Returns:
            Dict mapping cluster_id → list of top word strings.
        """
        import re
        from collections import Counter

        STOPWORDS = {
            "the", "a", "an", "in", "is", "it", "of", "to", "and", "i",
            "that", "this", "was", "for", "on", "are", "as", "at", "be",
            "by", "or", "with", "not", "but", "from", "have", "he", "she",
            "they", "we", "you", "do", "did", "has", "had", "his", "her",
            "can", "will", "so", "if", "my", "your", "our", "who", "been",
            "would", "could", "should", "than", "then", "also", "its",
            "there", "what", "which", "one", "no", "more", "about", "up",
            "out", "were", "all", "just", "their", "me", "him", "write",
            "re", "com", "edu", "get", "get", "may",
        }

        summaries: dict[int, list[str]] = {}
        for c in range(self.n_components):
            col = cluster_probs[:, c]
            top_indices = np.argsort(col)[::-1][:top_docs]
            words: list[str] = []
            for idx in top_indices:
                tokens = re.findall(r"[a-z]{3,}", texts[idx].lower())
                words.extend([t for t in tokens if t not in STOPWORDS])
            counter = Counter(words)
            summaries[c] = [w for w, _ in counter.most_common(top_words)]
        return summaries

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = CLUSTER_MODEL_PATH) -> None:
        """Save clusterer state to disk with pickle."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_components": self.n_components,
                    "pca": self.pca,
                    "gmm": self.gmm,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("FuzzyClusterer saved → %s", path)

    @classmethod
    def load(cls, path: Path = CLUSTER_MODEL_PATH) -> "FuzzyClusterer":
        """
        Load a previously saved FuzzyClusterer from disk.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Cluster model not found at {path}.")
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.n_components = state["n_components"]
        obj.pca = state["pca"]
        obj.gmm = state["gmm"]
        logger.info(
            "FuzzyClusterer loaded (n_components=%d) from %s",
            obj.n_components,
            path,
        )
        return obj
