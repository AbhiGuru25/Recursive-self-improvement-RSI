"""Embedding-based solution-path diversity (PRD 5.5, metric 2).

Effective number of clusters ``N_eff`` over embedded reasoning traces. Uses a
lazy sentence-transformers import; callers may pass precomputed embeddings so
this is testable without the heavy dependency.
"""

from __future__ import annotations

import numpy as np


def effective_clusters(
    embeddings: np.ndarray,
    *,
    n_clusters: int | None = None,
    method: str = "kmeans",
    seed: int = 0,
) -> float:
    """Return the effective number of clusters via inverse Simpson's index.

    ``N_eff = 1 / sum(p_i^2)`` where ``p_i`` are cluster size fractions. This is
    robust to the exact cluster count and comparable across rounds.
    """
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim != 2 or emb.shape[0] == 0:
        return 0.0
    if emb.shape[0] == 1:
        return 1.0
    k = n_clusters or max(2, min(10, round(np.sqrt(emb.shape[0]))))

    if method == "kmeans":
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=min(k, emb.shape[0]), n_init=10, random_state=seed).fit_predict(emb)
    else:  # pragma: no cover - optional path
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(min_cluster_size=2).fit_predict(emb)

    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 / np.sum(p**2))


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Lazy sentence-transformers embedding (BGE-small or MiniLM)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, normalize_embeddings=True))
