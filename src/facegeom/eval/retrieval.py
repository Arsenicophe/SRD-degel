"""Recherche par plus proches voisins : P@1, R-Precision, MAP@R (Musgrave et al. 2020, § 3.2).

Chaque image est requête, les références sont les autres images du même
ensemble. R = nombre d'autres images de la même identité. Les fonctions
renvoient des valeurs par requête (pour le bootstrap) ; la moyenne est la métrique.
"""

from __future__ import annotations

import numpy as np
import torch


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Lignes de longueur 1, en float32."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Vecteur nul : impossible de le normaliser.")
    return x / norms


def ap_at_r(hits: np.ndarray, r: int) -> float:
    """MAP@R d'une requête : (1/R) Σ_{i≤R} P(i) · hits[i]."""
    hits = np.asarray(hits[:r], dtype=np.float64)
    precision_at_i = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision_at_i * hits).sum() / r)


def chi2_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``1 − χ²(a, b) / masse`` entre histogrammes : 1 si identiques, plus bas si différents.

    ``χ² = Σ (a − b)² / (a + b)`` ; la masse (somme d'un histogramme) ramène
    le score dans [−1, 1], comme un cosinus. ``a`` : (m, d), ``b`` : (n, d).
    """
    d = a[:, None, :] - b[None, :, :]
    s = (a[:, None, :] + b[None, :, :]).clamp_min_(1e-12)
    return 1 - d.square_().div_(s).sum(-1) / a.sum(1, keepdim=True)


def _chi2_topk_numba(x: np.ndarray, k: int) -> np.ndarray:
    """Voisins au khi-deux compilés avec numba : environ dix fois plus rapide que torch ici.

    Parallélisme par fils Python (fonction sans GIL) : le mode parallèle de numba
    entre en conflit avec l'OpenMP de torch sur macOS.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    from numba import njit

    @njit(nogil=True, fastmath=True)
    def rows(x, lo, hi, k, out):
        n, d = x.shape
        sim = np.empty(n, dtype=np.float32)
        for i in range(lo, hi):
            a = x[i]
            mass = a.sum()
            for j in range(n):
                b = x[j]
                s = np.float32(0.0)
                for t in range(d):
                    u = a[t] + b[t]
                    if u > 0:
                        v = a[t] - b[t]
                        s += v * v / u
                sim[j] = 1 - s / mass
            sim[i] = -np.inf
            out[i] = np.argsort(-sim)[:k]

    n = len(x)
    out = np.empty((n, k), dtype=np.int64)
    step = max(1, min(64, n // 8))
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        list(pool.map(lambda lo: rows(x, lo, min(lo + step, n), k, out), range(0, n, step)))
    return out


def nearest_neighbors(x: np.ndarray, k: int, chunk: int = 2048, metric: str = "cosine") -> np.ndarray:
    """Indices des ``k`` plus proches voisins de chaque ligne de ``x``, elle exclue.

    ``metric="cosine"`` : ``x`` normalisé ; ``"chi2"`` : histogrammes, avec numba s'il est installé.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if metric == "chi2":
        try:
            return _chi2_topk_numba(x, k)
        except ImportError:
            chunk = max(1, int(2e8 // x.size))  # repli torch : petits morceaux, lent
    xt = torch.from_numpy(x)
    n = xt.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sim = chi2_similarity(xt[lo:hi], xt) if metric == "chi2" else xt[lo:hi] @ xt.T
        sim[torch.arange(hi - lo), torch.arange(lo, hi)] = -float("inf")
        out[lo:hi] = torch.topk(sim, k, dim=1).indices.numpy()
    return out


def retrieval_per_query(x: np.ndarray, labels: np.ndarray, metric: str = "cosine") -> dict[str, np.ndarray]:
    """P@1, R-Precision et MAP@R par requête ; les identités à une seule image sont exclues."""
    x = l2_normalize(x) if metric == "cosine" else x
    labels = np.asarray(labels)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    r = counts[inverse] - 1
    keep = r > 0
    nn = nearest_neighbors(x, int(r.max()), metric=metric)
    hits = labels[nn] == labels[:, None]

    rprec = np.array([hits[i, : r[i]].mean() if r[i] else np.nan for i in range(len(r))])
    mapr = np.array([ap_at_r(hits[i], r[i]) if r[i] else np.nan for i in range(len(r))])
    return {
        "p_at_1": hits[keep, 0].astype(np.float64),
        "r_precision": rprec[keep],
        "map_at_r": mapr[keep],
        "labels": labels[keep],
        "query_mask": keep,
    }


def summarize(per_query: dict[str, np.ndarray]) -> dict[str, float]:
    """Moyennes par requête, en pourcentage."""
    return {m: float(100 * per_query[m].mean()) for m in ["p_at_1", "r_precision", "map_at_r"]}
