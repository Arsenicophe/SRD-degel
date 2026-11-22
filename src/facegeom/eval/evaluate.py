"""Évaluation complète d'un jeu de vecteurs : recherche, vérification, géométrie.

Tout sort des similarités cosinus. Les paires sont tirées une fois par partie
et servent à toutes les méthodes (comparaison appariée).
"""

from __future__ import annotations

import numpy as np

from facegeom.eval.retrieval import retrieval_per_query, summarize
from facegeom.eval.verification import (
    make_pairs,
    pair_scores,
    rates_at_threshold,
    verification_summary,
)


def random_pairs(n: int, n_pairs: int, seed: int = 0) -> np.ndarray:
    """``n_pairs`` paires d'indices distincts tirées uniformément parmi ``n`` images."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, size=n_pairs)
    b = (a + rng.integers(1, n, size=n_pairs)) % n  # décalage non nul : jamais a == b
    return np.stack([a, b], axis=1)


def alignment(pos_scores: np.ndarray) -> float:
    """Wang & Isola : moyenne de ‖f(x) − f(y)‖² sur les paires positives (= 2 − 2 cos)."""
    return float(np.mean(2 - 2 * pos_scores.astype(np.float64)))


def uniformity(rand_scores: np.ndarray, t: float = 2.0) -> float:
    """Wang & Isola : log de la moyenne de exp(−t ‖f(x) − f(y)‖²) sur des paires au hasard."""
    return float(np.log(np.mean(np.exp(-t * (2 - 2 * rand_scores.astype(np.float64))))))


def make_eval_pairs(labels: np.ndarray, vcfg: dict, ucfg: dict) -> dict:
    """Paires de vérification et paires au hasard d'une partie, à réutiliser pour toutes les méthodes."""
    pairs, y = make_pairs(labels, vcfg["n_negative"], vcfg["seed"])
    return {"pairs": pairs, "y": y, "fars": tuple(vcfg["fars"]), "t": ucfg["t"],
            "unif_pairs": random_pairs(len(labels), ucfg["n_pairs"], ucfg["seed"])}


def evaluate(x: np.ndarray, labels: np.ndarray, ep: dict, threshold: float | None = None,
             metric: str = "cosine", per_query: dict | None = None) -> tuple[dict, np.ndarray]:
    """Toutes les métriques (en %, sauf alignement et uniformité) et les scores des paires.

    ``threshold`` : seuil de vérification choisi sur la validation ; s'il est
    donné, ajoute TAR, FAR et exactitude équilibrée à ce seuil.
    ``metric="chi2"`` (histogrammes LBP) : pas de sphère, donc ni alignement ni uniformité.
    ``per_query`` : si un dictionnaire est donné, il reçoit les valeurs de
    recherche par requête (pour le bootstrap sur les identités).
    """
    scores = pair_scores(x, ep["pairs"], metric=metric)
    y = ep["y"]
    pq = retrieval_per_query(x, labels, metric)
    if per_query is not None:
        per_query.update(pq)
    m = {**summarize(pq), **verification_summary(y, scores, ep["fars"])}
    if metric == "cosine":
        m["alignment"] = alignment(scores[y == 1])
        m["uniformity"] = uniformity(pair_scores(x, ep["unif_pairs"]), ep["t"])
    else:
        m["alignment"] = m["uniformity"] = float("nan")
    if threshold is not None:
        m.update(rates_at_threshold(y, scores, threshold))
    return m, scores
