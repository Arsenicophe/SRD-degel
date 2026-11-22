"""Méthodes classiques sur les pixels : Eigenfaces, Fisherfaces, histogrammes LBP.

Même recadrage que pour le réseau, en niveaux de gris. Eigenfaces et
Fisherfaces donnent des vecteurs comparés par cosinus ; LBP donne des
histogrammes comparés par khi-deux (``metric="chi2"`` dans l'évaluation).
"""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA

from facegeom.eval.baselines import CenteredLDA


def lbp_bins(points: int) -> int:
    """Nombre de motifs « uniformes » distincts, plus une case pour les autres (59 pour 8 points)."""
    return points * (points - 1) + 3


def lbp_histogram(gray: np.ndarray, points: int, radius: int, grid: int) -> np.ndarray:
    """Histogrammes LBP uniformes par case d'une grille ``grid × grid``, concaténés.

    Chaque case somme à 1 : le vecteur a une masse totale de ``grid²``.
    """
    codes = local_binary_pattern(gray, points, radius, method="nri_uniform").astype(np.int64)
    bins = lbp_bins(points)
    rows = np.repeat(np.arange(grid), np.diff(np.linspace(0, gray.shape[0], grid + 1).round().astype(int)))
    cols = np.repeat(np.arange(grid), np.diff(np.linspace(0, gray.shape[1], grid + 1).round().astype(int)))
    cell = rows[:, None] * grid + cols[None, :]
    hist = np.bincount((cell * bins + codes).ravel(), minlength=grid * grid * bins).reshape(grid * grid, bins)
    return (hist / hist.sum(axis=1, keepdims=True)).astype(np.float32).ravel()


_CACHE = None  # cache des carrés, ouvert une fois par processus


def _open(cache_path: str) -> None:
    global _CACHE
    _CACHE = np.load(cache_path, mmap_mode="r")


def _one(job) -> tuple[np.ndarray, np.ndarray | None]:
    i, gray_size, lbp = job
    face = Image.fromarray(np.asarray(_CACHE[i])).convert("L")  # même carré que le réseau
    if lbp and face.size[0] != lbp["size"]:
        face_lbp = face.resize((lbp["size"],) * 2, Image.BILINEAR)
    else:
        face_lbp = face
    hist = (lbp_histogram(np.asarray(face_lbp), lbp["points"], lbp["radius"], lbp["grid"])
            if lbp else None)
    pixels = np.asarray(face.resize((gray_size, gray_size), Image.BILINEAR), dtype=np.uint8).ravel()
    return pixels, hist


def extract_pixels(cache_path: str, n_images: int, gray_size: int, lbp: dict | None,
                   workers: int = 4) -> tuple[np.ndarray, np.ndarray | None]:
    """Pixels gris ``(N, gray_size²)`` en uint8 et, si ``lbp`` est donné, histogrammes LBP ``(N, D)``.

    Les images viennent du cache des carrés de 128 px (``scripts/01_build_cache.py``).
    """
    jobs = [(i, gray_size, lbp) for i in range(n_images)]
    with Pool(workers, initializer=_open, initargs=(str(cache_path),)) as pool:
        out = pool.map(_one, jobs, chunksize=256)
    pixels = np.stack([p for p, _ in out])
    return pixels, (np.stack([h for _, h in out]) if lbp else None)


class PixelProjections:
    """Eigenfaces (ACP) et Fisherfaces (ACP puis LDA), ajustées sur les pixels d'entraînement."""

    def __init__(self, dim: int, fisher_pca_dim: int, seed: int = 0):
        self.dim, self.fisher_pca_dim, self.seed = dim, fisher_pca_dim, seed

    @staticmethod
    def _scale(pixels: np.ndarray) -> np.ndarray:
        return np.asarray(pixels, dtype=np.float32) / 255.0

    def fit(self, pixels: np.ndarray, labels: np.ndarray) -> PixelProjections:
        # Une seule ACP : ses premières composantes sont les eigenfaces
        self.pca_ = PCA(n_components=self.fisher_pca_dim, random_state=self.seed).fit(self._scale(pixels))
        self.lda_ = CenteredLDA(self.dim).fit(self.pca_.transform(self._scale(pixels)), labels)
        return self

    def eigenfaces(self, pixels: np.ndarray) -> np.ndarray:
        return self.pca_.transform(self._scale(pixels))[:, : self.dim].astype(np.float32)

    def fisherfaces(self, pixels: np.ndarray) -> np.ndarray:
        return self.lda_.transform(self.pca_.transform(self._scale(pixels))).astype(np.float32)


def load_classical(root, method: str, split: str, names: list[str]) -> np.ndarray:
    """Vecteurs d'une méthode classique, remis dans l'ordre de ``names``."""
    from pathlib import Path

    root = Path(root)
    stored = (root / f"{split}_images.txt").read_text().split()
    x = np.load(root / f"{method}_{split}.npy")
    if stored == list(names):
        return x
    pos = {n: i for i, n in enumerate(stored)}
    return x[[pos[n] for n in names]]
