"""Méthodes classiques : histogrammes LBP, khi-deux, projections sur pixels."""

import numpy as np
import torch

from facegeom.classical import PixelProjections, lbp_bins, lbp_histogram
from facegeom.eval.evaluate import evaluate, make_eval_pairs
from facegeom.eval.retrieval import chi2_similarity, retrieval_per_query, summarize
from facegeom.eval.verification import pair_scores

VCFG = {"n_negative": 2000, "seed": 0, "fars": [0.1, 0.01]}
UCFG = {"n_pairs": 2000, "seed": 0, "t": 2}


def _histograms(n_id=12, per_id=4, dim=30, seed=0):
    """Histogrammes de masse 1, presque identiques au sein d'une identité."""
    rng = np.random.default_rng(seed)
    centers = rng.dirichlet(np.full(dim, 0.3), size=n_id)
    x = np.repeat(centers, per_id, axis=0) + 1e-4 * rng.random((n_id * per_id, dim))
    return (x / x.sum(1, keepdims=True)).astype(np.float32), np.repeat(np.arange(n_id), per_id)


def test_lbp_histogram_shape_and_mass():
    gray = np.random.default_rng(0).integers(0, 256, size=(128, 128), dtype=np.uint8)
    h = lbp_histogram(gray, points=8, radius=2, grid=7)
    assert lbp_bins(8) == 59 and h.shape == (7 * 7 * 59,)
    assert np.allclose(h.reshape(49, 59).sum(1), 1.0, atol=1e-5)  # chaque case somme à 1
    assert (h == lbp_histogram(gray, 8, 2, 7)).all()


def test_chi2_similarity_bounds_and_pairs_agree():
    x, _ = _histograms()
    sim = chi2_similarity(torch.from_numpy(x), torch.from_numpy(x)).numpy()
    assert np.allclose(np.diag(sim), 1.0, atol=1e-6)  # identiques : 1
    assert sim.min() >= -1 - 1e-6 and np.allclose(sim, sim.T, atol=1e-5)
    pairs = np.array([[0, 1], [0, 20], [5, 40]])
    assert np.allclose(pair_scores(x, pairs, metric="chi2"), sim[pairs[:, 0], pairs[:, 1]], atol=1e-5)


def test_chi2_metrics_are_perfect_on_separable_histograms():
    x, y = _histograms()
    assert summarize(retrieval_per_query(x, y, metric="chi2"))["map_at_r"] == 100.0
    m, _ = evaluate(x, y, make_eval_pairs(y, VCFG, UCFG), metric="chi2")
    assert m["auc"] == 100.0 and np.isnan(m["alignment"]) and np.isnan(m["uniformity"])


def test_pixel_projections_shapes_and_fisher_separates():
    rng = np.random.default_rng(0)
    n_id, per_id, d = 12, 20, 64
    centers = rng.integers(60, 200, size=(n_id, d))
    # Bruit fort commun à toutes les classes sur les 8 premiers pixels : l'ACP le garde, la LDA l'ignore
    noise = np.zeros((n_id * per_id, d))
    noise[:, :8] = rng.normal(0, 60, size=(n_id * per_id, 8))
    pixels = np.clip(np.repeat(centers, per_id, axis=0) + noise + rng.normal(0, 2, (n_id * per_id, d)), 0, 255)
    y = np.repeat(np.arange(n_id), per_id)
    proj = PixelProjections(dim=5, fisher_pca_dim=30).fit(pixels.astype(np.uint8), y)
    eig, fis = proj.eigenfaces(pixels), proj.fisherfaces(pixels)
    assert eig.shape == fis.shape == (n_id * per_id, 5)
    mapr = lambda v: summarize(retrieval_per_query(v, y))["map_at_r"]
    assert mapr(fis) > mapr(eig) and mapr(fis) > 95
