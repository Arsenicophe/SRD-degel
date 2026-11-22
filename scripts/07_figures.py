"""Étape 6 : figures d'une évaluation (dossier 6_eval_val ou 6_eval_test).

- histogrammes de similarité « même personne / personnes différentes » (graine 0) ;
- plan alignement / uniformité (un point par graine et par méthode) ;
- MAP@R et TAR à FAR = 10⁻³ avec leur IC à 95 % ;
- courbes ROC (graine 0) ;
- PCA + t-SNE sur 20 identités (illustration seulement) ;
- planche des plus proches voisins (graine 0).

    python scripts/07_figures.py --eval runs/<date>_6_eval_val --cache-dir /content/cache
    python scripts/07_figures.py --eval ... --en          # étiquettes en anglais (poster)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from facegeom.cli import load_configs, parser
from facegeom.data.dataset import load_images
from facegeom.eval.retrieval import l2_normalize, nearest_neighbors
from facegeom.utils import read_json
from facegeom.viz import (
    EN_LABELS,
    NAMES_EN,
    NAMES_FR,
    align_uniform_plane,
    metric_intervals,
    neighbors_sheet,
    roc_curves,
    similarity_histograms,
    tsne_panels,
)

ORDER = ["brut", "brut_acp", "lda", "nsoftmax", "arcface", "contrastive", "triplet",
         "eigenfaces", "fisherfaces", "lbp"]
EMBEDDED = ["brut_acp", "lda", "nsoftmax", "arcface", "contrastive", "triplet"]


def tsne_xy(x: np.ndarray, pca_dim: int, perplexity: float, seed: int) -> np.ndarray:
    z = PCA(n_components=min(pca_dim, x.shape[1], len(x)), random_state=seed).fit_transform(x)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(z)


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--eval", required=True, help="dossier 6_eval_<partie>")
    p.add_argument("--en", action="store_true", help="étiquettes en anglais")
    p.add_argument("--out", default=None, help="dossier des figures (défaut : <eval>/figures)")
    args = p.parse_args()
    _, _, ecfg = load_configs(args)
    ev = Path(args.eval)
    out = Path(args.out) if args.out else ev / "figures"
    out.mkdir(parents=True, exist_ok=True)
    split = read_json(ev / "meta.json")["args"]["split"]
    names = NAMES_EN if args.en else NAMES_FR

    per_seed = pd.read_csv(ev / "per_seed.csv")
    summary = pd.read_csv(ev / "summary.csv")
    methods = [m for m in ORDER if m in set(per_seed.method)]
    scores = np.load(ev / "scores_seed0.npz")
    y = scores["y"]
    mapr = summary[summary.metric == "map_at_r"].set_index("method")["mean"]

    panels = [{"title": names.get(m, m), "same": scores[m][y == 1].astype(np.float32),
               "diff": scores[m][y == 0].astype(np.float32), "note": f"MAP@R {mapr[m]:.1f}"}
              for m in methods if m != "lbp"]  # LBP : khi-deux, pas un cosinus
    similarity_histograms(panels, out / f"{split}_histogrammes.png", ncols=3,
                          labels=EN_LABELS if args.en else None)
    align_uniform_plane(per_seed, out / f"{split}_alignement_uniformite.png", en=args.en)
    order = pd.CategoricalDtype(methods, ordered=True)
    summary = summary[summary.method.isin(methods)].assign(
        method=lambda d: d.method.astype(order)).sort_values("method")
    summary["method"] = summary["method"].astype(str)
    metric_intervals(summary, "map_at_r", out / f"{split}_map_at_r.png", "MAP@R (%)", en=args.en)
    metric_intervals(summary, "tar_at_far_0.001", out / f"{split}_tar_far_1e-3.png",
                     "TAR à FAR = 10⁻³ (%)" if not args.en else "TAR at FAR = 1e-3 (%)", en=args.en)
    roc_curves({m: scores[m] for m in methods if m != "lbp"}, y, out / f"{split}_roc.png",
               en=args.en)

    # PCA + t-SNE : mêmes 20 personnes pour toutes les méthodes
    emb = np.load(ev / "embeddings_seed0.npz")
    labels = emb["labels"]
    tc = ecfg["tsne"]
    rng = np.random.default_rng(tc["seed"])
    chosen = rng.choice(np.unique(labels), size=tc["n_identities"], replace=False)
    keep = np.isin(labels, chosen)
    tsne = [{"title": names.get(m, m), "labels": labels[keep],
             "xy": tsne_xy(l2_normalize(emb[m][keep].astype(np.float32)), tc["pca_dim"],
                           tc["perplexity"], tc["seed"])}
            for m in EMBEDDED if m in emb.files]
    tsne_panels(tsne, out / f"{split}_tsne.png", en=args.en)

    # Planche des plus proches voisins : mêmes requêtes pour toutes les méthodes
    images, _ = load_images(load_configs(args)[0], args.cache_dir, split)
    nc = ecfg["neighbors"]
    queries = np.random.default_rng(nc["seed"]).choice(len(labels), nc["n_queries"], replace=False)
    for m in EMBEDDED:
        if m not in emb.files:
            continue
        x = l2_normalize(emb[m].astype(np.float32))
        nn = nearest_neighbors(x, nc["n_neighbors"])
        rows = [{"query": images[q], "neighbors": [images[j] for j in nn[q]],
                 "correct": [bool(labels[j] == labels[q]) for j in nn[q]],
                 "sims": [float(x[q] @ x[j]) for j in nn[q]]} for q in queries]
        hits = sum(sum(r["correct"]) for r in rows)
        neighbors_sheet(rows, out / f"{split}_voisins_{m}.png",
                        f"{names.get(m, m)} : {hits}/{len(rows) * nc['n_neighbors']} bons voisins")
    print("Figures dans", out)


if __name__ == "__main__":
    main()
