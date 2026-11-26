"""Figures du poster, en anglais, à partir des résultats de l'étape 6 (validation).

Lit le dossier d'évaluation (summary.csv, scores_seed0.npz, embeddings_seed0.npz)
et le zip de CelebA pour les vignettes. N'entraîne rien, ne lit pas le test.

    python poster/make_figures.py --eval-dir <runs>/<date>_6_eval_val \
        --celeba-zip <chemin>/img_align_celeba.zip --split val
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"

# Okabe-Ito : distinguables en cas de daltonisme. Une couleur par perte, partout.
COLORS = {"nsoftmax": "#0072B2", "arcface": "#D55E00", "contrastive": "#009E73",
          "triplet": "#CC79A7", "classical": "#9A9A9A", "untrained": "#5E5E5E"}
NAMES = {"eigenfaces": "Eigenfaces", "lbp": "LBP", "fisherfaces": "Fisherfaces",
         "brut_acp": "ImageNet\n+ PCA", "lda": "ImageNet\n+ LDA", "triplet": "Triplet",
         "contrastive": "Contras-\ntive", "nsoftmax": "Norm.\nsoftmax", "arcface": "ArcFace"}
YEARS = {"eigenfaces": "1991", "fisherfaces": "1997", "lbp": "2004", "contrastive": "2006",
         "triplet": "2015", "nsoftmax": "2017", "arcface": "2019"}
LOSSES = ["triplet", "contrastive", "nsoftmax", "arcface"]
INK, MUTED = "#1F1F1F", "#6B6B6B"
GOOD, BAD = "#1A9850", "#D73027"
BOX = (24, 52, 152, 180)
MM = 1 / 25.4  # tailles en millimètres, à l'échelle du poster A0

plt.rcParams.update({"font.family": "Avenir Next", "font.size": 20, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.edgecolor": "#BBBBBB", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("écrit", OUT / name)


def color_of(m: str) -> str:
    if m in COLORS:
        return COLORS[m]
    return COLORS["untrained"] if m in ("brut_acp", "lda") else COLORS["classical"]


def scores_figure(summary: pd.DataFrame) -> None:
    """MAP@R de toutes les méthodes, en trois familles, du plus ancien au plus récent."""
    mapr = summary[summary.metric == "map_at_r"].set_index("method")
    groups = [("before deep learning", ["eigenfaces", "lbp", "fisherfaces"]),
              ("pretrained,\nno face training", ["brut_acp", "lda"]),
              ("trained on faces", LOSSES)]
    fig, ax = plt.subplots(figsize=(390 * MM, 215 * MM))
    x, ticks, labels = 0.0, [], []
    for title, methods in groups:
        start = x
        for m in methods:
            v = mapr.loc[m, "mean"]
            ax.bar(x, v, width=0.78, color=color_of(m), zorder=2)
            lo, hi = mapr.loc[m, "ci_low"], mapr.loc[m, "ci_high"]
            if np.isfinite(lo):
                ax.plot([x, x], [lo, hi], color=INK, lw=1.6, zorder=3)
            ax.text(x, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=26,
                    fontweight="bold" if m in LOSSES else "normal")
            ticks.append(x)
            year = YEARS.get(m)
            labels.append(NAMES[m] + (f"\n({year})" if year else ""))
            x += 1
        ax.text((start + x - 1) / 2, -13.5, title, ha="center", va="top", fontsize=19,
                color=MUTED, style="italic")
        x += 0.6
    ax.set_xticks(ticks, labels, fontsize=15)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 84)
    ax.set_ylabel("MAP@R (%)", fontsize=22)
    ax.yaxis.grid(True, color="#E6E6E6", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "scores.png")


def hills_figure(scores, summary: pd.DataFrame) -> None:
    """Similarités des paires « même personne » et « personnes différentes »."""
    y = scores["y"]
    mapr = summary[summary.metric == "map_at_r"].set_index("method")["mean"]
    rng = np.random.default_rng(0)
    panels = [("brut_acp", "Pretrained, not trained on faces"), ("arcface", "Trained with ArcFace")]
    fig, axes = plt.subplots(1, 2, figsize=(390 * MM, 150 * MM), sharey=True)
    bins = np.linspace(-0.6, 1.0, 81)
    for ax, (m, title) in zip(axes, panels):
        s = scores[m].astype(np.float32)
        same, diff = s[y == 1], s[y == 0]
        diff = rng.choice(diff, size=min(len(diff), 200_000), replace=False)
        hs, _ = np.histogram(same, bins, density=True)
        hd, _ = np.histogram(diff, bins, density=True)
        c = (bins[:-1] + bins[1:]) / 2
        ax.fill_between(c, hd, color="#C8C8C8", lw=0, label="different people")
        ax.fill_between(c, hs, color=color_of(m), alpha=0.75, lw=0, label="same person")
        ax.fill_between(c, np.minimum(hs, hd), facecolor="none", edgecolor=INK, hatch="///",
                        lw=0, label="overlap = errors")
        ax.set_ylim(0, max(hs.max(), hd.max()) * 1.75)
        ax.legend(frameon=False, fontsize=17, loc="upper left", handlelength=1.4)
        ax.set_title(f"{title}\nMAP@R {mapr[m]:.1f}", fontsize=21, loc="left")
        ax.set_xlabel("similarity of two photos", fontsize=20)
        ax.set_xticks([-0.5, 0, 0.5, 1])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "hills.png")


def plane_figure(summary: pd.DataFrame) -> None:
    """Alignement contre uniformité (Wang et Isola, 2020) : plus bas = mieux, sur les deux axes."""
    t = summary[summary.metric.isin(["alignment", "uniformity"])].pivot(
        index="method", columns="metric", values="mean")
    fig, ax = plt.subplots(figsize=(390 * MM, 215 * MM))
    offsets = {"arcface": (-22, -34), "nsoftmax": (18, 10), "contrastive": (20, -6),
               "triplet": (20, -6), "brut_acp": (-20, 14)}
    for m in ["brut_acp", *LOSSES]:
        a, u = t.loc[m, "alignment"], t.loc[m, "uniformity"]
        big = m in LOSSES
        ax.scatter(a, u, s=900 if big else 450, color=color_of(m), zorder=3,
                   edgecolor="white", linewidth=2)
        name = NAMES[m].replace("-\n", "").replace("\n", " ")
        dx, dy = offsets[m]
        ax.annotate(name, (a, u), xytext=(dx, dy), textcoords="offset points", fontsize=24,
                    ha="right" if dx < 0 else "left", color=INK)
    ax.set_xlim(0.2, 1.6)
    ax.set_ylim(-4.2, -1.8)
    ax.set_xlabel("alignment: distance within a person (lower = tighter groups)", fontsize=20)
    ax.set_ylabel("uniformity\n(lower = more spread out)", fontsize=20)
    ax.grid(True, color="#EEEEEE")
    save(fig, "plane.png")


def read_faces(zip_path: Path, names: list[str]) -> dict[str, np.ndarray]:
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for n in set(names):
            img = Image.open(io.BytesIO(z.read(f"img_align_celeba/{n}"))).convert("RGB")
            out[n] = np.asarray(img.crop(BOX))
    return out


def neighbors_figure(emb, names: list[str], zip_path: Path, n_queries: int, k: int,
                     seed: int, show: int) -> dict[str, str]:
    """Une requête au hasard (même règle que l'étape 6) et ses k plus proches voisins."""
    labels = emb["labels"]
    queries = np.random.default_rng(seed).choice(len(labels), n_queries, replace=False)
    rows, counts = [], {}
    for m in ["brut_acp", *LOSSES]:
        x = emb[m].astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        hits = 0
        for qi, q in enumerate(queries):
            sim = x @ x[q]
            sim[q] = -np.inf
            nn = np.argsort(-sim)[:k]
            hits += int((labels[nn] == labels[q]).sum())
            if qi == show:
                rows.append((m, q, nn))
        counts[m] = f"{hits}/{n_queries * k}"
    faces = read_faces(zip_path, [names[i] for _, q, nn in rows for i in [q, *nn]])
    fig, axes = plt.subplots(len(rows), k + 2, figsize=(390 * MM, 290 * MM),
                             gridspec_kw={"width_ratios": [1.9, 1, *[1] * k]})
    for r, (m, q, nn) in enumerate(rows):
        ax = axes[r, 0]
        ax.text(1.0, 0.6, NAMES[m].replace("-\n", "").replace("\n", " "), ha="right", va="center", fontsize=22,
                color=color_of(m), fontweight="bold", transform=ax.transAxes)
        ax.text(1.0, 0.32, f"{counts[m]} correct", ha="right", va="center", fontsize=18,
                color=MUTED, transform=ax.transAxes)
        ax.axis("off")
        cells = [(q, INK)] + [(j, GOOD if labels[j] == labels[q] else BAD) for j in nn]
        for c, (i, col) in enumerate(cells, start=1):
            ax = axes[r, c]
            ax.imshow(faces[names[i]])
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_edgecolor(col)
                side.set_linewidth(5 if c > 1 else 3)
        if r == 0:
            axes[r, 1].set_title("query", fontsize=19, color=MUTED)
            axes[r, 3].set_title(f"its {k} nearest photos", fontsize=19, color=MUTED, loc="left")
    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    save(fig, "neighbors.png")
    Image.fromarray(faces[names[rows[0][1]]]).save(OUT / "face_query.png")
    return counts


def tsne_figure(emb, n_ids: int, seed: int, perplexity: float, pca_dim: int) -> None:
    """Avant / après : 20 personnes jamais vues, une couleur par personne."""
    labels = emb["labels"]
    chosen = np.random.default_rng(seed).choice(np.unique(labels), size=n_ids, replace=False)
    keep = np.isin(labels, chosen)
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(260 * MM, 120 * MM))
    for ax, (m, title) in zip(axes, [("brut_acp", "Before training on faces"),
                                     ("arcface", "After: trained with ArcFace")]):
        x = emb[m][keep].astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x = PCA(pca_dim, random_state=seed).fit_transform(x)
        xy = TSNE(2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(x)
        _, idx = np.unique(labels[keep], return_inverse=True)
        ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(i % 20) for i in idx], s=40,
                   edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=19, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    save(fig, "tsne.png")


def sphere_panel(ax, pts, cols, title: str, yaw=0.5, pitch=-0.35) -> None:
    """Sphère éclairée vue de face ; points de la face visible seulement."""
    n = 600
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, -g)
    inside = X ** 2 + Y ** 2 <= 1
    Z = np.sqrt(np.clip(1 - X ** 2 - Y ** 2, 0, None))
    light = np.array([-0.45, 0.55, 0.7])
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.clip(X * light[0] + Y * light[1] + Z * light[2], 0, 1)
    base = np.array([0.93, 0.94, 0.97])
    img = np.ones((n, n, 4))
    img[..., :3] = base * shade[..., None]
    img[..., 3] = inside
    ax.imshow(img, extent=(-1, 1, -1, 1), zorder=1)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#B8BCC8", lw=1.5, zorder=2))
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    x = pts[:, 0] * cy - pts[:, 2] * sy
    z = pts[:, 0] * sy + pts[:, 2] * cy
    y = pts[:, 1] * cp - z * sp
    z2 = pts[:, 1] * sp + z * cp
    order = np.argsort(z2)
    for i in order:
        if z2[i] < -0.05:
            continue
        ax.scatter(x[i], y[i], s=60 + 140 * z2[i], color=cols[i], edgecolor="white",
                   linewidth=0.8, zorder=3, alpha=0.45 + 0.55 * max(z2[i], 0))
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=28, pad=10)


def sphere_figure() -> None:
    """Schéma, pas des données : 8 personnes, 14 photos chacune, sur une sphère en 3D."""
    rng = np.random.default_rng(11)

    def nm(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    cube = nm(np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float))
    tight = nm(cube * 0.38 + np.array([0, 1, 0]))
    pal = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7F3C8D", "#8C6D31"]
    k, p = 8, 14
    cols = [pal[i] for i in range(k) for _ in range(p)]
    mixed = nm(rng.normal(size=(k * p, 3)))
    crowded = nm(np.repeat(tight, p, 0) + 0.06 * rng.normal(size=(k * p, 3)))
    spread = nm(np.repeat(cube, p, 0) + 0.2 * rng.normal(size=(k * p, 3)))
    fig, axes = plt.subplots(1, 3, figsize=(520 * MM, 165 * MM))
    views = [(0.5, -0.35), (0.3, 1.05), (0.5, -0.35)]
    for ax, pts, title, (yaw, pitch) in zip(axes, [mixed, crowded, spread],
                                            ["Untrained: people mixed", "Tight but crowded",
                                             "Spread over the sphere"], views):
        sphere_panel(ax, pts, cols, title, yaw, pitch)
    save(fig, "sphere.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--celeba-zip", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    args = p.parse_args()
    ev = Path(args.eval_dir)
    summary = pd.read_csv(ev / "summary.csv")
    emb = np.load(ev / "embeddings_seed0.npz")
    names = pd.read_csv(ROOT / "splits" / f"{args.split}.csv")
    assert (names["identity"].to_numpy() == emb["labels"]).all(), "ordre des vecteurs ≠ partition"
    names = names["image"].tolist()

    scores_figure(summary)
    hills_figure(np.load(ev / "scores_seed0.npz"), summary)
    plane_figure(summary)
    sphere_figure()
    tsne_figure(emb, n_ids=20, seed=0, perplexity=30, pca_dim=50)
    counts = neighbors_figure(emb, names, Path(args.celeba_zip), n_queries=10, k=5, seed=0,
                              show=2)
    print("bons voisins sur les 10 requêtes :", counts)


if __name__ == "__main__":
    main()
