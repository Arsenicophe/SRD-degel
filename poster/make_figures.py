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
    """Similarités des paires « même personne » et « personnes différentes », avant et après IA."""
    y = scores["y"]
    val = summary.set_index(["method", "metric"])["mean"]
    rng = np.random.default_rng(0)
    panels = [("fisherfaces", "Before AI: Fisherfaces"), ("contrastive", "Contrastive"),
              ("arcface", "ArcFace")]
    fig, axes = plt.subplots(1, 3, figsize=(390 * MM, 150 * MM), sharey=False)
    bins = np.linspace(-0.6, 1.0, 81)
    for i, (ax, (m, title)) in enumerate(zip(axes, panels)):
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
        ax.set_ylim(0, max(hs.max(), hd.max()) * (1.9 if i == 0 else 1.15))
        if i == 0:
            ax.legend(frameon=False, fontsize=15, loc="upper left", handlelength=1.3)
        tar = val[(m, "tar_at_far_0.001")]
        ax.set_title(f"{title}\nTAR {tar:.1f}%", fontsize=19, loc="left", color=color_of(m)
                     if m in LOSSES else INK)
        ax.set_xlabel("similarity of two photos", fontsize=17)
        ax.set_xticks([-0.5, 0, 0.5, 1], ["−0.5", "0", "0.5", "1"], fontsize=15)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "hills.png")


def plane_figure(summary: pd.DataFrame) -> None:
    """Alignement contre uniformité (Wang et Isola, 2020), toutes les méthodes à cosinus."""
    t = summary[summary.metric.isin(["alignment", "uniformity"])].pivot(
        index="method", columns="metric", values="mean")
    fig, ax = plt.subplots(figsize=(480 * MM, 300 * MM))
    labels = {"eigenfaces": ("Eigenfaces (1991)", (22, 0), "left"),
              "fisherfaces": ("Fisherfaces (1997) and ImageNet + LDA", (0, -42), "center"),
              "brut_acp": ("ImageNet + PCA", (-24, 0), "right"),
              "triplet": ("Triplet", (26, 0), "left"),
              "contrastive": ("Contrastive", (26, 0), "left"),
              "nsoftmax": ("Norm. softmax", (24, 18), "left"),
              "arcface": ("ArcFace", (-24, -26), "right")}
    for m in ["eigenfaces", "fisherfaces", "lda", "brut_acp", *LOSSES]:
        a, u = t.loc[m, "alignment"], t.loc[m, "uniformity"]
        ax.scatter(a, u, s=1300 if m in LOSSES else 700, color=color_of(m), zorder=3,
                   edgecolor="white", linewidth=2.5)
        if m in labels:
            name, (dx, dy), ha = labels[m]
            ax.annotate(name, (a, u), xytext=(dx, dy), textcoords="offset points", fontsize=25,
                        ha=ha, va="center", color=color_of(m) if m in LOSSES else INK,
                        fontweight="bold" if m in LOSSES else "normal")
    ax.annotate("", xy=(1.05, -3.35), xytext=(1.5, -3.35),
                arrowprops={"arrowstyle": "-|>,head_width=0.5,head_length=0.9", "lw": 3,
                            "color": MUTED})
    ax.text(1.275, -3.29, "training on faces", ha="center", va="bottom", fontsize=21,
            color=MUTED, style="italic")
    ax.text(0.24, -4.08, "ideal corner: each person tight,\ndifferent people far apart",
            ha="left", va="bottom", fontsize=19, color=MUTED, style="italic")
    ax.text(1.80, -2.06, "before AI or\nno face training", ha="right", va="top", fontsize=19,
            color=COLORS["classical"], style="italic")
    ax.set_xlim(0.2, 1.85)
    ax.set_ylim(-4.15, -2.0)
    ax.set_xlabel("alignment: distance between photos of the same person   (lower = tighter)",
                  fontsize=21)
    ax.set_ylabel("uniformity   (lower = people more spread out)", fontsize=21)
    ax.tick_params(labelsize=18)
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
    """Avant IA, réseau non entraîné, après IA : 20 personnes jamais vues, une couleur chacune."""
    labels = emb["labels"]
    chosen = np.random.default_rng(seed).choice(np.unique(labels), size=n_ids, replace=False)
    keep = np.isin(labels, chosen)
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 3, figsize=(390 * MM, 140 * MM))
    for ax, (m, title) in zip(axes, [("fisherfaces", "Before AI: Fisherfaces"),
                                     ("brut_acp", "ImageNet, no face training"),
                                     ("arcface", "Trained with ArcFace")]):
        x = emb[m][keep].astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x = PCA(pca_dim, random_state=seed).fit_transform(x)
        xy = TSNE(2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(x)
        _, idx = np.unique(labels[keep], return_inverse=True)
        ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(i % 20) for i in idx], s=40,
                   edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=17, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    save(fig, "tsne.png")


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
    tsne_figure(emb, n_ids=20, seed=0, perplexity=30, pca_dim=50)
    counts = neighbors_figure(emb, names, Path(args.celeba_zip), n_queries=10, k=5, seed=0,
                              show=2)
    print("bons voisins sur les 10 requêtes :", counts)


if __name__ == "__main__":
    main()
