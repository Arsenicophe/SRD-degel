"""Mesures de base : vecteurs du ResNet-18 de départ, sans entraînement de perte.

Trois versions, comme dans l'ancien protocole (section 5) :

- ``brut``     : vecteurs de 512 tels quels ;
- ``brut_acp`` : réduits à 128 par ACP ajustée sur l'entraînement (ligne
  « Pretrained » de Musgrave) ;
- ``lda``      : LDA à 128 ajustée sur les identités d'entraînement.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from facegeom.data.augment import GpuAugment
from facegeom.eval.baselines import apply_projection, fit_lda, fit_pca
from facegeom.eval.retrieval import l2_normalize
from facegeom.models.network import autocast_dtype, embed_images, load_backbone

REFERENCES = ("brut", "brut_acp", "lda")


def eval_augment(dcfg: dict, tcfg: dict, device: str) -> GpuAugment:
    """Préparation des images (seule la transformation d'évaluation sert ici)."""
    a = tcfg["augment"]
    return GpuAugment(dcfg["input_size"], dcfg["normalize"]["mean"], dcfg["normalize"]["std"],
                      a["scale"], a["ratio"], a["flip_p"], device)


def base_features(tcfg: dict, dcfg: dict, images: np.ndarray, device: str,
                  log: Callable[[str], None] | None = print) -> np.ndarray:
    """Vecteurs de 512 du réseau de départ (poids ImageNet), transformation d'évaluation."""
    model = load_backbone(tcfg["backbone"], pretrained=True).to(device).eval()
    return embed_images(model, images, eval_augment(dcfg, tcfg, device),
                        int(tcfg["eval_batch_size"]), autocast_dtype(device, tcfg.get("amp")),
                        log=log)


def cached_base_features(path: Path, tcfg: dict, dcfg: dict, images: np.ndarray,
                         device: str, log: Callable[[str], None] = print) -> np.ndarray:
    """Vecteurs de base lus dans ``path`` s'ils y sont déjà, sinon calculés et écrits (float16)."""
    path = Path(path)
    if path.exists():
        x = np.load(path)
        if x.shape[0] == len(images):
            log(f"{path.name} : déjà calculé")
            return x.astype(np.float32)
    x = base_features(tcfg, dcfg, images, device, log=log)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, x.astype(np.float16))
    tmp.replace(path)
    log(f"{path.name} : {x.shape[0]} vecteurs calculés")
    return x


def reference_projections(train: np.ndarray, train_labels: np.ndarray,
                          dim: int) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Fonctions vecteurs de base → vecteurs évalués, ajustées sur l'entraînement."""
    pca = fit_pca(train, dim)
    lda = fit_lda(train, train_labels, dim)
    return {
        "brut": lambda x: l2_normalize(x),
        "brut_acp": lambda x: apply_projection(pca, x),
        "lda": lambda x: apply_projection(lda, x),
    }
