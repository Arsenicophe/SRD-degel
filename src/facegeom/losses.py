"""Les quatre fonctions de coût, via pytorch-metric-learning (version 3, marges en radians).

``nsoftmax`` est la softmax normalisée, c'est-à-dire ArcFace avec une marge
nulle (vérifié par un test). ``s`` est l'échelle : la température de
``NormalizedSoftmaxLoss`` vaut ``1 / s``.
"""

from __future__ import annotations

import torch
from pytorch_metric_learning import losses, miners

LOSSES = ("nsoftmax", "arcface", "contrastive", "triplet")
# Pertes qui ont leur propre matrice de poids (une ligne par personne d'entraînement)
WEIGHTED = ("nsoftmax", "arcface")


def make_loss(name: str, params: dict, n_classes: int, dim: int):
    """Perte et mineur éventuel (triplets semi-difficiles pour ``triplet``)."""
    if name == "nsoftmax":
        return losses.NormalizedSoftmaxLoss(n_classes, dim, temperature=1 / params["s"]), None
    if name == "arcface":
        return losses.ArcFaceLoss(n_classes, dim, margin=params["m"], scale=params["s"]), None
    if name == "contrastive":
        return losses.ContrastiveLoss(pos_margin=params["pos_margin"],
                                      neg_margin=params["neg_margin"]), None
    if name == "triplet":
        return (losses.TripletMarginLoss(margin=params["margin"]),
                miners.TripletMarginMiner(margin=params["margin"], type_of_triplets="semihard"))
    raise ValueError(f"Perte inconnue : {name}")


def compute_loss(loss_fn, miner, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Valeur de la perte sur un lot ; le mineur choisit les triplets s'il y en a un."""
    if miner is not None:
        return loss_fn(embeddings, labels, miner(embeddings, labels))
    return loss_fn(embeddings, labels)


def set_arcface_margin(loss_fn, margin: float) -> None:
    """Change la marge d'ArcFace en cours d'entraînement (montée progressive)."""
    loss_fn.margin = float(margin)


def margin_at(step: int, steps_per_epoch: int, warmup_epochs: float, margin: float) -> float:
    """Marge d'ArcFace au pas ``step`` : montée linéaire de 0 à ``margin``."""
    if warmup_epochs <= 0:
        return float(margin)
    return float(margin) * min(1.0, step / (warmup_epochs * steps_per_epoch))
