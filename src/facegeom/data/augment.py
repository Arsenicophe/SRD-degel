"""Augmentations et redimensionnement sur la carte graphique, par lots.

Les images arrivent du cache en octets, de forme (lot, 128, 128, 3). Tout le
reste se fait sur la carte graphique, en un seul appel par lot :

- entraînement : recadrage aléatoire redimensionné (la méthode de
  ``RandomResizedCrop`` de torchvision, avec les réglages de Musgrave) et
  symétrie horizontale, puis passage en 224 × 224 ;
- évaluation : simple redimensionnement du carré en 224 × 224, sans hasard.

Les tirages (taille, forme et position des recadrages, symétries) sont faits sur
le processeur avec un générateur dédié : ils ne dépendent que de la graine, pas
de la carte graphique.

Les recadrages sont continus (pas d'arrondi au pixel) : un recadrage est une
transformation affine de l'image, appliquée avec ``grid_sample`` en
interpolation bilinéaire.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def sample_crop_boxes(
    n: int,
    generator: torch.Generator,
    scale: Sequence[float],
    ratio: Sequence[float],
    attempts: int = 10,
) -> torch.Tensor:
    """Tire ``n`` recadrages dans une image carrée.

    Même règle que ``RandomResizedCrop`` : surface tirée uniformément dans
    ``scale`` (fraction de la surface de l'image), rapport largeur / hauteur tiré
    uniformément en échelle logarithmique dans ``ratio`` ; jusqu'à ``attempts``
    essais pour trouver un recadrage qui tient dans l'image, sinon l'image
    entière. La position est ensuite uniforme parmi les positions possibles.

    Retourne un tenseur (n, 4) sur le processeur : x0, y0, largeur, hauteur, en
    fraction du côté de l'image (entre 0 et 1).
    """
    area = torch.empty(n, attempts).uniform_(scale[0], scale[1], generator=generator)
    log_ratio = torch.empty(n, attempts).uniform_(
        math.log(ratio[0]), math.log(ratio[1]), generator=generator
    )
    aspect = torch.exp(log_ratio)
    w = torch.sqrt(area * aspect)
    h = torch.sqrt(area / aspect)
    fits = (w <= 1.0) & (h <= 1.0)
    first = fits.int().argmax(dim=1)  # premier essai qui tient (0 si aucun)
    rows = torch.arange(n)
    found = fits.any(dim=1)
    w = torch.where(found, w[rows, first], torch.ones(n))
    h = torch.where(found, h[rows, first], torch.ones(n))
    x0 = torch.rand(n, generator=generator) * (1.0 - w)
    y0 = torch.rand(n, generator=generator) * (1.0 - h)
    return torch.stack([x0, y0, w, h], dim=1)


def crop_theta(boxes: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
    """Matrices affines (n, 2, 3) qui envoient la sortie sur le recadrage.

    Convention de ``F.affine_grid`` avec ``align_corners=False`` : les bords de
    la sortie (-1 et 1) tombent sur les bords du recadrage. ``flip`` (booléens)
    retourne le recadrage de gauche à droite.
    """
    x0, y0, w, h = boxes.unbind(dim=1)
    sign = 1.0 - 2.0 * flip.to(boxes.dtype)
    theta = torch.zeros(boxes.shape[0], 2, 3, dtype=boxes.dtype)
    theta[:, 0, 0] = w * sign
    theta[:, 0, 2] = 2.0 * x0 + w - 1.0
    theta[:, 1, 1] = h
    theta[:, 1, 2] = 2.0 * y0 + h - 1.0
    return theta


def to_float(images: torch.Tensor) -> torch.Tensor:
    """(lot, H, L, 3) en octets vers (lot, 3, H, L) en flottants entre 0 et 1."""
    return images.permute(0, 3, 1, 2).float().div_(255.0)


class GpuAugment:
    """Prépare les lots d'images pour le réseau, sur l'appareil de calcul.

    Paramètres
    ----------
    out_size : côté de l'image donnée au réseau (224).
    mean, std : normalisation ImageNet, une valeur par canal.
    scale, ratio, flip_p : réglages des augmentations d'entraînement.
    device : appareil de calcul (``cuda``, ``mps`` ou ``cpu``).
    """

    def __init__(
        self,
        out_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        scale: Sequence[float],
        ratio: Sequence[float],
        flip_p: float,
        device: torch.device | str,
    ) -> None:
        self.out_size = int(out_size)
        self.scale = tuple(float(v) for v in scale)
        self.ratio = tuple(float(v) for v in ratio)
        self.flip_p = float(flip_p)
        self.device = torch.device(device)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def train(self, images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Recadrage aléatoire, symétrie, passage en ``out_size``, normalisation."""
        n = images.shape[0]
        boxes = sample_crop_boxes(n, generator, self.scale, self.ratio)
        flip = torch.rand(n, generator=generator) < self.flip_p
        theta = crop_theta(boxes, flip).to(self.device)
        x = to_float(images.to(self.device, non_blocking=True))
        grid = F.affine_grid(theta, [n, 3, self.out_size, self.out_size], align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return self._normalize(x)

    def eval(self, images: torch.Tensor) -> torch.Tensor:
        """Redimensionnement du carré entier en ``out_size``, normalisation."""
        x = to_float(images.to(self.device, non_blocking=True))
        x = F.interpolate(
            x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
        )
        return self._normalize(x)
