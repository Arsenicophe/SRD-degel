"""Réseau entraîné : ResNet-18 ImageNet, tête linéaire 512 → 128, BatchNorm gelés.

Poids téléchargés directement depuis leur source officielle et vérifiés par
SHA-256 (pas de ``torch.hub.load``, qui exécuterait du code distant).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tvm

from facegeom.data.augment import GpuAugment
from facegeom.utils import file_hash

# Clés absentes tolérées : la couche fc (retirée) et les compteurs BatchNorm
# (absents des checkpoints V1, inutiles puisque les BatchNorm sont gelés).
ALLOWED_MISSING = {"fc.weight", "fc.bias"}
ALLOWED_MISSING_SUFFIX = ".num_batches_tracked"
ARCHS = {"resnet18": tvm.resnet18, "resnet50": tvm.resnet50}


class WeightsError(RuntimeError):
    """Fichier de poids inattendu (empreinte ou clés)."""


def download_verified(url: str, sha256: str) -> Path:
    """Télécharge (ou reprend du cache torch) un checkpoint et vérifie son SHA-256."""
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    path = cache / f"{sha256[:12]}_{Path(url).name}"
    usual = cache / Path(url).name  # nom utilisé par torchvision, déjà présent sur certains postes
    if not path.exists() and usual.exists() and file_hash(usual, "sha256") == sha256:
        path = usual
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path), progress=False)
    got = file_hash(path, "sha256")
    if got != sha256:
        path.unlink()
        raise WeightsError(f"{url} : SHA-256 {got}, attendu {sha256}.")
    return path


def load_backbone(bcfg: dict, pretrained: bool = True) -> nn.Module:
    """ResNet sans sa couche de classification : sortie = vecteur après la moyenne globale.

    ``bcfg`` est la section ``backbone`` de ``configs/train.yaml`` (``arch``,
    ``url``, ``sha256``). ``pretrained=False`` sert aux tests (poids au hasard).
    """
    model = ARCHS[bcfg["arch"]](weights=None)
    if pretrained:
        state = torch.load(download_verified(bcfg["url"], bcfg["sha256"]),
                           map_location="cpu", weights_only=True)
        expected = set(model.state_dict())
        missing = expected - set(state)
        unexpected = set(state) - expected
        bad = [k for k in missing
               if k not in ALLOWED_MISSING and not k.endswith(ALLOWED_MISSING_SUFFIX)]
        if unexpected or bad:
            raise WeightsError(
                f"Clés inattendues : {sorted(unexpected)[:5]} ; manquantes : {sorted(bad)[:5]}"
            )
        model.load_state_dict(state, strict=False)
    model.fc = nn.Identity()
    return model


def freeze_batchnorm(model: nn.Module) -> None:
    """BatchNorm figés : statistiques d'ImageNet (mode évaluation) et paramètres sans gradient."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


class EmbeddingNet(nn.Module):
    """Réseau + tête linéaire + normalisation L2.

    ``train()`` laisse toujours les BatchNorm en mode évaluation.
    """

    def __init__(self, backbone: nn.Module, in_dim: int, out_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(in_dim, out_dim)
        freeze_batchnorm(self.backbone)

    def train(self, mode: bool = True) -> EmbeddingNet:
        super().train(mode)
        freeze_batchnorm(self.backbone)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.backbone(x)), dim=1)


def autocast_dtype(device: torch.device | str, amp: str | bool) -> torch.dtype | None:
    """Type de calcul de la précision mixte, ou ``None`` pour du float32 partout.

    ``amp`` : ``auto`` (sur CUDA : bfloat16 si la carte le gère nativement,
    capacité ≥ 8, sinon float16 avec mise à l'échelle des gradients ; ailleurs :
    float32), ``bf16``, ``fp16`` ou ``false``. Sur le GPU
    d'un Mac (``mps``), seul ``fp16`` active la précision mixte (essais locaux).
    """
    device = torch.device(device)
    if amp in (False, "false", "off", None):
        return None
    if device.type == "mps":
        return torch.float16 if amp == "fp16" else None
    if device.type != "cuda":
        return None
    if amp == "bf16" or (amp == "auto" and torch.cuda.get_device_capability(device)[0] >= 8):
        return torch.bfloat16  # A100, L4… : bfloat16 natif ; T4 (7.5) : float16 + GradScaler
    return torch.float16


@torch.no_grad()
def embed_images(
    model: nn.Module,
    images: np.ndarray,
    augment: GpuAugment,
    batch_size: int,
    amp_dtype: torch.dtype | None = None,
    channels_last: bool = False,
    normalize: bool = False,
    log: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Vecteurs (float32) de toutes les ``images`` du cache, transformation d'évaluation.

    ``model`` est passé en mode évaluation le temps du calcul, puis remis dans
    son mode d'origine. ``log`` : messages d'avancement (environ tous les 10 %).
    """
    was_training = model.training
    model.eval()
    device = augment.device
    out = []
    n_batches = (len(images) + batch_size - 1) // batch_size
    every = max(1, n_batches // 10)
    t0 = time.perf_counter()
    for b, lo in enumerate(range(0, len(images), batch_size), start=1):
        batch = torch.from_numpy(np.array(images[lo : lo + batch_size]))  # copie : memmap en lecture seule
        x = augment.eval(batch)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            z = model(x)
        z = z.float()
        if normalize:
            z = F.normalize(z, dim=1)
        out.append(z.cpu().numpy())
        if log and (b % every == 0 or b == n_batches):
            done = min(lo + batch_size, len(images))
            log(f"   {done}/{len(images)} images, {done / (time.perf_counter() - t0):.0f} images/s")
    model.train(was_training)
    result = np.concatenate(out) if out else np.empty((0, 0), dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("vecteurs non finis en sortie du réseau")
    if len(result) and (np.abs(result).max(axis=1) == 0).any():
        raise FloatingPointError("vecteurs nuls en sortie du réseau (réseau dégénéré)")
    return result
