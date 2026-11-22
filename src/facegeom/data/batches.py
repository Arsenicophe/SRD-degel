"""Lots d'entraînement : composition, ordre, lecture dans le cache.

Chaque lot contient ``classes_per_batch`` personnes × ``images_per_class``
images (64 × 4 = 256), tirées par ``MPerClassSampler`` de
pytorch-metric-learning, comme dans l'ancien protocole.

Les lots d'une époque ne dépendent que de la graine et du numéro de l'époque :
un entraînement repris après une coupure voit exactement les mêmes lots qu'un
entraînement sans coupure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from pytorch_metric_learning import samplers
from torch.utils.data import DataLoader, Dataset


def derived_seed(*keys: int) -> int:
    """Graine dérivée de plusieurs entiers (graine du run, époque, usage)."""
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(1)[0])


def epoch_batches(
    labels: np.ndarray,
    images_per_class: int,
    batch_size: int,
    num_batches: int,
    seed: int,
    epoch: int,
) -> list[np.ndarray]:
    """Lots (tableaux d'indices) de l'époque ``epoch`` pour la graine ``seed``."""
    gen = torch.Generator().manual_seed(derived_seed(seed, epoch, 0))
    sampler = samplers.MPerClassSampler(
        labels, images_per_class, batch_size, num_batches=num_batches, generator=gen
    )
    return [np.asarray(b, dtype=np.int64) for b in sampler]


def augment_generator(seed: int, epoch: int) -> torch.Generator:
    """Générateur des tirages d'augmentation de l'époque ``epoch``."""
    return torch.Generator().manual_seed(derived_seed(seed, epoch, 1))


class CachedBatches(Dataset):
    """Lots lus dans le cache d'images (memmap), un lot par élément.

    Les indices d'un lot sont triés avant la lecture (accès disque plus
    régulier) ; l'ordre dans le lot n'a pas d'effet sur les pertes. Le cache est
    ouvert à la demande, une fois par processus de lecture.
    """

    def __init__(self, cache_path: str, batches: Sequence[np.ndarray], labels: np.ndarray):
        self.cache_path = str(cache_path)
        self.batches = list(batches)
        self.labels = np.asarray(labels, dtype=np.int64)
        self._images: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._images is None:
            self._images = np.load(self.cache_path, mmap_mode="r")
        idx = np.sort(self.batches[i])
        images = torch.from_numpy(np.ascontiguousarray(self._images[idx]))
        return images, torch.from_numpy(self.labels[idx])


def batch_loader(dataset: CachedBatches, num_workers: int, pin_memory: bool) -> DataLoader:
    """Chargeur qui lit les lots en avance, sur ``num_workers`` processus."""
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=False,
    )
