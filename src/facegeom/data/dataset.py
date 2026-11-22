"""Parties de CelebA telles qu'utilisées par les scripts : noms, identités, cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from facegeom.data.cache import cache_file, open_cache
from facegeom.data.crop import crop_box
from facegeom.train import TrainData

SPLITS = ("train", "val", "test")


def read_split(splits_dir: str | Path, split: str) -> tuple[list[str], np.ndarray]:
    """Noms d'images et identités d'une partie, dans l'ordre de ``splits/<partie>.csv``."""
    df = pd.read_csv(Path(splits_dir) / f"{split}.csv")
    return df["image"].tolist(), df["identity"].to_numpy()


def cache_box(dcfg: dict) -> tuple[int, int, int, int]:
    """Boîte du carré recadré, calculée comme dans l'ancien dépôt et vérifiée."""
    box = crop_box(dcfg["crop"])
    expected = tuple(dcfg["crop"]["box"])
    if box != expected:
        raise ValueError(f"Boîte de recadrage {box}, attendu {expected} (configs/data.yaml).")
    return box


def load_images(dcfg: dict, cache_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Cache d'une partie (memmap en lecture seule) et identités alignées."""
    names, ids = read_split(dcfg["splits"]["output_dir"], split)
    images = open_cache(cache_file(cache_dir, split, dcfg["crop"]["size"]), names)
    return images, ids


def load_train_data(dcfg: dict, cache_dir: str | Path) -> TrainData:
    """Données d'entraînement : cache d'entraînement et de validation, jamais le test."""
    names, ids = read_split(dcfg["splits"]["output_dir"], "train")
    path = cache_file(cache_dir, "train", dcfg["crop"]["size"])
    open_cache(path, names)  # vérifie que le cache est complet et correspond à la partition
    _, labels = np.unique(ids, return_inverse=True)
    val_images, val_ids = load_images(dcfg, cache_dir, "val")
    return TrainData(train_cache=str(path), train_labels=labels.astype(np.int64),
                     val_images=val_images, val_labels=val_ids)
