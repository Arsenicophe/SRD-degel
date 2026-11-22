"""Cache des images recadrées : chaque carré est stocké une fois, en octets.

Disposition dans ``<cache_dir>`` :

- ``<partie>_<côté>.npy`` : tableau (N, côté, côté, 3) en ``uint8``, lignes dans
  l'ordre exact de ``splits/<partie>.csv`` ;
- ``<partie>_<côté>.progress`` : morceaux déjà écrits (reprise après coupure) ;
- ``<partie>_<côté>.done.json`` : marqueur de fin, avec le nombre d'images et une
  empreinte de la liste des noms.

Les images sont lues dans le dossier décompressé de CelebA ou directement dans
le zip, décodées et recadrées en parallèle sur plusieurs processus.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from facegeom.data.crop import crop_face
from facegeom.data.images import ZIP_PREFIX

_ZIP_HANDLES: dict[str, zipfile.ZipFile] = {}


def cache_file(cache_dir: str | Path, split: str, size: int) -> Path:
    return Path(cache_dir) / f"{split}_{size}.npy"


def names_digest(names: Sequence[str]) -> str:
    """Empreinte de la liste ordonnée des noms d'images."""
    return hashlib.sha1("\n".join(names).encode()).hexdigest()


def _read_rgb(source: str, name: str) -> Image.Image:
    """Image ``name`` en RVB, depuis un dossier ou un zip (une poignée par processus)."""
    if source.endswith(".zip"):
        zf = _ZIP_HANDLES.get(source)
        if zf is None:
            zf = _ZIP_HANDLES[source] = zipfile.ZipFile(source)
        with zf.open(ZIP_PREFIX + name) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")
    with Image.open(Path(source) / name) as img:
        return img.convert("RGB")


def load_crops(source: str, names: Sequence[str], box: tuple[int, int, int, int]) -> np.ndarray:
    """Carrés recadrés de ``names``, en tableau (n, côté, côté, 3) ``uint8``."""
    size = box[2] - box[0]
    out = np.empty((len(names), size, box[3] - box[1], 3), dtype=np.uint8)
    for i, name in enumerate(names):
        out[i] = np.asarray(crop_face(_read_rgb(source, name), box), dtype=np.uint8)
    return out


def _done_path(path: Path) -> Path:
    return path.with_name(path.stem + ".done.json")


def _progress_path(path: Path) -> Path:
    return path.with_name(path.stem + ".progress")


def is_complete(path: Path, names: Sequence[str]) -> bool:
    """Vrai si le cache ``path`` est terminé et correspond exactement à ``names``."""
    done = _done_path(path)
    if not (path.exists() and done.exists()):
        return False
    info = json.loads(done.read_text())
    return info.get("n_images") == len(names) and info.get("names_sha1") == names_digest(names)


def build_cache(
    names: Sequence[str],
    source: str | Path,
    box: tuple[int, int, int, int],
    path: str | Path,
    chunk_size: int = 1024,
    workers: int = 4,
    log: Callable[[str], None] = print,
) -> Path:
    """Écrit le cache de ``names`` dans ``path``, en reprenant un travail coupé.

    Les morceaux de ``chunk_size`` images sont répartis sur ``workers``
    processus ; chaque morceau écrit est noté dans le fichier de progression.
    """
    path = Path(path)
    names = list(names)
    if is_complete(path, names):
        log(f"{path.name} : déjà complet ({len(names)} images)")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    side_w, side_h = box[2] - box[0], box[3] - box[1]
    shape = (len(names), side_h, side_w, 3)
    progress = _progress_path(path)
    digest = names_digest(names)

    done_chunks: set[int] = set()
    resume = path.exists() and progress.exists()
    if resume:
        header = progress.read_text().splitlines()
        if header and header[0] == digest:
            done_chunks = {int(c) for c in header[1:] if c.strip()}
        else:
            resume = False
    if resume:
        arr = np.lib.format.open_memmap(path, mode="r+")
        if arr.shape != shape or arr.dtype != np.uint8:
            resume, done_chunks = False, set()
            del arr
    if not resume:
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
        progress.write_text(digest + "\n")

    n_chunks = (len(names) + chunk_size - 1) // chunk_size
    todo = [c for c in range(n_chunks) if c not in done_chunks]
    log(f"{path.name} : {len(names)} images, {n_chunks} morceaux, {len(todo)} à faire")
    source = str(source)

    def chunk_names(c: int) -> list[str]:
        return names[c * chunk_size : (c + 1) * chunk_size]

    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {c: pool.submit(load_crops, source, chunk_names(c), box) for c in todo}
        for k, c in enumerate(todo, start=1):
            crops = futures.pop(c).result()
            lo = c * chunk_size
            arr[lo : lo + len(crops)] = crops
            arr.flush()
            with open(progress, "a") as f:
                f.write(f"{c}\n")
            if k % 10 == 0 or k == len(todo):
                log(f"{path.name} : {k}/{len(todo)} morceaux écrits")
    del arr

    check = np.load(path, mmap_mode="r")
    if check.shape != shape:
        raise ValueError(f"{path} : forme {check.shape}, attendu {shape}.")
    empty = [i for i in range(0, len(names), max(1, len(names) // 200)) if not check[i].any()]
    if empty:
        raise ValueError(f"{path} : images entièrement noires, par exemple la ligne {empty[0]}.")
    _done_path(path).write_text(json.dumps(
        {"n_images": len(names), "names_sha1": digest, "box": list(box), "shape": list(shape)},
        indent=2,
    ))
    progress.unlink(missing_ok=True)
    log(f"{path.name} : terminé")
    return path


def open_cache(path: str | Path, names: Sequence[str]) -> np.ndarray:
    """Cache terminé, ouvert en lecture seule (memmap) ; vérifie qu'il correspond à ``names``."""
    path = Path(path)
    if not is_complete(path, names):
        raise FileNotFoundError(
            f"Cache absent ou incomplet : {path}. Lancer d'abord scripts/01_build_cache.py."
        )
    return np.load(path, mmap_mode="r")
