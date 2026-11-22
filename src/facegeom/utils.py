"""Outils communs : configuration, appareil, graines, dossiers de résultats, traçabilité."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import secrets
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml

PACKAGES = ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "scikit-image", "pandas",
            "pillow", "matplotlib", "pytorch-metric-learning"]
REPO_DIR = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    """Lit un fichier YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_device() -> str:
    """cuda (Colab), sinon mps (Mac), sinon cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_workers(value, cap: int = 8, reserve: int = 1) -> int:
    """Nombre de processus : ``auto`` = cœurs disponibles moins ``reserve``, au plus ``cap``."""
    if value in (None, "auto"):
        return max(0, min(cap, (os.cpu_count() or 1) - reserve))
    return int(value)


def set_seed(seed: int) -> None:
    """Fixe les graines de ``random``, ``numpy`` et ``torch``."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def file_hash(path: str | Path, algo: str = "md5") -> str:
    """Empreinte d'un fichier, lue par morceaux."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_run_dir(runs_dir: str | Path, name: str) -> Path:
    """Crée ``<runs_dir>/<date-heure>_<name>/`` ; échoue s'il existe déjà."""
    d = Path(runs_dir) / f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_json(path: str | Path, obj) -> None:
    """Écrit ``obj`` en JSON, d'un seul coup (fichier temporaire puis renommage)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_torch(obj, path: str | Path) -> None:
    """``torch.save`` sans jamais laisser un fichier à moitié écrit."""
    import torch

    path = Path(path)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_revision() -> str:
    """Révision git du dépôt, suffixée de « -dirty » s'il y a des modifications."""
    try:
        git = ["git", "-C", str(REPO_DIR)]
        rev = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True,
                               check=True)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "inconnue"


def run_info(device: str | None = None) -> dict:
    """Date, révision, versions et matériel, à écrire avec chaque résultat."""
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    info = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "host": platform.node(),
        "versions": versions,
        "device": device,
        "cpu_count": os.cpu_count(),
    }
    if device and device.startswith("cuda"):
        import torch

        info["gpu"] = torch.cuda.get_device_name(0)
    return info
