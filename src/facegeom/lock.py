"""Verrou d'un dossier de travail : une seule session à la fois.

Leçon de l'incident triplet du 15 novembre : deux copies d'une même cellule
Colab avaient écrit dans le même dossier. Ici, une session pose un fichier
``LOCK.json`` dans le dossier et le rafraîchit régulièrement (battement). Une
autre session qui trouve un verrou rafraîchi récemment refuse de démarrer. Un
verrou qui n'a pas été rafraîchi depuis ``stale_after_s`` secondes est
considéré comme abandonné (session coupée) et peut être repris.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import time
from pathlib import Path

LOCK_NAME = "LOCK.json"


class LockedError(RuntimeError):
    """Le dossier est déjà utilisé par une autre session."""


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


class DirLock:
    """Verrou posé dans ``directory``.

    Utilisation ::

        with DirLock(stage_dir) as lock:
            ...
            lock.heartbeat()   # à appeler régulièrement pendant le travail
    """

    def __init__(
        self,
        directory: str | Path,
        stale_after_s: float = 900.0,
        heartbeat_every_s: float = 30.0,
        force: bool = False,
    ) -> None:
        self.path = Path(directory) / LOCK_NAME
        self.stale_after_s = float(stale_after_s)
        self.heartbeat_every_s = float(heartbeat_every_s)
        self.force = force
        self.token = secrets.token_hex(8)
        self.owner = {"host": socket.gethostname(), "pid": os.getpid(), "token": self.token}
        self._last_beat = 0.0

    def _read(self) -> dict | None:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def acquire(self) -> None:
        """Pose le verrou, ou lève ``LockedError`` si une autre session l'a."""
        current = self._read()
        if current is not None and current.get("token") != self.token and not self.force:
            age = time.time() - float(current.get("heartbeat", 0.0))
            if age < self.stale_after_s:
                raise LockedError(
                    f"Dossier déjà utilisé par une autre session ({current.get('host')}, "
                    f"pid {current.get('pid')}, dernier signe de vie il y a {age:.0f} s) : "
                    f"{self.path.parent}. Attendre {self.stale_after_s - age:.0f} s ou "
                    "relancer avec --force si cette session est bien arrêtée."
                )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._beat()

    def _beat(self) -> None:
        now = time.time()
        _write_json_atomic(self.path, {**self.owner, "heartbeat": now})
        self._last_beat = now

    def heartbeat(self) -> None:
        """Rafraîchit le verrou, au plus une fois toutes les ``heartbeat_every_s`` s."""
        if time.time() - self._last_beat >= self.heartbeat_every_s:
            current = self._read()
            if current is not None and current.get("token") != self.token:
                raise LockedError(
                    f"Le verrou de {self.path.parent} a été repris par une autre session "
                    f"({current.get('host')}, pid {current.get('pid')}) : arrêt."
                )
            self._beat()

    def release(self) -> None:
        """Retire le verrou s'il nous appartient encore."""
        current = self._read()
        if current is not None and current.get("token") == self.token:
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> DirLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
