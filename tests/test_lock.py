"""Verrou de dossier : une seule session à la fois."""

import json
import time

import pytest

from facegeom.lock import LOCK_NAME, DirLock, LockedError


def test_second_session_is_refused(tmp_path):
    with DirLock(tmp_path), pytest.raises(LockedError):
        DirLock(tmp_path).acquire()
    # Une fois relâché, le dossier est de nouveau libre.
    with DirLock(tmp_path):
        pass
    assert not (tmp_path / LOCK_NAME).exists()


def test_stale_lock_can_be_taken_over(tmp_path):
    (tmp_path / LOCK_NAME).write_text(
        json.dumps({"host": "ancienne", "pid": 1, "token": "x", "heartbeat": time.time() - 3600})
    )
    with DirLock(tmp_path, stale_after_s=900):
        pass


def test_force_takes_over_a_fresh_lock(tmp_path):
    other = DirLock(tmp_path)
    other.acquire()
    with DirLock(tmp_path, force=True):
        pass


def test_heartbeat_detects_takeover(tmp_path):
    lock = DirLock(tmp_path, heartbeat_every_s=0.0)
    lock.acquire()
    DirLock(tmp_path, force=True).acquire()
    with pytest.raises(LockedError):
        lock.heartbeat()
