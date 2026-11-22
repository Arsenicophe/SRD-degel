"""Garde du test (règle 2) : toutes les pertes et graines exigées, une seule lecture."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _final_stage(runs: Path, loss: str, seeds: list[int], status: str = "done") -> Path:
    d = runs / f"20261122-120000_5_final_{loss}"
    for s in seeds:
        (d / f"seed{s}").mkdir(parents=True)
        (d / f"seed{s}" / "best.pt").write_bytes(b"")
        (d / f"seed{s}" / "result.json").write_text(json.dumps({"status": status}))
    (d / "DONE.json").write_text(json.dumps({"seeds": seeds}))
    return d


def test_test_split_requires_every_loss_and_seed(tmp_path):
    ev = _script("06_evaluate")
    for loss in ["nsoftmax", "arcface", "contrastive"]:
        _final_stage(tmp_path, loss, [0, 1, 2])
    with pytest.raises(SystemExit):  # triplet manque
        ev.final_runs(tmp_path, [0, 1, 2], require_all=True)
    _final_stage(tmp_path, "triplet", [0, 1])
    with pytest.raises(SystemExit):  # graine 2 de triplet manque
        ev.final_runs(tmp_path, [0, 1, 2], require_all=True)
    # Sur la validation, on évalue ce qui existe
    found = ev.final_runs(tmp_path, [0, 1, 2], require_all=False)
    assert sorted(found["triplet"]) == [0, 1]


def test_second_test_reading_is_refused(tmp_path, monkeypatch):
    ev = _script("06_evaluate")
    (tmp_path / ev.MARKER).write_text(json.dumps({"status": "terminé"}))
    monkeypatch.setattr(sys, "argv", ["06_evaluate.py", "--split", "test", "--confirm-test",
                                      "--runs-dir", str(tmp_path), "--cache-dir", str(tmp_path)])
    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit, match="déjà consulté"):
        ev.main()


def test_test_reading_needs_confirmation(tmp_path, monkeypatch):
    ev = _script("06_evaluate")
    monkeypatch.setattr(sys, "argv", ["06_evaluate.py", "--split", "test",
                                      "--runs-dir", str(tmp_path), "--cache-dir", str(tmp_path)])
    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit, match="confirm-test"):
        ev.main()
    assert not (tmp_path / ev.MARKER).exists()


def test_diverged_final_runs_are_refused_for_test(tmp_path):
    ev = _script("06_evaluate")
    for loss in ["nsoftmax", "arcface", "contrastive"]:
        _final_stage(tmp_path, loss, [0, 1, 2])
    _final_stage(tmp_path, "triplet", [0, 1, 2], status="diverged")
    with pytest.raises(SystemExit, match="divergé"):
        ev.final_runs(tmp_path, [0, 1, 2], require_all=True)


def test_interrupted_test_reading_needs_explicit_retry(tmp_path):
    ev = _script("06_evaluate")
    (tmp_path / ev.MARKER).write_text(json.dumps({"status": "commencé", "attempts": [{"run": "a"}]}))
    with pytest.raises(SystemExit, match="interrompue"):
        ev.check_test_marker(tmp_path, retry=False)
    assert ev.check_test_marker(tmp_path, retry=True) == [{"run": "a"}]
