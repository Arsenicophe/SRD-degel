"""Entraînement : pertes, lots, taux, BatchNorm gelés, même graine même résultat, reprise.

Les essais tournent sur processeur, avec un ResNet-18 aux poids tirés au hasard
et de petites images synthétiques : quelques secondes chacun.
"""

import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from facegeom.data.batches import epoch_batches
from facegeom.losses import make_loss, margin_at, set_arcface_margin
from facegeom.models.network import EmbeddingNet, load_backbone
from facegeom.stages import center_params, run_tune, sample_params
from facegeom.train import ConfigMismatchError, TrainData, build_optimizer, lr_factor, train_run
from facegeom.utils import load_config, read_json

ROOT = Path(__file__).resolve().parents[1]
TCFG = load_config(ROOT / "configs" / "train.yaml")
DCFG = {"input_size": 32, "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}}
PARAMS = {
    "nsoftmax": {"lr": 0.05, "lr_weights": 0.05, "s": 16},
    "arcface": {"lr": 0.05, "lr_weights": 0.05, "s": 16, "m": 0.3},
    "contrastive": {"lr": 0.05, "pos_margin": 0.0, "neg_margin": 0.5},
    "triplet": {"lr": 0.05, "margin": 0.1},
}


def small_cfg() -> dict:
    cfg = copy.deepcopy(TCFG)
    cfg.update({"batch": {"classes_per_batch": 4, "images_per_class": 2}, "eval_batch_size": 16,
                "num_workers": 0, "amp": False, "channels_last": False})
    return cfg


def _images(n_id: int, per_id: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Images 16 × 16 : un motif par personne, plus du bruit."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(n_id, 16, 16, 3))
    noise = rng.integers(-30, 31, size=(n_id * per_id, 16, 16, 3))
    x = np.clip(np.repeat(base, per_id, axis=0) + noise, 0, 255).astype(np.uint8)
    return x, np.repeat(np.arange(n_id), per_id)


@pytest.fixture()
def data(tmp_path) -> TrainData:
    xtr, ytr = _images(8, 6, 0)
    xva, yva = _images(5, 4, 1)
    np.save(tmp_path / "train.npy", xtr)
    return TrainData(train_cache=str(tmp_path / "train.npy"), train_labels=ytr,
                     val_images=xva, val_labels=yva + 100)


def _run(run_dir, data, loss="nsoftmax", epochs=2, **kw):
    return train_run(run_dir, small_cfg(), DCFG, loss, PARAMS[loss], 0, epochs, data, "cpu",
                     pretrained=False, log=lambda s: None, **kw)


# --- Pertes -------------------------------------------------------------------


def test_arcface_with_zero_margin_equals_nsoftmax():
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
    y = torch.arange(10) % 5
    a, _ = make_loss("arcface", {"s": 20, "m": 0.0}, 5, 8)
    b, _ = make_loss("nsoftmax", {"s": 20}, 5, 8)
    b.weight.data.copy_(a.weight.data)  # mêmes poids de classes
    assert a(z, y).item() == pytest.approx(b(z, y).item(), rel=1e-5)


def test_arcface_margin_ramp_takes_effect():
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
    y = torch.arange(10) % 5
    a, _ = make_loss("arcface", {"s": 20, "m": 0.5}, 5, 8)
    full = a(z, y).item()
    set_arcface_margin(a, 0.0)
    zero = a(z, y).item()
    assert zero < full  # la marge rend la perte plus exigeante
    assert margin_at(0, 100, 3, 0.5) == 0.0
    assert margin_at(150, 100, 3, 0.5) == pytest.approx(0.25)
    assert margin_at(10_000, 100, 3, 0.5) == 0.5


# --- Lots, tirages, taux ----------------------------------------------------------


def test_batches_have_m_per_class_and_depend_on_seed_and_epoch():
    labels = np.repeat(np.arange(20), 5)
    a = epoch_batches(labels, 4, 32, 10, seed=0, epoch=1)
    for idx in a:
        counts = np.unique(labels[idx], return_counts=True)[1]
        assert len(idx) == 32 and set(counts) == {4}
    b = epoch_batches(labels, 4, 32, 10, seed=0, epoch=1)
    c = epoch_batches(labels, 4, 32, 10, seed=0, epoch=2)
    d = epoch_batches(labels, 4, 32, 10, seed=1, epoch=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))
    assert not all(np.array_equal(x, y) for x, y in zip(a, d))


def test_sample_params_respects_ranges_and_seed():
    ranges = TCFG["losses"]["arcface"]["params"]
    a, b = sample_params(ranges, 10, 100), sample_params(ranges, 10, 100)
    assert a == b
    for p in a:
        for k, r in ranges.items():
            assert float(r[0]) <= p[k] <= float(r[1])


def test_center_params():
    c = center_params({"lr": [0.03, 0.3, "log"], "m": [0.25, 0.55]})
    assert c["lr"] == pytest.approx(np.sqrt(0.03 * 0.3))
    assert c["m"] == pytest.approx(0.40)


def test_lr_factor_warmup_then_cosine():
    total, warm = 100, 10
    f = [lr_factor(t, warm, total) for t in range(total)]
    assert f[0] == pytest.approx(0.1)
    assert max(f) == pytest.approx(lr_factor(9, warm, total))
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(f[:10]))   # montée
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(f[10:]))   # descente
    assert f[-1] < 0.01


def test_optimizer_groups_and_frozen_batchnorm():
    model = EmbeddingNet(load_backbone(TCFG["backbone"], pretrained=False), 512, 128)
    loss_fn, _ = make_loss("arcface", PARAMS["arcface"], 5, 128)
    opt = build_optimizer(model, loss_fn, "arcface", PARAMS["arcface"], TCFG)
    lrs = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lrs["head"] == 0.05 and lrs["loss_weights"] == 0.05
    assert lrs["backbone"] == pytest.approx(0.05 * TCFG["optimizer"]["backbone_lr_ratio"])
    assert all(g["weight_decay"] == TCFG["optimizer"]["weight_decay"] for g in opt.param_groups)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    bn = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bn and all(id(p) not in in_opt for m in bn for p in m.parameters())

    # En mode entraînement, les BatchNorm gardent leurs statistiques
    model.train()
    before = bn[0].running_mean.clone()
    model(torch.randn(4, 3, 32, 32) * 3 + 1)
    assert not bn[0].training and torch.equal(before, bn[0].running_mean)


# --- Boucle d'entraînement --------------------------------------------------------


@pytest.mark.parametrize("loss", ["nsoftmax", "arcface", "contrastive", "triplet"])
def test_run_writes_results_and_same_seed_same_result(tmp_path, data, loss):
    r1 = _run(tmp_path / "a", data, loss)
    _run(tmp_path / "b", data, loss)
    h1 = pd.read_csv(tmp_path / "a" / "history.csv")
    h2 = pd.read_csv(tmp_path / "b" / "history.csv")
    cols = ["train_loss", "map_at_r", "p_at_1"]
    assert h1[cols].equals(h2[cols])
    s1 = torch.load(tmp_path / "a" / "best.pt", weights_only=True)
    s2 = torch.load(tmp_path / "b" / "best.pt", weights_only=True)
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert r1["status"] == "done" and r1["epochs_run"] == 2
    assert not (tmp_path / "a" / "last.pt").exists()
    assert read_json(tmp_path / "a" / "result.json")["best_map_at_r"] == r1["best_map_at_r"]


class _Stop(Exception):
    pass


def test_resume_after_interruption_is_identical(tmp_path, data):
    """Coupure après l'époque 1, puis reprise : même historique et mêmes poids."""
    ref = _run(tmp_path / "ref", data, "arcface", epochs=3)
    calls = {"n": 0}

    def cut():
        calls["n"] += 1
        if calls["n"] == 2:  # 2e signe de vie = après l'écriture du point de reprise de l'époque 1
            raise _Stop

    with pytest.raises(_Stop):
        _run(tmp_path / "cut", data, "arcface", epochs=3, heartbeat=cut)
    assert (tmp_path / "cut" / "last.pt").exists()
    assert not (tmp_path / "cut" / "result.json").exists()
    res = _run(tmp_path / "cut", data, "arcface", epochs=3)
    h_ref = pd.read_csv(tmp_path / "ref" / "history.csv")
    h_cut = pd.read_csv(tmp_path / "cut" / "history.csv")
    cols = ["epoch", "train_loss", "map_at_r", "margin"]
    assert h_ref[cols].equals(h_cut[cols])
    a = torch.load(tmp_path / "ref" / "best.pt", weights_only=True)
    b = torch.load(tmp_path / "cut" / "best.pt", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert res["best_map_at_r"] == ref["best_map_at_r"]


def test_finished_run_is_skipped_and_params_are_checked(tmp_path, data):
    _run(tmp_path / "r", data, epochs=1)
    again = _run(tmp_path / "r", data, epochs=1)
    assert again["epochs_run"] == 1
    (tmp_path / "r" / "result.json").unlink()
    with pytest.raises(ConfigMismatchError):
        _run(tmp_path / "r", data, epochs=2)


def test_tuning_stage_writes_best_and_detects_changed_ranges(tmp_path, data):
    cfg = small_cfg()
    cfg["tune"] = {"n_trials": 2, "epochs": 1, "seed": 100, "keep_weights": False}
    best = run_tune(tmp_path, cfg, DCFG, "triplet", data, "cpu", pretrained=False,
                    log=lambda s: None)
    assert best is not None and (tmp_path / "DONE.json").exists()
    assert best["tag"] in ("trial00", "trial01")
    assert not (tmp_path / "trial00" / "best.pt").exists()  # réglage : poids non gardés
    cfg["losses"]["triplet"]["params"]["margin"] = [0.5, 0.6]
    with pytest.raises(ConfigMismatchError):
        run_tune(tmp_path, cfg, DCFG, "triplet", data, "cpu", pretrained=False,
                 log=lambda s: None)


def test_diverging_run_is_recorded_and_tuning_goes_on(tmp_path, data, monkeypatch):
    """Un run dont le réseau dégénère est consigné « diverged » ; le réglage continue."""
    import facegeom.train as train_module

    calls = {"n": 0}
    real_embed = train_module.embed_images

    def degenerate_first_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # premier tirage : vecteurs nuls
            raise FloatingPointError("vecteurs nuls en sortie du réseau (réseau dégénéré)")
        return real_embed(*args, **kwargs)

    monkeypatch.setattr(train_module, "embed_images", degenerate_first_run)
    cfg = small_cfg()
    cfg["tune"] = {"n_trials": 2, "epochs": 1, "seed": 100, "keep_weights": False}
    cfg["losses"]["arcface"]["params"] = {"lr": [0.01, 0.02], "lr_weights": [0.01, 0.02],
                                          "s": [10, 12], "m": [0.1, 0.2]}
    best = run_tune(tmp_path, cfg, DCFG, "arcface", data, "cpu", pretrained=False,
                    log=lambda s: None)
    first = read_json(tmp_path / "trial00" / "result.json")
    assert first["status"] == "diverged" and first["best_map_at_r"] is None
    assert best is not None and best["tag"] == "trial01"


def test_non_finite_loss_marks_run_diverged(tmp_path, data):
    params = {**PARAMS["nsoftmax"], "lr": 1e12, "lr_weights": 1e12}
    res = train_run(tmp_path / "r", small_cfg(), DCFG, "nsoftmax", params, 0, 2, data, "cpu",
                    pretrained=False, log=lambda s: None)
    assert res["status"] == "diverged"
    assert (tmp_path / "r" / "result.json").exists() and not (tmp_path / "r" / "last.pt").exists()


def test_tuning_with_only_diverged_trials_writes_no_best(tmp_path, data):
    cfg = small_cfg()
    cfg["tune"] = {"n_trials": 2, "epochs": 1, "seed": 100, "keep_weights": False}
    cfg["losses"]["nsoftmax"]["params"] = {"lr": [1e12, 2e12], "lr_weights": [1e12, 2e12],
                                           "s": [10, 12]}
    best = run_tune(tmp_path, cfg, DCFG, "nsoftmax", data, "cpu", pretrained=False,
                    log=lambda s: None)
    assert best is None and not (tmp_path / "best.json").exists()
    assert read_json(tmp_path / "DONE.json")["valid_trials"] == 0


def test_gradient_clipping_option_changes_training(tmp_path, data):
    """L'écrêtage (désactivé par défaut) agit quand il est demandé, et entre dans la recette."""
    assert TCFG["grad_clip_norm"] is None
    ref = _run(tmp_path / "ref", data, epochs=1)
    cfg = small_cfg()
    cfg["grad_clip_norm"] = 1e-3
    clipped = train_run(tmp_path / "clip", cfg, DCFG, "nsoftmax", PARAMS["nsoftmax"], 0, 1, data,
                        "cpu", pretrained=False, log=lambda s: None)
    assert read_json(tmp_path / "clip" / "params.json")["recipe"]["grad_clip_norm"] == 1e-3
    assert clipped["best_map_at_r"] != ref["best_map_at_r"] or not torch.equal(
        torch.load(tmp_path / "ref" / "best.pt", weights_only=True)["head.weight"],
        torch.load(tmp_path / "clip" / "best.pt", weights_only=True)["head.weight"])


def test_cli_overrides_are_applied():
    from facegeom.cli import apply_overrides

    cfg = apply_overrides({"a": 1, "b": {"c": 2}}, ["a=5", "b.c=null", "d=0.01"])
    assert cfg == {"a": 5, "b": {"c": None}, "d": 0.01}


@pytest.mark.parametrize("loss", ["nsoftmax", "contrastive"])
def test_head_batchnorm_option_trains_and_reloads(tmp_path, data, loss):
    from facegeom.train import load_trained

    cfg = small_cfg()
    cfg["head"] = {**cfg["head"], "batchnorm": True, "dropout": 0.1}
    r = train_run(tmp_path / "r", cfg, DCFG, loss, PARAMS[loss], 0, 1, data, "cpu",
                  pretrained=False, log=lambda s: None)
    assert r["status"] == "done"
    model = load_trained(tmp_path / "r", cfg)
    assert isinstance(model.head[-1], torch.nn.BatchNorm1d) and not model.training
    model.train()
    assert model.head[-1].training  # la BatchNorm de la tête s'entraîne, celles du réseau non
