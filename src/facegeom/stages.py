"""Étapes qui enchaînent plusieurs runs : pilote, réglage, entraînements finaux.

Chaque étape a son dossier dans ``runs/``, jamais écrasé
(``<date-heure>_4_tune_arcface`` par exemple). Relancer la même commande
reprend le dernier dossier de l'étape s'il n'est pas terminé (après une coupure
de Colab) : les runs finis sont sautés, le run coupé reprend à son dernier point
de reprise. ``new=True`` crée un nouveau dossier.

Un verrou par étape et par perte (``runs/.locks/``) empêche deux sessions de
travailler en même temps sur la même perte.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from facegeom.lock import DirLock
from facegeom.train import ConfigMismatchError, TrainData, train_run
from facegeom.utils import new_run_dir, read_json, run_info, write_json

KINDS = {"baselines": "2_baselines", "pilot": "3_pilot", "tune": "4_tune", "final": "5_final",
         "classical": "8_classical"}
DONE = "DONE.json"


# --- Tirages des réglages ------------------------------------------------------


def sample_params(ranges: dict, n: int, seed: int) -> list[dict]:
    """``n`` réglages tirés au hasard dans ``ranges`` (``[min, max]`` ou ``[min, max, log]``)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = {}
        for k, r in ranges.items():
            lo, hi = float(r[0]), float(r[1])
            p[k] = (float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if len(r) > 2
                    else float(rng.uniform(lo, hi)))
        out.append(p)
    return out


def center_params(ranges: dict) -> dict:
    """Centre des plages : milieu (linéaire) ou moyenne géométrique (logarithmique)."""
    return {k: (float(np.sqrt(float(r[0]) * float(r[1]))) if len(r) > 2
                else (float(r[0]) + float(r[1])) / 2) for k, r in ranges.items()}


# --- Dossiers d'étape ----------------------------------------------------------


ESSAI = "essai"  # suffixe des dossiers d'essais rapides (--max-steps), jamais pris pour le protocole


def stage_suffix(kind: str, loss: str | None = None, essai: bool = False) -> str:
    return KINDS[kind] + (f"_{loss}" if loss else "") + (f"_{ESSAI}" if essai else "")


def find_stages(runs_dir: str | Path, kind: str, loss: str | None = None,
                essai: bool = False) -> list[Path]:
    """Dossiers de l'étape, du plus ancien au plus récent (essais rapides à part)."""
    suffix = stage_suffix(kind, loss, essai)
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    return sorted(d for d in runs_dir.iterdir()
                  if d.is_dir() and d.name.split("_", 1)[-1] == suffix)


def is_done(stage_dir: Path) -> bool:
    return (Path(stage_dir) / DONE).exists()


def latest_done(runs_dir: str | Path, kind: str, loss: str | None = None,
                essai: bool = False) -> Path | None:
    done = [d for d in find_stages(runs_dir, kind, loss, essai) if is_done(d)]
    return done[-1] if done else None


def select_stage(runs_dir: str | Path, kind: str, loss: str | None, new: bool, essai: bool = False,
                 log: Callable[[str], None] = print) -> Path | None:
    """Dossier où travailler, ou ``None`` si l'étape est déjà terminée.

    - le dossier le plus récent de l'étape n'est pas terminé : on le reprend
      (après une coupure, même s'il avait été créé avec ``--new``) ;
    - il est terminé : rien à faire, sauf avec ``new=True`` (nouveau dossier) ;
    - aucun dossier : nouveau dossier.
    """
    stages = find_stages(runs_dir, kind, loss, essai)
    if stages and not is_done(stages[-1]):
        log(f"Reprise du dossier {stages[-1]}")
        return stages[-1]
    if stages and not new:
        log(f"Étape déjà terminée : {stages[-1]} (--new pour la refaire dans un nouveau dossier)")
        return None
    d = new_run_dir(runs_dir, stage_suffix(kind, loss, essai))
    log(f"Nouveau dossier {d}")
    return d


def stage_lock(runs_dir: str | Path, kind: str, loss: str | None, force: bool,
               essai: bool = False) -> DirLock:
    """Verrou d'une étape pour une perte, posé avant de choisir ou créer le dossier."""
    return DirLock(Path(runs_dir) / ".locks" / stage_suffix(kind, loss, essai), force=force)


def write_stage_meta(stage_dir: Path, config: dict, args: dict, device: str) -> None:
    """Ajoute un passage à ``meta.json`` sans effacer les précédents (reprises)."""
    path = stage_dir / "meta.json"
    sessions = read_json(path)["sessions"] if path.exists() else []
    write_json(path, {"sessions": [*sessions, {"config": config, "args": args,
                                               "run": run_info(device)}]})


def summarize_runs(stage_dir: Path, pattern: str) -> pd.DataFrame:
    """Une ligne par run terminé du dossier (``result.json``), réglages à plat."""
    rows = []
    for d in sorted(stage_dir.glob(pattern)):
        if (d / "result.json").exists():
            r = read_json(d / "result.json")
            rows.append({"run": d.name, **{f"p_{k}": v for k, v in r["params"].items()},
                         **{k: v for k, v in r.items() if k != "params"}})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(stage_dir / "summary.csv", index=False)
    return df


# --- Réglage -------------------------------------------------------------------


def run_tune(
    stage_dir: Path,
    tcfg: dict,
    dcfg: dict,
    loss: str,
    data: TrainData,
    device: str,
    trials: list[int] | None = None,
    pretrained: bool = True,
    log: Callable[[str], None] = print,
    heartbeat: Callable[[], None] | None = None,
    max_steps_per_epoch: int | None = None,
) -> dict | None:
    """Tirages de réglage d'une perte ; écrit ``best.json`` quand tous sont finis."""
    tune = tcfg["tune"]
    planned = sample_params(tcfg["losses"][loss]["params"], int(tune["n_trials"]), int(tune["seed"]))
    path = stage_dir / "trials.json"
    if path.exists():
        stored = read_json(path)
        same = (len(stored["params"]) == len(planned) and stored["epochs"] == int(tune["epochs"])
                and all(a.keys() == b.keys() and np.allclose([a[k] for k in a], [b[k] for k in a],
                                                              rtol=1e-12, atol=0)
                        for a, b in zip(stored["params"], planned)))
        if not same:
            raise ConfigMismatchError(
                f"{stage_dir} : plages ou nombre d'époques changés depuis le début du réglage. "
                "Relancer avec --new pour un nouveau réglage."
            )
    else:
        write_json(path, {"loss": loss, "seed": int(tune["seed"]), "epochs": int(tune["epochs"]),
                          "params": planned})

    for i in (trials if trials is not None else range(len(planned))):
        log(f"--- {loss} tirage {i} : {planned[i]}")
        train_run(stage_dir / f"trial{i:02d}", tcfg, dcfg, loss, planned[i], int(tune["seed"]),
                  int(tune["epochs"]), data, device, pretrained=pretrained,
                  keep_weights=bool(tune.get("keep_weights", False)), log=log,
                  heartbeat=heartbeat, max_steps_per_epoch=max_steps_per_epoch)

    df = summarize_runs(stage_dir, "trial*")
    if len(df) < len(planned):
        log(f"{loss} : {len(df)}/{len(planned)} tirages terminés")
        return None
    scores = df["best_map_at_r"].astype(float).fillna(-np.inf)
    if not np.isfinite(scores).any():
        write_json(stage_dir / DONE, {"n_trials": len(planned), "valid_trials": 0})
        log(f"{loss} : AUCUN tirage valide (tous ont divergé) : plages à revoir en groupe")
        return None
    i_best = int(scores.idxmax())
    tag = df.loc[i_best, "run"]
    best = {"loss": loss, "params": planned[int(tag[-2:])], "tag": tag,
            "best_map_at_r": float(df.loc[i_best, "best_map_at_r"]),
            "best_epoch": int(df.loc[i_best, "best_epoch"])}
    write_json(stage_dir / "best.json", best)
    write_json(stage_dir / DONE, {"n_trials": len(planned),
                                  "valid_trials": int(np.isfinite(scores).sum())})
    log(f"{loss} : MEILLEUR {tag}, MAP@R {best['best_map_at_r']:.2f}, {best['params']}")
    return best


# --- Entraînements finaux --------------------------------------------------------


def run_final(
    stage_dir: Path,
    tcfg: dict,
    dcfg: dict,
    loss: str,
    best: dict,
    source: str,
    data: TrainData,
    device: str,
    seeds: list[int] | None = None,
    pretrained: bool = True,
    log: Callable[[str], None] = print,
    heartbeat: Callable[[], None] | None = None,
    max_steps_per_epoch: int | None = None,
) -> pd.DataFrame | None:
    """Meilleur réglage × graines finales ; ``DONE.json`` quand toutes sont finies."""
    fcfg = tcfg["final"]
    planned_seeds = [int(s) for s in fcfg["seeds"]]
    expected = {"loss": loss, "params": best["params"], "source": source,
                "epochs": int(fcfg["epochs"]), "seeds": planned_seeds}
    path = stage_dir / "params.json"
    if path.exists():
        if read_json(path) != expected:
            raise ConfigMismatchError(
                f"{stage_dir} : réglage, graines ou époques changés. Relancer avec --new."
            )
    else:
        write_json(path, expected)

    for seed in (seeds if seeds is not None else planned_seeds):
        if seed not in planned_seeds:
            raise ValueError(f"Graine {seed} hors de la configuration {planned_seeds}.")
        log(f"--- {loss} graine {seed}")
        train_run(stage_dir / f"seed{seed}", tcfg, dcfg, loss, best["params"], seed,
                  int(fcfg["epochs"]), data, device, pretrained=pretrained, keep_weights=True,
                  log=log, heartbeat=heartbeat, max_steps_per_epoch=max_steps_per_epoch)

    df = summarize_runs(stage_dir, "seed*")
    if len(df) < len(planned_seeds):
        log(f"{loss} : {len(df)}/{len(planned_seeds)} graines terminées")
        return None
    write_json(stage_dir / DONE, {"seeds": planned_seeds})
    log(f"{loss} : graines terminées, MAP@R {df['best_map_at_r'].round(2).tolist()}")
    return df
