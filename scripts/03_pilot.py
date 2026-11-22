"""Étape 3 : pilote. Une époque par perte, réglages au centre des plages, graine 100.

L'époque du pilote est la première époque d'un tirage de réglage : montée du
taux pendant l'époque, puis cosinus calculé sur toute la durée d'un tirage
(20 époques). Deux mesures, avant de lancer le réglage :
- la durée d'une époque (entraînement, puis MAP@R sur la validation), d'où le
  temps estimé du réglage et des entraînements finaux ;
- si chaque perte apprend : MAP@R de validation après une époque, comparé au
  brut (ACP 128) des mesures de base.

    python scripts/03_pilot.py --cache-dir /content/cache --runs-dir $RUNS
    python scripts/03_pilot.py --max-steps 50     # essai rapide (dossier « essai » à part)
"""

import time
from pathlib import Path

import pandas as pd

from facegeom.cli import check_device, load_configs, parser, require_baselines
from facegeom.data.dataset import load_train_data
from facegeom.lock import LockedError
from facegeom.losses import LOSSES
from facegeom.stages import (
    DONE,
    center_params,
    latest_done,
    select_stage,
    stage_lock,
    write_stage_meta,
)
from facegeom.train import train_run
from facegeom.utils import read_json, write_json


def budget(rows: list[dict], tcfg: dict, full_steps: int) -> dict | None:
    """Heures prévues au plus (sans arrêt précoce) pour le réglage et les graines finales.

    Les pertes sans durée mesurée (divergées) prennent la durée moyenne des autres.
    """
    timed = [r for r in rows if r["seconds_per_step"] is not None]
    if not timed:
        return None
    per_step = sum(r["seconds_per_step"] for r in timed) / len(timed)
    eval_s = sum(r["seconds_eval"] for r in timed) / len(timed)
    tune, final = tcfg["tune"], tcfg["final"]
    out = {"tune_hours": 0.0, "final_hours": 0.0}
    for r in rows:
        step_s = r["seconds_per_step"] if r["seconds_per_step"] is not None else per_step
        ev_s = r["seconds_eval"] if r["seconds_eval"] is not None else eval_s
        epoch_s = step_s * full_steps + ev_s
        out["tune_hours"] += int(tune["n_trials"]) * int(tune["epochs"]) * epoch_s / 3600
        out["final_hours"] += len(final["seeds"]) * int(final["epochs"]) * epoch_s / 3600
    out["total_hours"] = out["tune_hours"] + out["final_hours"]
    return {k: round(v, 1) for k, v in out.items()}


def pilot_row(loss: str, r: dict, brut: float, wall: float) -> dict:
    steps = r["steps_per_epoch"]
    t = r["seconds_per_epoch_train"]
    return {
        "loss": loss, "status": r["status"], "map_at_r": r["best_map_at_r"], "brut_acp": brut,
        "apprend": bool(r["best_map_at_r"] is not None and r["best_map_at_r"] > brut),
        "precision": r.get("precision"), "steps": steps,
        "seconds_per_step": round(t / steps, 3) if t is not None else None,
        "seconds_eval": r["seconds_per_epoch_eval"], "wall_seconds": round(wall),
    }


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--losses", nargs="+", default=list(LOSSES), choices=list(LOSSES))
    p.add_argument("--max-steps", type=int, default=None, help="essai rapide : pas par époque")
    p.add_argument("--new", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    check_device(args)
    dcfg, tcfg, _ = load_configs(args)
    runs = Path(args.runs_dir)
    essai = args.max_steps is not None
    baselines = require_baselines(runs)
    brut = next(r["map_at_r"] for r in baselines["rows"] if r["method"] == "brut_acp")

    data = load_train_data(dcfg, args.cache_dir)
    bcfg = tcfg["batch"]
    full_steps = len(data.train_labels) // (bcfg["classes_per_batch"] * bcfg["images_per_class"])
    pcfg = tcfg["pilot"]
    try:
        lock = stage_lock(runs, "pilot", None, args.force, essai)
        lock.acquire()
    except LockedError as err:
        raise SystemExit(str(err)) from None
    try:
        stage = select_stage(runs, "pilot", None, args.new, essai)
        if stage is None:
            done = latest_done(runs, "pilot", essai=essai)
            print(pd.DataFrame(read_json(done / "pilot.json")["rows"]).to_string(index=False))
            return
        write_stage_meta(stage, {"data": dcfg, "train": tcfg}, vars(args), args.device)
        rows = []
        for loss in args.losses:
            params = center_params(tcfg["losses"][loss]["params"])
            print(f"--- pilote {loss} : {params}")
            t = time.perf_counter()
            r = train_run(stage / loss, tcfg, dcfg, loss, params, int(pcfg["seed"]),
                          int(pcfg["epochs"]), data, args.device, keep_weights=False,
                          heartbeat=lock.heartbeat, max_steps_per_epoch=args.max_steps,
                          schedule_epochs=int(tcfg["tune"]["epochs"]))
            rows.append(pilot_row(loss, r, brut, time.perf_counter() - t))
            print(rows[-1])
        estimate = budget(rows, tcfg, full_steps)
        write_json(stage / "pilot.json", {"rows": rows, "budget": estimate,
                                          "full_steps_per_epoch": full_steps,
                                          "max_steps": args.max_steps})
        if set(args.losses) == set(LOSSES):
            write_json(stage / DONE, {"losses": args.losses})
        print()
        print(pd.DataFrame(rows).to_string(index=False))
        if estimate:
            print(f"\nTemps prévu au plus (sans arrêt précoce, avec les époques complètes) : "
                  f"réglage {estimate['tune_hours']} h, graines finales "
                  f"{estimate['final_hours']} h, total {estimate['total_hours']} h")
        print("Résultats dans", stage)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
