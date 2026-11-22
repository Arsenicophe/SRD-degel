"""Étape 3 : pilote. Une époque par perte, réglages au centre des plages, graine 100.

Deux mesures, avant de lancer le réglage :
- la durée d'une époque (entraînement, puis MAP@R sur la validation), d'où le
  temps estimé du réglage et des entraînements finaux ;
- si chaque perte apprend : MAP@R de validation après une époque, comparé au
  brut (ACP 128) des mesures de base.

    python scripts/03_pilot.py --cache-dir /content/cache --runs-dir $RUNS
    python scripts/03_pilot.py --max-steps 50     # essai rapide (époques raccourcies)
"""

import time
from pathlib import Path

import pandas as pd

from facegeom.cli import load_configs, parser, require_baselines
from facegeom.data.dataset import load_train_data
from facegeom.losses import LOSSES
from facegeom.stages import (
    DONE,
    center_params,
    latest_done,
    open_stage,
    stage_lock,
    write_stage_meta,
)
from facegeom.train import train_run
from facegeom.utils import read_json, write_json


def budget(rows: list[dict], tcfg: dict, full_steps: int) -> dict:
    """Heures prévues au plus (sans arrêt précoce) pour le réglage et les graines finales."""
    tune, final = tcfg["tune"], tcfg["final"]
    out = {"tune_hours": 0.0, "final_hours": 0.0}
    for r in rows:
        epoch_s = r["seconds_per_step"] * full_steps + r["seconds_eval"]
        out["tune_hours"] += int(tune["n_trials"]) * int(tune["epochs"]) * epoch_s / 3600
        out["final_hours"] += len(final["seeds"]) * int(final["epochs"]) * epoch_s / 3600
    out["total_hours"] = out["tune_hours"] + out["final_hours"]
    return {k: round(v, 1) for k, v in out.items()}


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--losses", nargs="+", default=list(LOSSES), choices=list(LOSSES))
    p.add_argument("--max-steps", type=int, default=None, help="essai rapide : pas par époque")
    p.add_argument("--new", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    dcfg, tcfg, _ = load_configs(args)
    runs = Path(args.runs_dir)
    baselines = require_baselines(runs)
    brut = next(r["map_at_r"] for r in baselines["rows"] if r["method"] == "brut_acp")

    data = load_train_data(dcfg, args.cache_dir)
    bcfg = tcfg["batch"]
    full_steps = len(data.train_labels) // (bcfg["classes_per_batch"] * bcfg["images_per_class"])
    pcfg = tcfg["pilot"]
    with stage_lock(runs, "pilot", None, args.force) as lock:
        done = latest_done(runs, "pilot")
        if done is not None and not args.new:
            print(f"Pilote déjà fait : {done}")
            print(pd.DataFrame(read_json(done / "pilot.json")["rows"]).to_string(index=False))
            return
        stage = open_stage(runs, "pilot", None, args.new)
        write_stage_meta(stage, {"data": dcfg, "train": tcfg}, vars(args), args.device)
        rows = []
        for loss in args.losses:
            params = center_params(tcfg["losses"][loss]["params"])
            print(f"--- pilote {loss} : {params}")
            t = time.perf_counter()
            r = train_run(stage / loss, tcfg, dcfg, loss, params, int(pcfg["seed"]),
                          int(pcfg["epochs"]), data, args.device, keep_weights=False,
                          heartbeat=lock.heartbeat, max_steps_per_epoch=args.max_steps)
            steps = r["steps_per_epoch"]
            rows.append({
                "loss": loss, "map_at_r": r["best_map_at_r"], "brut_acp": brut,
                "apprend": bool(r["best_map_at_r"] is not None and r["best_map_at_r"] > brut),
                "steps": steps,
                "seconds_per_step": round(r["seconds_per_epoch_train"] / steps, 3),
                "seconds_epoch_full": round(r["seconds_per_epoch_train"] / steps * full_steps),
                "seconds_eval": r["seconds_per_epoch_eval"],
                "wall_seconds": round(time.perf_counter() - t),
            })
            print(rows[-1])
        estimate = budget(rows, tcfg, full_steps)
        write_json(stage / "pilot.json", {"rows": rows, "budget": estimate,
                                          "full_steps_per_epoch": full_steps,
                                          "max_steps": args.max_steps})
        if set(args.losses) == set(LOSSES):
            write_json(stage / DONE, {"losses": args.losses})
        print()
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"\nTemps prévu au plus (sans arrêt précoce) : réglage {estimate['tune_hours']} h, "
              f"graines finales {estimate['final_hours']} h, total {estimate['total_hours']} h")
        print("Résultats dans", stage)


if __name__ == "__main__":
    main()
