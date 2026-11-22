"""Étape 5 : entraînements finaux. Meilleur tirage du réglage × graines 0, 1, 2,
30 époques au plus, arrêt précoce sur le MAP@R de validation.

Le réglage pris est le dernier réglage terminé de la perte (ou ``--tune-dir``).
Relancer la même commande reprend là où le travail a été coupé.

    python scripts/05_final.py --loss arcface --cache-dir /content/cache --runs-dir $RUNS
    python scripts/05_final.py --loss all ...
    python scripts/05_final.py --loss triplet --seeds 2   # une seule graine
"""

from pathlib import Path

from facegeom.cli import load_configs, parse_index_list, parser, require_baselines
from facegeom.data.dataset import load_train_data
from facegeom.losses import LOSSES
from facegeom.stages import latest_done, open_stage, run_final, stage_lock, write_stage_meta
from facegeom.utils import read_json


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--loss", required=True, choices=[*LOSSES, "all"])
    p.add_argument("--seeds", default=None, help="ex. « 0-2 » ; défaut : graines de la configuration")
    p.add_argument("--tune-dir", default=None, help="dossier de réglage (défaut : le dernier terminé)")
    p.add_argument("--new", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-steps", type=int, default=None, help="essai rapide seulement")
    args = p.parse_args()
    dcfg, tcfg, _ = load_configs(args)
    runs = Path(args.runs_dir)
    require_baselines(runs)
    losses = list(LOSSES) if args.loss == "all" else [args.loss]
    if args.tune_dir and len(losses) > 1:
        raise SystemExit("--tune-dir ne va qu'avec une seule perte.")
    data = load_train_data(dcfg, args.cache_dir)

    for loss in losses:
        tune_dir = Path(args.tune_dir) if args.tune_dir else latest_done(runs, "tune", loss)
        if tune_dir is None or not (tune_dir / "best.json").exists():
            print(f"{loss} : aucun réglage terminé, graines finales impossibles pour l'instant")
            continue
        best = read_json(tune_dir / "best.json")
        if best["loss"] != loss:
            raise SystemExit(f"{tune_dir} est le réglage de {best['loss']}, pas de {loss}.")
        with stage_lock(runs, "final", loss, args.force) as lock:
            done = latest_done(runs, "final", loss)
            if done is not None and not args.new:
                print(f"{loss} : graines finales déjà terminées dans {done}")
                continue
            stage = open_stage(runs, "final", loss, args.new)
            write_stage_meta(stage, {"data": dcfg, "train": tcfg}, vars(args), args.device)
            run_final(stage, tcfg, dcfg, loss, best, tune_dir.name, data, args.device,
                      seeds=parse_index_list(args.seeds), heartbeat=lock.heartbeat,
                      max_steps_per_epoch=args.max_steps)


if __name__ == "__main__":
    main()
