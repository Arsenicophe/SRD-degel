"""Étape 4 : réglage. 10 tirages par perte dans les plages fixées, graine 100,
20 époques au plus par tirage, critère MAP@R de validation.

Relancer la même commande reprend le réglage là où il a été coupé. Une perte
par session à la fois (verrou) ; plusieurs sessions peuvent travailler en
parallèle sur des pertes différentes.

    python scripts/04_tune.py --loss arcface --cache-dir /content/cache --runs-dir $RUNS
    python scripts/04_tune.py --loss all ...            # les 4 pertes à la suite
    python scripts/04_tune.py --loss triplet --trials 0-4   # une partie des tirages
"""

from pathlib import Path

from facegeom.cli import load_configs, parse_index_list, parser, require_baselines
from facegeom.data.dataset import load_train_data
from facegeom.losses import LOSSES
from facegeom.stages import latest_done, open_stage, run_tune, stage_lock, write_stage_meta


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--loss", required=True, choices=[*LOSSES, "all"])
    p.add_argument("--trials", default=None, help="ex. « 0-4 » ; défaut : tous")
    p.add_argument("--new", action="store_true", help="nouveau réglage dans un nouveau dossier")
    p.add_argument("--force", action="store_true", help="passer outre le verrou d'une autre session")
    p.add_argument("--max-steps", type=int, default=None, help="essai rapide seulement")
    args = p.parse_args()
    dcfg, tcfg, _ = load_configs(args)
    runs = Path(args.runs_dir)
    require_baselines(runs)
    data = load_train_data(dcfg, args.cache_dir)
    losses = list(LOSSES) if args.loss == "all" else [args.loss]
    trials = parse_index_list(args.trials)

    for loss in losses:
        with stage_lock(runs, "tune", loss, args.force) as lock:
            done = latest_done(runs, "tune", loss)
            if done is not None and not args.new:
                print(f"{loss} : réglage déjà terminé dans {done}")
                continue
            stage = open_stage(runs, "tune", loss, args.new)
            write_stage_meta(stage, {"data": dcfg, "train": tcfg}, vars(args), args.device)
            run_tune(stage, tcfg, dcfg, loss, data, args.device, trials=trials,
                     heartbeat=lock.heartbeat, max_steps_per_epoch=args.max_steps)


if __name__ == "__main__":
    main()
