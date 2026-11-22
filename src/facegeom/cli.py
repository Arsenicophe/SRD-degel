"""Éléments communs aux scripts : options de ligne de commande, garde des mesures de base."""

from __future__ import annotations

import argparse
from pathlib import Path

from facegeom.stages import latest_done
from facegeom.utils import REPO_DIR, default_device, load_config, read_json

BASELINES_EXPORT = REPO_DIR / "results" / "2_baselines_val.json"


def parser(doc: str) -> argparse.ArgumentParser:
    """Options partagées : configurations, cache, dossier des runs, appareil."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--train-config", default="configs/train.yaml")
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--cache-dir", default="cache", help="cache des carrés (Colab : /content/cache)")
    p.add_argument("--runs-dir", default="runs", help="résultats (Colab : dossier runs sur Drive)")
    p.add_argument("--device", default=default_device())
    p.add_argument("--allow-cpu", action="store_true",
                   help="autoriser un entraînement sans carte graphique (très lent)")
    return p


def check_device(args) -> None:
    """Refuse d'entraîner sur processeur par mégarde (session Colab sans GPU)."""
    if args.device == "cpu" and not args.allow_cpu:
        raise SystemExit("Aucune carte graphique : sur Colab, Exécution → Modifier le type "
                         "d'exécution → GPU. (--allow-cpu pour forcer.)")


def load_configs(args) -> tuple[dict, dict, dict]:
    """Configurations données, entraînement et évaluation."""
    return (load_config(args.data_config), load_config(args.train_config),
            load_config(args.eval_config))


def require_baselines(runs_dir: str | Path) -> dict:
    """Résultats des mesures de base (règle 6 : aucune perte entraînée avant elles).

    Cherche une étape ``2_baselines`` terminée dans ``runs_dir``, sinon le résumé
    exporté dans le dépôt (``results/2_baselines_val.json``, quand les mesures ont
    été faites sur une autre machine).
    """
    done = latest_done(runs_dir, "baselines")
    if done is not None:
        return read_json(done / "results.json")
    if BASELINES_EXPORT.exists():
        return read_json(BASELINES_EXPORT)
    raise SystemExit(
        "Règle 6 : aucune perte n'est entraînée avant les mesures de base. "
        "Lancer d'abord scripts/02_baselines.py."
    )


def parse_index_list(text: str | None) -> list[int] | None:
    """« 0-4 » ou « 3 5 7 » ou « 0-2 8 » vers une liste d'entiers ; ``None`` = tout."""
    if not text:
        return None
    out: list[int] = []
    for part in str(text).replace(",", " ").split():
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out
