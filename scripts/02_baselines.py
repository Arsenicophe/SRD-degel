"""Étape 2 : mesures de base sur le ResNet-18 de départ, sans entraînement de perte.

Vecteurs de 512 du réseau de départ (poids ImageNet, transformation
d'évaluation) pour l'entraînement et la validation. Trois versions mesurées sur
la validation, avec les mêmes paires : brut (512), brut réduit par ACP à 128,
LDA à 128 (ACP et LDA ajustées sur l'entraînement). Le test n'est jamais lu.

    python scripts/02_baselines.py --cache-dir /content/cache --runs-dir $RUNS
    python scripts/02_baselines.py --export-results   # copie le résumé dans results/ (dépôt)
"""

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from facegeom.cli import BASELINES_EXPORT, load_configs, parser
from facegeom.data.dataset import load_images
from facegeom.eval.evaluate import evaluate, make_eval_pairs
from facegeom.eval.verification import eer_threshold, rates_at_threshold
from facegeom.reference import REFERENCES, cached_base_features, reference_projections
from facegeom.stages import DONE, latest_done, select_stage, stage_lock, write_stage_meta
from facegeom.utils import write_json
from facegeom.viz import NAMES_FR, similarity_histograms


def export(stage: Path) -> None:
    """Copie le résumé dans le dépôt, pour qu'une autre machine sache l'étape faite."""
    BASELINES_EXPORT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stage / "results.json", BASELINES_EXPORT)
    shutil.copyfile(stage / "results.csv", BASELINES_EXPORT.with_suffix(".csv"))
    print("Résumé copié dans", BASELINES_EXPORT)


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--new", action="store_true", help="refaire les mesures dans un nouveau dossier")
    p.add_argument("--force", action="store_true", help="passer outre le verrou d'une autre session")
    p.add_argument("--export-results", action="store_true")
    args = p.parse_args()
    dcfg, tcfg, ecfg = load_configs(args)
    runs = Path(args.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)

    with stage_lock(runs, "baselines", None, args.force) as lock:
        stage = select_stage(runs, "baselines", None, args.new)
        if stage is None:
            if args.export_results:
                export(latest_done(runs, "baselines"))
            return
        write_stage_meta(stage, {"data": dcfg, "train": tcfg, "eval": ecfg}, vars(args), args.device)

        feats, labels = {}, {}
        for split in ["train", "val"]:
            t = time.perf_counter()
            images, labels[split] = load_images(dcfg, args.cache_dir, split)
            feats[split] = cached_base_features(stage / f"features_{split}.npy", tcfg, dcfg, images,
                                                args.device)
            print(f"{split} : {feats[split].shape} en {time.perf_counter() - t:.0f} s")
            lock.heartbeat()

        t = time.perf_counter()
        projections = reference_projections(feats["train"], labels["train"], int(ecfg["dim"]))
        print(f"ACP et LDA ajustées en {time.perf_counter() - t:.0f} s")
        yva = labels["val"]
        ep = make_eval_pairs(yva, ecfg["verification"], ecfg["uniformity"])

        rows, panels, per_query = [], [], {"labels": None}
        for name in REFERENCES:
            x = projections[name](feats["val"])
            pq: dict = {}
            m, scores = evaluate(x, yva, ep, per_query=pq)
            thr = eer_threshold(ep["y"], scores)
            m.update(rates_at_threshold(ep["y"], scores, thr))
            rows.append({"method": name, "split": "val", "threshold": thr, **m})
            per_query[name] = pq["map_at_r"]
            per_query["labels"] = pq["labels"]
            panels.append({"title": NAMES_FR.get(name, name), "same": scores[ep["y"] == 1],
                           "diff": scores[ep["y"] == 0],
                           "note": f"MAP@R {m['map_at_r']:.1f} %   AUC {m['auc']:.1f} %"})
            print(f"{name:9} : MAP@R {m['map_at_r']:5.2f}  P@1 {m['p_at_1']:5.2f}  "
                  f"AUC {m['auc']:5.2f}  TAR@1e-3 {m['tar_at_far_0.001']}  "
                  f"align {m['alignment']:.3f}  unif {m['uniformity']:.3f}")
            lock.heartbeat()

        pd.DataFrame(rows).to_csv(stage / "results.csv", index=False)
        write_json(stage / "results.json", {"stage": stage.name, "split": "val", "rows": rows})
        np.savez_compressed(stage / "per_query_map_at_r.npz", **per_query)
        similarity_histograms(panels, stage / "histogrammes.png", ncols=3)
        write_json(stage / DONE, {"methods": list(REFERENCES)})
        print("Résultats dans", stage)
        if args.export_results:
            export(stage)


if __name__ == "__main__":
    main()
