"""Étapes 6 et 7 : évaluation de toutes les méthodes, sur la validation ou, une seule
fois, sur le test.

Méthodes : mesures de base (brut, brut ACP 128, LDA 128 du ResNet-18 de départ),
les 4 pertes × graines (derniers entraînements finaux terminés) et, si elles ont
été calculées (scripts/08_classical.py), Eigenfaces, Fisherfaces et LBP.

Par méthode : recherche (MAP@R, R-Precision, P@1), vérification (AUC, EER, TAR
à FAR fixé, seuil au point d'égale erreur choisi sur la validation), géométrie
(alignement, uniformité). Par perte : moyenne et IC de Student sur les graines.
Entre pertes : écarts appariés graine par graine, et bootstrap sur les identités.

    python scripts/06_evaluate.py --split val --cache-dir /content/cache --runs-dir $RUNS
    python scripts/06_evaluate.py --split test --confirm-test ...   # une seule fois, à la fin

Garde du test : un fichier TEST_CONSULTED.json est écrit dans le dossier des runs
avant toute lecture du test ; une seconde évaluation du test est refusée.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from facegeom.classical import load_classical
from facegeom.cli import load_configs, parser
from facegeom.data.dataset import load_images, read_split
from facegeom.eval.evaluate import evaluate, make_eval_pairs
from facegeom.eval.report import bootstrap_table, paired_table, seed_summary
from facegeom.eval.verification import eer_threshold, pair_scores
from facegeom.losses import LOSSES
from facegeom.models.network import autocast_dtype, embed_images
from facegeom.reference import REFERENCES, cached_base_features, eval_augment, reference_projections
from facegeom.stages import latest_done
from facegeom.train import load_trained
from facegeom.utils import new_run_dir, run_info, write_json

MARKER = "TEST_CONSULTED.json"
CLASSICAL = [("eigenfaces", "cosine"), ("fisherfaces", "cosine"), ("lbp", "chi2")]


def final_runs(runs: Path, seeds: list[int], require_all: bool) -> dict[str, dict[int, Path]]:
    """``{perte: {graine: dossier}}`` des derniers entraînements finaux terminés."""
    out: dict[str, dict[int, Path]] = {}
    for loss in LOSSES:
        stage = latest_done(runs, "final", loss)
        if stage is None:
            if require_all:
                raise SystemExit(f"Test refusé : pas d'entraînements finaux terminés pour {loss}.")
            print(f"{loss} : pas d'entraînements finaux terminés, perte ignorée")
            continue
        out[loss] = {s: stage / f"seed{s}" for s in seeds if (stage / f"seed{s}" / "best.pt").exists()}
        if require_all and sorted(out[loss]) != sorted(seeds):
            raise SystemExit(f"Test refusé : graines manquantes pour {loss} dans {stage}.")
    return out


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--confirm-test", action="store_true", help="obligatoire avec --split test")
    p.add_argument("--classical-dir", default=None,
                   help="dossier de scripts/08_classical.py (défaut : le dernier terminé)")
    args = p.parse_args()
    dcfg, tcfg, ecfg = load_configs(args)
    runs = Path(args.runs_dir)
    seeds = [int(s) for s in tcfg["final"]["seeds"]]
    is_test = args.split == "test"

    if is_test:
        if not args.confirm_test:
            raise SystemExit("Le test n'est évalué qu'une fois : ajouter --confirm-test.")
        if (runs / MARKER).exists():
            raise SystemExit(f"Test déjà consulté : {(runs / MARKER).read_text()}")
    finals = final_runs(runs, seeds, require_all=is_test)
    classical_dir = (Path(args.classical_dir) if args.classical_dir
                     else latest_done(runs, "classical"))
    baselines_dir = latest_done(runs, "baselines")

    out = new_run_dir(runs, f"6_eval_{args.split}")
    write_json(out / "meta.json", {
        "config": {"data": dcfg, "train": tcfg, "eval": ecfg}, "args": vars(args),
        "run": run_info(args.device), "baselines": baselines_dir and baselines_dir.name,
        "classical": classical_dir and classical_dir.name,
        "finals": {k: {s: str(d) for s, d in v.items()} for k, v in finals.items()}})
    if is_test:  # avant toute lecture du test
        write_json(runs / MARKER, {"status": "commencé", "run": out.name, "date": run_info()["date"]})

    splits = ["val", "test"] if is_test else ["val"]
    images, labels, names, ep = {}, {}, {}, {}
    for sp in splits:
        images[sp], labels[sp] = load_images(dcfg, args.cache_dir, sp)
        names[sp], _ = read_split(dcfg["splits"]["output_dir"], sp)
        ep[sp] = make_eval_pairs(labels[sp], ecfg["verification"], ecfg["uniformity"])

    # Mesures de base : vecteurs du réseau de départ, ACP et LDA ajustées sur l'entraînement
    def base_path(sp: str) -> Path:
        stored = baselines_dir / f"features_{sp}.npy" if baselines_dir else None
        return stored if stored is not None and stored.exists() else out / f"features_{sp}.npy"

    train_images, train_ids = load_images(dcfg, args.cache_dir, "train")
    base = {"train": cached_base_features(base_path("train"), tcfg, dcfg, train_images, args.device)}
    for sp in splits:
        base[sp] = cached_base_features(base_path(sp), tcfg, dcfg, images[sp], args.device)
    projections = reference_projections(base["train"], train_ids, int(ecfg["dim"]))
    del base["train"]

    # Méthodes : (nom, graine) → vecteurs par partie, et mesure de comparaison
    methods: dict[tuple[str, int | None], tuple[dict[str, np.ndarray], str]] = {}
    for name in REFERENCES:
        methods[(name, None)] = ({sp: projections[name](base[sp]) for sp in splits}, "cosine")
    if classical_dir is not None:
        for name, metric in CLASSICAL:
            methods[(name, None)] = ({sp: load_classical(classical_dir, name, sp, names[sp])
                                      for sp in splits}, metric)
    else:
        print("Méthodes classiques absentes (scripts/08_classical.py pas encore lancé)")
    augment = eval_augment(dcfg, tcfg, args.device)
    amp = autocast_dtype(args.device, tcfg.get("amp"))
    for loss, by_seed in finals.items():
        for seed, run_dir in sorted(by_seed.items()):
            model = load_trained(run_dir, tcfg, args.device)
            methods[(loss, seed)] = ({sp: embed_images(model, images[sp], augment,
                                                       int(tcfg["eval_batch_size"]), amp)
                                      for sp in splits}, "cosine")
            print(f"{loss} graine {seed} : vecteurs calculés")

    rows, scores_seed0, emb_seed0, per_query = [], {}, {}, {}
    sp = args.split
    for (name, seed), (vecs, metric) in methods.items():
        # Seuil de vérification : toujours choisi sur les paires de validation
        val_scores = pair_scores(vecs["val"], ep["val"]["pairs"], metric=metric)
        thr = eer_threshold(ep["val"]["y"], val_scores)
        pq: dict = {}
        m, scores = evaluate(vecs[sp], labels[sp], ep[sp], threshold=thr, metric=metric,
                             per_query=pq)
        rows.append({"method": name, "seed": seed, "threshold": thr, **m})
        per_query.setdefault(name, []).append(pq["map_at_r"])
        per_query["__labels__"] = [pq["labels"]]
        if seed in (None, seeds[0]):
            scores_seed0[name] = scores.astype(np.float16)
            emb_seed0[name] = vecs[sp].astype(np.float16)
        print(f"{name:12} {'' if seed is None else seed!s:>2} : MAP@R {m['map_at_r']:5.2f}  "
              f"P@1 {m['p_at_1']:5.2f}  AUC {m['auc']:5.2f}  TAR@1e-3 {m['tar_at_far_0.001']}  "
              f"align {m['alignment']:.3f}  unif {m['uniformity']:.3f}")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    seed_summary(per_seed).to_csv(out / "summary.csv", index=False)
    losses = [loss for loss in LOSSES if loss in finals]
    paired_table(per_seed, losses).to_csv(out / "paired.csv", index=False)

    q_labels = per_query.pop("__labels__")[0]
    mean_pq = {name: np.mean(v, axis=0) for name, v in per_query.items()}
    pairs = [(a, b) for i, a in enumerate(losses) for b in losses[i + 1:]]
    pairs += [(loss, ref) for loss in losses for ref in ("brut_acp", "lda")]
    boot = bootstrap_table(mean_pq, q_labels, pairs, int(ecfg["bootstrap"]["n"]),
                           int(ecfg["bootstrap"]["seed"]))
    boot.to_csv(out / "bootstrap.csv", index=False)
    np.savez_compressed(out / "scores_seed0.npz", y=ep[sp]["y"], **scores_seed0)
    np.savez_compressed(out / "embeddings_seed0.npz", labels=labels[sp], **emb_seed0)
    np.savez_compressed(out / "per_query_map_at_r.npz", labels=q_labels, **mean_pq)

    print(f"\n{sp.upper()} : MAP@R, moyenne [IC 95 % graines] et [IC 95 % identités]")
    summary = seed_summary(per_seed, ["map_at_r"]).set_index("method")
    for _, r in boot[boot.kind == "mean"].iterrows():
        s = summary.loc[r.a]
        print(f"  {r.a:12} {s['mean']:5.2f}  [{s.ci_low:5.2f} ; {s.ci_high:5.2f}]  "
              f"[{r.ci_low:5.2f} ; {r.ci_high:5.2f}]")
    if is_test:
        write_json(runs / MARKER, {"status": "terminé", "run": out.name, "date": run_info()["date"]})
        print(f"\nTest consulté : {runs / MARKER}. Aucune autre évaluation du test.")
    print("Résultats dans", out)


if __name__ == "__main__":
    main()
