"""Tableaux de l'évaluation : IC sur les graines, écarts appariés, bootstrap sur les identités."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from facegeom.eval.bootstrap import bootstrap_diff, bootstrap_mean
from facegeom.eval.stats import paired_diff, student_ci

METRICS = ["map_at_r", "p_at_1", "r_precision", "auc", "eer", "tar_at_far_0.1",
           "tar_at_far_0.01", "tar_at_far_0.001", "thr_tar", "thr_far", "thr_balanced_acc",
           "alignment", "uniformity"]


def seed_summary(per_seed: pd.DataFrame, metrics: list[str] = METRICS) -> pd.DataFrame:
    """Moyenne et IC de Student à 95 % par méthode (une seule valeur : pas d'IC)."""
    rows = []
    for method, g in per_seed.groupby("method", sort=False):
        for metric in metrics:
            if metric not in g:
                continue
            v = g[metric].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            if len(v) > 1:
                ci = student_ci(v)
            else:
                ci = {"mean": float(v[0]), "ci_low": np.nan, "ci_high": np.nan, "n": 1}
            rows.append({"method": method, "metric": metric, "mean": ci["mean"],
                         "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": ci["n"]})
    return pd.DataFrame(rows)


def paired_table(per_seed: pd.DataFrame, losses: list[str],
                 metrics: list[str] = METRICS) -> pd.DataFrame:
    """Écarts appariés graine par graine entre pertes (graines communes, au moins 2)."""
    rows = []
    for a, b in combinations(losses, 2):
        ga = per_seed[per_seed.method == a].set_index("seed")
        gb = per_seed[per_seed.method == b].set_index("seed")
        common = sorted(set(ga.index) & set(gb.index))
        if len(common) < 2:
            continue
        for metric in metrics:
            if metric not in ga or ga[metric].isna().any():
                continue
            rows.append({"a": a, "b": b, "metric": metric, "seeds": len(common),
                         **paired_diff(ga.loc[common, metric], gb.loc[common, metric])})
    return pd.DataFrame(rows)


def bootstrap_table(per_query: dict[str, np.ndarray], labels: np.ndarray,
                    pairs: list[tuple[str, str]], n_boot: int, seed: int) -> pd.DataFrame:
    """IC par bootstrap sur les identités : MAP@R de chaque méthode, puis écarts entre méthodes.

    ``per_query`` : MAP@R par requête de chaque méthode (moyenne sur les graines
    pour les pertes), mêmes requêtes et même ordre que ``labels``. En %.
    """
    rows = []
    for method, v in per_query.items():
        r = bootstrap_mean(100 * v, labels, n_boot, seed)
        rows.append({"kind": "mean", "a": method, "b": "", 
            "value": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    for a, b in pairs:
        if a in per_query and b in per_query:
            r = bootstrap_diff(100 * per_query[a], 100 * per_query[b], labels, n_boot, seed)
            rows.append({"kind": "diff", "a": a, "b": b, "value": r["diff"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                         "contains_zero": r["contains_zero"]})
    return pd.DataFrame(rows)
