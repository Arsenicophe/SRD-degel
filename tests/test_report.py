"""Tableaux de l'évaluation : IC sur les graines, écarts appariés, bootstrap sur les identités."""

import numpy as np
import pandas as pd
import pytest

from facegeom.eval.bootstrap import bootstrap_diff, bootstrap_mean
from facegeom.eval.report import bootstrap_table, paired_table, seed_summary


def _per_seed():
    rows = []
    for seed, (a, b) in enumerate([(10.0, 8.0), (11.0, 8.5), (12.0, 9.5)]):
        rows += [{"method": "nsoftmax", "seed": seed, "map_at_r": a},
                 {"method": "triplet", "seed": seed, "map_at_r": b}]
    rows.append({"method": "lda", "seed": None, "map_at_r": 5.0})
    return pd.DataFrame(rows)


def test_seed_summary_student_ci_and_single_value():
    s = seed_summary(_per_seed(), ["map_at_r"]).set_index("method")
    assert s.loc["nsoftmax", "mean"] == pytest.approx(11.0)
    assert s.loc["nsoftmax", "ci_low"] < 11.0 < s.loc["nsoftmax", "ci_high"]
    assert s.loc["lda", "n"] == 1 and np.isnan(s.loc["lda", "ci_low"])


def test_paired_table_uses_common_seeds():
    p = paired_table(_per_seed(), ["nsoftmax", "triplet"], ["map_at_r"])
    assert len(p) == 1 and p.loc[0, "seeds"] == 3
    assert p.loc[0, "diff"] == pytest.approx(np.mean([2.0, 2.5, 2.5]))
    assert bool(p.loc[0, "different"])


def test_bootstrap_mean_contains_estimate_and_matches_diff_draws():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(50), 4)
    a = rng.random(200)
    b = a - 0.1
    ma = bootstrap_mean(a, labels, 500, seed=0)
    assert ma["ci_low"] <= ma["mean"] <= ma["ci_high"]
    d = bootstrap_diff(a, b, labels, 500, seed=0)
    assert d["diff"] == pytest.approx(0.1) and d["ci_low"] == pytest.approx(0.1)


def test_bootstrap_table_rows():
    labels = np.repeat(np.arange(20), 3)
    pq = {"lda": np.full(60, 0.05), "nsoftmax": np.full(60, 0.15)}
    t = bootstrap_table(pq, labels, [("nsoftmax", "lda")], 200, 0)
    assert set(t.kind) == {"mean", "diff"}
    diff = t[t.kind == "diff"].iloc[0]
    assert diff.value == pytest.approx(10.0) and not diff.contains_zero
