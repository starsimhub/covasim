"""Shared parity-gate helper for multi-seed v4-vs-v3.1.8 metric comparison.

The z-score formula is

    z = (v4_mean - v3_mean) / sqrt(v3_SE^2 + v4_SE^2)

where SE is the standard error of the mean across seeds (std with ddof=1
divided by sqrt(n)). A metric fails when |z| >= z_threshold.

Per-seed rows are dicts of {metric_name: value}. Non-metric bookkeeping keys
(e.g. _seed, _total_pop, n_alive) should be passed via skip_keys.

Ported essentially verbatim from the hpvsim v2->v3 regression harness; only the
v2/v3 argument names became v3/v4 for the Covasim port.
"""
import math

import numpy as np


def _mean_se(rows, key):
    """Return (mean, standard_error) of rows[*][key], or None if absent."""
    vals = np.array([float(r[key]) for r in rows if key in r], dtype=float)
    if vals.size == 0:
        return None
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, se


def parity_gate(v4_seeds, v3_seeds, z_threshold=3.0, skip_keys=frozenset()):
    """Return [(metric_name, z)] for metrics exceeding |z| >= z_threshold.

    Args:
        v4_seeds (list): per-seed summary dicts from the v4 run.
        v3_seeds (list): per-seed summary dicts from the v3.1.8 baseline.
        z_threshold (float): failure threshold; metrics with |z| >= it fail.
        skip_keys (set): metric names to ignore (bookkeeping fields).

    Returns:
        list of (metric_name, z) tuples for metrics that fail the gate.

    Two degenerate-distribution policies:
      - both v3 and v4 have zero spread AND exactly equal means -> pass.
      - both have zero spread AND unequal means -> fail with z=inf.
    """
    metric_keys = sorted((set(v3_seeds[0]) & set(v4_seeds[0])) - skip_keys)
    failures = []
    for key in metric_keys:
        v3_stats = _mean_se(v3_seeds, key)
        v4_stats = _mean_se(v4_seeds, key)
        if v3_stats is None or v4_stats is None:
            continue
        v3_mean, v3_se = v3_stats
        v4_mean, v4_se = v4_stats
        se_combo = math.sqrt(v3_se ** 2 + v4_se ** 2)
        if se_combo == 0:
            if v3_mean != v4_mean:
                failures.append((key, float('inf')))
            continue
        z = (v4_mean - v3_mean) / se_combo
        if abs(z) >= z_threshold:
            failures.append((key, z))
    return failures
