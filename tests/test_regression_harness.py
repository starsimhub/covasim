"""Tests for the v3.1.8 -> v4.0 regression harness.

Fast unit + smoke tests for the harness machinery:
  - anchor smoke test
  - parity_gate unit tests
  - compute_drift unit tests

The heavy multi-seed z-score release gate lives in tests/test_m0_parity.py.
"""

import sys
from pathlib import Path

import pytest

# tests/ is on sys.path when pytest runs from tests/, but be robust:
sys.path.insert(0, str(Path(__file__).parent))

from regression.anchor import run_and_summarize  # noqa: E402
from regression.short_summary import METRIC_KEYS  # noqa: E402
from regression.parity import parity_gate  # noqa: E402
from regression.compare import compute_drift  # noqa: E402


# --- Anchor smoke test --------------------------------------------------------

@pytest.mark.skip(reason='M0 anchor (hybrid + waning) needs M2+ features; superseded by anchor_m1 in M1.')
def test_anchor_runs():
    short = run_and_summarize()
    missing = set(METRIC_KEYS) - set(short.keys())
    assert not missing, f'short summary missing keys: {missing}'
    for k in METRIC_KEYS:
        assert isinstance(short[k], float), f'{k} is not a float: {type(short[k])}'
    assert short['cum_infections'] > 0, \
        f'cum_infections should be positive, got {short["cum_infections"]}'
    return short


# --- Unit tests for tests/regression/parity.py:parity_gate --------------------

def _rows(**series):
    """Build per-seed row dicts from {metric: [values]} columns of equal length."""
    n = len(next(iter(series.values())))
    return [{k: series[k][i] for k in series} for i in range(n)]


def test_parity_gate_overlapping_passes():
    v3 = _rows(cum_infections=[100.0, 102.0, 98.0, 101.0])
    v4 = _rows(cum_infections=[101.0, 99.0, 103.0, 100.0])
    failures = parity_gate(v4, v3, z_threshold=3.0)
    assert failures == [], f'expected no failures, got {failures}'


def test_parity_gate_separated_fails():
    v3 = _rows(cum_infections=[100.0, 102.0, 98.0, 101.0])
    v4 = _rows(cum_infections=[200.0, 202.0, 198.0, 201.0])
    failures = parity_gate(v4, v3, z_threshold=3.0)
    names = [name for name, z in failures]
    assert 'cum_infections' in names, f'expected cum_infections to fail, got {failures}'


def test_parity_gate_degenerate_equal_passes():
    v3 = _rows(x=[5.0, 5.0, 5.0])
    v4 = _rows(x=[5.0, 5.0, 5.0])
    assert parity_gate(v4, v3) == []


def test_parity_gate_degenerate_unequal_fails_inf():
    v3 = _rows(x=[5.0, 5.0, 5.0])
    v4 = _rows(x=[7.0, 7.0, 7.0])
    failures = parity_gate(v4, v3)
    assert failures and failures[0][0] == 'x'
    assert failures[0][1] == float('inf')


def test_parity_gate_skips_bookkeeping_keys():
    skip = frozenset({'_seed'})
    v3 = _rows(x=[1.0, 1.0], _seed=[0.0, 1.0])
    v4 = _rows(x=[1.0, 1.0], _seed=[100.0, 200.0])  # _seed differs wildly but is skipped
    assert parity_gate(v4, v3, skip_keys=skip) == []


# --- Unit tests for tests/regression/compare.py:compute_drift -----------------

def test_compute_drift_within_threshold():
    baseline = {'a': 100.0, 'b': 50.0}
    current  = {'a': 105.0, 'b': 49.0}
    rows = compute_drift(baseline, current, threshold=0.10)
    by_key = {r['key']: r for r in rows}
    assert by_key['a']['rel_diff'] == 0.05
    assert by_key['a']['over_threshold'] is False
    assert by_key['b']['rel_diff'] == -0.02
    assert by_key['b']['over_threshold'] is False


def test_compute_drift_over_threshold():
    rows = compute_drift({'a': 100.0}, {'a': 120.0}, threshold=0.10)
    assert rows[0]['rel_diff'] == 0.20
    assert rows[0]['over_threshold'] is True


def test_compute_drift_zero_baseline_flagged():
    rows = compute_drift({'a': 0.0}, {'a': 1.0}, threshold=0.10)
    assert rows[0]['rel_diff'] is None
    assert rows[0]['abs_diff'] == 1.0
    assert rows[0]['over_threshold'] is True


def test_compute_drift_skips_keys_missing_from_current():
    rows = compute_drift({'a': 1.0, 'b': 2.0}, {'a': 1.05})  # 'b' missing
    assert [r['key'] for r in rows] == ['a']
