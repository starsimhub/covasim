# Covasim v4.0 — M0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **CRITICAL — commit discipline.** This is a local effort with a **pause-for-review-and-commit** cadence. The assistant (Claude) **prepares and stages** each piece of work and then **pauses for Cliff Kerr (the Covasim author) to review and commit**. **The assistant NEVER runs `git commit` and NEVER runs `git push`.** Where this plan reaches a check-in boundary it says **PAUSE FOR CLIFF** and lists what to stage; it does not contain `git commit` commands. Check-ins happen 2–5 times across M0 (the five boundaries below).

**Goal:** Stand up the self-contained, multi-seed z-score regression harness on `starsim-port` so every later milestone has a deterministic v3.1.8-vs-v4 parity gate, an anchor scenario, a lightweight drift CLI, the `_v2_legacy` quarantine scaffold, a runnable stub `cv.Sim(ss.Sim)`, and CI smoke coverage — without landing any real migration code in `covasim/`.

**Architecture:** A small importable package under `tests/regression/` forms the harness. `anchor.py` defines the pinned vanilla scenario; `short_summary.py` builds the flat metric dict; `parity.py` is the z-score gate helper (ported ~verbatim from hpvsim); `multi_seed_v3.py` sweeps the anchor across N seeds in a frozen v3.1.8 env (gitignored output); `compare.py` is the fast one-seed ±10% drift CLI with a no-baseline mode. `tests/test_m0_parity.py` is the `@pytest.mark.slow` z-score release gate; `tests/test_regression_harness.py` holds fast unit tests. CI runs the existing pytest suite plus a one-line no-baseline invocation of `compare.py`. A trivial stub `cv.Sim(ss.Sim)` proves the continuous-runnability invariant. The `covasim/_v2_legacy/` + `tests/_legacy/` quarantines are scaffolded empty.

**Tech Stack:** Python 3.9–3.13, pytest + pytest-xdist, covasim (v3.1.8 on `starsim-port`), starsim 3.3.x, sciris, GitHub Actions.

**Authority:** the M0 section of `MIGRATION_PLAN.md` (Demo, Acceptance test, Sub-tasks, and the pinned anchor + summary set in its final paragraph). This plan implements exactly that scope. The design rationale is in `docs/superpowers/specs/2026-05-29-covasim-m0-foundation-design.md`.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `tests/regression/__init__.py` | Create | Empty; makes `tests/regression/` an importable package |
| `tests/regression/short_summary.py` | Create | `build_summary(sim)` → flat `{metric: float}`; `METRIC_KEYS`, `SKIP_KEYS` |
| `tests/regression/anchor.py` | Create | `PARS` + `make_sim()` + `run_and_summarize()` + `__main__`; the pinned vanilla anchor |
| `tests/regression/parity.py` | Create | `_mean_se` + `parity_gate(z_threshold=3.0)`; ported ~verbatim from hpvsim |
| `tests/regression/multi_seed_v3.py` | Create | CLI: sweep anchor across N seeds in a v3.1.8 env → gitignored `v3_seeds_n{N}.json` |
| `tests/regression/multi_seed_v4.py` | Create | (optional) same sweep in-env → gitignored `v4_seeds_n{N}.json` |
| `tests/regression/compare.py` | Create | `compute_drift()` pure fn + CLI: one-seed ±10% drift table; no-baseline mode |
| `tests/regression/README.md` | Create | Full usage doc: anchor pars, baseline-generation, parity + drift workflow, gate behavior |
| `tests/test_m0_parity.py` | Create | `@pytest.mark.slow` z-score release gate; skips when baseline absent |
| `tests/test_regression_harness.py` | Create | Fast unit tests: `parity_gate`, `compute_drift`, anchor smoke |
| `tests/README.md` | Modify | Append one paragraph pointing to `tests/regression/README.md` |
| `covasim/_v2_legacy/__init__.py` | Create | Pure-docstring quarantine marker, NO imports (empty in M0) |
| `tests/_legacy/__init__.py` | Create | Quarantine marker for v3 tests exercising removed APIs (empty in M0) |
| `covasim/_v4.py` | Create | Trivial stub `cv.Sim(ss.Sim)` that runs (continuous-runnability invariant) |
| `tests/test_v4_stub.py` | Create | Fast test: `cv.v4.Sim().run()` returns results |
| `.gitignore` | Modify | Add `tests/regression/v3_seeds_n*.json` and `tests/regression/v4_seeds*.json` |
| `.github/workflows/tests.yaml` | Modify | Add a no-baseline `compare.py` smoke step after the pytest step |
| `pyproject.toml`/`setup.py` | Verify | Confirm `starsim` is a dependency (add if missing — escalate, don't guess version) |

The generated sweep JSON (`tests/regression/v3_seeds_n*.json`, `v4_seeds*.json`) is gitignored and stays local-only.

---

## Task 0: Scaffold + gitignore + quarantine (CHECK-IN 1)

**Files:** `.gitignore` (modify), `covasim/_v2_legacy/__init__.py` (create), `tests/_legacy/__init__.py` (create). Work happens on the existing `starsim-port` branch — no branch is created.

- [ ] **Step 1: Confirm the starting point**

```bash
git -C /home/cliffk/idm/covasim status --short
git -C /home/cliffk/idm/covasim branch --show-current
git -C /home/cliffk/idm/covasim rev-parse --short HEAD
```

Expected: on `starsim-port` (the branch Cliff created off `main`). Record the HEAD short SHA for the report. The tree already carries the uncommitted planning docs (`MIGRATION_PLAN.md`, `docs/superpowers/`) awaiting Cliff's review.

- [ ] **Step 2: Confirm the existing `starsim-port` branch is checked out**

The `starsim-port` branch already exists (created off `main` by Cliff) and is the current branch. The assistant does **not** create or switch branches, and **never commits or pushes** — just verify:

```bash
git -C /home/cliffk/idm/covasim branch --show-current   # expect: starsim-port
```

If this does not print `starsim-port`, stop and ask Cliff rather than creating or switching branches.

- [ ] **Step 3: Add the gitignore entries for generated baselines**

Append to the end of `.gitignore` (use `Edit` to append after the final `node_modules` line):

```
# Regression baselines (regenerate from a v3.1.8 env via tests/regression/multi_seed_v3.py)
tests/regression/v3_seeds_n*.json
tests/regression/v4_seeds*.json
```

- [ ] **Step 4: Scaffold the `covasim/_v2_legacy/` quarantine (empty)**

Create `covasim/_v2_legacy/__init__.py` with a pure docstring and **no imports**:

```python
"""Quarantine for Covasim v3 modules during the v4.0 Starsim port.

During the port, v3 modules that the current milestone has not yet replaced are
moved here as a porting reference. Active code in ``covasim/`` NEVER imports from
this package — it exists purely so the v3 implementation stays readable alongside
the v4 reimplementation. This package is deleted wholesale at M10.

Empty in M0: no migration code has landed yet, so nothing is quarantined.
"""
```

- [ ] **Step 5: Scaffold the `tests/_legacy/` quarantine (empty)**

Create `tests/_legacy/__init__.py`:

```python
"""Quarantine for Covasim v3 tests that exercise removed/replaced v4 APIs.

When a milestone replaces a v3 subsystem, the v3 tests that exercise its removed
API move here so they neither run nor block CI, while staying available as a
reference for what behavior the v4 port must preserve. Deleted wholesale at M10.

Empty in M0: nothing has been removed yet.
"""
```

Note: `tests/_legacy/` holds `test_*.py` files in later milestones. CI's `pytest test_*.py` glob runs from `tests/` and does NOT recurse into subdirectories, so quarantined tests there are not collected — that is the intended behavior.

- [ ] **Step 6: Verify the working tree**

```bash
git -C /home/cliffk/idm/covasim status --short
git -C /home/cliffk/idm/covasim check-ignore tests/regression/v3_seeds_n30.json   # expect the path echoed back
```

Expected: `.gitignore` modified; `covasim/_v2_legacy/__init__.py` and `tests/_legacy/__init__.py` untracked; the check-ignore probe prints the path (confirming the ignore rule works even though the dir doesn't exist yet).

- [ ] **PAUSE FOR CLIFF (check-in 1).** Report the gitignore edit and the two quarantine `__init__.py` files (and note the planning docs already in the tree from the planning step). Suggested staging: `git add .gitignore covasim/_v2_legacy/__init__.py tests/_legacy/__init__.py`. Do NOT commit. Wait for Cliff to review and commit before continuing.

---

## Task 1: Short-summary builder + anchor (TDD via smoke test)

**Files:**
- Create: `tests/regression/__init__.py`
- Create: `tests/regression/short_summary.py`
- Create: `tests/regression/anchor.py`
- Create: `tests/test_regression_harness.py` (anchor smoke test at this stage; `parity`/`compute_drift` unit tests come in Tasks 2/4)

- [ ] **Step 1: Write the failing anchor smoke test**

Create `tests/test_regression_harness.py`:

```python
"""Tests for the v3.1.8 -> v4.0 regression harness.

Fast unit + smoke tests for the harness machinery:
  - anchor smoke test (this task)
  - parity_gate unit tests (Task 2)
  - compute_drift unit tests (Task 4)

The heavy multi-seed z-score release gate lives in tests/test_m0_parity.py.
"""

import sys
from pathlib import Path

# tests/ is on sys.path when pytest runs from tests/, but be robust:
sys.path.insert(0, str(Path(__file__).parent))

from regression.anchor import run_and_summarize  # noqa: E402
from regression.short_summary import METRIC_KEYS  # noqa: E402


def test_anchor_runs():
    short = run_and_summarize()
    missing = set(METRIC_KEYS) - set(short.keys())
    assert not missing, f'short summary missing keys: {missing}'
    for k in METRIC_KEYS:
        assert isinstance(short[k], float), f'{k} is not a float: {type(short[k])}'
    assert short['cum_infections'] > 0, \
        f'cum_infections should be positive, got {short["cum_infections"]}'
    return short
```

- [ ] **Step 2: Run the test to verify it fails with ImportError**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_regression_harness.py -v
```

Expected: collection error / `ModuleNotFoundError: No module named 'regression'`.

- [ ] **Step 3: Create the package init**

Create `tests/regression/__init__.py` as an empty file (zero bytes).

- [ ] **Step 4: Create `tests/regression/short_summary.py`**

The metric set and sources are pinned by `MIGRATION_PLAN.md` and the design spec. Gated metrics come from `sim.summary` (the flat objdict of `result_keys()` at the last timepoint, the same dict `tests/baseline.json` stores) plus two peak metrics from the `sim.results` time series.

```python
"""Short-summary builder for the M0 vanilla anchor.

Returns a flat {metric: float} dict pinned to Covasim's own sim.summary (the
end-of-run flat dict of result_keys()) plus two epidemic-shape metrics computed
from the sim.results time series (peak prevalence, peak n_infectious).

Testing/vaccination cumulatives (cum_tests/cum_diagnoses/cum_doses) are omitted
because the M0 anchor has no interventions, so they are identically zero; they
re-enter via the M5/M6 capability anchors. r_eff is omitted from the gated set
because it is version-sensitive (Covasim's own test_regression.py skips it).
"""

# Gated metrics: subject to the |z| < 3 parity gate.
METRIC_KEYS = (
    'cum_infections',
    'cum_reinfections',
    'cum_symptomatic',
    'cum_severe',
    'cum_critical',
    'cum_deaths',
    'peak_prevalence',
    'peak_n_infectious',
    'prevalence',
    'incidence',
)

# Bookkeeping keys written by the sweeps but ignored by the gate.
SKIP_KEYS = frozenset({'_seed', '_total_pop', 'n_alive'})


def build_summary(sim):
    """Return the M0 anchor short summary as a flat dict of floats.

    Args:
        sim (cv.Sim): a run Covasim sim (sim.summary populated, sim.results ready).

    Returns:
        dict of {metric_name: float}: the gated metrics in METRIC_KEYS plus the
        bookkeeping key 'n_alive'.
    """
    summary = sim.summary  # objdict of result_keys() at the last timepoint
    out = {
        'cum_infections':   float(summary['cum_infections']),
        'cum_reinfections': float(summary['cum_reinfections']),
        'cum_symptomatic':  float(summary['cum_symptomatic']),
        'cum_severe':       float(summary['cum_severe']),
        'cum_critical':     float(summary['cum_critical']),
        'cum_deaths':       float(summary['cum_deaths']),
        'prevalence':       float(summary['prevalence']),
        'incidence':        float(summary['incidence']),
        # Epidemic shape: peaks over the full time series (not in the end-of-run summary).
        'peak_prevalence':   float(sim.results['prevalence'].values.max()),
        'peak_n_infectious': float(sim.results['n_infectious'].values.max()),
        # Bookkeeping (skipped by the gate, kept for diagnostics):
        'n_alive': float(summary['n_alive']),
    }
    return out
```

Note on `.values`: a Covasim `Result` is array-like; `.values` is its underlying numpy array. If `sim.results['prevalence'].values` raises in the installed version, fall back to `np.asarray(sim.results['prevalence'])`. Verify in Step 6 and adjust if needed — do not guess.

- [ ] **Step 5: Create `tests/regression/anchor.py`**

```python
"""M0 anchor scenario for the v3.1.8 -> v4.0 migration regression harness.

Representative-but-clean vanilla Covasim sim: hybrid population, waning immunity
ON, no interventions, no analyzers. This isolates core-dynamics + immunity drift
from intervention-port bugs. Intervention/vaccine anchors are added in M5/M6.

Pinned anchor pars. Do NOT change without coordinating with the gitignored
v3.1.8 baselines (regenerate them via multi_seed_v3.py if you do).

Run as a script to print the summary:
    python tests/regression/anchor.py
"""
import sys
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary  # noqa: E402


PARS = dict(
    pop_size     = 20_000,    # large enough that per-seed CV is moderate
    pop_infected = 100,
    pop_type     = 'hybrid',  # household/school/work + the random layer
    n_days       = 120,       # full epidemic peak + decline
    use_waning   = True,      # exercises the NAb / immunity core
    rand_seed    = 0,         # base seed only; the sweep overrides it 0..N-1
    verbose      = 0,
    # NO interventions, NO analyzers.
)


def make_sim(**kwargs):
    """Build (but do not run) the M0 anchor sim. kwargs override PARS (e.g. rand_seed)."""
    pars = sc.mergedicts(sc.dcp(PARS), kwargs)
    return cv.Sim(pars)


def run_and_summarize(**kwargs):
    """Run the M0 anchor sim and return the short summary dict."""
    sim = make_sim(**kwargs)
    sim.run()
    return build_summary(sim)


if __name__ == '__main__':
    short = run_and_summarize()
    print('M0 anchor short summary (vanilla hybrid, waning ON):')
    for k, v in short.items():
        print(f'  {k:<24} {v:>14.4g}')
```

- [ ] **Step 6: Run the smoke test to verify it passes**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_regression_harness.py::test_anchor_runs -v
```

Expected: PASS. Wall-clock ~10–30s (20k-agent, 120-day, single-variant waning sim). If far slower, do NOT change the anchor pars — escalate as a concern in the report.

- [ ] **Step 7: Sanity-check the `__main__` runner**

```bash
python /home/cliffk/idm/covasim/tests/regression/anchor.py
```

Expected: a printed table of the 10 metric keys plus `n_alive`. Plausibility checks: `cum_infections` positive and a sizeable fraction of 20k; `peak_prevalence` between 0 and ~0.5; `n_alive` near 20000 minus deaths. Confirm `peak_prevalence`/`peak_n_infectious` printed without error (this validates the `.values.max()` access from Step 4; if it errored, switch to `np.asarray(...).max()` and re-run).

- [ ] **PAUSE FOR CLIFF (check-in 2 starts here; held until Task 3).** Continue to Tasks 2–3 before the next pause — the harness package (anchor + short-summary + parity + sweeps) is one logical reviewable unit. Stage at the Task 3 pause.

---

## Task 2: Parity gate helper (TDD)

**Files:**
- Create: `tests/regression/parity.py`
- Modify: `tests/test_regression_harness.py` (add `parity_gate` unit tests)

- [ ] **Step 1: Add failing `parity_gate` unit tests**

Append to `tests/test_regression_harness.py`:

```python


# --- Unit tests for tests/regression/parity.py:parity_gate --------------------

from regression.parity import parity_gate  # noqa: E402


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
```

- [ ] **Step 2: Run to verify they fail with ImportError**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_regression_harness.py -v
```

Expected: collection error / `ModuleNotFoundError: No module named 'regression.parity'`.

- [ ] **Step 3: Create `tests/regression/parity.py` (ported ~verbatim from hpvsim)**

This is hpvsim's `tests/regression/parity.py` with argument names retargeted from v2/v3 to v3/v4. The z-formula and the two degenerate-distribution policies are unchanged.

```python
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
```

- [ ] **Step 4: Run all tests to confirm they pass**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_regression_harness.py -v
```

Expected: `test_anchor_runs` plus the 5 `test_parity_gate_*` tests PASS.

---

## Task 3: Multi-seed sweep generators

**Files:**
- Create: `tests/regression/multi_seed_v3.py`
- Create: `tests/regression/multi_seed_v4.py` (optional but recommended for symmetry)

- [ ] **Step 1: Create `tests/regression/multi_seed_v3.py`**

This is the v3.1.8 baseline generator — intended to run **in a frozen v3.1.8 environment** (a separate venv). It sweeps the anchor across N seeds and writes a gitignored JSON list of per-seed summaries. It imports `PARS` and `build_summary` from the harness; since the harness uses only the public Covasim API (`cv.Sim`, `sim.summary`, `sim.results`), the same files work under both v3.1.8 and v4.

```python
"""Multi-seed sweep of the M0 anchor, to be run from a FROZEN v3.1.8 env.

Sweeps the anchor across N seeds and writes a JSON list of per-seed short-summary
dicts (the v3.1.8 baseline for the parity gate). The output is gitignored.

Run from a v3.1.8 env at the repo root:
    "<v3.1.8 env>/python" tests/regression/multi_seed_v3.py --n 30

DO NOT commit the output. The v4 parity gate (tests/test_m0_parity.py) loads it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor import PARS  # noqa: E402
from short_summary import build_summary  # noqa: E402


def run_seed(seed):
    """Run the anchor at one seed; return build_summary + bookkeeping keys."""
    pars = sc.mergedicts(sc.dcp(PARS), dict(rand_seed=int(seed)))
    sim = cv.Sim(pars)
    sim.run()
    summary = build_summary(sim)
    summary['_seed'] = int(seed)
    summary['_total_pop'] = float(sim.summary['n_alive'])
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate the v3.1.8 multi-seed baseline.')
    p.add_argument('--n', type=int, default=30, help='Number of seeds (default 30).')
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=None,
                   help='Output path (default tests/regression/v3_seeds_n{N}.json).')
    args = p.parse_args(argv)

    out = args.out or (Path(__file__).resolve().parent / f'v3_seeds_n{args.n}.json')
    seeds = list(range(args.start_seed, args.start_seed + args.n))
    rows = []
    t0 = time.time()
    print(f'Sweeping anchor over {args.n} seeds with covasim {cv.__version__} ...')
    for seed in seeds:
        ts = time.time()
        row = run_seed(seed)
        rows.append(row)
        print(f'  seed {seed}: done in {time.time()-ts:.1f}s '
              f'(cum_infections={row["cum_infections"]:.0f})')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'Wrote {len(rows)} seed summaries to {out} in {time.time()-t0:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Create `tests/regression/multi_seed_v4.py` (optional)**

A thin wrapper that defaults to the `v4_seeds_n{N}.json` name. The pytest gate generates v4 seeds in-process, so this file is for ad-hoc local diffing only.

```python
"""Multi-seed sweep of the M0 anchor in the CURRENT (v4) env, for ad-hoc diffing.

Identical sweep to multi_seed_v3.py but defaults to writing v4_seeds_n{N}.json.
The pytest parity gate (tests/test_m0_parity.py) generates v4 seeds in-process
and does not require this file; it exists for manual local comparison.

    python tests/regression/multi_seed_v4.py --n 10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_seed_v3 import main  # noqa: E402


if __name__ == '__main__':
    # Reuse the v3 sweep machinery; just change the default output name.
    argv = sys.argv[1:]
    if not any(a.startswith('--out') for a in argv):
        # Derive N from --n (default 30) for the default v4 filename.
        n = 30
        if '--n' in argv:
            n = int(argv[argv.index('--n') + 1])
        argv = argv + ['--out', str(Path(__file__).resolve().parent / f'v4_seeds_n{n}.json')]
    sys.exit(main(argv))
```

- [ ] **Step 3: Smoke-run a tiny v4 sweep in the current env**

This validates the generator end-to-end (the output will be gitignored). Use a small `--n` to keep it fast.

```bash
python /home/cliffk/idm/covasim/tests/regression/multi_seed_v3.py --n 2 --out /tmp/v3_smoke.json
python -c "import json; d=json.load(open('/tmp/v3_smoke.json')); print(len(d), sorted(d[0]))"
```

Expected: 2 rows; each row has the 10 `METRIC_KEYS` plus `n_alive`, `_seed`, `_total_pop`. (Running the v3 generator in the v4 env here is fine — it just exercises the script; the *real* v3.1.8 baseline is generated later from a frozen v3.1.8 env.)

- [ ] **Step 4: Confirm the default output path is gitignored**

```bash
git -C /home/cliffk/idm/covasim check-ignore tests/regression/v3_seeds_n30.json tests/regression/v4_seeds_n10.json
```

Expected: both paths echoed back (ignored).

- [ ] **PAUSE FOR CLIFF (check-in 2).** Report the harness package: `tests/regression/{__init__.py, short_summary.py, anchor.py, parity.py, multi_seed_v3.py, multi_seed_v4.py}` and the anchor + parity unit tests in `tests/test_regression_harness.py`. Suggested staging: `git add tests/regression/__init__.py tests/regression/short_summary.py tests/regression/anchor.py tests/regression/parity.py tests/regression/multi_seed_v3.py tests/regression/multi_seed_v4.py tests/test_regression_harness.py`. Do NOT commit. Wait for Cliff.

---

## Task 4: Comparison CLI (development gate) + drift unit tests (TDD)

**Files:**
- Create: `tests/regression/compare.py`
- Modify: `tests/test_regression_harness.py` (add `compute_drift` unit tests)

- [ ] **Step 1: Add failing `compute_drift` unit tests**

Append to `tests/test_regression_harness.py`:

```python


# --- Unit tests for tests/regression/compare.py:compute_drift -----------------

from regression.compare import compute_drift  # noqa: E402


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
```

- [ ] **Step 2: Run to verify they fail with ImportError**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_regression_harness.py -v -k compute_drift
```

Expected: `ModuleNotFoundError: No module named 'regression.compare'`.

- [ ] **Step 3: Create `tests/regression/compare.py` (pure fn + CLI)**

```python
"""Compare one v4 run of the M0 anchor against a stored one-seed v3.1.8 snapshot.

This is the lightweight DEVELOPMENT gate: a per-metric +/-10% relative-drift
table, always exit 0, informational only. The hard scientific gate is the
multi-seed z-score parity gate in tests/test_m0_parity.py.

No-baseline mode: if the snapshot file is missing, print a notice and exit 0
WITHOUT running the anchor (CLI-integrity check; this is the mode CI runs). The
anchor-runs check is the pytest smoke test's job.

Usage:
    python tests/regression/compare.py
    python tests/regression/compare.py --baseline path/to/snapshot.json
    python tests/regression/compare.py --threshold 0.05
"""
import argparse
import json
import sys
from pathlib import Path

THRESHOLD = 0.10  # 10% relative drift

# Default snapshot: a single-seed v3.1.8 summary written by --save-snapshot.
DEFAULT_BASELINE = Path(__file__).resolve().parent / 'anchor_snapshot.json'


def compute_drift(baseline_summary, current_summary, threshold=THRESHOLD):
    """Compute per-key relative-drift records.

    Args:
        baseline_summary (dict): {key: number} stored v3.1.8 one-seed snapshot.
        current_summary (dict): {key: number} the current run's summary.
        threshold (float): relative-drift threshold (default 0.10 = 10%).

    Returns:
        list of dicts with keys: key, baseline, current, abs_diff, rel_diff,
        over_threshold. Keys in baseline but missing from current are skipped.
        If the baseline value is zero, rel_diff is None and over_threshold is True.
    """
    rows = []
    for k in baseline_summary.keys():
        if k not in current_summary:
            continue
        b = float(baseline_summary[k])
        c = float(current_summary[k])
        abs_diff = c - b
        if b == 0:
            rel_diff = None
            over = True
        else:
            rel_diff = abs_diff / b
            over = abs(rel_diff) > threshold
        rows.append({
            'key': k, 'baseline': b, 'current': c, 'abs_diff': abs_diff,
            'rel_diff': rel_diff, 'over_threshold': over,
        })
    return rows


def format_table(rows, threshold=THRESHOLD):
    """Format drift rows as a printable table (str)."""
    out = [f'{"key":<24} {"baseline":>14} {"current":>14} {"rel_diff":>10} {"over":>6}',
           '-' * 72]
    for r in rows:
        rel = f'{r["rel_diff"]*100:+.2f}%' if r['rel_diff'] is not None else 'n/a'
        flag = 'YES' if r['over_threshold'] else ''
        out.append(f'{r["key"]:<24} {r["baseline"]:>14.4g} {r["current"]:>14.4g} '
                   f'{rel:>10} {flag:>6}')
    n_over = sum(1 for r in rows if r['over_threshold'])
    out.append('')
    out.append(f'{n_over}/{len(rows)} keys exceed +/- {threshold*100:.0f}% relative '
               f'drift (informational; exit 0 regardless).')
    return '\n'.join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description='Compare anchor run vs. v3.1.8 snapshot.')
    p.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE,
                   help=f'One-seed snapshot JSON (default: {DEFAULT_BASELINE}).')
    p.add_argument('--threshold', type=float, default=THRESHOLD,
                   help='Relative drift threshold (default 0.10).')
    p.add_argument('--save-snapshot', action='store_true',
                   help='Run the anchor once and write the snapshot to --baseline, then exit.')
    args = p.parse_args(argv)

    # Local imports: only touch the anchor when we actually need to run it.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if args.save_snapshot:
        from anchor import run_and_summarize  # noqa: E402
        snapshot = {k: float(v) for k, v in run_and_summarize().items()}
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        with open(args.baseline, 'w') as f:
            json.dump({'summary': snapshot}, f, indent=2)
        print(f'Wrote one-seed snapshot to {args.baseline}')
        return 0

    if not args.baseline.exists():
        print(f'No baseline at {args.baseline}; skipping diff.')
        print('To create one (from a v3.1.8 env): '
              'python tests/regression/compare.py --save-snapshot')
        return 0

    from anchor import run_and_summarize  # noqa: E402
    with open(args.baseline) as f:
        baseline_summary = json.load(f)['summary']
    current_summary = {k: float(v) for k, v in run_and_summarize().items()}
    print(format_table(compute_drift(baseline_summary, current_summary,
                                     threshold=args.threshold), threshold=args.threshold))
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

Note: `compare.py` is gitignore-compatible — `anchor_snapshot.json` is a one-seed dev artifact. Add it to the gitignore pattern if Cliff wants it ignored; by default it sits outside the `v[34]_seeds_*` patterns. Flag this as an open question in the report (whether to also gitignore `anchor_snapshot.json`).

- [ ] **Step 4: Run all harness tests**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_regression_harness.py -v
```

Expected: all pass — `test_anchor_runs`, 5× `test_parity_gate_*`, 4× `test_compute_drift_*`.

- [ ] **Step 5: Exercise the no-baseline CLI path**

```bash
python /home/cliffk/idm/covasim/tests/regression/compare.py --baseline /tmp/does_not_exist.json
```

Expected (<1s, does NOT run the anchor):

```
No baseline at /tmp/does_not_exist.json; skipping diff.
To create one (from a v3.1.8 env): python tests/regression/compare.py --save-snapshot
```

Exit 0.

- [ ] **Step 6: Exercise the full snapshot+compare round-trip locally**

```bash
python /home/cliffk/idm/covasim/tests/regression/compare.py --save-snapshot --baseline /tmp/anchor_snap.json
python /home/cliffk/idm/covasim/tests/regression/compare.py --baseline /tmp/anchor_snap.json
```

Expected: snapshot written, then a drift table where every `rel_diff` is exactly `+0.00%` (same code, same seed) and the footer reads `0/N keys exceed +/- 10% relative drift`. If any row shows non-zero drift, the anchor is non-deterministic at fixed seed — escalate as a real bug.

---

## Task 5: The z-score release gate

**Files:**
- Create: `tests/test_m0_parity.py`

- [ ] **Step 1: Create `tests/test_m0_parity.py`**

Ports hpvsim's `test_m01_short_summary_parity.py`, retargeted to Covasim and the M0 anchor. `@pytest.mark.slow` so the existing 5-minute PR job (which does not opt into slow tests) skips it; it runs in the nightly/optional job or locally.

```python
"""M0 acceptance gate: multi-seed mean parity vs. the v3.1.8 anchor baseline.

Runs N_V4_SEEDS v4 seeds of the M0 anchor, then gates each pinned metric on

    z = (v4_mean - v3_mean) / sqrt(v3_SE^2 + v4_SE^2)

failing any metric with |z| >= Z_THRESHOLD. The v3.1.8 baseline is the gitignored
multi-seed sweep at tests/regression/v3_seeds_n{M}.json, regenerated via
`python tests/regression/multi_seed_v3.py --n 30` from a FROZEN v3.1.8 env.

Marked slow so the 5-minute PR job skips it; run it locally or in the nightly job:
    cd tests && pytest test_m0_parity.py -m slow -v
"""
import json
import sys
from pathlib import Path

import pytest
import sciris as sc

sys.path.insert(0, str(Path(__file__).parent))
from regression.anchor import PARS, make_sim  # noqa: E402
from regression.short_summary import build_summary, SKIP_KEYS  # noqa: E402
from regression.parity import parity_gate  # noqa: E402

N_V4_SEEDS = 10                 # 10 v4 seeds vs 30 v3 seeds (hpvsim's committed ratio)
M_V3_SEEDS = 30
Z_THRESHOLD = 3.0
BASELINE_PATH = Path(__file__).parent / 'regression' / f'v3_seeds_n{M_V3_SEEDS}.json'


def _run_v4_seeds(n, start_seed=0):
    rows = []
    for seed in range(start_seed, start_seed + n):
        sim = make_sim(rand_seed=int(seed))
        sim.run()
        rows.append(build_summary(sim))
    return rows


@pytest.mark.slow
def test_m0_anchor_parity():
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'Missing v3.1.8 M0 baseline at {BASELINE_PATH}. Regenerate via '
            f'`python tests/regression/multi_seed_v3.py --n {M_V3_SEEDS}` from a '
            f'frozen v3.1.8 covasim env.'
        )
    v3_rows = json.loads(BASELINE_PATH.read_text())
    v4_rows = _run_v4_seeds(N_V4_SEEDS)

    failures = parity_gate(v4_rows, v3_rows, z_threshold=Z_THRESHOLD, skip_keys=SKIP_KEYS)
    if failures:
        details = '\n'.join(f'  {name:<24} z={z:+.2f}' for name, z in failures)
        pytest.fail(
            f'M0 anchor parity drift exceeds |z|>={Z_THRESHOLD} on {len(failures)} '
            f'metrics (v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):\n{details}'
        )
    return v4_rows
```

- [ ] **Step 2: Confirm the gate SKIPS cleanly when the baseline is absent**

The real v3.1.8 baseline isn't generated yet (it needs a frozen v3.1.8 env), so the gate must skip, not fail.

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_m0_parity.py -m slow -v
```

Expected: `1 skipped` with the "Missing v3.1.8 M0 baseline" message. (Without `-m slow`, confirm the existing default config either deselects or still skips it — see Step 3.)

- [ ] **Step 3: Confirm the slow gate does not run in the default PR invocation**

The existing CI runs `pytest -v test_*.py unittests/test_*.py -n auto` with no `-m` selection. Check whether `pytest.ini`/`pyproject.toml` registers the `slow` marker and whether it is deselected by default.

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_m0_parity.py -v 2>&1 | tail -20
grep -n "markers\|slow\|addopts" /home/cliffk/idm/covasim/tests/pytest.ini /home/cliffk/idm/covasim/pyproject.toml 2>/dev/null
```

If `slow` is not a registered marker, `COVASIM_WARNINGS=error` under `run_tests` could turn the `PytestUnknownMarkWarning` into a failure. To stay safe, **register the marker**: add to `tests/pytest.ini` under `[pytest]`:

```ini
markers =
    slow: long-running multi-seed parity gates (deselected from the fast PR job)
```

The default PR job still *collects and runs* `test_m0_parity.py`, but since the baseline is absent the single test SKIPS in <1s — acceptable for the 5-minute budget. (The heavy path only triggers when a developer has generated the baseline and runs `-m slow` locally / in the nightly job.) Confirm the marker registration removes any warning, then re-run Step 2.

---

## Task 6: Stub `cv.Sim(ss.Sim)` (continuous-runnability invariant)

**Files:**
- Verify/Modify: `pyproject.toml` or `setup.py` (ensure `starsim` is a dependency)
- Create: `covasim/_v4.py`
- Create: `tests/test_v4_stub.py`

- [ ] **Step 1: Confirm `starsim` is importable and a declared dependency**

```bash
python -c "import starsim as ss; print('starsim', ss.__version__)"
grep -n "starsim" /home/cliffk/idm/covasim/pyproject.toml /home/cliffk/idm/covasim/setup.py 2>/dev/null
```

Expected: `starsim 3.3.x` prints. If `starsim` is not a declared dependency, add it to the install requirements (pin to `starsim>=3.3,<3.4` to match the migration target) — but **flag this as a change for Cliff to confirm**, do not guess a version beyond 3.3.x.

- [ ] **Step 2: Create `covasim/_v4.py` — a trivial runnable stub**

The stub must subclass `ss.Sim`, construct, and run to completion. It must NOT touch or alter the existing v3.1.8 `cv.Sim` (in `covasim/sim.py`) that the existing suite depends on. It is exposed as `cv.v4.Sim`, a *new* symbol path, so the existing `cv.Sim` is unchanged.

```python
"""Stub v4.0 Sim on the Starsim base (continuous-runnability invariant).

M0 ships NO real port code. This module exists solely to prove, from day one,
that the Starsim base imports and a Covasim-namespaced Sim subclass constructs
and runs to completion (Implementation conventions item 1: cv.Sim().run() must
return results at every commit on starsim-port). It is NOT the real port: it runs a
degenerate, disease-free sim. The real cv.Sim is built out starting in M1.

Exposed as cv.v4.Sim to avoid disturbing the existing v3.1.8 cv.Sim that the
current test suite depends on. The two coexist until M1 begins the in-place
replacement.

    import covasim as cv
    sim = cv.v4.Sim(n_agents=100).run()   # returns a run ss.Sim
"""
import starsim as ss


class Sim(ss.Sim):
    """Minimal v4 Sim stub: a disease-free Starsim sim that runs.

    Args:
        n_agents (int): number of agents (default 100, small for a fast smoke run).
        kwargs: forwarded to ss.Sim (e.g. start, stop, dt).

    **Example**::

        sim = cv.v4.Sim(n_agents=100).run()
    """

    def __init__(self, n_agents=100, **kwargs):
        super().__init__(n_agents=n_agents, **kwargs)
        return
```

- [ ] **Step 3: Expose the stub as `cv.v4` without disturbing existing imports**

Append a single guarded import to the end of `covasim/__init__.py` (after the existing flat imports). Read the file first to find the end of the import block.

```python
from . import _v4 as v4   # v4.0 port stub (cv.v4.Sim); does not affect the v3.1.8 cv.Sim
```

This adds the `cv.v4` submodule namespace only. It does **not** rebind `cv.Sim`. If importing `starsim` at package import time is undesirable (e.g. it slows `import covasim`), wrap the import in a try/except and flag the trade-off for Cliff — but the default is the plain import above, since `starsim` is now a declared dependency.

- [ ] **Step 4: Create `tests/test_v4_stub.py`**

```python
"""Continuous-runnability invariant: the v4 stub Sim constructs and runs.

M0 ships no real port, but cv.v4.Sim().run() must return results so the
invariant (MIGRATION_PLAN Implementation conventions item 1) holds from day one.
"""
import covasim as cv


def test_v4_stub_runs():
    sim = cv.v4.Sim(n_agents=100)
    sim.run()
    assert sim.results is not None, 'v4 stub sim produced no results'
    return sim


if __name__ == '__main__':
    test_v4_stub_runs()
    print('v4 stub Sim ran successfully.')
```

- [ ] **Step 5: Run the stub test and confirm the existing suite is undisturbed**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_v4_stub.py -v
python -c "import covasim as cv; print('cv.Sim is v3.1.8:', cv.Sim.__module__); print('cv.v4.Sim:', cv.v4.Sim.__module__)"
```

Expected: `test_v4_stub_runs` PASS; `cv.Sim.__module__` is `covasim.sim` (the unchanged v3.1.8 Sim) and `cv.v4.Sim.__module__` is `covasim._v4`. Then run a representative slice of the existing suite to confirm no regression from the new import:

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_sim.py -n auto -q
```

Expected: passes as before. If `import starsim` introduced any warning that `test_*` modules surface under strict warnings, escalate (do not silence globally).

- [ ] **PAUSE FOR CLIFF (check-in 3).** Report the comparison CLI + drift tests (Task 4), the z-score gate (Task 5), the marker registration, and the stub `cv.Sim` (Task 6). Suggested staging: `git add tests/regression/compare.py tests/test_m0_parity.py tests/test_regression_harness.py tests/pytest.ini covasim/_v4.py covasim/__init__.py tests/test_v4_stub.py` (and `pyproject.toml`/`setup.py` if the starsim dependency was added). Do NOT commit. Wait for Cliff.

---

## Task 7: CI workflow update

**Files:**
- Modify: `.github/workflows/tests.yaml`

- [ ] **Step 1: Add a no-baseline `compare.py` smoke step after the pytest step**

Open `.github/workflows/tests.yaml`. The existing test step is:

```yaml
      - name: Run integration tests
        working-directory: ./tests
        run: pytest -v test_*.py unittests/test_*.py -n auto --durations=0 --junitxml=test-results.xml
```

Insert a new step immediately after it (before `Publish test Results`). Use `Edit` with:

`old_string`:
```
      - name: Run integration tests
        working-directory: ./tests
        run: pytest -v test_*.py unittests/test_*.py -n auto --durations=0 --junitxml=test-results.xml
      - name: Publish test Results
```

`new_string`:
```
      - name: Run integration tests
        working-directory: ./tests
        run: pytest -v test_*.py unittests/test_*.py -n auto --durations=0 --junitxml=test-results.xml
      - name: Smoke-check regression compare CLI (no-baseline mode)
        working-directory: ./tests
        run: python regression/compare.py --baseline /tmp/no_baseline.json
      - name: Publish test Results
```

Pointing `--baseline` at a non-existent path forces the no-baseline path explicitly, so the step is guaranteed to exit in <1s without running the anchor (keeping well within the 5-minute timeout). Do **not** add the slow parity sweep here — it stays out of the PR job (a separate/nightly job is future work, out of scope for M0).

- [ ] **Step 2: Validate the YAML parses**

```bash
python -c "import yaml; yaml.safe_load(open('/home/cliffk/idm/covasim/.github/workflows/tests.yaml'))"
```

Expected: no output, exit 0. A traceback means malformed YAML — fix indentation.

- [ ] **Step 3: Locally simulate the new CI step exactly as CI runs it**

```bash
cd /home/cliffk/idm/covasim/tests && python regression/compare.py --baseline /tmp/no_baseline.json
```

Expected:
```
No baseline at /tmp/no_baseline.json; skipping diff.
To create one (from a v3.1.8 env): python tests/regression/compare.py --save-snapshot
```
Exit 0.

- [ ] **Step 4: Confirm the new test files are collected by the existing glob**

The existing CI runs `pytest -v test_*.py ...` from `tests/`. Confirm the new root-level test files are collected and the slow gate skips:

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_*.py --collect-only -q 2>&1 | grep -E "test_regression_harness|test_m0_parity|test_v4_stub"
```

Expected: all three appear in collection. Then time the fast subset to confirm it fits the budget:

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_regression_harness.py test_m0_parity.py test_v4_stub.py -n auto --durations=0
```

Expected: harness unit tests + stub sub-second; `test_anchor_runs` ~10–30s; `test_m0_anchor_parity` skipped (no baseline). Comfortably within 5 minutes.

---

## Task 8: Documentation + migration-plan pointer

**Files:**
- Create: `tests/regression/README.md`
- Modify: `tests/README.md`

- [ ] **Step 1: Create `tests/regression/README.md`**

```markdown
# Regression harness (v3.1.8 -> v4.0 migration)

This directory holds the self-contained regression harness for the Covasim v4.0
Starsim port. It does two complementary jobs:

- **Development gate (`compare.py`)** — a fast, informational one-seed +/-10%
  drift table for day-to-day porting feedback. Always exits 0; never blocks.
- **Release gate (`../test_m0_parity.py`)** — the scientific gate: a multi-seed
  z-score parity check (`|z| < 3`) comparing N v4 seeds to M v3.1.8 seeds with
  overlapping uncertainty intervals. NOT bit-for-bit (the RNG stream differs
  between v3.1.8's global numba RNG and v4's Starsim CRN).

This layers ON TOP OF Covasim's existing baseline machinery (`../baseline.json`
+ `../test_baselines.py` + `../update_baseline` + `../../covasim/regression/`),
which stays the v4-internal bit-for-bit self-consistency gate. The two answer
different questions and both stay.

## What's here

| File | Role |
|---|---|
| `anchor.py` | Pinned vanilla anchor: hybrid pop, waning ON, seed 0, no interventions. `PARS`, `make_sim()`, `run_and_summarize()`; runs as `__main__`. |
| `short_summary.py` | `build_summary(sim)` -> flat `{metric: float}` from `sim.summary` + peak metrics from `sim.results`. `METRIC_KEYS`, `SKIP_KEYS`. |
| `parity.py` | `parity_gate(v4, v3, z_threshold=3.0)` z-score helper (ported ~verbatim from hpvsim). |
| `multi_seed_v3.py` | CLI: sweep the anchor across N seeds **in a frozen v3.1.8 env** -> gitignored `v3_seeds_n{N}.json`. |
| `multi_seed_v4.py` | (optional) same sweep in-env -> gitignored `v4_seeds_n{N}.json`, for ad-hoc diffing. |
| `compare.py` | `compute_drift()` + CLI: one-seed `+/-10%` drift table; `--save-snapshot`; no-baseline mode. |
| `__init__.py` | Empty; makes this an importable package. |

## Anchor scenario

Pinned in `anchor.py:PARS`:

| Par | Value |
|---|---|
| `pop_size` | `20_000` |
| `pop_infected` | `100` |
| `pop_type` | `'hybrid'` |
| `n_days` | `120` |
| `use_waning` | `True` |
| `rand_seed` | `0` (the sweep overrides 0..N-1) |
| `verbose` | `0` |

No interventions, no analyzers. `hybrid` + `use_waning` exercises the
population-structure and NAb/immunity machinery — the highest-risk parts of the
port — without confounding from intervention ports. Intervention/vaccine anchors
are added in M5/M6.

## Pinned summary set

Gated metrics (`METRIC_KEYS`): `cum_infections`, `cum_reinfections`,
`cum_symptomatic`, `cum_severe`, `cum_critical`, `cum_deaths`, `peak_prevalence`,
`peak_n_infectious`, `prevalence`, `incidence`. The cumulative/derived metrics
come from `sim.summary`; the two peaks are computed from the `sim.results` time
series. `r_eff` is omitted from the gate (version-sensitive; Covasim's own
`test_regression.py` skips it). Bookkeeping keys `_seed`, `_total_pop`, `n_alive`
are written but skipped by the gate.

## Generating the v3.1.8 baseline (gitignored)

The baseline is a 30-seed sweep, regenerated from a FROZEN v3.1.8 environment and
never committed:

1. In a separate v3.1.8 venv (the frozen reference build):

   ```bash
   python tests/regression/multi_seed_v3.py --n 30
   ```

2. This writes `tests/regression/v3_seeds_n30.json` (gitignored).
3. Back in the v4 env, the release gate consumes it automatically.

## Running the release gate (z-score parity)

```bash
cd tests && pytest test_m0_parity.py -m slow -v
```

Runs 10 v4 seeds, loads the 30-seed v3.1.8 baseline, fails any metric with
`|z| >= 3`. **Skips cleanly** (does not fail) if the baseline JSON is absent, so
contributors without a v3.1.8 env can still run the rest of the suite.

z-formula: `z = (v4_mean - v3_mean) / sqrt(v3_SE^2 + v4_SE^2)`, `SE = std(ddof=1)/sqrt(n)`.
Degenerate distributions: zero combined spread + equal means passes; zero spread
+ unequal means fails (`z = inf`).

## Running the development gate (drift)

```bash
# From a v3.1.8 env, snapshot one seed:
python tests/regression/compare.py --save-snapshot
# From the v4 env, diff against it:
python tests/regression/compare.py
```

Output: a `key | baseline | current | rel_diff | over` table; always exit 0.
No-baseline mode (missing snapshot) prints a notice and exits without running the
anchor — this is the mode CI smoke-runs.

## When to refresh the baselines

- After a v3.1.8 patch-equivalent change lands and is forward-merged into `starsim-port`.
- After an explicit decision that drift introduced by a milestone is the new target.
- Otherwise: don't. Stable baseline = stable signal.

## CI

CI runs the pytest suite (which collects the harness unit tests + anchor smoke
test + the skipped slow gate) plus `python regression/compare.py` in no-baseline
mode (CLI-integrity only). Neither fails on drift. The heavy multi-seed sweep
runs only locally or in a future nightly job, never in the 5-minute PR job.
```

- [ ] **Step 2: Append a pointer to `tests/README.md`**

Read `tests/README.md` first, then append at the end (after the `update_baseline` section):

```markdown

## regression (v4.0 migration harness)

There is also a self-contained regression harness under `tests/regression/` used
for the v3.1.8 -> v4.0 Starsim port. It runs partly outside the standard pytest
flow and is documented in [`tests/regression/README.md`](regression/README.md).
It compares a v4 run of a pinned anchor scenario against a locally-generated,
gitignored v3.1.8 baseline: a fast informational `+/-10%` drift CLI
(`compare.py`) and a multi-seed z-score parity gate (`test_m0_parity.py`,
`|z| < 3`). It layers on top of `baseline.json` / `test_baselines.py` rather than
replacing them.
```

- [ ] **PAUSE FOR CLIFF (check-in 4).** Report the CI edit (Task 7) and the docs (Task 8). Suggested staging: `git add .github/workflows/tests.yaml tests/regression/README.md tests/README.md`. Do NOT commit. Wait for Cliff.

---

## Task 9: Confirm the plan/spec docs are present (CHECK-IN 5)

**Files:** (already written by the planning step that produced this document)
- `docs/superpowers/plans/2026-05-29-covasim-m0-foundation.md` (this file)
- `docs/superpowers/specs/2026-05-29-covasim-m0-foundation-design.md`

- [ ] **Step 1: Confirm both docs exist and are linked from `MIGRATION_PLAN.md`**

```bash
ls -l /home/cliffk/idm/covasim/docs/superpowers/plans/2026-05-29-covasim-m0-foundation.md \
      /home/cliffk/idm/covasim/docs/superpowers/specs/2026-05-29-covasim-m0-foundation-design.md
grep -n "2026-05-29-covasim-m0-foundation" /home/cliffk/idm/covasim/MIGRATION_PLAN.md
```

Expected: both files present; `MIGRATION_PLAN.md`'s "Linked documents" already references both paths (it does — lines ~284–285).

- [ ] **PAUSE FOR CLIFF (check-in 5).** Report the plan + spec docs. Suggested staging: `git add docs/superpowers/plans/2026-05-29-covasim-m0-foundation.md docs/superpowers/specs/2026-05-29-covasim-m0-foundation-design.md`. Do NOT commit. Wait for Cliff.

---

## Task 10: End-to-end verification (no commits)

A manual verification pass. Contains no staging; the assistant never commits.

- [ ] **Step 1: Confirm the branch and working tree**

```bash
git -C /home/cliffk/idm/covasim branch --show-current   # expect starsim-port
git -C /home/cliffk/idm/covasim status --short
```

Expected: on `starsim-port`; remaining unstaged/uncommitted items are exactly the M0 deliverables Cliff has not yet committed (depending on how many check-ins he's processed).

- [ ] **Step 2: Run the full fast suite**

```bash
cd /home/cliffk/idm/covasim/tests && ./run_tests
```

Expected: all tests pass under `COVASIM_WARNINGS=error` and `COVASIM_INTERACTIVE=0`, including the new `test_regression_harness.py`, `test_v4_stub.py`, and a skipped `test_m0_parity.py`. If a new warning from `import starsim` or the harness fails the strict bar, fix it at the source — escalate rather than silencing globally.

- [ ] **Step 3: Run the full developer parity workflow (requires a v3.1.8 env)**

```bash
# In a frozen v3.1.8 venv (the reference build):
<v3.1.8-python> tests/regression/multi_seed_v3.py --n 30
# Back in the v4 env:
cd tests && pytest test_m0_parity.py -m slow -v
```

Expected: the v3 sweep writes `tests/regression/v3_seeds_n30.json` (gitignored); the gate then runs 10 v4 seeds and reports per-metric z-scores. With the M0 stub (no real port yet), the gate still requires a v4 *covasim* run of the anchor — at M0 the anchor uses the existing v3.1.8 `cv.Sim`, so v4 ≈ v3 and all metrics should pass (`|z|` small). This confirms the gate machinery end-to-end. NOTE: if no v3.1.8 env is available, this step is skipped and the gate's skip path (Task 5 Step 2) is the verification instead — record that in the report.

- [ ] **Step 4: Confirm `.gitignore` excludes the generated sweeps**

```bash
git -C /home/cliffk/idm/covasim check-ignore tests/regression/v3_seeds_n30.json tests/regression/v4_seeds_n10.json
```

Expected: both echoed back (ignored).

- [ ] **Step 5: Final report to Cliff**

Summarize: the branch, the five check-in boundaries and what was staged at each, test results, any escalations (slow-run timings, the `anchor_snapshot.json` gitignore question, the starsim dependency declaration, any strict-warnings issues from `import starsim`), and confirmation that the assistant committed nothing. **Milestone completion is Cliff's call: acceptance test green locally AND Cliff has reviewed and committed the work.**

---

## Self-review checklist

After completing all tasks, verify against `MIGRATION_PLAN.md` §M0 and the spec at `docs/superpowers/specs/2026-05-29-covasim-m0-foundation-design.md`:

| Requirement | Implementing task |
|---|---|
| Work on the existing `starsim-port` branch (assistant never creates branches or commits) | Task 0 |
| CI adapted: existing pytest step left as-is + no-baseline `compare.py` smoke step | Task 7 |
| Heavy multi-seed parity sweep kept out of the 5-minute PR job | Tasks 5, 7 |
| `tests/regression/` importable package | Task 1 |
| `anchor.py` — pinned vanilla anchor (hybrid, waning, seed 0, no interventions, 120 days, 20k) | Task 1 |
| `short_summary.py` — `build_summary` from `sim.summary` + peak metrics; pinned `METRIC_KEYS` | Task 1 |
| `parity.py` — `_mean_se` + `parity_gate(z_threshold=3.0)`, ported ~verbatim from hpvsim | Task 2 |
| `multi_seed_v3.py` — N-seed sweep CLI for a frozen v3.1.8 env (gitignored output) | Task 3 |
| optional `multi_seed_v4.py` | Task 3 |
| `compare.py` — one-seed `+/-10%` drift CLI with no-baseline mode | Task 4 |
| z-score release gate `test_m0_parity.py`, `@pytest.mark.slow`, skips when baseline absent | Task 5 |
| `slow` marker registered (strict-warnings safe) | Task 5 |
| Reuse (not replace) `baseline.json` / `test_baselines.py` / `update_baseline` / `covasim/regression/` | Spec decision 6; harness layers on top (no edits to those files) |
| `_v2_legacy` quarantine scaffold (`covasim/_v2_legacy/__init__.py`, pure docstring, no imports) | Task 0 |
| `tests/_legacy/` quarantine scaffold | Task 0 |
| stub `cv.Sim(ss.Sim)` that runs (continuous-runnability invariant) | Task 6 |
| stub exposed without disturbing the v3.1.8 `cv.Sim` | Task 6 |
| Gitignore `tests/regression/v3_seeds_n*.json` + `v4_seeds*.json` | Task 0 |
| Doc pointers: `tests/regression/README.md` + `tests/README.md` pointer | Task 8 |
| M0 plan + design spec under `docs/superpowers/` | Task 9 |
| Acceptance: pytest passes + no-baseline CLI smoke exits clean + local parity workflow works | Task 10 |
| Assistant never commits / never pushes; 5 PAUSE-FOR-CLIFF check-ins | Tasks 0,3,6,8,9 boundaries |
| No real migration code in `covasim/` (only the trivial stub) | Spec "Out of scope"; only `covasim/_v4.py` + one `__init__.py` line added |
```
