# Covasim v4.0 — M0 Foundation: design spec

## Goal

Stand up the self-contained regression infrastructure that every later milestone (M1–M10) will rely on, plus the branch and quarantine scaffolding for the port. M0 is foundation work only — **no migration code lands in `covasim/`** in this milestone, and the package on `starsim-port` stays v3.1.8-equivalent until M1 begins. The one exception to "no code" is a deliberately trivial stub `cv.Sim(ss.Sim)` that exists solely to prove the continuous-runnability invariant can be met on a Starsim base (see "Stub `cv.Sim`" below); it is not wired into the model and does not change any existing behavior.

The deliverables are:

1. Work staged on the existing long-lived `starsim-port` branch (already created off `main` and checked out; the assistant never creates branches, **never commits, never pushes** — Cliff commits). No per-milestone branches.
2. CI (`.github/workflows/tests.yaml`) extended to cover the new regression harness in **no-baseline mode** (the heavy multi-seed parity sweep stays out of the 5-minute PR job).
3. A self-contained regression harness under `tests/regression/` (an importable package) implementing the **multi-seed z-score parity** pattern adapted from hpvsim: a pinned **anchor scenario**, a **short-summary builder**, a **v3.1.8 multi-seed sweep generator** (run in a frozen v3.1.8 env), a **`parity.py` z-score helper** (ported essentially verbatim from hpvsim), an optional v4 sweep, and a `README.md`.
4. A **comparison CLI** (`compare.py`) that diffs a current run against a locally-stored baseline and prints a per-summary-result drift table, with a **no-baseline mode** that exits clean for CI.
5. Reuse — not replacement — of Covasim's existing baseline machinery (`tests/baseline.json` / `test_baselines.py` / `tests/update_baseline` / `covasim/regression/`).
6. The `_v2_legacy` quarantine scaffold (`covasim/_v2_legacy/` + `tests/_legacy/`), empty in M0.
7. A stub `cv.Sim(ss.Sim)` that runs, satisfying the continuous-runnability invariant on the Starsim base.
8. Gitignore entries for the generated (large, env-specific) baseline sweeps.
9. Documentation: `tests/regression/README.md`, a pointer from `tests/README.md`, and the M0 plan + this spec under `migration_plan/`.

## Problem statement

Covasim v3.1.8 is a mature, self-contained ABM with its own engine. Porting it onto Starsim 3.3.x changes the random-number model fundamentally — from a single global numpy/numba stream (`cv.utils.set_seed`) to Starsim's per-distribution Common Random Numbers (CRN), seeded by each distribution's *path within the sim*. The exact numeric stream therefore changes, so **bit-for-bit equivalence is impossible and not the bar**. We need a way to prove the port is *scientifically faithful* despite a different RNG: that a v4 run reproduces v3.1.8 on the headline epidemiological metrics, within the noise of finite seeds.

The chosen validation bar is **self-contained**: it depends only on this repo and a frozen v3.1.8 reference environment. There are **no external analysis/validation repos** in the validation set (this is a deliberate divergence from the hpvsim plan, which had an "analysis-repo suite"). "Done" for v4.0 has two parts:

1. The existing Covasim test suite passes against the v4/Starsim build (`test_sim`, `test_immunity`, `test_interventions`, `test_baselines`, `test_regression`, `test_analysis`, `test_run`, etc.), under the strict-warnings bar (`run_tests` sets `COVASIM_WARNINGS=error`).
2. New **multi-seed z-score parity gates** compare a v4 run to v3.1.8-generated baselines and require **overlapping uncertainty intervals** (`|z| < 3`), *not* bit-for-bit equivalence.

M0 builds the machinery for part 2 (the parity harness) and wires the lightweight comparison CLI into CI. The first concrete parity gate against the M0 anchor is enabled here; the per-milestone capability parity gates and the retrofit of M1/M2/M3/M5/M6 onto z-score gates happen in M8.

## Design decisions

1. **Multi-seed z-score parity, not single-seed bit-for-bit, is the cross-version gate.** Because the RNG stream differs between v3.1.8 (global numba) and v4 (Starsim CRN), a single fixed seed cannot match. The gate compares the *mean across many seeds* of each pinned metric, with a z-score over the combined standard error:

   `z = (mean_v4 − mean_v3) / sqrt(SE_v3² + SE_v4²)`, where `SE = std(ddof=1)/sqrt(n)`.

   A metric passes when `|z| < 3`. Degenerate distributions follow hpvsim's `parity.py` policy: zero combined spread with equal means passes; zero spread with unequal means fails (`z = inf`).

2. **The harness mirrors hpvsim's `tests/regression/` multi-seed structure, not hpvsim's simpler baseline/compare scripts.** The hpvsim M0 plan shipped `baseline.py`/`compare.py` (single-seed snapshot + ±10% drift). HPVsim later evolved a multi-seed z-score layer (`anchor_m01.py`, `short_summary_m01.py`, `multi_seed_v2.py`, `parity.py`, `test_m01_short_summary_parity.py`). Covasim M0 adopts the *evolved* structure directly, since the locked validation bar is the z-score gate. The `compare.py` ±10% drift table is retained as the lightweight **development gate** (informational, exit-0, the per-check-in signal) and as the CI smoke target.

3. **Two coexisting comparison layers, answering different questions:**
   - **Development gate (`compare.py`, per check-in):** runs *one* v4 seed of the anchor against a *one*-seed v3.1.8 snapshot and prints a ±10% relative-drift table. Fast, informational, exit 0. This is the day-to-day signal during porting.
   - **Release gate (`test_m0_parity.py`, per milestone / nightly):** runs N v4 seeds vs M v3.1.8 seeds through `parity_gate()` and fails on `|z| ≥ 3`. Marked `@pytest.mark.slow`; excluded from the 5-minute PR job.

4. **The anchor is a representative-but-clean *vanilla* sim**, deliberately *not* the overloaded `test_baselines.py` sim. `test_baselines.make_sim()` wires in four interventions (`change_beta`, `test_prob`, `contact_tracing`, `vaccinate_prob`) for coverage breadth — exactly what we do *not* want for a parity anchor, where intervention-port bugs would confound core-dynamics drift. The M0 anchor isolates core dynamics: `pop_type='hybrid'` + `use_waning=True` exercises population structure and the immunity/NAb machinery (the parts most likely to drift in a port) with no interventions or analyzers. Intervention/vaccine anchors are added in their owning milestones (M5/M6) and retrofitted onto z-score gates in M8.

5. **The pinned summary set is drawn from Covasim's own `sim.summary`** — the same flat objdict (`compute_summary`, all `result_keys()` at the last timepoint) that `tests/baseline.json` already stores — plus two epidemic-shape metrics computed from the `sim.results` time series (peak prevalence, peak `n_infectious`). This keeps the parity metrics aligned with Covasim's established headline outputs and with the existing baseline machinery.

6. **Existing Covasim baseline machinery is reused, not replaced.** `tests/baseline.json` + `test_baselines.py` stays the v4-*internal* bit-for-bit self-consistency gate (regenerated for v4 via `tests/update_baseline` once the port lands, since v4's RNG stream produces different exact numbers). The new parity harness is the *cross-implementation* v3↔v4 scientific-equivalence gate. `tests/update_baseline` touches only `baseline.json`/`benchmark.json`/`covasim/regression/pars_*.json` and never the parity baselines, so the two regeneration workflows are independent. A `pars_v3.1.8.json` snapshot (currently missing) is added opportunistically on the next `update_baseline`; it is forensic only.

7. **v3.1.8 baselines are generated locally and gitignored, never committed.** The generator script, anchor, short-summary builder, parity helper, comparison CLI, and pytest gate are committed; only the generated per-seed sweep JSON is gitignored. The parity gate **skips** (does not fail) when the baseline file is absent, so a contributor without a v3.1.8 env can still run the rest of the suite.

8. **The `_v2_legacy` quarantine is scaffolded now but empty.** No migration code lands in M0, so nothing is quarantined yet. The scaffold exists so M1 can immediately move v3 modules into `covasim/_v2_legacy/` and v3 tests into `tests/_legacy/`. Active code never imports from either quarantine.

9. **A trivial stub `cv.Sim(ss.Sim)` proves the continuous-runnability invariant on the Starsim base.** Implementation conventions item 1 (`cv.Sim().run()` must return results at every commit on `starsim-port`) needs to hold from day one. M0 ships a minimal `cv.Sim` that subclasses `ss.Sim` and runs a degenerate (no-disease) sim to completion. It is not the real port — it asserts only that the Starsim base imports, a `cv.Sim` constructs, and `.run()` returns. It lives behind the existing `cv.Sim` only as a *new* symbol path that does not disturb the v3.1.8 `cv.Sim` used by the existing suite (see plan Task 6 for the exact coexistence mechanism).

10. **CI stays minimal and within the 5-minute budget.** The existing `pytest -v test_*.py unittests/test_*.py -n auto` step is left as-is (new `test_*.py` files at the `tests/` root are auto-collected; the slow parity gate is `@pytest.mark.slow` and deselected). One new step runs `compare.py` in no-baseline mode (exits in <1s, proving the CLI doesn't bitrot). The heavy multi-seed sweep runs only as a separate/optional/nightly job, never in the PR job.

### Style note (Starsim vs. Covasim conventions)

New harness code follows the **Starsim style guide**. There is little tension on the biggest item — both Starsim and Covasim forbid type annotations in signatures and put type info in docstrings only (Google `Args:` style) — so signatures stay annotation-free. The one flagged tension for *model* code (not harness code): Starsim uses CRN/`ss.Dist` sampling and `ss.uids`/`BoolArr` set-like indexing in place of Covasim's global-RNG numba helpers and integer-index arrays; new model code uses the Starsim idioms. Harness code keeps Covasim's "optimize for the scientist-reader" ethos: clear logic, sensible defaults, every function ending in an explicit `return`.

## Anchor scenario (pinned)

Lives in `tests/regression/anchor.py`. The module exposes `make_sim()` (an unrun v3.1.8 `cv.Sim`), `run_and_summarize()` (run + `build_summary`), and a `__main__` that prints the summary. Pinned pars:

```python
PARS = dict(
    pop_size     = 20_000,    # large enough that per-seed CV is moderate
    pop_infected = 100,
    pop_type     = 'hybrid',  # exercises household/school/work + the random layer
    n_days       = 120,       # captures a full epidemic peak + decline
    use_waning   = True,      # exercises the NAb / immunity core
    rand_seed    = 0,         # base seed only; the sweep overrides it 0..N-1
    verbose      = 0,
    # NO interventions, NO analyzers — isolate core dynamics from intervention-port bugs.
)
```

`hybrid` + `use_waning` is chosen so the anchor stresses the population-structure and NAb/immunity machinery — the highest-risk parts of the port — without confounding from testing/tracing/vaccination intervention ports. A second anchor that *adds* the four interventions is deferred to M5/M6 (mirroring hpvsim's staged M01 → M05/M06 anchors).

## Pinned summary-result set

`build_summary(sim)` (in `tests/regression/short_summary.py`) returns a flat `{metric: float}` dict. Every value is a plain `float`. Sources:

**Gated metrics** (subject to `|z| < 3`):
- Cumulative burden (from `sim.summary`): `cum_infections`, `cum_reinfections`, `cum_symptomatic`, `cum_severe`, `cum_critical`, `cum_deaths`.
- Epidemic shape (computed from the `sim.results` time series, *not* in the end-of-run summary): `peak_prevalence` = `sim.results['prevalence'].max()`, `peak_n_infectious` = `sim.results['n_infectious'].max()`.
- Derived end-of-run rates (from `sim.summary`): `prevalence`, `incidence`.

Testing/vaccination cumulatives (`cum_tests`, `cum_diagnoses`, `cum_doses`) are **excluded** for the vanilla anchor — with no interventions they are identically 0. They re-enter via the M5/M6 capability anchors.

`r_eff` is treated as a **soft/optional metric** — Covasim's own `tests/test_regression.py` *skips* `r_eff` because its calculation is version-sensitive. It may be reported diagnostically but is excluded from the hard gate in M0.

**Skip/bookkeeping keys** (written by the v3 sweep, ignored by the gate via `_SKIP_KEYS`): `_seed`, `_total_pop`, and `n_alive`.

## Regression-harness architecture + file layout

```
tests/
├── regression/
│   ├── __init__.py            # empty; makes tests/regression an importable package
│   ├── anchor.py              # PARS + make_sim() + run_and_summarize() + __main__
│   ├── short_summary.py       # build_summary(sim) -> flat {metric: float}; METRIC_KEYS
│   ├── parity.py              # _mean_se + parity_gate(z_threshold=3.0); ported ~verbatim from hpvsim
│   ├── multi_seed_v3.py       # CLI: sweep anchor across N seeds in a v3.1.8 env -> v3_seeds_n{N}.json
│   ├── multi_seed_v4.py       # (optional) same sweep in-env for ad-hoc diffing
│   ├── compare.py             # compute_drift() + CLI: one-seed ±10% drift table; no-baseline mode
│   └── README.md              # anchor pars, generate-baseline / parity / drift workflow, gate behavior
├── test_m0_parity.py          # @pytest.mark.slow z-score gate (skips when baseline absent)
├── test_regression_harness.py # fast unit tests: parity_gate, compute_drift, anchor smoke
└── README.md                  # append one paragraph pointing to tests/regression/README.md

tests/regression/v3_seeds_n*.json   # GITIGNORED — generated from a frozen v3.1.8 env
tests/regression/v4_seeds*.json     # GITIGNORED — generated in-env for ad-hoc diffing

covasim/_v2_legacy/
└── __init__.py                # pure-docstring quarantine marker, NO imports (empty in M0)
tests/_legacy/
└── __init__.py                # quarantine marker for v3 tests that exercise removed APIs (empty in M0)
```

**Why the pytest files live at the `tests/` root, not under `tests/regression/`:** CI invokes `pytest -v test_*.py unittests/test_*.py -n auto` from the `tests/` directory; that glob only collects `test_*.py` at the `tests/` root, not in subdirectories. So `test_m0_parity.py` and `test_regression_harness.py` sit at the root and import from the `regression` package. `anchor.py` is the scientific definition; everything else imports from it.

**Module responsibilities:**

- `anchor.py` — the single source of truth for the M0 scenario. `PARS`, `make_sim()`, `run_and_summarize()` (run + `build_summary`), `__main__` print. Imports `build_summary` from `short_summary.py`.
- `short_summary.py` — `build_summary(sim)` reads `sim.summary` for the cumulative/derived metrics and `sim.results` for the peak metrics, returns a flat `{metric: float}` dict. Exposes `METRIC_KEYS` (the gated set) and `SKIP_KEYS`.
- `parity.py` — `_mean_se(rows, key)` and `parity_gate(v4_seeds, v3_seeds, z_threshold=3.0, skip_keys=...)`, ported essentially verbatim from hpvsim (the z-formula and the two degenerate-distribution policies are identical; only the argument names v2/v3 → v3/v4 change).
- `multi_seed_v3.py` — argparse CLI (`--n` default 30, `--start-seed`, `--out`) that sweeps the anchor across seeds and writes a JSON list of per-seed summary dicts. Intended to run **in a frozen v3.1.8 environment** (separate venv); writes `v3_seeds_n{N}.json`. Each row is `build_summary(sim)` plus `_seed`/`_total_pop`.
- `multi_seed_v4.py` (optional) — the same sweep in the current env, writing `v4_seeds_n{N}.json`, for ad-hoc local diffing. The pytest gate generates v4 seeds in-process and does not require this file.
- `compare.py` — `compute_drift(baseline_summary, current_summary, threshold=0.10)` (pure, per-key relative drift with a zero-baseline guard) + `format_table()` + an argparse CLI. The CLI loads a one-seed baseline JSON, runs one v4 seed of the anchor, prints the drift table, and **always exits 0**. No-baseline mode prints a notice and exits 0 *without running the anchor* (the mode CI uses).
- `test_m0_parity.py` — `@pytest.mark.slow`; runs `N_V4_SEEDS` v4 seeds in-process, loads `v3_seeds_n{M}.json`, calls `parity_gate(..., z_threshold=3.0, skip_keys=_SKIP_KEYS)`, and `pytest.fail`s with a per-metric `z=±x.xx` table on any `|z| ≥ 3`. Skips (does not fail) when the baseline JSON is absent.
- `test_regression_harness.py` — fast unit tests (sub-second): `parity_gate` (pass/fail/degenerate cases), `compute_drift` (within/over threshold, zero-baseline, missing-key), and an anchor smoke check (`run_and_summarize()` returns the expected keys with a positive population).

## Comparison rules (development gate)

`compute_drift` semantics (per-key, vs. a one-seed baseline):
- **Relative drift:** `(current − baseline) / baseline`. A row is flagged when `|rel_diff| > threshold` (default 0.10).
- **Zero-baseline guard:** if the baseline value is zero, `rel_diff` is `None`, `abs_diff` is reported, and the row is flagged.
- **Missing keys:** keys present in the baseline but absent from the current run are skipped (not reported).
- **Output:** a table `key | baseline | current | abs_diff | rel_diff | over_threshold?`, always exit 0.
- **No-baseline mode:** if the baseline file is missing, print a notice and exit 0 *without running the anchor* — CLI-integrity check only (the pytest harness covers anchor execution). This is the mode CI runs.

## Parity rules (release gate)

`parity_gate` (multi-seed, vs. M v3.1.8 seeds):
- `z = (mean_v4 − mean_v3) / sqrt(SE_v3² + SE_v4²)`, `SE = std(ddof=1)/sqrt(n)`.
- Metric fails on `|z| ≥ z_threshold` (default 3.0).
- Degenerate: zero combined SE with equal means → pass; zero combined SE with unequal means → fail (`z = inf`).
- `_SKIP_KEYS = {'_seed', '_total_pop', 'n_alive'}` are ignored.
- Recommended seed counts: **N ≥ 10 v4 seeds** vs **M = 30 v3.1.8 seeds** (hpvsim's committed gate uses 10×30; 30×30 is cleaner if runtime allows). The 20k-agent × 120-day anchor makes a 30-v4-seed sweep too heavy for the 5-minute PR job, hence `@pytest.mark.slow` + nightly/optional.

## Acceptance test

- The existing `pytest test_*.py` step passes against the M0 working tree (the stub `cv.Sim` and harness do not perturb the v3.1.8 suite; the slow parity gate is deselected from the PR job).
- The new no-baseline CLI smoke step (`python regression/compare.py`) exits clean, proving the comparison tooling doesn't bitrot.
- `cv.Sim().run()` returns results on the Starsim base (continuous-runnability invariant), demonstrated by a fast test.
- A developer can, locally: (a) generate a v3.1.8 baseline sweep from a frozen v3.1.8 env via `multi_seed_v3.py`; (b) run `test_m0_parity.py` to get a per-metric z-score parity result (or skip cleanly if the baseline is absent); (c) run `compare.py` against a one-seed snapshot to get a per-metric drift table.
- The `starsim-port` branch exists off `main`; the harness, parity helper, comparison CLI, sweep generator, quarantine scaffold, gitignore entries, and docs are all present (uncommitted, staged for Cliff's review/commit).

**Milestone completion** = acceptance test green locally *and* Cliff has reviewed and committed the work. There are no PRs or issue numbers as gates (this is a local effort).

## Workflow (pause-for-review-and-commit)

This is a local effort with no GitHub RACI table and no issues/PRs as gates. The assistant (Claude) prepares each piece of work and leaves it **uncommitted** in the working tree, then pauses for Cliff Kerr to review and commit. **The assistant never commits and never pushes.** Check-ins happen **2–5 times per milestone** at natural sub-task boundaries. M0's natural check-in points: (1) branch + scaffold + gitignore + quarantine; (2) the harness package (anchor, short-summary, parity, sweeps); (3) the comparison CLI + tests; (4) the stub `cv.Sim` + CI step; (5) docs + plan/spec.

## Out of scope for M0

- Any real migration code in `covasim/` — the package stays v3.1.8-equivalent until M1. (The trivial stub `cv.Sim(ss.Sim)` is the only new symbol, and it does not alter existing behavior.)
- Capability anchors beyond the vanilla M0 anchor (testing/tracing/quarantine, vaccination, multi-variant) — added by their owning milestones (M3/M5/M6) under `tests/regression/`.
- The retrofit of M1/M2/M3/M5/M6 acceptance tests onto z-score parity gates — done in M8.
- Regenerating `tests/baseline.json` / `benchmark.json` for v4 — done in M10 once the port lands (v4's RNG stream changes the exact numbers).
- A second-venv CI job that installs v3.1.8 and runs the parity sweep in CI — deferred; the v3.1.8 baseline generation stays a developer-local action, and the parity gate runs nightly/optionally.
- Reconciling or retiring `example_regression.sim` (the frozen v1.7.0 pickle that will likely break under the Starsim object model) — deferred to M10.
- Branch-protection rules on `starsim-port` — none (local effort).

## Linked documents

- [`MIGRATION_PLAN.md`](../MIGRATION_PLAN.md) — overall migration plan, of which this is the M0 deliverable spec.
- [`plans/2026-05-29-covasim-m0-foundation.md`](../plans/2026-05-29-covasim-m0-foundation.md) — the task-by-task M0 implementation plan behind this spec.
- [`tests/baseline.json`](../../tests/baseline.json) + [`tests/test_baselines.py`](../../tests/test_baselines.py) — the existing v4-internal bit-for-bit self-consistency gate (reused, not replaced).
- [`covasim/regression/README.md`](../../covasim/regression/README.md) — the forensic per-version default-parameter snapshots (`pars_v*.json`); `pars_v3.1.8.json` to be added opportunistically.
