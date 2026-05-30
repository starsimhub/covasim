# Covasim v4.0 — M7 Calibration + Fit: design spec

> **Status:** drafted 2026-05-30 (autonomous session, after M3-M6 landed). Grounded in the v3
> `cv.Fit`/`compute_gof` (`_v2_legacy/analysis.py`, `misc.py`) and Starsim's `ss.Calibration` (Optuna).

## Goal

Port the model-vs-data fit (`cv.Fit` + `cv.compute_gof`) and integrate Starsim's Optuna calibration
as `cv.Calibration`. Demo: calibrate a small model to data and reproduce a v3.1.8 fit. Acceptance:
a small calibration converges and reproduces a v3.1.8 fit.

## Findings framing M7

1. **`cv.compute_gof` is already active** (`misc.py`, kept-from-v3) -- the goodness-of-fit core.
2. **`cv.Fit` reads flat top-level results** (`sim.results['cum_deaths']`, `sim.result_keys()`,
   `sim.npts`, `sim.datevec`) + `sim.data`. The v4 port namespaces aggregate results under
   `sim.results['covid']` (the deferred **Open Q E**), so M7 must first **bridge the flat aggregate
   results** to the sim top level (M3 only bridged `['variant']` + `n_imports`).
3. **`ss.Calibration(sim, calib_pars, n_workers, total_trials, build_fn, eval_fn, components, ...)`**
   uses Optuna with a build_fn (apply trial pars to a sim) + an eval (a likelihood/mismatch). This is
   a different paradigm from v3's `Calibration` (which minimised `cv.Fit.mismatch`); `cv.Calibration`
   wraps `ss.Calibration` with a build_fn that applies `calib_pars=dict(key=[best,low,high])` and an
   eval that returns `cv.Fit(...).mismatch`.

## Design decisions

### 1. Flat aggregate-results bridge (completes Open Q E)

`cv.Sim.finalize` already bridges `results['variant']` + `n_imports`. Extend it to reference **every
top-level `ss.Result` of the covid module** at the sim root (`sim.results[key] = covid.results[key]`),
so `sim.results['cum_deaths']` etc. resolve as in v3. Plus `t`/`date`/`datevec` helpers for `cv.Fit`.
Additive (references, no dynamics change) -> M1-M6 unaffected.

### 2. `cv.Fit` -- ported, results-structure-adapted

Lift-and-shift the v3 `cv.Fit` (reconcile_inputs -> compute_diffs -> compute_gofs -> compute_losses
-> compute_mismatch), reading the bridged flat results. Two adaptations:
- **Data source:** v3 reads `sim.data` (a DataFrame loaded from a datafile). v4 `cv.Sim` has no
  datafile loader, so `cv.Fit(sim, data=<DataFrame>)` accepts the data explicitly (falling back to
  `sim.data` if present). Data is a DataFrame indexed by date with columns like `cum_deaths`.
- **Dates:** use the sim's date vector (`sim.timevec`/`datevec`) to align data dates to sim indices.
Default weights `cum_deaths:10`, `cum_diagnoses:5`, else 1 (v3). `custom` series supported.
`compute_gof` is the shared core (already active). **Validation:** on IDENTICAL synthetic
results+data, `cv.Fit.mismatch` equals the v3 value (the logic is engine-independent; compute_gof is
shared) -- a deterministic check that does not depend on sim-trajectory parity.

### 3. `cv.Calibration` -- wraps `ss.Calibration` (Optuna)

`cv.Calibration(sim, calib_pars, data, n_trials=..., n_workers=..., **kw)`:
- `calib_pars = dict(key=[best, low, high])` (v3 form); a build_fn maps an Optuna trial's sampled
  values onto a fresh `cv.Sim` (resolving dotted keys into `sim.pars` / `sim.diseases['covid'].pars`,
  the `dynamic_pars` resolution noted in the architecture map).
- eval: run the sim, build `cv.Fit(sim, data)`, return `.mismatch` (minimised by Optuna).
- `.calibrate()` runs `ss.Calibration.calibrate()`; expose `.best_pars`/`.df`.
Confirm a small calibration (few trials) converges (best mismatch <= initial) and its best pars move
toward the v3 calibrated values on the same target data.

## Scope for THIS session vs deferred

- **This session:** flat results bridge + `cv.Fit` (+ deterministic v3-parity validation). These are
  contained and engine-independent.
- **Deferred (next session, design captured above):** `cv.Calibration` (the Optuna build_fn/eval
  integration + the dotted-path `calib_pars` resolution) and the end-to-end calibration test. This is
  a different paradigm (Optuna components) and is best implemented fresh.

## Out of scope for M7

- `cv.Fit.plot` (Covasim-specific plotting) -- M9.
- Datafile loading into `cv.Sim` -- `cv.Fit` accepts a DataFrame directly.

## Linked documents

- `../plans/2026-05-30-covasim-m7-calibration-fit.md` -- task plan.
- `MIGRATION_PLAN.md` §M7. v3 reference: `_v2_legacy/analysis.py` (Fit/Calibration), `misc.py` (compute_gof).
