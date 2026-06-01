# Covasim v4.0 — M7 Calibration + Fit Implementation Plan

> Implements the M7 spec (`../specs/2026-05-30-covasim-m7-calibration-fit-design.md`). Additive
> (Fit/Calibration are analysis layers; no engine change). Commit at each working increment.

## Task 1: flat aggregate-results bridge + cv.Fit (CHECK-IN 1)

- [ ] `cv.Sim.finalize`: reference every top-level `ss.Result` of the covid module at the sim root
      (`sim.results[key] = covid.results[key]`), completing Open Q E. Additive (M1-M6 unaffected).
- [ ] Port `cv.Fit(Analyzer)` into a new active `covasim/analysis.py`: reconcile_inputs / compute_diffs
      / compute_gofs / compute_losses / compute_mismatch / summarize, reading the bridged flat results;
      `cv.Fit(sim, data=<DataFrame>, weights=..., keys=..., custom=...)`. Default weights
      cum_deaths:10/cum_diagnoses:5/else 1. `compute_gof` already active (misc.py).
- [ ] Export `cv.Fit` (and `cv.Analyzer` base if not already).
- [ ] Tests: deterministic Fit-vs-v3 on identical synthetic results+data (mismatch matches); Fit on a
      real cv.Sim run + toy data computes a finite mismatch; weights applied. **Commit.**

## Task 2 (next session, design in spec): cv.Calibration via ss.Calibration (Optuna)

- [ ] `cv.Calibration(sim, calib_pars, data, n_trials, n_workers)`: build_fn applies
      `calib_pars=dict(key=[best,low,high])` to a fresh cv.Sim (dotted-path resolution into
      sim.pars / diseases['covid'].pars); eval returns `cv.Fit(sim, data).mismatch`; `.calibrate()`
      wraps `ss.Calibration.calibrate()`; expose best_pars/df.
- [ ] Tests: a small calibration converges (best mismatch <= initial); best pars move toward the v3
      calibrated values on the same target data.

## Self-review checklist

- [ ] Flat bridge additive -> M1-M6 byte-identical (verify the M2-M6 gates still pass).
- [ ] cv.Fit mismatch matches v3 on identical inputs (engine-independent; compute_gof shared).
- [ ] No datafile-loading dependency (Fit accepts a DataFrame).
