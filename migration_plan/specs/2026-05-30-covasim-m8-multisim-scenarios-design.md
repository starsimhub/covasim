# Covasim v4.0 — M8 MultiSim / Scenarios / parallel: design spec + plan

> **Status:** drafted 2026-05-30 (autonomous session, after M3-M7 landed). Combined spec+plan.

## Goal

Port the multi-run UQ layer: `cv.MultiSim` (multi-seed runs + median/quantile aggregation),
`cv.parallel`/`cv.multi_run`/`cv.single_run`, and `cv.Scenarios` (named parameter-set comparison),
wrapping Starsim's `ss.MultiSim`/`ss.parallel`. Demo: run N seeds, combine into uncertainty intervals,
and run a `Scenarios` comparison. Acceptance: multi-seed CIs are produced; `Scenarios` works.

## Key finding

`ss.MultiSim(base_sim=cv.Sim(...), n_runs=N).run()` **runs cv.Sim multi-seed correctly** (verified:
N independent seeds with varying results). But `ss.MultiSim.reduce()` raises a shape error on the
cv.Sim results structure (the M3 bridged top-level references + the nested `['variant']` sub-dict
confuse its result-stacking). So `cv.MultiSim` uses `ss.MultiSim` **to run** the seeds, but implements
its **own reduction** over the per-seed COVID-module results (stacking the 1D time-series Results ->
median/mean + low/high quantile bands). This is robust, gives full control, and matches the v3
`cv.MultiSim` UQ pattern (median trajectory + bands).

## Design

### `cv.MultiSim(sim, n_runs=4, **kwargs)`
- `.run(**kw)`: run `n_runs` seeds via `ss.MultiSim(base_sim=sim, n_runs=n_runs)`; store `self.sims`.
- `.reduce(quantiles=(0.1,0.9), use_mean=False, keys=None)`: stack each 1D COVID result across sims ->
  `self.results[key] = objdict(best, low, high)` (best = median or mean). Also handles the aggregate
  `cum_infections` as the events sum-over-variants (matching the parity definition).
- `.mean()` / `.median()`: convenience wrappers.
- `.plot(keys=...)`: median line + `fill_between(low, high)` per key.

### `cv.single_run(sim)` / `cv.multi_run(sim, n_runs)` / `cv.parallel(*sims_or_sim)`
Thin wrappers: `single_run` runs one copy; `multi_run` returns the list of run seeds (via ss.MultiSim);
`parallel` runs multiple distinct sims (wraps `ss.parallel`), returning a `cv.MultiSim`.

### `cv.Scenarios(sim, scenarios, n_runs=...)`
A named-scenario comparison: `scenarios = {name: {'name':..., 'pars':{...}}}` (v3 form). For each
scenario, build a base sim with the par overrides applied, run a `cv.MultiSim(n_runs)`, reduce, and
store `self.results[name]` = the reduced (median + band) results. `.plot()` overlays scenarios.

## Acceptance test

A multi-seed `cv.MultiSim` produces median + 10/90 bands (band width > 0 where seeds differ); a
2-scenario `cv.Scenarios` (e.g. baseline vs an intervention) produces distinct per-scenario results
with the expected ordering (the intervention scenario has fewer infections). A UQ demo
(`examples/m8_uq_sweep.py`) plots the median + band, with a smoke variant in the test suite.

## Out of scope

- `noise`/`iterpars` (v3 multi_run extras) -- add opportunistically.
- Full ss.MultiSim.reduce compatibility (a starsim-side fix) -- cv.MultiSim reduces itself.

## Linked documents

- `MIGRATION_PLAN.md` §M8. v3 reference: `_v2_legacy/run.py`.
