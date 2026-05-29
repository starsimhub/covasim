# Regression harness (v3.1.8 -> v4.0 migration)

This directory holds the self-contained regression harness for the Covasim v4.0
Starsim port. It does two complementary jobs:

- **Development gate (`compare.py`)** -- a fast, informational one-seed +/-10%
  drift table for day-to-day porting feedback. Always exits 0; never blocks.
- **Release gate (`../test_m0_parity.py`)** -- the scientific gate: a multi-seed
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
population-structure and NAb/immunity machinery -- the highest-risk parts of the
port -- without confounding from intervention ports. Intervention/vaccine anchors
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
anchor -- this is the mode CI smoke-runs.

## When to refresh the baselines

- After a v3.1.8 patch-equivalent change lands and is forward-merged into `starsim-port`.
- After an explicit decision that drift introduced by a milestone is the new target.
- Otherwise: don't. Stable baseline = stable signal.

## CI

CI runs the pytest suite (which collects the harness unit tests + anchor smoke
test + the skipped slow gate) plus `python regression/compare.py` in no-baseline
mode (CLI-integrity only). Neither fails on drift. The heavy multi-seed sweep
runs only locally or in a future nightly job, never in the 5-minute PR job.
