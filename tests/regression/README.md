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

## M1 anchors (basic transmission)

`anchor_m1.py` adds the M1 single-variant basic-transmission anchor for the
`random` and `hybrid` backends. The same file runs under v3.1.8 (configured to a
transmission+recovery-only SEIR: `use_waning=False`, all-asymptomatic prognoses,
`asymp_factor=1.0`) and under v4 (`cv.Sim`), so one anchor serves both the baseline
and the gate. `build_summary_m1` (in `short_summary.py`) extracts the M1 metrics
(`cum_infections`, `peak_prevalence`, `peak_n_infectious`) from either engine.

Generate the gitignored v3.1.8 M1 baselines from a frozen v3.1.8 env:

```bash
python tests/regression/multi_seed_v3.py --anchor m1_random --n 30
python tests/regression/multi_seed_v3.py --anchor m1_hybrid --n 30
```

Then the release gate `../test_m1_parity.py` (slow, `|z| < 3`, per backend) compares
v4 to those baselines; it skips cleanly when they are absent. The contact-structure
equivalence half (per-layer degree + age-mixing) lives in `../test_network.py` and
consumes a `v3_m1_contacts.json` baseline (also v3.1.8-env-generated).

## M2 anchors (full natural history)

`anchor_m2.py` adds the M2 single-variant **full-natural-history** anchor (the
prognosis tree + viral_load/beta_dist). Unlike `anchor_m1`, the v3 branch keeps the
**default** age-based prognoses (the symptomatic disease course is the point) -- only
`use_waning=False` + `n_variants=1`. `build_summary_m2` extracts both the burden
metrics (`cum_symptomatic`/`cum_severe`/`cum_critical`/`cum_deaths`) and the
re-converged transmission metrics. Generate the gitignored v3.1.8 baselines from a
frozen v3.1.8 env (the worktree method):

```bash
git worktree add /tmp/cov-v3 main
cd /tmp/cov-v3 && python -c "import sys; sys.path.append('<repo>/tests/regression'); \
  from multi_seed_v3 import main; main(['--anchor','m2_random','--n','30']); main(['--anchor','m2_hybrid','--n','30'])"
```

The release gate is `../test_m2_parity.py` (slow, per backend), skipping when the
baseline is absent. **M2 uses `|z| < 5`** (not the default `|z| < 3`) by an explicit
documented decision (MIGRATION_PLAN.md Open Q G, signed off 2026-05-29): after matching
v3's integer duration rounding, every metric agrees within ~3% in magnitude, but the
40-seed standard error is so small that a ~3% *systematic* offset (Starsim CRN vs v3
numba RNG + per-day viral-load discretization, irreducible without bit-for-bit
equivalence) still reads as `|z|` up to ~3.5. `|z| < 5` admits that scientifically
negligible band while still catching genuine regressions; see the rationale block at
the top of `../test_m2_parity.py`.

## M3 anchors (multi-variant + cross-immunity)

`anchor_m3.py` adds the M3 **multi-variant** anchor for the `random` and `hybrid`
backends: wild seeded at t0, **alpha introduced at day 10** and **delta at day 30**
(`n_imports=20` each), `pop_size=20_000`, `n_days=120`. The same file runs under
v3.1.8 (with **cross-immunity active**, i.e. `use_waning=True` — the realistic
multi-variant regime, M3 design-spec Open Q D) and under v4 (`cv.Sim(variants=[...])`).
`build_summary_m3` (in `short_summary.py`) extracts aggregate metrics
(`cum_infections`, `cum_deaths`, `peak_n_infectious`, `peak_prevalence`) plus the
per-variant `cum_infections_<v>` / `peak_n_infectious_<v>` for wild/alpha/delta.

Generate the gitignored v3.1.8 M3 baselines from a frozen v3.1.8 env (worktree method):

```bash
git worktree add /tmp/cov-v3 main
PYTHONPATH=/tmp/cov-v3 python tests/regression/multi_seed_v3.py --anchor m3_random --n 30
PYTHONPATH=/tmp/cov-v3 python tests/regression/multi_seed_v3.py --anchor m3_hybrid --n 30
```

(`PYTHONPATH=/tmp/cov-v3` makes `import covasim` resolve to the v3.1.8 worktree, not the
editable v4 install; the harness duck-types on `cv.COVID` to pick the v3-vs-v4 branch.)

The release gate is `../test_m3_parity.py` (slow, per backend), skipping when the
baseline is absent.

**Documented static-vs-NAb divergence (the M3 acceptance boundary).** M3 ships a
*static, NAb-free* cross-immunity: the connector writes `sus_imm = matrix[target, source]`
directly, whereas v3 weights it by the per-agent neutralizing-antibody titre
(`sus_imm = calc_VE(nab × matrix)`). Consequently the v4 and v3 trajectories agree where
the static matrix suffices but diverge where NAb kinetics dominate:

  - **Converges (GATED, `|z| < 5`):** `cum_infections_wild` (≈ `|z| 0`, even with
    multi-variant reinfection feedback), `peak_n_infectious`, `peak_prevalence`. These
    validate the core multi-variant machinery (per-variant transmission, host
    exclusivity, the cross-immunity connector, reinfection).
  - **Diverges (INFORMATIONAL, not gated):** the per-variant alpha/delta absolute counts
    and aggregate `cum_infections`. The gap is largest for the **late-introduced escape
    variant delta** (`matrix[delta, wild]=0.374`, so v4 wild-recovered are only ~37%
    protected and delta finds a large susceptible pool): v4 has ~7–10× more delta and
    ~55% more total infections than v3 (`|z|` up to ~46). This is by design; the NAb
    engine (M4) re-converges these. A related divergence: same-variant reinfection is
    **exactly 0** in M3 (`matrix[v,v]=1.0`), whereas v3's `calc_VE(nab×1.0) < 1` permits
    a small amount.

The gate therefore hard-gates only the convergent subset and prints the full per-metric
table (`[GATE]`/`[info]`) for diagnostics; see the rationale block at the top of
`../test_m3_parity.py` and the demo in `NOTES_FOR_CLIFF.md`.

## M4 anchor (waning immunity + NAbs)

`anchor_m4.py` is the M3 multi-variant anchor (wild + alpha@d10 + delta@d30) **with the NAb engine
on** (`use_waning=True`). The v3.1.8 side is byte-for-byte the M3 anchor's v3 branch (which already
ran `use_waning=True`), so **M4 reuses the M3 v3.1.8 baseline** (`v3_m3_<pt>_seeds_n*.json`) — no
new baseline. `build_summary_m3` serves both milestones.

The release gate is `../test_m4_parity.py` (slow, per backend). Where M3's *static* cross-immunity
diverged from v3 on the per-variant escape dynamics (delta `|z|~25-46`, gated only on a convergent
subset), M4's NAb-weighted cross-immunity (`sus_imm = calc_VE(nab × matrix)`) **re-converges every
pinned metric** — aggregate burden AND per-variant wild/alpha/delta counts — to within `|z|<3.5` of
the same v3 baseline. So M4 hard-gates the WHOLE metric set at `|z|<5`. This is the M4 acceptance:
the documented M3 static-vs-NAb divergence closes once NAbs are wired.

## M5 anchor (testing / tracing / quarantine)

`anchor_m5.py` is the M2 single-variant scenario plus a `test_prob` testing intervention and a
`contact_tracing` intervention (same public API in v3.1.8 and v4). `build_summary_m5` pins the burden
(`cum_infections`/`cum_deaths`/`peak_n_infectious`) plus the testing/quarantine outcomes
(`cum_tests`/`cum_diagnoses`/`peak_n_quarantined`/`peak_n_isolated`). Generate the gitignored v3.1.8
baseline:

```bash
PYTHONPATH=/tmp/cov-v3 python tests/regression/multi_seed_v3.py --anchor m5_random --n 30
PYTHONPATH=/tmp/cov-v3 python tests/regression/multi_seed_v3.py --anchor m5_hybrid --n 30
```

The release gate is `../test_m5_parity.py` (slow, per backend). Once quarantine reduces both
transmissibility AND susceptibility (the v3 `quar_factor` semantics), **every gated metric matches
v3 within |z|<2** (cum_infections z≈−0.1, cum_diagnoses |z|<1.3, quarantine/isolation/deaths/peak all
< 2). `cum_tests` is **informational** (not gated): the testing volume matches to ~2%, but its
cross-seed SE is so tiny that the residual reads as |z|~8 on the random backend — the irreducible
Starsim-CRN-vs-v3-RNG offset (analogous to M2's documented residual). The iso/quar transmissibility
factors are a scalar M5 approximation of v3's per-layer values (spec Open Q A; the per-layer
refinement would tighten hybrid further but the aggregate already matches).
