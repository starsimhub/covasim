# Regression harness (v3.1.8 -> v4.0 migration)

This directory holds the self-contained regression harness for the Covasim v4.0 Starsim port. It does two complementary jobs:

- **Development gate (`compare.py`)** -- a fast, informational one-seed +/-10% drift table for day-to-day porting feedback. Always exits 0; never blocks.
- **Release gates (`test_parity_*.py`)** -- the scientific gates: a multi-seed z-score parity check comparing N v4 seeds to M v3.1.8 seeds with overlapping uncertainty intervals, one gate per feature area. NOT bit-for-bit (the RNG stream differs between v3.1.8's global numba RNG and v4's Starsim CRN).

This layers ON TOP OF Covasim's existing baseline machinery (`../baseline.json` + `../test_baselines.py` + `../update_baseline` + `../../covasim/regression/`), which stays the v4-internal bit-for-bit self-consistency gate. The two answer different questions and both stay.

## What's here

| File | Role |
|---|---|
| `anchor.py` | Pinned vanilla anchor: hybrid pop, waning ON, seed 0, no interventions. `PARS`, `make_sim()`, `run_and_summarize()`; runs as `__main__`. |
| `anchor_transmission.py` | Basic-transmission anchor (single variant, no waning, pure SEIR). |
| `anchor_natural_history.py` | Full natural-history anchor (single variant, no waning, full prognosis tree). |
| `anchor_variants.py` | Multi-variant + cross-immunity anchor (wild + alpha + delta). |
| `anchor_waning.py` | The multi-variant anchor with waning immunity + NAbs on. |
| `anchor_testing.py` | Natural-history anchor plus testing + contact tracing. |
| `anchor_vaccination.py` | Single-variant anchor with waning on plus a pfizer vaccination campaign. |
| `short_summary.py` | `build_summary(sim)` and the per-anchor `build_summary_<feature>(sim)` -> flat `{metric: float}` dicts. `METRIC_KEYS`, `METRIC_KEYS_<FEATURE>`, `SKIP_KEYS`. |
| `parity.py` | `parity_gate(v4, v3, z_threshold=3.0)` z-score helper (ported ~verbatim from hpvsim). |
| `contact_stats.py` | Contact-structure metrics (per-layer degree, age-mixing matrix) used by `../test_network.py`. |
| `multi_seed_v3.py` | CLI: sweep an anchor across N seeds **in a frozen v3.1.8 env** -> gitignored `v3_*seeds_n{N}.json`. |
| `multi_seed_v4.py` | (optional) same sweep in-env -> gitignored `v4_seeds_n{N}.json`, for ad-hoc diffing. |
| `compare.py` | `compute_drift()` + CLI: one-seed `+/-10%` drift table; `--save-snapshot`; no-baseline mode. |
| `__init__.py` | Empty; makes this an importable package. |
| `test_parity_*.py` | The release gates: one multi-seed z-score parity check against v3 per feature area (slow). |
| `test_regression_harness.py` | Fast unit and smoke tests of the harness itself (anchor, `parity_gate`, `compute_drift`). |
| `test_v3_compat.py` | Typical v3 user code run against v4, one test per snippet; `v3_saved.sim` is a small sim saved by v3.1.9, used to check that v3 objects can be loaded. |
| `test_sim_v3.py` | The v3 version of `../test_sim.py`, run against v4. |
| `test_migration.py` | Tests of the script for migrating v3 code to v4 (`cv.migrate3to4`). |

## Anchor names and baseline files

Both `multi_seed_v3.py` and `compare.py` take `--anchor <name>`. The names, and the gitignored v3.1.8 baselines they produce, are:

| `--anchor` | Anchor file | Baseline file | Parity gate |
|---|---|---|---|
| `vanilla` (default) | `anchor.py` | `v3_seeds_n30.json` | `test_parity_anchor.py` (skipped) |
| `transmission_random` / `transmission_hybrid` | `anchor_transmission.py` | `v3_transmission_<pt>_seeds_n30.json` | `test_parity_transmission.py` |
| `natural_history_random` / `natural_history_hybrid` | `anchor_natural_history.py` | `v3_natural_history_<pt>_seeds_n30.json` | `test_parity_natural_history.py` |
| `variants_random` / `variants_hybrid` | `anchor_variants.py` | `v3_variants_<pt>_seeds_n30.json` | `test_parity_variants.py` |
| `waning_random` / `waning_hybrid` (`compare.py` only) | `anchor_waning.py` | reuses `v3_variants_<pt>_seeds_n30.json` | `test_parity_waning.py` |
| `testing_random` / `testing_hybrid` | `anchor_testing.py` | `v3_testing_<pt>_seeds_n30.json` | `test_parity_testing.py` |
| `vaccination_random` / `vaccination_hybrid` | `anchor_vaccination.py` | `v3_vaccination_<pt>_seeds_n30.json` | `test_parity_vaccination.py` |

The contact-structure equivalence check in `../test_network.py` consumes a separate `v3_contacts.json` baseline (also v3.1.8-env-generated).

## Vanilla anchor scenario

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

No interventions, no analyzers. `hybrid` + `use_waning` exercises the population-structure and NAb/immunity machinery -- the highest-risk parts of the port -- without confounding from intervention ports. Intervention/vaccine behavior is covered by the testing and vaccination anchors.

## Pinned summary set

Gated metrics (`METRIC_KEYS`): `cum_infections`, `cum_reinfections`, `cum_symptomatic`, `cum_severe`, `cum_critical`, `cum_deaths`, `peak_prevalence`, `peak_n_infectious`, `prevalence`, `incidence`. The cumulative/derived metrics come from `sim.summary`; the two peaks are computed from the `sim.results` time series. `r_eff` is omitted from the gate (version-sensitive; Covasim's own `test_regression.py` skips it). Bookkeeping keys `_seed`, `_total_pop`, `n_alive` are written but skipped by the gate.

## Generating the v3.1.8 baselines (gitignored)

Each baseline is a 30-seed sweep, regenerated from a FROZEN v3.1.8 environment and never committed. The simplest way to get a v3.1.8 build is a worktree of `main`:

```bash
git worktree add /tmp/cov-v3 main
PYTHONPATH=/tmp/cov-v3 python tests/migrate3to4/multi_seed_v3.py --n 30                                  # vanilla
PYTHONPATH=/tmp/cov-v3 python tests/migrate3to4/multi_seed_v3.py --anchor transmission_random --n 30
PYTHONPATH=/tmp/cov-v3 python tests/migrate3to4/multi_seed_v3.py --anchor transmission_hybrid --n 30
# ... and likewise for natural_history_*, variants_*, testing_*, vaccination_*
```

(`PYTHONPATH=/tmp/cov-v3` makes `import covasim` resolve to the v3.1.8 worktree, not the editable v4 install; the harness duck-types on `cv.COVID` to pick the v3-vs-v4 branch.) This writes `tests/migrate3to4/v3_*seeds_n30.json` (gitignored). Back in the v4 env, the release gates consume them automatically.

## Running the release gates (z-score parity)

```bash
cd tests && pytest migrate3to4/test_parity_transmission.py -m slow -v   # or any other test_parity_*.py
```

Each gate runs 10 v4 seeds per backend, loads the 30-seed v3.1.8 baseline, and fails any gated metric with `|z|` at or above its threshold. **Skips cleanly** (does not fail) if the baseline JSON is absent, so contributors without a v3.1.8 env can still run the rest of the suite.

z-formula: `z = (v4_mean - v3_mean) / sqrt(v3_SE^2 + v4_SE^2)`, `SE = std(ddof=1)/sqrt(n)`. Degenerate distributions: zero combined spread + equal means passes; zero spread + unequal means fails (`z = inf`).

## Running the development gate (drift)

```bash
# From a v3.1.8 env, snapshot one seed:
python tests/migrate3to4/compare.py --save-snapshot
# From the v4 env, diff against it:
python tests/migrate3to4/compare.py
```

Output: a `key | baseline | current | rel_diff | over` table; always exit 0. No-baseline mode (missing snapshot) prints a notice and exits without running the anchor -- this is the mode CI smoke-runs. Use `--anchor` and `--baseline` to snapshot/diff a feature anchor instead of the vanilla one.

## When to refresh the baselines

- After a v3.1.8 patch-equivalent change lands and is forward-merged into `starsim-port`.
- After an explicit decision that drift introduced by a change is the new target.
- Otherwise: don't. Stable baseline = stable signal.

## CI

CI runs the pytest suite (which collects the harness unit tests + anchor smoke test + the skipped slow gates) plus `python migrate3to4/compare.py` in no-baseline mode (CLI-integrity only). Neither fails on drift. The heavy multi-seed sweeps run only locally or in a future nightly job, never in the 5-minute PR job.

## Basic transmission

`anchor_transmission.py` is the single-variant basic-transmission anchor for the `random` and `hybrid` backends. The same file runs under v3.1.8 (configured to a transmission+recovery-only SEIR: `use_waning=False`, all-asymptomatic prognoses, `asymp_factor=1.0`) and under v4 (`cv.Sim`), so one anchor serves both the baseline and the gate. `build_summary_transmission` (in `short_summary.py`) extracts the transmission metrics (`cum_infections`, `peak_prevalence`, `peak_n_infectious`) from either engine.

The release gate `test_parity_transmission.py` (slow, `|z| < 3`, per backend) compares v4 to those baselines. The contact-structure equivalence half (per-layer degree + age-mixing) lives in `../test_network.py` and consumes the `v3_contacts.json` baseline.

## Natural history

`anchor_natural_history.py` is the single-variant **full-natural-history** anchor (the prognosis tree + viral_load/beta_dist). Unlike `anchor_transmission`, the v3 branch keeps the **default** age-based prognoses (the symptomatic disease course is the point) -- only `use_waning=False` + `n_variants=1`. `build_summary_natural_history` extracts both the burden metrics (`cum_symptomatic`/`cum_severe`/`cum_critical`/`cum_deaths`) and the transmission metrics.

The release gate is `test_parity_natural_history.py` (slow, per backend). **It uses `|z| < 5`** (not the default `|z| < 3`) by an explicit documented decision (signed off 2026-05-29): after matching v3's integer duration rounding, every metric agrees within ~3% in magnitude, but the 40-seed standard error is so small that a ~3% *systematic* offset (Starsim CRN vs v3 numba RNG + per-day viral-load discretization, irreducible without bit-for-bit equivalence) still reads as `|z|` up to ~3.5. `|z| < 5` admits that scientifically negligible band while still catching genuine regressions; see the rationale block at the top of `test_parity_natural_history.py`.

## Variants and cross-immunity

`anchor_variants.py` is the **multi-variant** anchor for the `random` and `hybrid` backends: wild seeded at t0, **alpha introduced at day 10** and **delta at day 30** (`n_imports=20` each), `pop_size=20_000`, `n_days=120`. The same file runs under v3.1.8 (with **cross-immunity active**, i.e. `use_waning=True` — the realistic multi-variant regime) and under v4 (`cv.Sim(variants=[...])`). `build_summary_variants` (in `short_summary.py`) extracts aggregate metrics (`cum_infections`, `cum_deaths`, `peak_n_infectious`, `peak_prevalence`) plus the per-variant `cum_infections_<v>` / `peak_n_infectious_<v>` for wild/alpha/delta.

The release gate is `test_parity_variants.py` (slow, per backend).

**Documented static-vs-NAb divergence.** The static (NAb-free) cross-immunity path writes `sus_imm = matrix[target, source]` directly, whereas v3 weights it by the per-agent neutralizing-antibody titre (`sus_imm = calc_VE(nab × matrix)`). Consequently the v4 and v3 trajectories agree where the static matrix suffices but diverge where NAb kinetics dominate:

- **Converges (GATED, `|z| < 5`):** `cum_infections_wild` (≈ `|z| 0`, even with multi-variant reinfection feedback), `peak_n_infectious`, `peak_prevalence`. These validate the core multi-variant machinery (per-variant transmission, host exclusivity, the cross-immunity connector, reinfection).
- **Diverges (INFORMATIONAL, not gated):** the per-variant alpha/delta absolute counts and aggregate `cum_infections`. The gap is largest for the **late-introduced escape variant delta** (`matrix[delta, wild]=0.374`, so v4 wild-recovered are only ~37% protected and delta finds a large susceptible pool): v4 has ~7–10× more delta and ~55% more total infections than v3 (`|z|` up to ~46). This is by design; the NAb engine (see below) re-converges these. A related divergence: same-variant reinfection is **exactly 0** on the static path (`matrix[v,v]=1.0`), whereas v3's `calc_VE(nab×1.0) < 1` permits a small amount.

The gate therefore hard-gates only the convergent subset and prints the full per-metric table (`[GATE]`/`[info]`) for diagnostics; see the rationale block at the top of `test_parity_variants.py`.

## Waning immunity and NAbs

`anchor_waning.py` is the multi-variant anchor (wild + alpha@d10 + delta@d30) **with the NAb engine on** (`use_waning=True`). The v3.1.8 side is identical to the variants anchor's v3 branch (which already runs `use_waning=True`), so **it reuses the variants v3.1.8 baseline** (`v3_variants_<pt>_seeds_n*.json`) — no separate baseline. `build_summary_variants` serves both gates.

The release gate is `test_parity_waning.py` (slow, per backend). Where the *static* cross-immunity diverges from v3 on the per-variant escape dynamics (delta `|z|~25-46`, gated only on a convergent subset), the NAb-weighted cross-immunity (`sus_imm = calc_VE(nab × matrix)`) **re-converges every pinned metric** — aggregate burden AND per-variant wild/alpha/delta counts — to within `|z|<3.5` of the same v3 baseline. So this gate hard-gates the WHOLE metric set at `|z|<5`: the static-vs-NAb divergence closes once NAbs are on.

## Testing, tracing and quarantine

`anchor_testing.py` is the single-variant natural-history scenario plus a `test_prob` testing intervention and a `contact_tracing` intervention (same public API in v3.1.8 and v4). `build_summary_testing` pins the burden (`cum_infections`/`cum_deaths`/`peak_n_infectious`) plus the testing/quarantine outcomes (`cum_tests`/`cum_diagnoses`/`peak_n_quarantined`/`peak_n_isolated`).

The release gate is `test_parity_testing.py` (slow, per backend). With quarantine reducing both transmissibility AND susceptibility (the v3 `quar_factor` semantics), **every gated metric matches v3 within |z|<2** (cum_infections z≈−0.1, cum_diagnoses |z|<1.3, quarantine/isolation/deaths/peak all < 2). `cum_tests` is **informational** (not gated): the testing volume matches to ~2%, but its cross-seed SE is so tiny that the residual reads as |z|~8 on the random backend — the irreducible Starsim-CRN-vs-v3-RNG offset (analogous to the natural-history residual). The iso/quar transmissibility factors are a scalar approximation of v3's per-layer values (the per-layer refinement would tighten hybrid further but the aggregate already matches).

## Vaccination

`anchor_vaccination.py` is the single-variant natural-history scenario with `use_waning=True` plus a pfizer vaccination campaign (`vaccinate_prob('pfizer', days=20, prob=0.05)`). `build_summary_vaccination` pins the burden (`cum_infections`/`cum_severe`/`cum_deaths`/`peak_n_infectious`) plus the vaccination outcomes (`cum_doses`/`cum_vaccinated`).

The release gate is `test_parity_vaccination.py` (slow, per backend). Because vaccine immunity shares the waning-immunity NAb pipeline (which re-converges to v3 at |z|<3.5), the vaccinated trajectory tracks v3: **every pinned metric matches within |z|<2.1** — burden, peak, doses, and vaccinated. (`cum_infections` counts infection EVENTS = sum of `cum_infections_by_variant`, matching v3's flow definition, since `use_waning=True` produces reinfections.) Gate threshold |z|<5 as for the other feature gates.
