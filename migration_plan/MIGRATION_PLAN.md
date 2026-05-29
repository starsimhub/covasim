# Covasim v4.0 Migration Plan

## Overview

Covasim v4.0 is a reimplementation of Covasim on the [Starsim](https://starsim.org/) agent-based modeling framework. The original Covasim (v3.1.8) is a mature, self-contained stochastic ABM with its own engine — `base.py` (`ParsObj`/`BaseSim`/`BasePeople`), a struct-of-arrays `People` health-state machine in `people.py`, contact-network construction in `population.py`, the NAb/waning/cross-immunity engine in `immunity.py`, `interventions.py`, `analysis.py`, `run.py`, and the integration loop in `sim.py`. v4.0 inherits Starsim's core classes (`ss.Sim`, `ss.Disease`/`ss.Infection`, `ss.Network`, `ss.Intervention`, `ss.Analyzer`, `ss.Connector`, `ss.MultiSim`) while keeping Covasim's domain-specific logic: the COVID natural-history chain (exposed → infectious → asymptomatic/mild/severe/critical → recovered/dead), multi-variant dynamics, neutralizing-antibody waning immunity, testing/tracing/quarantine, and vaccination. v4.0 targets Starsim 3.3.x (3.3.4 at time of writing).

Migration work happens on the existing `starsim-port` branch (created off `main` by Cliff, already checked out). No other branches are created — all milestones land on `starsim-port`. v4.0 is considered **done** when the validation bar is met (see below): the existing Covasim test suite passes against the Starsim-based build, *and* new multi-seed z-score parity gates show that a v4 run reproduces v3.1.8-generated baselines with overlapping uncertainty intervals on the headline epidemiological metrics. Bit-for-bit reproducibility is explicitly **not** required — the move from Covasim's single global RNG stream to Starsim's per-distribution Common Random Numbers changes the random stream, so exact numbers will differ.

## Validation criteria

Validation is **self-contained**: it depends only on this repo and a frozen v3.1.8 reference build. There are no external analysis or validation repos in the validation set. "Done" has two parts, both of which must hold:

1. **The existing Covasim test suite passes** against the v4/Starsim-based build — `test_sim`, `test_immunity`, `test_interventions`, `test_baselines`, `test_regression`, `test_analysis`, `test_run`, and the rest of `tests/test_*.py`. The strict-warnings bar applies: `run_tests` sets `COVASIM_WARNINGS=error`, so any new warning the port emits is a failure even when numbers match. The only except is `test_baselines.py`, which is allowed to fail and be updated with new baselines, as long as the z-scores overlap (see next point).
2. **Multi-seed z-score parity gates** compare a v4 Starsim-based run to v3.1.8-generated baselines and require **overlapping uncertainty intervals**, adapting the hpvsim multi-seed `|z| < 3` pattern. This is *not* bit-for-bit equivalence — it is statistical-equivalence on the headline metrics across many seeds.

The z-score gate (per pinned metric, over N v4 seeds vs M v3.1.8 seeds): `z = (mean_v4 − mean_v3) / sqrt(SE_v3² + SE_v4²)`, where `SE = std(ddof=1)/sqrt(n)`; a metric passes when `|z| < 3`. Degenerate distributions are handled as in the hpvsim `parity.py`: zero-spread with equal means passes; zero-spread with unequal means fails (`z = inf`). The pinned summary-result set is drawn from Covasim's own `sim.summary` (the same flat dict `tests/baseline.json` stores) plus epidemic-shape metrics computed from the results time series. The v3.1.8 baselines are generated locally from a frozen v3.1.8 environment and are **gitignored, never committed** (the generator scripts, anchor, parity helper, and pytest gate are committed). The parity gate **skips** (does not fail) when the baseline file is absent, so contributors without a v3.1.8 environment can still run the rest of the suite.

This validation layers on top of Covasim's existing regression machinery rather than replacing it: `tests/baseline.json` + `test_baselines.py` remains the v4-internal bit-for-bit self-consistency gate (regenerated for v4 via `tests/update_baseline` once the port lands), and the new parity gate is the cross-implementation v3↔v4 scientific-equivalence gate. They answer different questions and both stay.

## Scope decisions (settled)

Settled with the project owner (Cliff Kerr). Each bullet is a feature name + disposition.

**Backwards compatibility is the overriding constraint.** The target is **>99% backwards-compatible**: every public Covasim name keeps its name, call signature, and behavior — `cv.Sim`, `cv.People`, `cv.MultiSim`, `cv.Scenarios`, `cv.Result`, `cv.Layer`, `cv.Contacts`, `cv.options`, `cv.Intervention`, `cv.Analyzer`, every intervention/analyzer (`cv.test_num`, `cv.contact_tracing`, `cv.vaccinate_prob`, `cv.simple_vaccine`, `cv.snapshot`, `cv.TransTree`, …), `cv.variant`, `cv.Fit`, `cv.Calibration`, and the module-level helpers (`cv.get_prognoses`, `cv.get_cross_immunity`, `cv.get_variant_pars`, `cv.sample`, `cv.parallel`, `cv.multi_run`, …). **Nothing is renamed for the sake of renaming.** New names are introduced *only* where Covasim has no class at all (the disease module, the per-layer network, and the immunity connector — see the *New classes* note under Architecture mapping); each is flagged for Cliff's sign-off. If preserving a specific name turns out to be genuinely untenable, it is raised explicitly, never changed silently.

- **Multi-variant / strains** (`cv.variant`, alpha/beta/gamma/delta/wild): **Port.**
- **Waning immunity + neutralizing antibodies (NAbs):** **Port.** See the explicit divergence note below.
- **Cross-immunity matrix** (the `n_variants × n_variants` `immunity` matrix): **Port.**
- **Testing interventions** (`test_num`, `test_prob`): **Port.**
- **Contact tracing** (`contact_tracing`): **Port.**
- **Quarantine / isolation** (the People-level state machine driven by `quar_factor`/`iso_factor`): **Port.**
- **Vaccination** (`vaccinate_prob`, `vaccinate_num`, vaccine products, per-variant efficacy, NAb interaction): **Port.**
- **Scenarios class** (`cv.Scenarios`): **Port.**
- **SynthPops population backend** (`pop_type='synthpops'`): **Port.**
- **TransTree analysis** (`cv.TransTree`): **Port.**
- **`bin/covasim` CLI wrapper:** **Drop** (the only dropped subsystem; not Windows-compatible, rarely used, and superseded by direct API use).

### Explicit divergence from the hpvsim plan: waning immunity + NAbs are CORE and MUST be ported

The hpvsim migration plan **dropped waning immunity** ("never used in any published analysis") and reduced HPVsim's immunity to a static running-max of beta samples with no time decay. **Covasim is the opposite case.** Waning immunity and neutralizing-antibody (NAb) kinetics are central to COVID-19 dynamics — reinfection, vaccine waning, and variant escape all depend on them — so the entire `immunity.py` engine (per-agent `peak_nab`/`nab`/`t_nab_event`, the `nab_kin` precomputed waning kernel, `update_peak_nab` boosting, the NAb→protection mapping `calc_VE` along the sus/symp/sev axes, the cross-immunity matrix, and `check_immunity`) **must be ported in full**. There is no Starsim equivalent: Starsim 3.3.4 has only the trivial scalar exponential waning in `ss.SIS` and no multi-strain, NAb, or cross-immunity facility. The COVID `Immunity`/cross-immunity connector therefore cannot simply matrix-multiply a static immunity array once; it must, every step, (a) advance each agent's NAb along its kinetic curve from `peak_nab`/`t_nab_event`, (b) map current NAb to per-axis efficacy via `calc_VE`, then (c) apply the cross-reactivity matrix and write per-variant `rel_sus`/`symp`/`sev` protection back. This is the single largest piece of net-new porting work relative to the hpvsim template (M4 below).

## Workflow

This is a local effort. There is no GitHub RACI table and no GitHub issues/PRs as gates. The cadence is **pause-for-review-and-commit**:

- The assistant (Claude) prepares each piece of work and leaves it **uncommitted** in the working tree, then pauses for Cliff Kerr (the Covasim author) to review and commit. **The assistant never commits and never pushes.**
- Check-ins happen **2–5 times per milestone** — at natural sub-task boundaries — so Cliff reviews and commits incrementally rather than in one large batch at the end.
- **Milestone completion** = the acceptance test is green locally *and* Cliff has reviewed and committed the work. There are no issue numbers or PR merges as completion gates.

## Architecture mapping

All existing `cv.*` names are **preserved** (see Backwards compatibility above). The right column shows the Starsim base each one sits on; **bold "NEW"** marks the only three constructs with no Covasim original (provisional names — see the *New classes* note below the table).

| Covasim v3.1.8 (`main`) | Covasim v4.0 (`starsim-port`) — original names preserved |
|---|---|
| `cv.Sim` / `cv.BaseSim` / `cv.ParsObj` (`sim.py`, `base.py`) | `cv.Sim(ss.Sim)` — same public name; thin wrapper assembling the module stack (`cv.BaseSim`/`cv.ParsObj` kept) |
| `Sim.step` integration loop (`sim.py`) | Starsim `Loop` + module `step()`/`step_state()` ordering |
| `cv.People` / `cv.Person` / `cv.BasePeople` (`people.py`, `base.py`) | `cv.People(ss.People)` — same public name; `cv.Person`/`cv.BasePeople` kept |
| Health-state machine + transitions (`people.py:164-586`) | **NEW** disease module on `ss.Infection` — Covasim has no disease class today (provisional `cv.COVID`; see *New classes*). Exposed→infectious→asymp/mild/severe/critical→recovered/dead; states as `ss.BoolState`/`ss.FloatArr`; date-scheduled transitions via `ss.dur` timers |
| Prognoses, age-banded (`cv.get_prognoses`, `people.set_prognoses`) | `cv.get_prognoses` kept; `set_prognoses` on the disease module via age-dependent `ss.bernoulli`/lookup |
| `cv.Contacts` / `cv.Layer` (h/s/w/c/a/l) (`base.py:1509/1610`) | `cv.Contacts`/`cv.Layer` kept as the edge containers; a **NEW** `ss.Network` subclass per layer wraps them (provisional `cv.Network`) |
| Population build (`cv.make_randpop`/`cv.make_hybrid_contacts`/`cv.make_synthpop`) | same functions + names kept, now returning the network module(s) (random + hybrid + synthpops backends) |
| `compute_infections`/`compute_trans_sus` kernels (`utils.py:88/99`) | `ss.Infection.infect()` + CRN-safe `ss.multi_random('source','target')`; `cv.utils` helpers kept where still used |
| Variants (`cv.variant`, `cv.get_variant_pars`) (`immunity.py:18`) | `cv.variant`/`cv.get_variant_pars` kept; one disease-module instance per variant + a variant-introduction intervention |
| Cross-immunity matrix (`cv.get_cross_immunity`) (`immunity.py:269`) | `cv.get_cross_immunity` kept; a **NEW** `ss.Connector` applies it each step (provisional `cv.CrossImmunity`) |
| NAb waning + `calc_VE` + boosting + `check_immunity` (`immunity.py:138-350`) | **NEW** `ss.Connector`/module — live per-step NAb kinetics, **ported from scratch, no Starsim analog** (provisional `cv.Immunity`) |
| `cv.Intervention` base (`interventions.py:223`) | `cv.Intervention(ss.Intervention)` — same name |
| `cv.test_num` / `cv.test_prob` (`interventions.py:713/858`) | same names, on `ss.Intervention` |
| `cv.contact_tracing` + quarantine/isolation (`interventions.py:984`, `people.check_quar/check_enter_iso`) | `cv.contact_tracing` kept + host-state quarantine/isolation machine |
| `cv.vaccinate_prob`/`cv.vaccinate_num`/`cv.vaccinate`/`cv.BaseVaccination`/`cv.simple_vaccine`/`cv.historical_vaccinate_prob` (`interventions.py`) | all names kept (`ss.Intervention`); products via `ss.Vx`-style classes |
| `cv.dynamic_pars`/`cv.sequence`/`cv.change_beta`/`cv.clip_edges`/`cv.historical_wave`/`cv.prior_immunity` (`interventions.py:413-589`) | all names kept (`ss.Intervention`); `dynamic_pars` via dotted-path resolution into `sim.diseases`/`sim.interventions`/`sim.pars` |
| `cv.Analyzer` base (`analysis.py:23`) | `cv.Analyzer(ss.Analyzer)` — same name |
| `cv.snapshot`/`cv.age_histogram`/`cv.daily_age_stats`/`cv.daily_stats`/`cv.nab_histogram` (`analysis.py`) | all names kept, on `ss.Analyzer` |
| `cv.Fit` / `cv.Calibration` / `cv.compute_gof` (`analysis.py:991/1358`) | names kept; `cv.Calibration` runs on Starsim's Optuna calibration under the hood |
| `cv.TransTree` (`analysis.py:1772`) | `cv.TransTree` kept; built over the infection log |
| `cv.MultiSim`/`cv.parallel`/`cv.multi_run`/`cv.single_run` (`run.py:36/1406/1522`) | all names kept; wrap `ss.MultiSim` / `ss.parallel` |
| `cv.Scenarios` (`run.py:861`) | `cv.Scenarios` kept; adapted onto `ss.MultiSim` over parameter sets |
| `cv.Result` (`base.py:117`) | `cv.Result` kept (subclass/alias of `ss.Result`) |
| `cv.options` / `cv.Options` (`settings.py:626`) | `cv.options` kept (backed by `ss.options` where sensible) |
| `cv.sample` + `cv.true`/`cv.false`/index helpers (`utils.py:161-674`) | names kept; internally use `ss.Dist` (CRN) + `ss.uids`/`BoolArr` ops |
| dynamic rescaling (`pop_scale`, `rescale`, `make_naive`) (`sim.py:535`) | **ported in a later milestone** (no direct Starsim analog); M2 starts with absolute counts (`total_pop`/`pop_scale`) — kept, not dropped |
| `bin/covasim` CLI wrapper | **Dropped** (the only dropped subsystem) |

### New classes (no Covasim original — names need your sign-off)

Backwards compatibility means keeping every existing `cv.*` name (above). Only three constructs have **no** Covasim original, because Covasim packs that logic into `People`/`Sim`/`immunity.py` *functions* rather than dedicated classes. Starsim is module-based, so the port needs a class for each. Provisional names below are flagged for approval; nothing else is renamed:

1. **The disease module** (`ss.Infection` subclass holding the COVID natural-history chain). Provisional `cv.COVID`. Alternatives: `cv.SARSCoV2`, `cv.Covid`, `cv.Disease`, or keeping the chain inside `cv.People`. Users never instantiated a disease object in v3, so any choice is *additive*, not breaking.
2. **The per-layer network** (`ss.Network` subclass wrapping `cv.Layer`/`cv.Contacts`). Provisional `cv.Network`. Users build populations via `pop_type=`, not by instantiating a network, so this too is additive.
3. **The immunity / cross-immunity connector(s)** (`ss.Connector` running NAb kinetics + the cross-immunity matrix each step). Provisional `cv.Immunity` (optionally split into `cv.CrossImmunity`).

## Milestones

Each milestone produces a user-visible demo and must meet its acceptance test before the next begins. Tests written during a milestone stay in the suite from that point onward. Sub-tasks are concrete units of work; the always-runnable invariant (Implementation conventions item 1) holds at every commit. The validation progression mirrors hpvsim: early single-sim milestones validate against fixed-seed v3.1.8 baselines via the ±10% drift harness; M8 introduces the multi-seed z-score parity gates and retrofits earlier acceptance tests onto them.

### M0: Foundation

**Demo:** `starsim-port` branch exists with CI green on a stub `cv.Sim(ss.Sim)` that runs; `MIGRATION_PLAN.md` and the M0 plan/spec docs are present; the v3.1.8 baseline-generation script is committed (baseline files themselves are generated locally and gitignored, never committed); the anchor-scenario script, the multi-seed sweep script, the parity helper, and the comparison CLI are committed.

**Acceptance test:** CI is green on a stub `cv.Sim(ss.Sim)` that runs; the self-contained regression harness compares a v4 run to a local v3.1.8 baseline and reports drift. Concretely: the existing `pytest test_*.py` step passes; a new no-baseline-mode CLI smoke step proves the comparison tooling doesn't bitrot; a developer can locally generate a v3.1.8 baseline and run the comparison to get a per-metric drift table.

**Sub-tasks:**
- Use the existing `starsim-port` branch (already created off `main` and checked out; the assistant does not create or switch branches). Prepare the scaffolding for Cliff to commit. `main` is frozen for v4 work; all milestones land directly on `starsim-port` — there are no per-milestone branches.
- Adapt CI (`.github/workflows/tests.yaml`): leave the existing `pytest test_*.py unittests/test_*.py -n auto` step as-is (`test_regression.py`-style files at the `tests/` root are auto-collected); insert one new step running the comparison CLI in **no-baseline mode** so it exits clean. Keep the heavy multi-seed parity sweep out of the 5-minute PR job — it runs as a separate/optional/nightly job.
- Build the self-contained regression harness under `tests/regression/` (importable package): `anchor.py` (the pinned anchor scenario + `make_sim()` / `run_and_summarize()`), `short_summary.py` (`build_summary(sim)` → flat `{metric: float}` from `sim.summary` plus epidemic-shape metrics), `parity.py` (the `_mean_se` + `parity_gate(..., z_threshold=3.0)` helper, ported essentially verbatim from hpvsim's `parity.py`), `multi_seed_v3.py` (CLI sweeping the anchor across N seeds, intended to run in a frozen v3.1.8 env), an optional `multi_seed_v4.py`, and a `README.md`.
- Reuse Covasim's existing baseline machinery: the harness layers the v3↔v4 drift comparison *on top of* `tests/baseline.json` / `test_baselines.py` / `tests/update_baseline` / `covasim/regression/` rather than replacing them.
- Set up the `_v2_legacy` quarantine scaffold (`covasim/_v2_legacy/` with a pure-docstring `__init__.py`, no imports) and `tests/_legacy/`; no migration code lands in M0, so nothing is quarantined yet.
- Gitignore the generated baselines: add `tests/regression/v3_seeds_n*.json` and `tests/regression/v4_seeds*.json` to `.gitignore`.
- Write the M0 plan (`migration_plan/plans/`) and design spec (`migration_plan/specs/`); add a pointer from `tests/README.md` to `tests/regression/README.md`.

The pinned anchor scenario for M0 is a representative-but-clean vanilla sim (isolates core-dynamics drift, not intervention-port bugs): `pop_size=20_000`, `pop_infected=100`, `pop_type='hybrid'`, `n_days=120`, `use_waning=True`, `rand_seed=0` (the sweep overrides 0..N−1), `verbose=0`, no interventions, no analyzers. The pinned summary set is drawn from `sim.summary` — cumulative burden (`cum_infections`, `cum_reinfections`, `cum_symptomatic`, `cum_severe`, `cum_critical`, `cum_deaths`), epidemic shape (peak `prevalence`, peak `n_infectious`, computed from `sim.results`), and derived (`prevalence`, `incidence`); bookkeeping keys (`n_alive`, `_seed`, `_total_pop`) are written but skipped by the gate. `r_eff` is treated as a soft/optional metric (Covasim's own `test_regression.py` skips it as version-sensitive).

### M1: Basic transmission sim

**Demo:** Run an epidemic on a random or hybrid population and plot the infection curve over time.

**Acceptance test:** The infection trajectory overlaps v3.1.8 on matched parameters; contact-layer structure (degree by layer, age mixing) matches v3.1.8.

**Sub-tasks:**
- Add a thin `cv.Sim(ss.Sim)` wrapper that assembles the module stack and forwards to `ss.Sim`; rely on stock `ss.People` (or a thin `cv.People`) for now.
- Port the multilayer contact network as `cv.Network(ss.Network)` via lift-and-shift from `population.py` — household/school/work/community layers, with both the **random** (single layer `a`) and **hybrid** (`h`/`s`/`w`/`c`) backends; carry over the per-layer `beta_layer` and `contacts` parameters and the age restrictions (school 6–22, work 22–65).
- Add a minimal single-variant `cv.COVID(ss.Infection)` (transmission + recovery only; SEIR-style exposed→infectious→recovered), wiring per-edge transmission via `ss.Infection.infect()` with CRN-safe `ss.multi_random('source','target')` (replacing the `compute_infections`/`compute_trans_sus` numba kernels).
- Quarantine v3 modules untouched by M1 (the full engine) into `covasim/_v2_legacy/`; move v3 tests that exercise removed APIs into `tests/_legacy/`.
- Add tests: contact-layer structure equivalence (degree distribution per layer, age-mixing matrix) and infection-trajectory overlap vs. a new M1 single-variant baseline.

### M2: Natural-history parity

**Demo:** Single-variant disease progression — exposed → infectious → (asymptomatic / mild / severe / critical) → recovered / dead — matching v3.1.8.

**Acceptance test:** Cumulative infections / symptomatic / severe / critical / deaths are within tolerance of a v3.1.8 single-variant baseline.

**Sub-tasks:**
- Port the full single-variant progression into `cv.COVID(ss.Infection)`: add `symptomatic`/`severe`/`critical`/`dead`/`recovered` states and the `infect()` decision tree (symptomatic? severe? critical? death?), with outcome dates pre-drawn from the `dur` distributions and scheduled as `ss.dur` timers (this preserves Covasim's date-driven transitions; per the porting risks, do **not** convert them to per-step rate checks).
- Port age-based prognoses (`get_prognoses`, the age-banded `symp_prob`/`severe_prob`/`crit_prob`/`death_prob`, comorbidity and OR scalers) into `cv.COVID.set_prognoses` using age-dependent `ss.bernoulli`/lookup.
- Port the severity scalers and health-system feedbacks (`rel_symp_prob`/`rel_severe_prob`/`rel_crit_prob`/`rel_death_prob`, `no_hosp_factor`/`no_icu_factor`, hospital/ICU bed caps).
- Port population scaling (`pop_scale`/`total_pop`) onto `ss.Sim`'s absolute-agent-count model. Covasim's *dynamic* rescaling (`rescale`, `make_naive`/`make_nonnaive`) has no direct Starsim analog; M2 starts with absolute agent counts, and dynamic rescaling **is ported in a later milestone** (kept, not dropped) — see the scope-items table.
- Add tests: cumulative infections/symptomatic/severe/critical/deaths match a v3.1.8 single-variant baseline within tolerance.

### M3: Multi-variant + cross-immunity

**Demo:** Multi-variant sim (e.g. wild + alpha + delta) with cross-immunity, matching v3.1.8 per-variant trajectories.

**Acceptance test:** Multi-variant prevalence and per-variant trajectories overlap v3.1.8.

**Sub-tasks:**
- Replicate `cv.COVID(ss.Infection)` per variant — one disease instance per variant (`diseases = [cv.COVID(variant=v) for v in [...]]`), each carrying its own `rel_beta`/`rel_symp_prob`/`rel_severe_prob`/`rel_crit_prob`/`rel_death_prob` (from `get_variant_pars`). Eliminate the genotype/variant array dimension in favor of N instances, following the hpvsim genotype→variant analogy.
- **Enforce host-level exclusivity over the SEIR/severity/death chain.** Unlike hpvsim genotypes (biologically real co-infection), a COVID host has ONE exposed→infectious→recovered/dead trajectory. A per-variant-`Disease` design must prevent an agent being independently active with two variants on two separate `ti_dead` schedules — either a host-level state machine owning the chain with per-variant modules contributing only `rel_beta`/immunity/severity modifiers, or an "exclusive infection" connector that blocks the other variant modules from infecting an already-infected host. This is net-new vs. hpvsim (whose only cross-genotype state coupling was the cancer-cancellation hack); decide the mechanism in the M3 design spec.
- Add `cv.CrossImmunity(ss.Connector)` implementing the `n_variants × n_variants` cross-immunity matrix (`get_cross_immunity`): discover the variant modules in `init_pre` (`[m for m in sim.diseases.values() if isinstance(m, cv.COVID)]`), column-stack their per-agent immunity arrays, matrix-multiply by the cross-protection matrix, and write per-variant `rel_sus` back (the `np.dot(cross_immunity, nab)` → column-stack-then-matmul refactor).
- Add coordinated variant introduction (the staggered-import seeding from `cv.variant.apply`) as an `ss.Intervention` (or a seeding `ss.Connector` like hpvsim's `_ExclusiveSeeder`), using `ss.poisson`-style imports.
- Add a `Total`-style analyzer that unions per-variant infection states into the familiar aggregate results (`n_infectious`, `cum_infections`) with union-counting so co-infections aren't double-counted.
- Add tests: multi-variant prevalence + per-variant trajectories overlap a v3.1.8 multi-variant baseline.

### M4: Waning immunity + NAbs

**Demo:** Reinfection and immune waning over a long horizon — NAb levels rise to a peak then decay, and reinfection/breakthrough dynamics match v3.1.8. **(Covasim-specific; no hpvsim analog — see Scope decisions.)**

**Acceptance test:** Immunity / reinfection dynamics match v3.1.8 `test_immunity` baselines.

**Sub-tasks:**
- Add the NAb state arrays (`peak_nab`, `nab`, `t_nab_event`) via `define_states` on the immunity module (or on `cv.COVID`).
- Port NAb acquisition and boosting (`update_peak_nab`): draw `init_nab` from `nab_init`, scale natural infection by symptom severity (`rel_imm_symp`) and normalize by `nab_eff`, apply `nab_boost` to agents with prior NAbs.
- Port waning kinetics: precompute the `nab_kin` waning kernel once (`precompute_waning` / `nab_growth_decay` linear-growth-then-two-phase-decay) and apply per-step `nab += nab_kin[t − t_nab_event] × peak_nab`, clamped to `[0, peak_nab]`.
- Port the NAb→protection mapping (`calc_VE`, inverse-logit along sus/symp/sev axes parameterized by `nab_eff`) and `check_immunity`: per step, combine natural cross-immunity (M3's matrix) and vaccine immunity (`max` of the two), compute `effective_nabs = nab × imm`, and set per-variant `sus_imm`/`symp_imm`/`sev_imm`, feeding `rel_sus` (transmission) and the symp/severe probabilities (infection outcomes).
- Build this as a live per-step computation inside `cv.Immunity(ss.Connector)` (or fold into `cv.CrossImmunity`) — the connector cannot read a static array once; it must advance NAb kinetics, map to efficacy, then apply cross-reactivity every step (the key divergence from hpvsim's static running-max immunity). The connector runs at the loop slot between `step_state` and disease `step()`/transmission, matching where v3 `check_immunity` runs.
- Port reinfection (recovery setting `susceptible=True` and clearing `diagnosed` under waning) and the breakthrough `trans_redux` factor.
- Add tests: immunity/reinfection dynamics match v3.1.8 `test_immunity` baselines.

### M5: Testing, tracing, quarantine

**Demo:** Run testing + contact tracing + quarantine/isolation and show diagnoses and quarantine counts over time.

**Acceptance test:** Testing / diagnoses / quarantine counts match v3.1.8 with these interventions.

**Sub-tasks:**
- Port `cv.test_num(ss.Intervention)` (fixed tests/day, symptomatic weighting, quarantine policy, ILI prevalence, subtargets, rescaling correction) and `cv.test_prob(ss.Intervention)` (per-person test probability by symptom/quarantine state).
- Port the People-level testing action (`people.test`) and the diagnosis state machine (`check_diagnosed`, `date_pos_test` → `date_diagnosed`).
- Port `cv.contact_tracing(ss.Intervention)`: on diagnosis (or presumptive on test), trace per-layer contacts (`trace_probs`, `trace_time`), set `known_contact`, and schedule quarantine; preserve the capacity limit and the no-testing-no-effect behavior.
- Port the isolation/quarantine state machine and its interactions: `check_quar` / `check_enter_iso` / `check_exit_iso`, the `quar_factor`/`iso_factor` per-layer beta modifiers, `quar_period`, and the diagnosis→isolation and trace→quarantine triggers. Implement the factors by modifying network beta or agent `rel_sus`/`rel_trans` via the intervention/host module.
- Add tests: testing/diagnoses/quarantine counts match v3.1.8 with these interventions wired in.

### M6: Vaccination

**Demo:** Run a vaccination campaign (e.g. Pfizer) and show the infection/severe/death reduction, including per-variant efficacy.

**Acceptance test:** The vaccination-impact trajectory matches v3.1.8.

**Sub-tasks:**
- Port `cv.vaccinate_prob` and `cv.vaccinate_num` (and the `cv.vaccinate` wrapper) as `ss.Intervention` subclasses, plus the `BaseVaccination` shared logic (vaccine-par parsing, dose scheduling by `interval`, second-dose prioritization, booster support).
- Port the NAb-based vaccine products and their per-variant efficacy tables (`get_vaccine_pars`, `get_vaccine_variant_pars`, `get_vaccine_dose_pars`, predefined pfizer/moderna/etc.), and the `target_eff` back-calculation into `nab_init`/`nab_boost`.
- Wire the vaccine→NAb interaction: vaccination calls the M4 immunity module's boost function (the `update_peak_nab` analog), setting `vaccinated`/`vaccine_source`/`doses`. Keep vaccine immunity flowing through the M4 NAb→protection pipeline (vaccine and natural NAbs share one efficacy curve).
- Port `cv.simple_vaccine` (the non-NAb, `use_waning=False` path), preserving its public API; it shares the M6 vaccination scaffolding but applies a direct efficacy rather than going through the NAb pipeline.
- Add tests: vaccination-impact trajectory matches a v3.1.8 vaccination baseline.

### M7: Calibration + Fit

**Demo:** Calibrate a small model to data and show the fit converge and reproduce a v3.1.8 fit.

**Acceptance test:** A small calibration converges and reproduces a v3.1.8 fit.

**Sub-tasks:**
- Port the model-vs-data `cv.Fit` (reconcile inputs, compute diffs/gofs/losses/mismatch, default weights `cum_deaths:10`, `cum_diagnoses:5`, custom series) onto Starsim, and `compute_gof`.
- Integrate Starsim's Optuna-based calibration with `cv.Sim` (the `cv.Calibration` analog), with `calib_pars=dict(key=[best,low,high])` and parallel workers.
- Add calibration tests that run a small number of trials end-to-end.
- Confirm a small calibration's posterior is consistent with a v3.1.8 calibration on the same target data.

### M8: MultiSim, Scenarios, parallel

**Demo:** Run N seeds, combine into uncertainty intervals for the earlier milestones, and run a `Scenarios` comparison.

**Acceptance test:** Multi-seed CIs are produced; `Scenarios` comparison works.

**Sub-tasks:**
- Verify `ss.MultiSim` works with `cv.Sim` (multi-seed runs, result aggregation, median + quantiles via `reduce`/`mean`/`median`) and that `ss.parallel` handles seed/reproducibility/`copy_inputs`/`inplace` semantics.
- Adapt `cv.Scenarios` onto `ss.MultiSim` over parameter sets (Starsim has no exact `Scenarios` equivalent; build the named-scenario comparison + median/quantile result storage on top of `ss.MultiSim`).
- **Retrofit the M1, M2, M3, M5, M6 acceptance tests onto multi-seed z-score parity gates** (the `|z| < 3` standard from the validation criteria), consuming the shared `parity_gate()` helper from `tests/regression/parity.py`. Generate the v3.1.8 multi-seed baselines (gitignored) from a frozen v3.1.8 env and the v4 seeds in-process. Gates may be loosened per-metric (e.g. `|z| < 5`) only with a documented residual rationale.
- Add a multi-seed UQ demo (e.g. `examples/m8_uq_sweep.py`) producing a median + 10/90-quantile trajectory plot, with a smoke variant in the test suite.

### M9: Analyzers, TransTree, plotting, synthpops

**Demo:** All secondary analyzers work; the TransTree and key Covasim figures reproduce; the synthpops population backend runs.

**Acceptance test:** Each analyzer matches v3.1.8; key figures reproduce; the synthpops backend runs and matches v3.1.8 on a synthpops baseline.

**Sub-tasks:**
- Port `cv.snapshot`, `cv.age_histogram`, `cv.daily_age_stats`, and `cv.nab_histogram` as `ss.Analyzer` subclasses.
- Port `cv.TransTree` as a custom `ss.Analyzer` over the infection log (sources/targets, `make_detailed`, optional NetworkX graph, `r0`).
- Implement Covasim-specific result views in `sim.plot()` and the `Fit` plots, rebuilt on Starsim plotting patterns.
- Port the synthpops population backend (`pop_type='synthpops'`, the LTCF layer `l`) as a custom population/network constructor (optional dependency).
- Add tests: each analyzer matches v3.1.8; the synthpops backend matches a v3.1.8 synthpops baseline.

### M10: Release readiness

**Demo:** Migration guide published; tutorials updated; docs build on Quarto; `pip install covasim==4.0.0` works.

**Acceptance test:** The full Covasim test suite is green against the v4 build; the multi-seed parity gates pass with overlapping intervals; the migration guide is merged; `v4.0.0` is tagged.

**Sub-tasks:**
- Write the migration guide (v3 → v4) documenting API changes, parameter remapping, and script conversion.
- Update the Quarto docs/tutorials (under `docs/`) and auto-generate the API reference (quartodoc).
- Verify save/load works with the new architecture.
- Strip all remaining subclass-first delegations (the interim `class Foo(ss.X)` delegations tracked across M1–M9). No delegations ship in v4.0.0.
- Regenerate `tests/baseline.json` / `benchmark.json` / `covasim/regression/pars_v4.0.0.json` for v4 via `tests/update_baseline`; reconcile or retire `example_regression.sim` (the frozen v1.7.0 pickle will likely break under the Starsim object model — regenerate from v3.1.8 or retire in favor of the parity gate).
- Delete the `covasim/_v2_legacy/` and `tests/_legacy/` quarantines wholesale.
- Run the full test suite + multi-seed parity gates; confirm overlapping intervals; tag v4.0.0.

## Scope items not pinned to a milestone

| Item | Suggested home | Notes |
|---|---|---|
| Population scaling (`pop_scale` / `total_pop`) | M2 | Required for long-horizon and large-population runs |
| Dynamic rescaling (`rescale`, `make_naive`/`make_nonnaive`) | Later milestone (ported, not dropped) | No direct Starsim analog; M2 uses absolute agent counts, dynamic rescaling is ported afterward |
| `simple_vaccine` (non-NAb path) | M6 | Ported alongside the other vaccination paths (preserves the `use_waning=False` direct-efficacy behavior) |
| Historical/prior immunity (`historical_vaccinate_prob`, `historical_wave`, `prior_immunity`) | M6 | Pre-t=0 NAb imprinting; depends on the M4 immunity engine |
| `sequence` intervention | M5 or opportunistic | Generic intervention scheduler; small |
| Save/load | M10 or opportunistic | Not capability-blocking |
| `pars_v3.1.8.json` parameter snapshot | M0 or opportunistic | Currently missing from `covasim/regression/`; forensic only |
| `example_regression.sim` regeneration vs. retirement | M10 | The v1.7.0 pickle likely breaks under the Starsim object model |
| Global-RNG vs. CRN reproducibility semantics | M1 | Covasim's single global numpy/numba stream vs. Starsim per-distribution CRN; reproducibility differs — pin expectations early |
| `beta_dist` / `viral_dist` dispersion and time-varying viral load | M2 | Per-agent transmissibility heterogeneity; folds into `cv.COVID` `rel_trans` |
| Cross-repo naming sync with Starsim conventions | M10 | Align parameter/class names where Starsim has a standard (`ss.Pars`, network keys via `ss.standardize_netkey`) |

## Out of scope

Not ported to v4.0 (can revisit in a future release):

- **`bin/covasim` CLI wrapper** — not Windows-compatible, rarely used, superseded by direct API use. (The only dropped subsystem.)

## Implementation conventions

These conventions apply to every milestone; contributors should align on them from day one.

1. **Continuous runnability invariant.** `cv.Sim().run()` must return results at every commit on `starsim-port`. A change that breaks the invariant is not acceptable for Cliff to commit.
2. **Dual validation gates.**
   - **Development gate (per check-in).** An *anchor scenario* (vanilla natural history, no interventions, fixed seed) plus per-milestone *capability scenarios* are run against locally-stored v3.1.8 baselines. Target: ±10% relative drift per summary result. The pinned summary-result set, established in M0, is drawn from `sim.summary` (cumulative infections/symptomatic/severe/critical/deaths, peak prevalence, peak `n_infectious`, derived rates) plus total population. **On failure the gate is informational, not auto-blocking**: the work carries either a fix or an explicit note classifying the drift as expected feature-misalignment, with a follow-up to re-converge.
   - **Release gate (per milestone acceptance test and at v4.0.0).** The self-contained validation bar: the existing Covasim test suite passes *and* the multi-seed z-score parity gates (`|z| < 3` over N v4 seeds vs M v3.1.8 seeds) show overlapping uncertainty intervals on the headline metrics. This is the scientific gate.
3. **Subclass-first tactic permitted as an interim.** `class Foo(ss.X)` that delegates to v3 logic is allowed during a milestone. Every such delegation must be tracked so it is stripped before M10. No delegations ship in v4.0.0.
4. **Lift-and-shift exclusion — the immunity engine is re-implemented, not lifted.** The contact network *is* lift-and-shifted (M1). The one subsystem that is rebuilt from scratch rather than lifted is the **NAb/waning/cross-immunity engine** (M4): Starsim has no multi-strain/NAb/waning facility, so `immunity.py` is re-expressed as `ss.Arr` immunity states + an `ss.Connector` recomputing `rel_sus`/protection live each step, rather than a mechanical port of the numba/global-state code. (This is the analog of hpvsim's "HIV-only" lift-and-shift exclusion.)
5. **In-place replacement, with quarantines.** v4 work replaces `covasim/` in place. v3 modules untouched by the current milestone are moved to `covasim/_v2_legacy/`; v3 tests that exercise removed APIs are moved to `tests/_legacy/`. Active code never imports from either quarantine — they exist purely as a porting reference. M10 deletes both wholesale.

**Style note (Starsim vs. Covasim conventions).** New ported code follows the **Starsim style guide**. Where Starsim style differs from Covasim's historical conventions, defer to Starsim for new code. In practice there is little tension on the biggest item — both forbid type annotations in signatures and put type info in docstrings only (Google `Args:` style) — so continue writing annotation-free signatures. The one flagged tension: Starsim uses CRN/`ss.Dist` sampling and `ss.uids`/`BoolArr` set-like indexing (plain-integer-array indexing is disallowed) in place of Covasim's global-RNG numba helpers and integer-index arrays; new code uses the Starsim idioms. Throughout, keep Covasim's "optimize for the scientist-reader" ethos: clear scientific logic, sensible defaults, papers cited in comments.

## Branching strategy

- `starsim-port` already exists (created off `main` by Cliff, currently checked out) and is the **single long-lived branch** where all migration work converges. The assistant never creates or switches branches.
- `main` is effectively frozen for v4 work — no v4 development on it. Should a critical bug fix land on `main`, it is forward-merged into `starsim-port`; otherwise periodic merges from `main` are unnecessary.
- **There are no per-milestone branches.** Every milestone's work lands directly on `starsim-port`, committed incrementally by Cliff at the pause-for-review check-ins. (Milestone names like `m1-basic-transmission` are used only as commit-message / check-in labels, not branches.)
- The **pause-for-review-and-commit** cadence (see Workflow) governs commits: the assistant leaves work uncommitted and pauses 2–5 times per milestone for Cliff to review and commit. The assistant never commits and never pushes.
- **Milestone completion** = acceptance test green locally *and* Cliff has reviewed and committed the work. There are no PRs or issue numbers as gates (this is a local effort).
- `starsim-port` is always runnable (Implementation conventions item 1); each milestone's work is held to this before Cliff commits it.
- No merge from `starsim-port` into `main` until v4.0.0 at M10. At that point `starsim-port` merges to `main`.

## Linked documents

- [`plans/2026-05-29-covasim-m0-foundation.md`](./plans/2026-05-29-covasim-m0-foundation.md) — M0 foundation implementation plan (task-by-task: CI adaptation, the `tests/regression/` harness, the `_v2_legacy` quarantine scaffold).
- [`specs/2026-05-29-covasim-m0-foundation-design.md`](./specs/2026-05-29-covasim-m0-foundation-design.md) — M0 foundation design spec (the pinned anchor scenario, file layout, comparison/parity rules, CI integration, out-of-scope).
