# Covasim v4.0 — M1 Basic-Transmission: design spec

**Date:** 2026-05-29
**Milestone:** M1 (Basic transmission sim)
**Branch:** `starsim-port` (the single long-lived branch; the assistant never creates branches, never commits, never pushes — Cliff reviews and commits)
**Predecessor:** [M0 Foundation](2026-05-29-covasim-m0-foundation-design.md)
**Target:** Covasim v4.0.0 on Starsim 3.3.x (3.3.4 at time of writing)

## Goal

Stand up the **minimum runnable COVID epidemic on Starsim**: port Covasim's multilayer contact network and add a minimal single-variant disease module that does S→E→I→R transmission and recovery only. The user-visible demo is "run an epidemic on a random or hybrid population and plot the infection curve over time."

M1 is the first milestone that lands **real port code** in `covasim/`. M0 shipped only the regression harness, the quarantine scaffold (empty), and a degenerate `cv.v4.Sim` stub. M1 replaces the v3.1.8 `cv.Sim`/`cv.People` engine *in place* on the Starsim base, retires the M0 stub (the real `cv.Sim` now *is* the Starsim-based one), and quarantines the v3 modules M1 does not yet touch into `covasim/_v2_legacy/`. As with hpvsim M01, this is an **in-place replacement with quarantines**, not a coexisting `cv.v4` subpackage. The only hard invariant at every commit remains the **continuous-runnability invariant**: `cv.Sim().run()` returns results.

The deliverables are:

1. A thin `cv.Sim(ss.Sim)` that assembles the M1 module stack (people + network(s) + one disease) and forwards to `ss.Sim`.
2. A `cv.People(ss.People)` thin subclass (keeps the public name; see "People" below for the keep-vs-stock decision).
3. `cv.Network(ss.Network)` (provisional name — flagged for Cliff) — a lift-and-shift of `population.py`, reproducing the **random** single-layer (`a`) backend and the **hybrid** four-layer (`h`/`s`/`w`/`c`) backend, with per-layer `beta_layer`/`contacts`, the household-cluster vs random-graph generation, and the school (6–22) / work (22–65) age restrictions.
4. `cv.COVID(ss.Infection)` (provisional name — flagged for Cliff) — a minimal single-variant disease doing transmission + S→E→I→R recovery, with CRN-safe per-edge transmission inherited from `ss.Infection.infect()` (replacing the `compute_infections`/`compute_trans_sus` numba kernels).
5. The **quarantine move**: v3 modules M1 doesn't touch → `covasim/_v2_legacy/`; v3 tests that exercise removed APIs → `tests/_legacy/`; `tests/conftest.py` gains `collect_ignore_glob`; `covasim/__init__.py` is trimmed to the M1 surface.
6. A new **M1 anchor** (`tests/regression/anchor_m1.py`) + pinned metrics: contact-structure equivalence (per-layer degree distribution + age-mixing matrix) and infection-trajectory parity, both vs. a new single-variant v3.1.8 baseline, reusing the M0 `parity_gate`.
7. Tests: contact-structure equivalence, infection-trajectory overlap, plus the M1 anchor smoke/drift/parity gates.

## Problem statement

Covasim v3.1.8 runs transmission through three numba kernels (`compute_trans_sus`, `compute_viral_load`, `compute_infections` in `utils.py:39-133`), driven from `Sim.step()` (`sim.py:621-649`), over `cv.Layer`/`cv.Contacts` edge arrays built once in `population.py`. The whole thing is seeded by a single global numpy/numba stream (`cv.utils.set_seed`). Starsim replaces this with module-based composition: a `ss.Network` holds the edges, `ss.Infection.infect()` walks them once per disease per step, and per-distribution Common Random Numbers (CRN) — `ss.multi_random('source','target')` — make transmission decisions reproducible and independent of network ordering/size (Starsim API findings §5).

M1's problem is therefore two coupled ports:

1. **Contact structure** — reproduce, on a `ss.Network`, the *exact generation procedure* of the random and hybrid backends so the realized network is statistically identical to v3.1.8 (the per-layer degree distribution and the age-mixing matrix are the acceptance metrics, so "mean degree n" is not enough — the Poisson/2 halving, the household Poisson-cluster construction, and the age windows all matter; see Network findings §5).
2. **Minimal dynamics** — reproduce the S→E→I→R trajectory (seeding `pop_infected`, the `exp2inf` latency, the `asym2rec` recovery, permanent immunity with `use_waning=False`) so the infection curve overlaps v3.1.8 within multi-seed noise.

Because the RNG model changes fundamentally, **bit-for-bit equivalence is impossible and not the bar** (the M0 premise). The bar is: the existing (non-quarantined) tests pass, and the M1 anchor's metrics overlap a v3.1.8 baseline within the M0 `parity_gate`'s `|z| < 3` (multi-seed) or within ±10% drift (single-seed development gate).

## Design decisions

### 1. `cv.Network(ss.Network)` reproduces the random + hybrid backends (lift-and-shift)

`cv.Network` subclasses **`ss.Network`** (not `ss.DynamicNetwork`), because both M1 backends are **static by default** — every layer in the random and hybrid defaults has `dynam_layer = 0` (`parameters.py:179,186`), edges are generated once at population creation and never change. A plain `ss.Network` has no `step` override, so its edges are fixed after `init_post` — exactly the static behavior M1 needs (Starsim findings §1). The dynamic-rewire path (`Layer.update`, `dynam_layer=1`) is **deferred/stubbed** — it is off by default and not exercised by the degree-distribution / age-mixing acceptance metrics (Network findings §5 "MAY defer").

Edges are stored in `self.edges` (an `sc.objdict` with `p1`/`p2`/`beta` columns, dtypes from `self.meta`) and built by overriding **`add_pairs()`**, ending in `self.append(p1=..., p2=..., beta=...)` (Starsim findings §1). This is the same undirected, singly-stored edgelist semantics as `cv.Layer` (`base.py:1610-1805`), so the port is mechanical: `cv.Layer`'s `p1`/`p2`/`beta` arrays map directly onto `ss.Network.edges`. **`net_beta` is NOT overridden** (mirroring hpvsim M01, which inherited stock `net_beta`); the per-layer scalar β is carried on the disease side (decision 5).

**Multi-layer handling — one `cv.Network` instance per layer (the hpvsim M01 pattern).** Covasim's `cv.Contacts` is a dict-of-Layers; the Starsim analog is `sim.networks`, an `ndict` of one `cv.Network` per layer. So:

- **random backend** → one `cv.Network(layer='a')`. `add_pairs` ports `make_random_contacts(pop_size, n=20)` (`population.py:241-284`): pre-draw the pool `choose_r(max_n=pop_size, n=pop_size*20*1.2)`, per-person Poisson count `n_poisson(20, pop_size)`, **halve-and-round** (`p_count = round(p_count/2.0)` — essential to hit mean degree 20 for undirected singly-stored edges; Network findings hazards), then walk the pool slicing `p_count[p]` partners per person.
- **hybrid backend** → four instances `cv.Network(layer='h'|'s'|'w'|'c')`:
  - `h` (household): `make_microstructured_contacts(pop_size, cluster_size=contacts['h']=2.0)` (`population.py:287-329`) — Poisson-sized disjoint **fully-connected** clusters. **Hazard: household degree ≠ `contacts['h']`** — the value is a Poisson cluster-size parameter, not a mean degree; reproduce the actual procedure (Network findings §2 caveat + §5 hazards).
  - `c` (community): `make_random_contacts(pop_size, contacts['c']=20)` over everyone.
  - `s` (school): `make_random_contacts` over the school-age subset `ages∈[6,22)` with the `mapping` remap to global UIDs (`population.py:357,361`); mean 20.
  - `w` (work): same over the work-age subset `ages∈[22,65)` (`population.py:358,362`); mean 16.

**Ages and `make_people`.** The age sample (multinomial over `default_age_data` bins + uniform-within-bin, `population.py:198-204`) drives the age-mixing-matrix metric, so it is reproduced exactly. `make_randpop`/`make_random_contacts`/`make_hybrid_contacts`/`make_microstructured_contacts` keep their names and are ported to feed the `cv.Network` instances (the migration-plan architecture mapping preserves these function names). **Provisional structural question for Cliff (Open question A):** whether to (a) keep `population.py`'s functions as free functions that build edgelists which `cv.Network.add_pairs` consumes, or (b) move the generation logic into the `cv.Network` subclass per layer (the hpvsim shape). Recommendation: **(a)** — it preserves the public `cv.make_randpop`/`cv.make_hybrid_contacts` names (backwards-compat) and keeps age/contact generation in one readable place; `cv.Network.add_pairs` is then a thin "call the v3 builder for my layer, `self.append` the result."

**RNG note (Open question B, flagged for Cliff):** v3's contact generation uses the global-numpy helpers (`cvu.choose_r`, `cvu.n_poisson`, `cvu.poisson`). For the M1 acceptance metrics (degree distribution + age-mixing **distributions**, not per-agent identity) it is sufficient to reproduce the *procedure*; the network need not be CRN-per-agent-stable at M1 (that matters for scenario analysis, M8). Recommendation: build the network with `ss.Dist` draws (`ss.poisson`, `ss.choice`) so the network participates in Starsim's deterministic seeding, but do **not** invest in `ss.RandomSafeNet`-style single-agent stability yet — defer that to M8 if scenario differencing needs it. This keeps M1 honest about reproducibility expectations (the migration-plan "Global-RNG vs CRN" scope item is pinned at M1).

### 2. `cv.COVID(ss.Infection)` does minimal S→E→I→R transmission only

`cv.COVID` subclasses `ss.Infection` and — mirroring hpvsim M01's `HPV` — overrides **only `__init__`, `set_prognoses`, and `step_state`**. Transmission (`infect()`) is **inherited from stock `ss.Infection`**: the stock loop walks each network's edges, computes per-edge `p = rel_trans[src] * rel_sus[trg] * beta_per_dt`, draws CRN randvals via `self.trans_rng = ss.multi_random('source','target')`, and calls `set_outcomes → set_prognoses` (Starsim findings §2). This is exactly the M1 transmission kernel; **no custom transmission code is written** (the key M1 insight).

```python
class COVID(ss.Infection):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.probperday(0.016),                       # v3 pars['beta'] (parameters.py:62)
            init_prev = None,                                  # seeding handled via pop_infected (see below)
            dur_exp2inf = ss.lognorm_ex(mean=ss.days(4.5), std=ss.days(1.5)),  # E->I (parameters.py:85)
            dur_asym2rec = ss.lognorm_ex(mean=ss.days(8.0), std=ss.days(2.0)), # I->R (parameters.py:91)
        )
        self.update_pars(pars=pars, **kwargs)
        # ss.Infection provides: susceptible, infected, rel_sus, rel_trans, ti_infected.
        self.define_states(
            ss.BoolState('exposed',    label='Exposed'),
            ss.BoolState('infectious', label='Infectious'),
            ss.BoolState('recovered',  label='Recovered'),
            ss.FloatArr('ti_exposed'),
            ss.FloatArr('ti_infectious'),
            ss.FloatArr('ti_recovered'),
        )

    @property
    def infectious(self):
        return self._infectious   # only infectious (not merely exposed) agents transmit

    def set_prognoses(self, uids, sources=None):
        super().set_prognoses(uids, sources)        # logs the infection
        ti = self.ti
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = ti
        self.ti_infectious[uids] = ti + self.pars.dur_exp2inf.rvs(uids)   # E->I latency
        # asymptomatic-only path: schedule recovery off ti_infectious
        self.ti_recovered[uids] = self.ti_infectious[uids] + self.pars.dur_asym2rec.rvs(uids)

    def step_state(self):
        ti = self.ti
        new_inf = (self.exposed & (self.ti_infectious <= ti)).uids   # E->I
        self.exposed[new_inf] = False
        self.infectious[new_inf] = True
        rec = (self.infectious & (self.ti_recovered <= ti)).uids      # I->R
        self.infectious[rec] = False
        self.recovered[rec] = True   # use_waning=False => permanent immunity, NOT back to susceptible
```

Notes / grounding:

- **SEIR, not SIR.** Covasim is S→E→I→R: `infect()` sets `exposed`, schedules `date_infectious = t + dur_exp2inf` (`people.py:514-516`), `check_infectious` flips E→I when `date_infectious <= t` (`people.py:222-232`), `check_recovery` flips I→R (`people.py:256-291`). So M1 adds `exposed`/`infectious`/`recovered` BoolStates beyond `ss.Infection`'s `infected`/`susceptible`. The `infectious` **property override** ensures only infectious (not exposed) agents transmit — stock `ss.Infection.infect()` zeroes `rel_trans` for non-`infectious` agents (Starsim findings §2). The inherited `infected` BoolState is repurposed/aliased; **Open question C (flag for Cliff):** whether to map v3's `exposed` (= E or I) onto `ss.Infection.infected` and add only an `infectious` sub-state, or to add all three states fresh with `reset=True` (the `ss.SIR` pattern, diseases.py:647). Recommendation: **add all three fresh** with the `infectious` property override — it is the clearest reader-facing mapping of v3's state names and avoids subtle `infected`-vs-`infectious` confusion; the inherited `n_infected`-style result is then provided by the new BoolStates.

- **Asymptomatic-only path.** M1 collapses the prognosis tree to its asymptomatic branch (recovery via `asym2rec`, no death, `date_dead` NaN). In stock-Covasim terms this is `symp_prob=0` so every agent recovers via `asym2rec` (`people.py:524-532`; Transmission findings §3b, §5). Symptomatic/severe/critical/death are **deferred to M2**.

- **Permanent immunity.** With `use_waning=False`, v3's `check_recovery` does **not** return recovered agents to susceptible (the `people.py:285-289` reinfection branch is guarded by `use_waning`); M1's `step_state` sets `recovered=True` and never re-susceptibles (a true SEIR with absorbing R). Waning/NAbs/reinfection are M4.

- **Deferred transmission detail.** v3's `compute_trans_sus`/`compute_viral_load` fold in `viral_load`, `asymp_factor`, `beta_layer`, and the per-agent `beta_dist`/`rel_trans` overdispersion (`utils.py:39-95`). For M1, `asymp_factor=1.0` (no effect, since everyone is "asymptomatic"), `immunity_factors=0`, `iso/quar` absent. The per-agent `rel_trans` superspreading draw (`beta_dist`, mean≈1.0) and time-varying `viral_load` are **transmission-shape refinements deferrable to M2** (migration-plan scope item "beta_dist / viral_dist → M2"). **Recommendation for M1:** keep `rel_sus = rel_trans = 1.0` (stock `ss.Infection` defaults) for a clean SEIR; the parity gate's multi-seed CIs absorb the resulting variance difference, and M2 reintroduces the heterogeneity. Flag as **Open question D** if Cliff wants the `beta_dist` draw in at M1 for tighter trajectory parity.

### 3. `cv.Sim(ss.Sim)` and `cv.People(ss.People)` assemble the stack

`cv.Sim` is a thin wrapper (the hpvsim `Sim` pattern) that builds the M1 default stack and forwards to `ss.Sim`, using the `kwargs.pop('x', None)`-then-default override idiom so tests can inject their own networks/disease/people:

```python
class Sim(ss.Sim):
    def __init__(self, pars=None, people=None, pop_size=20_000, pop_infected=20,
                 pop_type='random', n_days=60, start_day='2020-03-01',
                 rand_seed=1, **kwargs):
        if people is None:
            people = cv.People(pop_size)                 # ages sampled from default_age_data
        networks = kwargs.pop('networks', None)
        if networks is None:
            networks = make_networks(pop_type, people)   # [cv.Network('a')] or [h,s,w,c]
        diseases = kwargs.pop('diseases', None)
        if diseases is None:
            diseases = cv.COVID(beta=...)                # per-layer beta wired via betamap (decision 5)
        super().__init__(pars=pars, people=people, networks=networks, diseases=diseases,
                         start=ss.date(start_day), dur=ss.days(n_days), dt=ss.days(1),
                         rand_seed=rand_seed, **kwargs)
        # seed pop_infected at t==0 (see "Seeding" below)
```

- **People.** Per the locked decisions, `cv.People` **keeps its name** and subclasses `ss.People`. M1's only real need from `cv.People` is the v3 **age sample** (multinomial over `default_age_data` + uniform-within-bin, `population.py:198-204`) so the age-mixing-matrix metric matches; `ss.People(n_agents, age_data=df)` already accepts an `[age, value]` pyramid (hpvsim used stock `ss.People` this way). **Decision:** ship a thin `cv.People(ss.People)` that (a) keeps the public name (backwards-compat) and (b) supplies the v3 age distribution as its default `age_data`. All disease state lives on `cv.COVID` via `define_states`; all network state on `cv.Network`; Starsim auto-aggregates module states onto People (so there is no v3 "god-object" `People` health-state machine in M1 — that logic moved to `cv.COVID`). Multi-scale/`pop_scale` state is deferred (M2).

- **Daily timestep.** `dt=ss.days(1)` is the unambiguous Starsim-3.3.x form for a daily step; betas are day-native (`ss.probperday`) and durations day-native (`ss.days`/`ss.lognorm_ex(mean=ss.days(...))`) (Starsim findings §4). A v3 60-day run = `start=ss.date('2020-03-01'), dur=ss.days(60), dt=ss.days(1)`.

- **Seeding `pop_infected`.** v3 seeds `pop_infected` agents uniformly without replacement at t==0 (`cvu.choose(pop_size, pop_infected)`, then `people.infect(...)`; `sim.py:505-532`). The Starsim idiom is `init_prev` as an `ss.bernoulli` seeded in `Infection.init_post` (diseases.py:150). But v3 seeds an **exact count**, not a per-agent probability. **Open question E (flag for Cliff):** reproduce the exact-count seed (e.g. an `ss.choice(pop_infected, replace=False)` selection wired into the disease's `init_post`, or an `init_prev` callback that draws exactly `pop_infected`) vs. accept a Bernoulli `init_prev = pop_infected/pop_size` (small variance at pop_size=20k). Recommendation: **exact-count seed** — it matches v3 precisely (the parity gate's `cum_infections` adds `pop_infected` back, per `sim.py:786-787`, so getting the seed count exactly right keeps the trajectory aligned) and is a small amount of code. Note the `frac_susceptible<1` path is irrelevant at M1 (`frac_susceptible=1.0` default).

- **Loop order.** Stock Starsim loop runs `diseases.step_state()` (E→I, I→R) at slot 5, `networks.step()` at 7 (no-op for static M1 networks), `diseases.step()` (transmission) at 9 (Starsim findings §3). This matches v3's "update states → update contacts → transmit" ordering in `Sim.step()` (`sim.py:577-649`), so an agent that becomes infectious this step can transmit this step — the v3 behavior is preserved for free.

### 4. Daily timestep + CRN transmission configuration

- **Timestep:** `dt=ss.days(1)`, `start=ss.date(start_day)`, `dur=ss.days(n_days)` (decision 3). Beta as `ss.probperday(0.016)` so `infect()` converts it per-dt via `beta.to_prob(self.t.dt)` (diseases.py:258); at `dt=1 day` this is the v3 per-contact probability 0.016 directly.

- **CRN transmission:** inherited `self.trans_rng = ss.multi_random('source','target')` (diseases.py:134). Per edge `(a,b)`, the random number is a deterministic XOR-combine of source `a`'s slot-draw and target `b`'s slot-draw (Starsim findings §5), so transmission decisions are reproducible and independent of edge ordering/network size — the migration-plan's stated replacement for the `compute_infections` binomial draw (`utils.py:128`). Per-step RNG jumping (`jump_dt`, 1000 draws/step) is automatic.

- **Reproducibility expectation pinned at M1 (the migration-plan scope item):** same `rand_seed` → identical v4 run; v4 ≠ v3 bit-for-bit (different RNG model); v4↔v3 equivalence is the multi-seed z-score gate. The network is built with `ss.Dist` draws (decision 1, Open question B) so it participates in deterministic per-distribution seeding.

### 5. Per-layer beta on the disease (not on the network)

v3 has two β quantities (Transmission findings §1d): the scalar `pars['beta']=0.016` and the per-layer `beta_layer` scalars (random `a=1.0`; hybrid `h=3.0, s=0.6, w=0.6, c=0.3`, `parameters.py:174-189`). In Starsim, the per-layer factor is expressed on the **disease's `beta` parameter as a dict keyed by network label** — `ss.Infection.validate_beta` accepts `beta=dict(a=...)` or `dict(h=..., s=..., w=..., c=...)` and builds the `betamap` matching the set of network keys (diseases.py:181; Starsim findings §1 "Per-layer beta"). So M1 wires:

- random: `cv.COVID(beta=dict(a=ss.probperday(0.016*1.0)))`
- hybrid: `cv.COVID(beta=dict(h=0.016*3.0, s=0.016*0.6, w=0.016*0.6, c=0.016*0.3))` (each as `ss.probperday`)

i.e. the v3 `pars['beta'] * beta_layer[lkey]` product becomes the per-layer disease beta. The per-edge `cv.Network.edges.beta` stays 1.0 (matching v3's default per-edge layer beta). `iso_factor`/`quar_factor` per-layer multipliers are **deferred to M5** (testing/tracing/quarantine; they are not network structure).

### 6. In-place replacement / `_v2_legacy` + `tests/_legacy` quarantine strategy for M1 — THE CONSEQUENTIAL DECISION

This mirrors hpvsim M01 exactly: a single quarantine commit (`git mv`, preserving history) moves every v3 module M1 does not touch into `covasim/_v2_legacy/`, every v3 test that exercises a removed API into `tests/_legacy/`, adds `collect_ignore_glob` to `tests/conftest.py`, and trims `covasim/__init__.py` to the M1 surface. Quarantines are **never imported by active code**; they are porting reference only; M10 deletes both wholesale.

#### v3 modules → `covasim/_v2_legacy/` (the recommendation for Cliff)

The hpvsim M01 rule was: **move only what the milestone doesn't touch; keep parameters/defaults/settings/utils/version/data active.** Applying that to Covasim's module list (`covasim/`: analysis, base, defaults, immunity, interventions, misc, parameters, people, plotting, population, requirements, run, settings, sim, utils, version + data/, regression/):

**RECOMMENDED — quarantine these (untouched by M1):**

| Module | Why quarantined |
|---|---|
| `analysis.py` | Analyzers / `Fit` / `Calibration` / `TransTree` — M7/M9. |
| `immunity.py` | Variants, waning, NAbs, cross-immunity — M3/M4. |
| `interventions.py` | Testing / tracing / quarantine / vaccination / `change_beta` etc. — M5/M6. |
| `run.py` | `MultiSim` / `Scenarios` / parallel — M8. |
| `plotting.py` | v3 plotting; M1 demo uses Starsim plotting (`sim.plot()`) — full v3 plotting restored M9. |

**RECOMMENDED — keep active (M1 depends on them, or they are utility/data plumbing the new code wraps):**

| Module | Why kept active |
|---|---|
| `parameters.py` | `make_pars`, `reset_layer_pars`, `get_prognoses`, the `dur`/`beta_layer`/`contacts` defaults, `default_age_data` lookups — M1 reads these for the network + disease pars. |
| `defaults.py` | `default_int`/`default_float`, `default_age_data`, colors/result keys. |
| `settings.py` | `cv.options`. |
| `misc.py` | helpers used by parameters (date handling, `git_info`, etc.). |
| `utils.py` | numerical helpers still used by the ported `population.py` builders (`choose_r`, `n_poisson`, `poisson`, `n_multinomial`, `sample`) — M1 lift-and-shifts these into `ss.Dist`-backed equivalents over the milestone, but `utils.py` stays active as the source until each helper is converted. |
| `version.py` | version/license. |
| `requirements.py` | import-time requirement check. |
| `data/` | demographic/age data loaders. |
| `regression/` | per-version pars JSON snapshots (forensic). |

**THE THREE CONSEQUENTIAL CASES — `base.py`, `people.py`, `population.py`, `sim.py`:** these are the modules M1 *replaces* rather than purely defers, and they need a clear decision:

- **`sim.py`** — **replaced in place.** The v3 `sim.py` (the 1900-line `Sim`/`BaseSim`/integration loop) is `git mv`'d to `covasim/_v2_legacy/sim.py` *before* the new thin `cv.Sim(ss.Sim)` is written at `covasim/sim.py`. This is precisely the hpvsim M01 move (v2 `sim.py` → `_v2_legacy/sim.py`, new tiny `sim.py` in its place). **The M0 `cv.v4.Sim` stub is retired here** (decision 7).
- **`people.py`** — **replaced in place.** v3 `people.py` (the `People` health-state machine — `check_infectious`, `check_recovery`, `infect`, `set_prognoses`, the whole transition engine) → `covasim/_v2_legacy/people.py`; a new thin `cv.People(ss.People)` is written at `covasim/people.py`. The health-state-machine logic does not return to `people.py`; it lives on `cv.COVID` (M1) and grows there (M2). This is the analog of hpvsim moving its `people.py` god-object to `_v2_legacy`.
- **`population.py`** — **kept active but rewritten over M1** (analog of hpvsim, where network logic came *out of* `population.py` into the new `network.py`). **Decision (recommended):** keep `population.py` active and port its builders in place to feed `cv.Network` (Open question A option (a)), preserving the public `cv.make_randpop`/`cv.make_hybrid_contacts`/`cv.make_random_contacts`/`cv.make_microstructured_contacts` names. The new `cv.Network` class lives in a new module `covasim/network.py` (the hpvsim `network.py` analog). `make_synthpop` (the synthpops backend) is **quarantined** (M9) — either left as a `# pragma: no cover` stub in `population.py` that raises "ported in M9", or moved to `_v2_legacy`; recommendation: leave a guarded stub in `population.py` so `pop_type='synthpops'` raises a clear "not yet ported (M9)" error rather than a missing-attribute error (continuous-runnability + good error hygiene).
- **`base.py`** — **the genuinely hard case.** v3 `base.py` holds `ParsObj`, `BaseSim`, `BasePeople`, `cv.Result`, **and `cv.Layer`/`cv.Contacts`** (`base.py:1509/1610`). The migration-plan architecture mapping says `cv.Layer`/`cv.Contacts`/`cv.Result`/`cv.BaseSim`/`cv.ParsObj`/`cv.BasePeople` are all **kept names**. But M1 replaces `BaseSim`/`BasePeople` (now `ss.Sim`/`ss.People` provide that) and replaces `cv.Layer`/`cv.Contacts`'s *role* with `cv.Network`. **Recommendation for Cliff:** do **not** quarantine `base.py` wholesale at M1 — instead keep `base.py` active and let the M1-unused classes (`BaseSim`, `BasePeople`) sit dormant (not imported by the new `cv.Sim`/`cv.People`), the same way hpvsim kept utility modules active. Keep `cv.Layer`/`cv.Contacts`/`cv.Result`/`cv.ParsObj` exported from `base.py` for backwards-compat name preservation (they are additive — nothing in M1 *removes* them). Reason: quarantining `base.py` would drag `cv.Result`/`cv.Layer`/`cv.Contacts` out of the active surface, which the migration plan commits to keeping; the cost of leaving dormant-but-importable classes active is low (they just aren't wired into the Starsim stack). **This is the single most consequential M1 quarantine call and is explicitly flagged for Cliff's sign-off (Open question F).** The alternative — quarantine `base.py` and re-export `cv.Layer`/`cv.Contacts`/`cv.Result` from a new slim module — is more work and risks subtle import-order breakage for no M1 benefit.

#### v3 tests → `tests/_legacy/`

Move every v3 `test_*.py` that exercises a now-quarantined/replaced API into `tests/_legacy/` (the hpvsim M01 move was "every v2 `test_*.py` except the regression harness tests"). Concretely, quarantine: `test_analysis.py`, `test_immunity.py`, `test_interventions.py`, `test_run.py`, `test_resume.py`, and `test_other.py`; and the parts of `test_sim.py`/`test_parameters.py`/`test_utils.py` that drive the removed v3 `Sim`/`People`/intervention APIs. **Recommendation:** rather than partial-file surgery, move the whole v3 `test_sim.py` to `tests/_legacy/` and write a new M1 `test_sim.py` (matching how hpvsim wrote fresh M01 `test_*.py` files for the new surface). Keep active: `test_baselines.py` (allowed to fail / be regenerated per the validation bar — `test_baselines` is the v4-internal gate, regenerated at M10), the M0 harness tests (`test_m0_parity.py`, `test_regression_harness.py`), and the new M1 tests. `test_v4_stub.py` is **deleted** (the stub it guards is retired — decision 7).

**Open question G (flag for Cliff):** `test_baselines.py` and `test_regression.py` will *fail* against the v4 build (different RNG → different exact numbers), but the validation bar explicitly allows `test_baselines.py` to fail-and-be-updated as long as z-scores overlap. Recommendation: mark the v3 bit-for-bit assertions in `test_baselines.py`/`test_regression.py` with `@pytest.mark.skip(reason='v3 baselines; regenerated for v4 at M10')` (hpvsim's M01 move skipped its multi-genotype regression smoke), rather than quarantining them — they document the target and are un-skipped/regenerated at M10.

#### Collection suppression + `__init__.py` trim

- **`tests/conftest.py`** (new): `collect_ignore_glob = ['_legacy/*', 'devtests/*']` (the exact hpvsim M01 mechanism). Note the CI invocation glob (`pytest -v test_*.py unittests/test_*.py`) is already root-only, so `tests/_legacy/test_*.py` is not collected by CI; the `conftest.py` makes a bare `pytest` (e.g. `run_tests`' `pytest test_*.py`, or a developer's `pytest .`) also skip the quarantine robustly.
- **`covasim/__init__.py`** trimmed to the M1 surface: keep the stable imports (`settings`/`options`, `version`, `defaults`, `misc`, `parameters`, `utils`, `data`, and `base`'s name-preserved exports per Open question F), add `from .network import Network`, `from .people import People`, `from .covid import COVID` (module name TBD — see Open question H), `from .sim import Sim`. Remove the imports of quarantined modules (`analysis`, `immunity`, `interventions`, `plotting`, `run`). **Retire the M0 stub line** `from . import _v4 as v4` (decision 7). Document the milestone-staged API expansion in the module docstring (hpvsim convention).

### 7. Fate of the M0 `cv.v4` stub

**Retired in M1.** The M0 stub (`covasim/_v4.py`, exposed as `cv.v4.Sim`) existed solely to prove the continuous-runnability invariant on a Starsim base without disturbing the v3.1.8 `cv.Sim` (M0 spec decision 9). In M1 the real `cv.Sim` *becomes* the Starsim-based one (it now subclasses `ss.Sim`), so the stub's reason to exist — a non-disturbing parallel symbol — is gone. Concretely:

- Delete `covasim/_v4.py` and remove `from . import _v4 as v4` from `covasim/__init__.py`.
- Delete `tests/test_v4_stub.py` (it asserts `cv.Sim.__module__ == 'covasim.sim'` *is the v3 Sim* and `cv.v4.Sim` is the stub — both assertions become false/obsolete once `cv.Sim` is the port).
- The continuous-runnability invariant is now carried by `cv.Sim().run()` itself (a fast M1 test asserts it returns results), which is strictly stronger than the stub.

This is a clean retirement — no `cv.v4` name is promised in the public API (it was always interim scaffolding), so removing it is not a backwards-compat concern.

## M1 anchor scenario + pinned metrics

A new anchor `tests/regression/anchor_m1.py` (sibling to M0's `anchor.py`), reusing the harness (`short_summary.py`, `parity.py`'s `parity_gate`, `multi_seed_v3.py`, `compare.py`). The M1 anchor is **single-variant, no interventions, no waning** — it isolates basic transmission, exactly the M1 capability:

```python
PARS = dict(
    pop_size     = 20_000,
    pop_infected = 20,
    pop_type     = 'random',   # plus a 'hybrid' variant of the anchor (see below)
    n_days       = 60,
    n_variants   = 1,          # single wild variant
    use_waning   = False,      # permanent immunity; skips NAb code
    rand_seed    = 0,          # sweep overrides 0..N-1
    verbose      = 0,
    # asymptomatic-only path (symp_prob -> 0); no interventions, no analyzers.
)
```

Two anchor variants are run: `pop_type='random'` and `pop_type='hybrid'` (the acceptance covers both backends). The v3.1.8 single-variant baseline is generated locally via the M0 `multi_seed_v3.py`-style sweep run in a frozen v3.1.8 env (the env IS covasim 3.1.8, so baselines are generated here and gitignored).

**Two metric families, both gated:**

1. **Contact-structure equivalence (the M1-specific gate, computed on the static post-creation network):**
   - **Per-layer degree distribution** — for each layer, the histogram of per-agent contact counts (built by `np.add.at` over `edges.p1`/`p2`, the hpvsim concurrency-histogram idiom). Compared to v3.1.8's `cv.Layer` degrees for the same pars/seed-sweep. Pass: the mean degree per layer within tolerance (random `a`≈20; hybrid `s`≈20, `w`≈16, `c`≈20; household `h` matched on its realized Poisson-cluster degree distribution, **not** assumed to equal `contacts['h']`), and the degree-distribution shape close (recommend a coarse-binned relative-difference or KS check; final tolerance pinned on first run, hpvsim-style).
   - **Age-mixing matrix** — 5-year-binned source-age × target-age contact matrix per layer (depends on the age sample + the school/work age windows). Compared via a bin-wise relative diff or cosine similarity (hpvsim M01 used cosine similarity > 0.85 after KS proved too strict for one seed; recommend **cosine similarity** with the threshold pinned on first run).

2. **Infection-trajectory parity (reuses the M0 `parity_gate`):** the pinned `short_summary` metrics restricted to the M1 capability — `cum_infections`, peak `prevalence`, peak `n_infectious`, and `prevalence`/`incidence` at end of run (`tests/regression/short_summary.py` already builds these from `sim.summary` + `sim.results`). The M2+ burden metrics (`cum_symptomatic`/`cum_severe`/`cum_critical`/`cum_deaths`) are identically 0 in M1 (asymptomatic-only) and are excluded from the M1 gate (they re-enter at M2). The `parity_gate` z-formula and degenerate-distribution policy are unchanged from M0.

## Acceptance test

1. **Sim runs end-to-end** on the M1 anchor (random and hybrid, single variant, seed 0), and `cv.Sim().run()` returns results (continuous-runnability invariant) — a fast test.
2. **Existing (non-quarantined) tests pass** under the strict-warnings bar (`COVASIM_WARNINGS=error`): the M0 harness tests, the new M1 tests, and `test_baselines.py`/`test_regression.py` either pass or carry the documented v4-skip (Open question G). The quarantined `tests/_legacy/` are not collected.
3. **Contact-structure equivalence** vs the v3.1.8 baseline: per-layer degree distribution and age-mixing matrix match within tolerance (final thresholds pinned on first run, hpvsim-style), for both random and hybrid. **This is the M1-specific acceptance gate.**
4. **Infection-trajectory overlap** vs the new single-variant v3.1.8 baseline: development gate (`compare.py`, single-seed ±10% drift, informational) green per check-in; release gate (`parity_gate`, multi-seed `|z| < 3`, `@pytest.mark.slow`) green or skips cleanly when the baseline is absent.
5. **Demo** plots the M1 anchor's infection curve over time (`sim.plot()` on the Starsim base; an `examples/`-style script or the anchor's `__main__`).

**Milestone completion** = acceptance test green locally *and* Cliff has reviewed and committed the work. No PRs/issues as gates (local effort).

## Workflow (pause-for-review-and-commit)

Local effort; the assistant prepares each piece, leaves it **uncommitted**, and pauses for Cliff to review and commit. **The assistant never commits and never pushes.** Check-ins **2–5 times per milestone** at natural sub-task boundaries. M1's natural check-in points:

1. **The quarantine commit** — `git mv` v3 modules → `_v2_legacy/`, v3 tests → `tests/_legacy/`, add `tests/conftest.py`, trim `covasim/__init__.py`, retire the `cv.v4` stub. (Reviewed first so the in-place replacement is a clean, history-preserving move before new code lands.)
2. **`cv.Network`** (+ ported `population.py` builders) and its contact-structure equivalence tests.
3. **`cv.COVID`** (minimal SEIR) and its functional tests (minimal-sim with stock `ss.RandomNet`, `copy_inputs=False`, hpvsim-style).
4. **`cv.Sim`/`cv.People`** assembly + the M1 anchor + trajectory parity/drift tests + the demo plot.
5. (If needed) tolerance-pinning pass once first-run variability is observed, plus the plan/spec docs.

## Out of scope for M1 (deferred to later milestones)

- **Symptomatic / severe / critical / death natural history** and the prognosis tree, age-based prognoses, severity scalers, bed caps (`people.py:235-579`, `parameters.py` prognoses) → **M2**.
- **`beta_dist` per-agent overdispersion + time-varying `viral_load`** (`utils.py:39-95`) → **M2** (Open question D may pull `beta_dist` into M1).
- **Population scaling** (`pop_scale`/`total_pop`) and dynamic rescaling (`rescale`, `make_naive`) → **M2** / later.
- **Multi-variant + cross-immunity** (`n_variants>1`, `cv.variant`, `get_cross_immunity`, host-level exclusivity) → **M3**.
- **Waning immunity / NAbs / reinfection** (`immunity.py`, `use_waning=True` path, `check_immunity`) → **M4**.
- **Testing / tracing / quarantine / isolation** (`interventions.py` testing/tracing, the quar/iso state machine, `iso_factor`/`quar_factor`) → **M5**.
- **Vaccination** → **M6**. **Calibration / Fit** → **M7**. **MultiSim / Scenarios / parallel** and the **retrofit of M1's acceptance onto z-score parity gates** → **M8**. **Analyzers / TransTree / plotting / synthpops backend** → **M9**.
- **Dynamic-layer rewiring** (`Layer.update`, `dynam_layer=1`) — off by default, not exercised by M1 metrics; stubbed/deferred.
- **Location-specific age/household data** (`cvdata.get_age_distribution`, `get_household_size`) — default Seattle `default_age_data` is sufficient for M1; location overrides later.
- **Negative-binomial contact overdispersion** (`make_random_contacts(dispersion=...)`) — unused by defaults; deferred.
- **CRN single-agent network stability** (`ss.RandomSafeNet`) — deferred to M8 if scenario differencing needs it (Open question B).

## Open questions for Cliff

- **A. `population.py` shape:** keep v3 builders as free functions feeding `cv.Network.add_pairs` (recommended, preserves `cv.make_*` names), or move generation into `cv.Network` per layer (hpvsim shape)?
- **B. Network RNG:** build with `ss.Dist` draws (recommended) and defer `ss.RandomSafeNet`-style single-agent CRN stability to M8?
- **C. SEIR state mapping:** add `exposed`/`infectious`/`recovered` fresh with `reset=True` + an `infectious` property override (recommended), or reuse `ss.Infection.infected` for v3's `exposed`?
- **D. `beta_dist` heterogeneity:** keep `rel_trans=1.0` at M1 (recommended; M2 reintroduces) or pull the per-agent `beta_dist` superspreading draw into M1 for tighter trajectory parity?
- **E. Seeding:** reproduce v3's exact `pop_infected` count seed (recommended) vs. a Bernoulli `init_prev = pop_infected/pop_size`?
- **F. `base.py` quarantine (the consequential call):** keep `base.py` active with dormant `BaseSim`/`BasePeople` and name-preserved `cv.Layer`/`cv.Contacts`/`cv.Result`/`cv.ParsObj` exports (recommended), or quarantine `base.py` and re-export those names from a new slim module?
- **G. `test_baselines.py`/`test_regression.py`:** skip the v3 bit-for-bit assertions with a documented "regenerated for v4 at M10" reason (recommended) vs. quarantine them to `tests/_legacy/`?
- **H. Class names (provisional, need sign-off):** `cv.COVID` (alternatives `cv.SARSCoV2`/`cv.Covid`) and `cv.Network` (the per-layer network). Also the disease module's *file* name (`covasim/covid.py`? `covasim/disease.py`?).

## Linked documents

- [`MIGRATION_PLAN.md`](../MIGRATION_PLAN.md) — overall plan; §M1 is this milestone's source.
- [`specs/2026-05-29-covasim-m0-foundation-design.md`](2026-05-29-covasim-m0-foundation-design.md) — M0 foundation design (the regression harness, anchor pattern, parity gate, quarantine scaffold that M1 reuses).
- hpvsim M01 design: `/home/cliffk/idm/hpvsim/docs/superpowers/specs/2026-04-28-hpvsim-m1-basic-transmission-design.md` — the in-place-replacement + quarantine template this spec mirrors.
- [`tests/regression/`](../../tests/regression/) — M0 harness (anchor, short_summary, parity, multi_seed_v3, compare); M1 adds `anchor_m1.py`.
- [`covasim/_v2_legacy/`](../../covasim/_v2_legacy/) + [`tests/_legacy/`](../../tests/_legacy/) — quarantines (empty in M0; populated in M1).
