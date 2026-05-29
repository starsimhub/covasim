# Covasim v4.0 — M2 Natural-History-Parity: design spec

**Date:** 2026-05-29
**Milestone:** M2 (Natural-history parity)
**Branch:** `starsim-port` (the single long-lived branch; the assistant never creates branches, never commits, never pushes — Cliff reviews and commits)
**Predecessor:** [M1 Basic transmission](2026-05-29-covasim-m1-basic-transmission-design.md)
**Target:** Covasim v4.0.0 on Starsim 3.3.x (3.3.4 at time of writing)

## Goal

Make the COVID disease module **epidemiologically complete for a single variant**: extend M1's asymptomatic-only S→E→I→R into Covasim's **full prognosis tree** — exposed → infectious → (asymptomatic / mild / severe / critical) → recovered / dead — and, in the same milestone, **close the M1 transmission gap** by adding the per-agent time-varying `rel_trans` (`viral_load × beta_dist`) that M1 deliberately deferred. The user-visible demo is "run a single-variant epidemic and plot the health-outcome curves (`cum_severe`, `cum_critical`, `cum_deaths`) alongside the re-converged infection curve."

M2 is a pure **extension** of the M1 surface — it adds states, parameters, and methods to the existing `cv.COVID(ss.Infection)` and adds absolute population scaling (`pop_scale`/`total_pop`) to `cv.Sim(ss.Sim)`. **No new public class is created, and nothing is quarantined or removed** (M2 extends; it does not retire). The locked public names from M1 — `cv.COVID`, `cv.Sim`, `cv.People`, `cv.Network` — are unchanged. The continuous-runnability invariant (`cv.Sim().run()` returns results at every commit) holds throughout, since M2 only grows the disease's state machine.

The deliverables are:

1. The **full prognosis tree** on `cv.COVID`: add `symptomatic`/`severe`/`critical`/`dead` BoolStates, the per-agent base-probability FloatArrs (`symp_prob`/`severe_prob`/`crit_prob`/`death_prob`), the `ti_symptomatic`/`ti_severe`/`ti_critical`/`ti_dead` timers, the durations + scalers + bed-cap pars, the **four-branch pre-scheduled decision tree** in `set_prognoses`, the multi-stage threshold transitions in `step_state`, death via `sim.people.request_death` + a `step_die` reset, and the new burden Results.
2. **Per-agent time-varying `rel_trans`** = `trans_OR × beta_dist_draw × viral_load(t)` (the M1-gap closure): a per-agent constant overdispersion draw at infection plus a per-step mean-preserving viral-load kernel write, applied to `self.rel_trans` in `step_state` (which precedes transmission in the loop).
3. **Absolute population scaling** on `cv.Sim`: forward `pop_scale`/`total_pop` to `ss.Sim` so Starsim auto-multiplies every `scale=True` Result by `pop_scale` at finalize. Dynamic rescaling (`rescale`/`make_naive`) stays **deferred** (kept, not dropped).
4. A new **M2 anchor** (`tests/regression/anchor_m2.py`, single-variant full natural history) + an extended parity gate covering **both** the new burden metrics (`cum_symptomatic`/`cum_severe`/`cum_critical`/`cum_deaths`) **and** the re-converged transmission metrics (`cum_infections`/`peak_prevalence`/`peak_n_infectious`), all vs. a v3.1.8 single-variant baseline.
5. Tests: prognosis-tree structural invariants (timer-chain ordering, sex/asymptomatic-cannot-die), the M2 anchor smoke/drift/parity gates, and a numeric check of the two mean-preserving normalizations (`beta_dist` mean ≈ 1, viral-load time-average ≈ 1).

## Problem statement

M1 landed a runnable single-variant epidemic but two ways short of v3.1.8 fidelity:

1. **Natural history is collapsed to its asymptomatic branch.** M1's `cv.COVID.set_prognoses` (covid.py:74–86) schedules only `ti_infectious = ti + dur_exp2inf` and `ti_recovered = ti_infectious + dur_asym2rec`; `step_state` (covid.py:88–98) flips only E→I and I→R. So `cum_symptomatic`/`cum_severe`/`cum_critical`/`cum_deaths` are identically 0 and were *excluded* from the M1 gate. The v3.1.8 health-state machine (the quarantined `covasim/_v2_legacy/people.py`) draws **all four branch outcomes — symptomatic? severe? critical? dead?** — once at infection inside `infect()` (people.py:435–586) and only flips booleans per step via `check_infectious/symptomatic/severe/critical/recovery/death` (people.py:164–312). The age-banded conditional prognoses (`get_prognoses`/`relative_prognoses`, parameters.py:230–294), the global severity scalers (`rel_symp_prob`/`rel_severe_prob`/`rel_crit_prob`/`rel_death_prob`), and the bed-capacity feedbacks (`no_hosp_factor`/`no_icu_factor`) are all unported.

2. **M1 was missing the symptomatic disease course** (and, secondarily, v3's transmission-shape factors). The reported "~15–33% smaller than v3" M1 drift was found during M2 design — by direct diagnostic against a v3.1.8 worktree — to be **mostly an artifact of the M1 v3 baseline, not a v4 defect.** The M1 baseline's all-asymptomatic config (`sim['prognoses']['symp_probs'][:]=0` applied after `initialize()`) was **not honored** by v3 (~68% of baseline agents were still symptomatic), so the baseline carried symptomatic agents whose longer recovery durations (`mild2rec`/`sev2rec`) inflated its mean infectious period to **9.6 days vs M1's correct 8.0**. R0 ∝ infectious duration, so that ~20% duration inflation accounts for essentially the whole `cum_infections` gap. **v4's M1 durations are correct** (measured latent 4.50 d, infectious 8.01 d, matching the nominal `dur` means) — M1 is not buggy. Disabling v3's `beta_dist`+`viral_load(t)` (both individually mean-preserving — `_v2_legacy/people.py:159`, `utils.py:39–84`) changed the baseline by only ~7%, confirming they are a *minor* contributor (superspreading variance + early-phase front-loading), **not** the source of the apparent gap. **Therefore M2 re-converges the transmission metrics primarily by adding the full prognosis tree** — symptomatic agents' longer infectious periods raise the average infectious duration to v3's — and adds `viral_load + beta_dist` for genuine (if small) fidelity. Both are in M2 scope. Crucially, **M2's v3 baseline must use the full *default* prognoses (a real symptomatic/asymptomatic mix), never a forced-asymptomatic config**, so the v4 and v3 sides are compared like-for-like; M2 re-convergence is then expected to be straightforward rather than a stretch.

M2's problem is therefore two coupled extensions on `cv.COVID`:

1. **Burden parity** — reproduce the four-branch pre-scheduled prognosis tree so the cumulative symptomatic/severe/critical/death counts match a v3.1.8 single-variant baseline within multi-seed noise.
2. **Transmission re-convergence** — add `viral_load × beta_dist` so the M1 basic-transmission metrics (`cum_infections`, `peak_prevalence`, `peak_n_infectious`) re-converge to v3.1.8 within `|z| < 3`.

As in M0/M1, **bit-for-bit equivalence is impossible and not the bar** (Starsim's per-distribution CRN replaces v3's global numba stream). The bar is: existing (non-quarantined) tests pass, and the M2 anchor's burden **and** re-converged transmission metrics overlap a v3.1.8 baseline within the `parity_gate`'s `|z| < 3` (multi-seed) or ±10% drift (single-seed development gate).

## Design decisions

### 1. Extend `cv.COVID` with the full prognosis tree — pre-scheduled, threshold-only transitions

The single most important fidelity decision (carried from the natural-history findings and confirmed against the quarantined `_v2_legacy/people.py`): v3 draws **every branch outcome once at infection** and only flips booleans per step. M2 preserves this exactly — all four `set_prognoses` branch draws happen in one call; `step_state` never re-draws a probability. This matches the trajectory-based (not per-step-Markov) pattern proven in the hpvsim M02 port, and is what keeps the numerical regression aligned.

#### 1a. New parameters (`define_pars`, extending covid.py:45–50)

M1 already carries `beta`, `init_prev`, `dur_exp2inf`, `dur_asym2rec`. M2 adds (defaults from parameters.py:83–120; lognormals are `mean`/`std` in days, the Starsim `ss.lognorm_ex(mean=ss.days(par1), std=ss.days(par2))` form M1 already uses):

```python
# Progression durations (parameters.py:85-88)
dur_inf2sym  = ss.lognorm_ex(mean=ss.days(1.1),  std=ss.days(0.9)),   # I -> symptomatic
dur_sym2sev  = ss.lognorm_ex(mean=ss.days(6.6),  std=ss.days(4.9)),   # symptomatic -> severe
dur_sev2crit = ss.lognorm_ex(mean=ss.days(1.5),  std=ss.days(2.0)),   # severe -> critical
dur_crit2die = ss.lognorm_ex(mean=ss.days(10.7), std=ss.days(4.8)),   # critical -> death
# Recovery durations (parameters.py:91-95)  [dur_asym2rec already in M1]
dur_mild2rec = ss.lognorm_ex(mean=ss.days(8.0),  std=ss.days(2.0)),
dur_sev2rec  = ss.lognorm_ex(mean=ss.days(18.1), std=ss.days(6.3)),
dur_crit2rec = ss.lognorm_ex(mean=ss.days(18.1), std=ss.days(6.3)),
# Global severity scalers (parameters.py:98-101), all 1.0
rel_symp_prob=1.0, rel_severe_prob=1.0, rel_crit_prob=1.0, rel_death_prob=1.0,
# Health-system capacity (parameters.py:116-120)
n_beds_hosp=None, n_beds_icu=None, no_hosp_factor=2.0, no_icu_factor=2.0,
# Transmission-shape pars (the M1-gap closure; parameters.py:60-63)
beta_dist = dict(dist='neg_binomial', par1=1.0, par2=0.45, step=0.01),
viral_dist = dict(frac_time=0.3, load_ratio=2, high_cap=4),
asymp_factor = 1.0,
```

The age-prognosis tables are loaded from Covasim's **active** `parameters.get_prognoses()` + `relative_prognoses()` (parameters.py:230–294) — kept active since M1. **Critical: store the CONDITIONAL probabilities** that `relative_prognoses` produces (`death_probs /= crit_probs`, `crit_probs /= severe_probs`, `severe_probs /= symp_probs`), not the absolute age tables. `comorbidities` (default 1.0 per band) folds into `severe_prob` at init, mirroring people.py:155.

**Lognormal-mean caveat ([VERIFY]):** v3's `dur` distributions are `lognormal_int` parameterized by the **mean/std of the lognormal itself**, then rounded to integer days (utils.py:228–236). M1 already chose `ss.lognorm_ex(mean=, std=)` as the equivalent and passed its trajectory gate, so M2 reuses that exact form. The implementer must **[VERIFY] at code-time** that `ss.lognorm_ex` interprets `mean`/`std` as the distribution's mean/std (not log-space params — `ss.lognorm_im` is the log-space form) and that the per-day rounding is either reproduced or absorbed by the parity tolerance, as hpvsim M02 found (it settled on `lognorm_ex` in years and added CRN-safe stochastic rounding of fractional `ti_*` to avoid a `np.ceil` bias).

#### 1b. New states (`define_states`, extending covid.py:54–66)

Add four BoolStates and four FloatArr timers (M1 already has `susceptible`/`infected`/`exposed`/`recovered` + `ti_infected`/`ti_exposed`/`ti_infectious`/`ti_recovered` + `rel_sus`/`rel_trans`). Per the natural-history findings (states are cumulative nested flags, not mutually exclusive — "mild" = `symptomatic & ~severe`, "asymptomatic" = `infectious & ~symptomatic`):

```python
ss.BoolState('symptomatic', label='Symptomatic'),
ss.BoolState('severe',      label='Severe (needs hospitalization)'),
ss.BoolState('critical',    label='Critical (needs ICU)'),
ss.BoolState('dead',        label='Dead'),
ss.FloatArr('ti_symptomatic'),
ss.FloatArr('ti_severe'),
ss.FloatArr('ti_critical'),
ss.FloatArr('ti_dead'),
# Per-agent base CONDITIONAL probabilities (filled by age-binning at init, mirroring people.py:152-157)
ss.FloatArr('symp_prob',   default=0.0),
ss.FloatArr('severe_prob', default=0.0),
ss.FloatArr('crit_prob',   default=0.0),
ss.FloatArr('death_prob',  default=0.0),
# Per-agent constant transmissibility draw (decision 2)
ss.FloatArr('rel_trans_base', default=1.0),
```

`ti_recovered` (already present) is reused for all four recovery paths; `ti_dead` is **mutually exclusive** with `ti_recovered` — set one, leave the other NaN (people.py:579). Because the death and recovery timers are exclusive branches, death never collides with a scheduled recovery (the Starsim SIR pattern; Starsim findings §2 "branch at prognosis time so the two timers are never both set").

**`exposed` semantics ([VERIFY] / Open question A).** M1's `exposed` means "pre-infectious" — `step_state` clears it at I-onset (covid.py:93). v3's `exposed` means "infected at all" — it stays true from infection through recovery/death, and `n_exposed` counts all infected (defaults.py:150; natural-history findings §1, §5.4). For `n_exposed` result parity, M2 should switch to v3's semantics: keep `exposed` (= `infected`) true through the whole infection. **Recommendation:** redefine `exposed` to track `infected` (do *not* clear it at I-onset), and rely on the `infectious` property (already overridden, covid.py:69–72: `infected & (ti_infectious <= ti)`) for the I-subset. This is a behavioral change from M1's covid.py:93 and is flagged for Cliff — it only matters if `n_exposed` is a gated metric (it is in v3's stock results but is *not* in the M0/M1 pinned summary set, so the gate may not require it; confirm before changing M1 behavior).

#### 1c. Per-agent base probabilities by age-binning at init

In `init_post` (extending covid.py:113–139, before/after seeding), fill the four `*_prob` FloatArrs by age-binning, mirroring people.py:152–157:

```python
progs = cvpar.get_prognoses(...)            # active parameters.py:230 (absolute age tables)
progs = cvpar.relative_prognoses(progs)     # active parameters.py:285 -> CONDITIONAL probs
age = np.asarray(self.sim.people.age)
inds = np.digitize(age, progs['age_cutoffs']) - 1
self.symp_prob[:]   = progs['symp_probs'][inds]
self.severe_prob[:] = progs['severe_probs'][inds] * progs['comorbidities'][inds]  # comorbidity folds into severe
self.crit_prob[:]   = progs['crit_probs'][inds]
self.death_prob[:]  = progs['death_probs'][inds]
```

(`rel_sus`/`trans_ORs` age odds-ratios are all 1.0 by default — parameters.py:258 — so `rel_sus` stays at M1's 1.0; the `trans_OR` factor for `rel_trans` is likewise 1.0, leaving `rel_trans_base` driven purely by the `beta_dist` draw, decision 2.) **[VERIFY]** the exact `get_prognoses`/`relative_prognoses` signatures and that `digitize` binning reproduces people.py's `np.digitize(age, age_cutoffs) - 1`.

#### 1d. The four-branch pre-scheduled decision tree (`set_prognoses`)

Replace M1's asymptomatic-only scheduling (covid.py:74–86) with v3's full tree (people.py:522–579). Each branch uses a dedicated reusable `ss.bernoulli` whose per-agent `p` is `.set()` from the conditional prob array (the hpvsim M02 pattern: declare scratch bernoullis once in `__init__` — `self._symp_bern = ss.bernoulli(p=0.5)`, etc. — to keep each draw on a stable per-Dist CRN slot; Starsim findings §5a, hpvsim M02 findings §3). On infection (after the M1 entry: `susceptible=False, infected=True, exposed=True, ti_infected=ti_exposed=ti, ti_infectious = ti + dur_exp2inf.rvs(uids)`), and after defensively resetting downstream `ti_*` to NaN:

1. **Symptomatic?** `p = rel_symp_prob × symp_prob[uids]`; `is_symp = self._symp_bern.rvs(uids)`. Asymptomatic → `ti_recovered = ti_infectious + dur_asym2rec.rvs`. (Asymptomatic cannot die.)
2. **Severe? (symptomatic only)** `ti_symptomatic = ti_infectious + dur_inf2sym.rvs`; `p = rel_severe_prob × severe_prob`; `is_sev`. Mild → `ti_recovered = ti_symptomatic + dur_mild2rec.rvs`.
3. **Critical? (severe only)** `ti_severe = ti_symptomatic + dur_sym2sev.rvs`; `p = rel_crit_prob × crit_prob × (no_hosp_factor if hosp_max else 1.0)`; `is_crit`. Non-critical → `ti_recovered = ti_severe + dur_sev2rec.rvs`.
4. **Die? (critical only)** `ti_critical = ti_severe + dur_sev2crit.rvs`; `p = rel_death_prob × death_prob × (no_icu_factor if icu_max else 1.0)`; `is_dead`. Survive → `ti_recovered = ti_critical + dur_crit2rec.rvs`. Die → `ti_dead = ti_critical + dur_crit2die.rvs`, leave `ti_recovered = NaN`.

All draws once per infection (CRN-safe via `uids`), each from its own bernoulli — exactly people.py's `binomial_arr` sequence. Use the *conditional* probs directly (do not re-divide). The `(1 - symp_imm)`/`(1 - sev_imm)` waning factors in people.py:523/539 are identity in M2 (no waning until M4) and are omitted. The variant `rel_*_prob` factors are 1.0 for single-variant M2 (they enter at M3).

**`hosp_max`/`icu_max` bed feedback (decision 1f, mostly inert at default).** Under the default `n_beds_hosp = n_beds_icu = None`, `hosp_max`/`icu_max` are falsy → factor 1.0 → no feedback (natural-history findings §4, §5.7). Recommendation: include the `(no_hosp_factor if hosp_max else 1.0)` / `(no_icu_factor if icu_max else 1.0)` **hooks** in the branch-3/branch-4 probability expressions so they are present for parity when beds are configured, but compute `hosp_max`/`icu_max` from the *current* severe/critical stock vs. the bed caps **at the infection timestep** (matching v3, where the constraint applied is the one in effect on the infection day, since branches are pre-scheduled). **[VERIFY]** how to read the live severe/critical counts inside `set_prognoses` (e.g. `np.count_nonzero(self.severe)` against `self.pars.n_beds_hosp`) — this is a within-step stock read, harmless at default `None`. Cliff may legitimately accept M2 with the bed hooks present-but-untested (they only matter when beds are set, which the anchor does not do); flag as **Open question D**.

#### 1e. Multi-stage threshold transitions (`step_state`) — no re-draws

Extend M1's two transitions (covid.py:88–98) to the full set, in v3's `update_states_pre` order (people.py:164–186): infectious → symptomatic → severe → critical → recovery → death. Each block masks `current-flag & (ti_<x> <= ti)` and flips the boolean (Starsim SIR pattern, diseases.py:651–662; hpvsim M02 `step_state`):

```python
ti = self.ti
# E semantics per Open question A: do NOT clear `exposed` at I-onset if exposed==infected.
# I -> symptomatic
to_symp = (self.infected & (self.ti_symptomatic <= ti)).uids
self.symptomatic[to_symp] = True
# -> severe
to_sev = (self.infected & (self.ti_severe <= ti)).uids
self.severe[to_sev] = True
# -> critical
to_crit = (self.infected & (self.ti_critical <= ti)).uids
self.critical[to_crit] = True
# -> recovered (clear symptomatic/severe/critical; people.py:271-276)
rec = (self.infected & (self.ti_recovered <= ti)).uids
self.infected[rec] = False; self.recovered[rec] = True
self.symptomatic[rec] = False; self.severe[rec] = False; self.critical[rec] = False
# -> dead (request the death; the people pipeline + step_die do the reset)
to_dead = (self.infected & (self.ti_dead <= ti)).uids
if len(to_dead):
    self.sim.people.request_death(to_dead)
```

Capture each transitioned-uid count per step to feed the `new_*` flow Results (decision 1g). Order matters for flow counting (clearance/recovery before death, as v3 counts deaths and recoveries in `update_states_pre` before transmission). Because death and recovery are exclusive branches (1b), a to-die agent has `ti_recovered = NaN` and can never satisfy the recovery mask — death wins automatically (no guard needed; Starsim findings §2).

#### 1f. Death via `request_death` + `step_die` reset

Per Starsim findings §2 and the hpvsim M02 pattern: **never write `people.ti_dead` directly** — call `self.sim.people.request_death(to_dead)` in `step_state` (people.py:417). `request_death` stamps `people.ti_dead = sim.ti`; the actual resolution happens later in `people.step_die` (loop slot 9, after transmission), which flips `people.alive=False` and calls each disease's `step_die(death_uids)` with the **reconciled** death set. M2 must add a `step_die` override that resets every custom BoolState so dead agents are not double-counted (Starsim findings §2 "disease pattern 3"):

```python
def step_die(self, uids):
    super().step_die(uids)
    self.infected[uids] = False; self.exposed[uids] = False
    self.symptomatic[uids] = False; self.severe[uids] = False; self.critical[uids] = False
    self.recovered[uids] = False; self.susceptible[uids] = False
    self.dead[uids] = True
```

**[VERIFY]** that `ss.Infection`/`ss.Disease` provides a base `step_die(self, uids)` to `super()` into (diseases.py:50; SIR overrides it at diseases.py:686 without calling super, so calling super may be optional — confirm at code-time). The `dead` BoolState is set here (the agent is confirmed dead this step), matching people.py:300–308. This `step_die` fires for *all* deaths (here only disease deaths exist; background `ss.Deaths` is not in the M2 stack), funneling them through one reset.

### 2. Per-agent time-varying `rel_trans` = `beta_dist × viral_load(t)` — transmission-shape fidelity

This adds v3's two per-agent transmissibility factors. Per the corrected problem statement, these are a **genuine but minor (~7%) contributor**, not the main M1-gap driver (that is the prognosis tree's symptomatic infectious periods, decision 1) — but they are real v3 mechanics and are ported for fidelity. v3's effective per-agent infectiousness on day `t` is the product `beta × beta_layer × rel_trans_i × viral_load_i(t) × f_asymp × f_iso × f_quar` (utils.py:87–95, line 93). M1 kept only `beta × beta_layer` (per-layer disease beta) with `rel_trans = 1.0`; M2 adds the two missing per-agent factors. `f_iso`/`f_quar` are M5; `f_asymp` is `asymp_factor` (default 1.0 → neutral) but should be exposed.

#### 2a. The per-agent constant overdispersion draw `rel_trans_base` (drawn once)

Each agent gets a constant `rel_trans_base_i = trans_OR_i × beta_dist_draw_i`. `trans_OR` is 1.0 by default (parameters.py:258), so `rel_trans_base_i` is the `beta_dist` draw alone. The draw is the v3 `beta_dist = dict(dist='neg_binomial', par1=1.0, par2=0.45, step=0.01)` — a negative-binomial, **mean exactly 1.0**, dispersion k=0.45 (high overdispersion → superspreaders), discretized to step 0.01 (utils.py:414–432). It is **mean-preserving** (verified: mean ≈ 1.001, std ≈ 1.50 over 2M draws); its effect is on superspreading *variance*, not the mean attack rate.

**Where/how to draw it ([VERIFY] / Open question B).** v3 draws it once per agent at init for the whole population (`_v2_legacy/people.py:159`, inside `set_prognoses` at population creation). Two faithful options in Starsim:
- **(i) draw at infection** in `set_prognoses` per infected `uids` (simplest; one draw per agent since each agent is infected at most once in M2's no-waning world).
- **(ii) draw at init** in `init_post` for all agents, store on `rel_trans_base`.

**Recommendation: (ii) draw all agents at init** — it matches v3's "drawn once per agent at initialization, held constant" semantics exactly and decouples the overdispersion draw from the per-step viral-load write. Use an `ss.Dist` so it participates in deterministic seeding. Starsim has no built-in negative-binomial-with-step sampler matching `cv.sample`'s exact discretization; **[VERIFY]** the closest `ss.Dist` (e.g. `ss.nbinom`/an `ss.Dist`-wrapped `n_neg_binomial`) or wrap the v3 `cvu.n_neg_binomial` (utils.py:414) inside an `ss.Dist` callback. The numeric check (decision 5) must confirm the realized sample mean ≈ 1.0 — a biased-mean overdispersion draw would *itself* shift the epidemic size and confound the gap closure.

#### 2b. The viral-load kinetic kernel `viral_load(t)` (precomputed per agent at infection)

`compute_viral_load` (utils.py:39–84) is a **two-level mean-preserving step function** (high plateau then low plateau, NOT a smooth rise/peak/decline), parameterized by `viral_dist = dict(frac_time=0.3, load_ratio=2, high_cap=4)`. With defaults the closed form is:

- normalizer `Z = 1 + frac_time × (load_ratio − 1)` = 1.3,
- HIGH value `load_ratio / Z` = 2/1.3 = **1.5385** for the early phase,
- LOW value `1 / Z` = **0.7692** afterward,
- the high→low switch is at `trans_point = min(frac_time, high_cap / infect_days_total)` of the infectious period (the `high_cap` caps the high phase at 4 days for long infections),
- 0 outside `[0, infect_days_total)`, where `infect_days_total = (ti_recovered or ti_dead) − ti_infectious`.

The time-average over the infectious period is **exactly 1.0** (verified: 0.3×1.5385 + 0.7×0.7692 = 1.000), so it is mean-preserving — it *redistributes* when transmission happens (front-loading into the high-load early phase) without changing the per-agent total. That front-loading, reaching susceptibles before depletion, is the dynamical effect expected to close the M1 gap.

**Recommendation:** precompute a small per-agent kernel at infection (in `set_prognoses`, once `ti_infectious` and the recovery/death date are known) via the closed form above, rather than recomputing the branch every step over the whole population (the hpvsim-style cached-kernel optimization; Transmission findings §5.2). Index it each step by `t − ti_infectious`.

#### 2c. Combine each step in `step_state` (which precedes transmission)

In `step_state` (decision 1e), before the stage transitions, write `self.rel_trans` for currently-infectious agents (Starsim findings §4 — `step_state` runs at loop slot 5, before `diseases.step()`/`infect()` at slot 9, so the write modulates this step's transmission):

```python
inf = self.infectious.uids
days_since = (self.ti - self.ti_infectious[inf]).astype(int)
days_since = np.clip(days_since, 0, kernel_len - 1)
vl = self.viral_load_kernel[inf, days_since]            # per-agent cached kernel
f_asymp = np.where(self.symptomatic[inf], 1.0, self.pars.asymp_factor)  # asymp_factor default 1.0
self.rel_trans[inf] = self.rel_trans_base[inf] * vl * f_asymp
```

Stock `ss.Infection.infect()` (diseases.py:237–290) then gates `rel_trans` by `infectious` (a BoolState→0/1 multiply) and computes per edge `p = rel_trans[src] × rel_sus[trg] × beta_per_dt` (diseases.py:228–235) — so the per-step `rel_trans` write directly modulates each source's per-contact transmission probability on the same step. `asymp_factor` applies to currently-asymptomatic (incl. pre-symptomatic) infectious agents only (utils.py:90); default 1.0 makes it neutral. **[VERIFY]** the `self.infectious.uids` access pattern (`infectious` is a property returning a BoolArr-like; confirm `.uids` works on the property result, or materialize via `(self.infected & (self.ti_infectious <= ti)).uids` as in covid.py:92).

**No custom transmission code** — exactly as M1, transmission stays the stock CRN-safe `ss.Infection.infect()`; M2 only writes the per-agent `rel_trans` state it consumes. The `beta` per-layer dict (M1 decision 5) is unchanged.

### 3. Absolute population scaling (`pop_scale`/`total_pop`) on `cv.Sim`

v3's `pop_scale` (default 1) means each agent represents `pop_scale` real people; `scaled_pop = pop_size × pop_scale` (base.py:316–320); at finalize every `scale=True` result is multiplied by `pop_scale` (sim.py:775). Starsim has the **same model built in**: `SimPars` carries `n_agents`/`total_pop`/`pop_scale` with `total_pop = n_agents × pop_scale` (parameters.py SimPars; Starsim findings §3), `validate_total_pop` reconciles them (set exactly one of `total_pop`/`pop_scale` — setting both raises), and every `ss.Result(scale=True)` is auto-multiplied by `pop_scale` once at `finalize_results` (sim.py:548; modules.py:724). So M2's scaling is almost free.

**Decision:** add `pop_scale=None`/`total_pop=None` to `cv.Sim.__init__` (sim.py:44) and forward them to `ss.Sim` (the hpvsim M02 pattern — forward `total_pop` straight through, let Starsim compute `pop_scale = total_pop / n_agents` and apply it). Map v3's `pop_size` → Starsim's `n_agents` (cv.Sim already passes `pop_size`; **[VERIFY]** the exact `ss.Sim` kwarg — M1's `cv.People(pop_size)` sets `n_agents`, and `cv.Sim` does not currently pass `total_pop`/`pop_scale`). Ensure every M2 burden Result is declared `scale=True` (M1's custom `n_infectious` already is — covid.py:104) so Starsim scales them; intensive metrics (prevalence, incidence, rates) stay `scale=False`.

**Deferred (kept, not dropped): dynamic rescaling.** v3's `rescale`/`rescale_threshold`/`rescale_factor` + `make_naive` (sim.py:535–555; _v2_legacy/people.py:378–411) — which starts at effective scale 1 and grows it as the epidemic spreads, converting non-naive agents back to naive — has **no direct Starsim analog** and is explicitly deferred to a later milestone (MIGRATION_PLAN.md scope table). M2 ships **static scaling only** (`rescale=False` equivalent): run the ABM at `n_agents`, multiply extensive results by the constant `pop_scale` at finalize. The multiplicative design means turning dynamic rescaling on later is additive. M2 validates the transmission fix at `pop_scale = 1` first, keeping scaling orthogonal to the gap closure.

### 4. The `infectious` property and recovery/death interaction

M1's `infectious` property (covid.py:69–72) is **kept unchanged**: `infectious = infected & (ti_infectious <= ti)`. M2 layers the prognosis tree on top without changing what "infectious" means — symptomatic/severe/critical agents are all still `infected` and past `ti_infectious`, so they remain infectious and transmit (with their stage-appropriate `rel_trans` from decision 2c, incl. the `asymp_factor` gate). Interaction rules:

- **Recovery** clears `infected` (so `infectious` becomes False automatically — the property derives from `infected`) and `symptomatic`/`severe`/`critical`, and sets `recovered=True`. With `use_waning=False` (M2), recovery confers **permanent immunity** — recovered agents never return to susceptible (matching M1 covid.py:94–97 and people.py's non-waning branch). Reinfection/waning is M4.
- **Death** is requested in `step_state` (`request_death`), resolved in `people.step_die` → `cv.COVID.step_die`, which sets `dead=True` and clears `infected`/`exposed`/all stage flags (decision 1f). Once `infected=False`, `infectious` is False and the agent stops transmitting and stops being counted alive (people sets `alive=False` and removes the agent at end of step — Starsim findings §0 loop slot 12).
- **Exclusivity of recovery vs. death** is guaranteed by the exclusive branch draw (1b): `ti_recovered` and `ti_dead` are never both set for one infection, so an agent destined to die can never trip the recovery mask — death wins without an explicit guard (Starsim findings §2).
- **`n_infectious` and the new stocks** are computed in `update_results` (extending covid.py:108–111) as `np.count_nonzero(self.<state>)` for `n_symptomatic`/`n_severe`/`n_critical` (the `infectious`-property pattern M1 established for `n_infectious`, since these need live boolean counts), all `scale=True`.

### 5. M2 result keys + the two normalization checks

#### 5a. New Results (`init_results`/`update_results`/`finalize_results`, extending covid.py:100–111)

The headline parity targets are the **cumulative burden flows**: `cum_symptomatic`, `cum_severe`, `cum_critical`, `cum_deaths` (plus `cum_recoveries`, and M1's `cum_infections`). Following the hpvsim M02 pattern (per-step event counters, cumulatives via `np.cumsum` in `finalize_results`) and the Starsim `Infection.init_results` convention (diseases.py:169–178):

- **New (daily) flow Results** (`scale=True`, `dtype=int`): `new_symptomatic`, `new_severe`, `new_critical`, `new_recoveries`, `new_deaths` (+ M1's `new_infections`/`new_infectious`). Written in `step_state` from the per-stage transitioned-uid counts (decision 1e).
- **Cumulative Results** (`scale=True`): `cum_symptomatic`/`cum_severe`/`cum_critical`/`cum_recoveries`/`cum_deaths`, filled in `finalize_results` as `np.cumsum(new_*)`.
- **Stock Results** (`scale=True`, count of agents with the flag true on that day): `n_symptomatic`/`n_severe`/`n_critical`/`n_dead`/`n_recovered` (+ M1's `n_infectious`), in `update_results` via `np.count_nonzero`.

These mirror v3's `result_flows`/`result_stocks` (defaults.py:145–199); the v3 default "Health outcomes" plot is exactly `cum_severe, cum_critical, cum_deaths, cum_known_deaths` (defaults.py:366–371). **Out of M2 scope** (intervention/immunity flows): `new_tests`/`new_diagnoses`/`new_quarantined`/`new_doses`/`new_vaccinated`, `known_deaths` (depends on testing/diagnosis), `*_by_variant`, `reinfections`, `pop_nabs`. **[VERIFY]** the exact `ss.Result` / `define_results` signature already used at covid.py:103.

#### 5b. The two mean-preserving normalization checks (fast unit test)

A fast unit test must confirm both factors are mean-preserving *before* integrating, since a biased mean in either would itself shift the epidemic and confound the gap closure (Transmission findings §5.4): (i) the `beta_dist` realized sample mean ≈ 1.0 (std ≈ 1.50), and (ii) the per-agent viral-load kernel time-average ≈ 1.0 (high 1.538, low 0.769, average 1.000). If a residual transmission gap remains after both are added and verified mean-1, the implementer checks (a) the high phase is correctly placed *early* (front-loading matters for susceptible depletion), and (b) overdispersion variance is preserved (not accidentally averaged across contacts).

## M2 anchor scenario + pinned metrics

A new anchor `tests/regression/anchor_m2.py` (sibling to M1's `anchor_m1.py`), reusing the harness (`short_summary.py`, `parity.py`'s `parity_gate`, `multi_seed_v3.py`, `compare.py`). The M2 anchor is **single-variant, full natural history, no interventions, no waning** — it isolates the prognosis tree + transmission-shape closure, exactly the M2 capability:

```python
PARS = dict(
    pop_size     = 20_000,
    pop_infected = 20,
    pop_type     = 'random',   # plus a 'hybrid' variant of the anchor
    n_days       = 120,        # long enough to accumulate severe/critical/deaths past the peak
    n_variants   = 1,          # single wild variant
    use_waning   = False,      # permanent immunity; no NAb code
    rand_seed    = 0,          # sweep overrides 0..N-1
    verbose      = 0,
    # FULL prognosis tree (symp/severe/crit/death enabled via the age prognoses);
    # beta_dist + viral_load ON; no interventions, no analyzers; n_beds_* = None (bed feedback inert).
)
```

Two anchor variants run: `pop_type='random'` and `pop_type='hybrid'`. The v3.1.8 single-variant baseline is generated from a frozen v3.1.8 reference (the worktree workflow below).

**Two metric families, both gated through the M0 `parity_gate` (`|z| < 3`):**

1. **Burden metrics (the M2-specific gate, new at M2):** `cum_symptomatic`, `cum_severe`, `cum_critical`, `cum_deaths`. All identically 0 in M1 (asymptomatic-only) and excluded from the M1 gate; they re-enter here as the headline natural-history parity targets. (`cum_recoveries` reported diagnostically.)
2. **Re-converged transmission metrics (the M1-gap-closure gate):** `cum_infections`, `peak_prevalence` = `sim.results['prevalence'].max()`, `peak_n_infectious` = `sim.results['n_infectious'].max()` — the same M1 metrics that came out ~15–33% small. With `viral_load + beta_dist` added, M2 **requires** these re-converge to v3.1.8 within `|z| < 3`. This is the explicit M1-acceptance condition Cliff carried into M2.

The `parity_gate` z-formula (`z = (mean_v4 − mean_v3) / sqrt(SE_v3² + SE_v4²)`), degenerate-distribution policy, and `_SKIP_KEYS` are unchanged from M0/M1. `build_summary` in `short_summary.py` is extended to emit the burden metrics (it already builds the transmission + derived-rate metrics).

## Acceptance test

1. **Sim runs end-to-end** on the M2 anchor (random and hybrid, single variant, seed 0), and `cv.Sim().run()` returns results (continuous-runnability invariant) — a fast test.
2. **Existing (non-quarantined) tests pass** under the strict-warnings bar (`COVASIM_WARNINGS=error`): the M0/M1 harness tests, the M1 anchor gates (still green — M2 must not regress basic transmission), the new M2 tests, and `test_baselines.py`/`test_regression.py` either pass or carry the documented v4-skip.
3. **Prognosis-tree structural invariants** (fast, no baseline): every infected agent gets either a recovery or a death timer; the timer chain is ordered `ti_infectious ≤ ti_symptomatic ≤ ti_severe ≤ ti_critical ≤ ti_dead`; asymptomatic/mild/severe-non-critical agents never get `ti_dead` (only the critical branch reaches death); `step_die` clears all stage flags. (The hpvsim M02 lifecycle-invariant pattern.)
4. **Burden parity** vs the v3.1.8 single-variant baseline: `cum_symptomatic`/`cum_severe`/`cum_critical`/`cum_deaths` overlap within `parity_gate`'s `|z| < 3` (multi-seed, `@pytest.mark.slow`); single-seed ±10% drift (`compare.py`) informational per check-in. **This is the M2-specific acceptance gate.**
5. **M1-gap closure** vs the same baseline: `cum_infections`/`peak_prevalence`/`peak_n_infectious` re-converge to `|z| < 3` (the same metrics that were ~15–33% small at M1). **This is the explicit re-convergence condition.** If a residual transmission gap persists after `viral_load + beta_dist` are verified mean-preserving, it is documented (with the front-loading/overdispersion diagnostics from 5b) and the per-metric gate may be loosened only with Cliff's sign-off and a written residual rationale (the MIGRATION_PLAN.md `|z| < 5` escape hatch).
6. **Demo** plots the M2 anchor's health-outcome curves (`cum_severe`/`cum_critical`/`cum_deaths`) and the re-converged infection curve (`sim.plot()` on the Starsim base, or the anchor's `__main__`).

**Milestone completion** = acceptance test green locally *and* Cliff has reviewed and committed the work. No PRs/issues as gates (local effort).

## v3.1.8-baseline-via-worktree workflow

The current env's editable `covasim` is now **v4** (the port), so it can no longer generate v3.1.8 baselines directly. v3 baselines are generated from a **frozen v3.1.8 reference via a git worktree**, whose cwd shadows the editable v4 install:

```
git worktree add /tmp/cov-v3 main        # main is v3.1.8-equivalent (the pre-port baseline)
cd /tmp/cov-v3 && python <harness>/multi_seed_v3.py --anchor m2 --n 30 --out <baseline.json>
```

Running with `cwd=/tmp/cov-v3` makes Python import the v3.1.8 `covasim` from the worktree (the local package shadows the editable v4 install on `sys.path`), so the M2 anchor's `make_sim()` builds a **v3.1.8** sim and the sweep writes genuine v3.1.8 per-seed summaries. The harness's `anchor_m2.py` must therefore build a *v3-compatible* sim when run under v3 (the M0/M1 anchors already do this — the anchor is the dual-version scenario definition; `multi_seed_v3.py --anchor m2` selects the M2 PARS). The generated `v3_seeds_m2_n{N}.json` is **gitignored** (large, env-specific); the parity gate **skips** (does not fail) when it is absent, so a contributor without the worktree can still run the rest of the suite. The v4 seeds are generated in-process by the slow gate. This is the same worktree method documented for M1; M2 adds the `--anchor m2` selector and the burden metrics to `build_summary`.

## Workflow (pause-for-review-and-commit)

Local effort; the assistant prepares each piece, leaves it **uncommitted**, and pauses for Cliff to review and commit. **The assistant never commits and never pushes.** Check-ins **2–5 times per milestone** at natural sub-task boundaries. M2's natural check-in points:

1. **The prognosis-tree extension to `cv.COVID`** — new pars/states/timers + the four-branch `set_prognoses` + multi-stage `step_state` + `request_death`/`step_die`, with the structural-invariant tests (no baseline needed).
2. **The transmission-shape closure** — `rel_trans_base` (`beta_dist`) draw + the viral-load kernel + the per-step `rel_trans` write, with the two mean-preserving normalization unit tests.
3. **`pop_scale`/`total_pop`** on `cv.Sim` + the burden Results (`new_*`/`cum_*`/`n_*`) and the `short_summary.py` extension.
4. **The M2 anchor + parity/drift gates + demo plot** + the v3.1.8 worktree baseline regeneration + the plan/spec docs.
5. (If needed) a tolerance/residual-gap pass once first-run variability and the re-convergence margin are observed.

## Out of scope for M2 (deferred to later milestones)

- **Multi-variant + cross-immunity** (`n_variants > 1`, one `cv.COVID` instance per variant, host-level exclusivity over the SEIR/severity chain, `cv.CrossImmunity(ss.Connector)`, the per-variant `rel_*_prob` factors) → **M3**.
- **Waning immunity / NAbs / reinfection** (`use_waning=True`, `symp_imm`/`sev_imm` reducing the branch-1/branch-2 probabilities, recovery returning to susceptible, `nab_kin`) → **M4**.
- **Testing / tracing / quarantine / isolation** (the `f_iso`/`f_quar` transmission factors, `known_dead`, the diagnosis state machine) → **M5**.
- **Vaccination** → **M6**. **Calibration / Fit** → **M7**. **MultiSim / Scenarios / parallel** and the **retrofit of M2's acceptance onto the shared z-score gates** → **M8**. **Analyzers / TransTree / full v3 plotting / synthpops backend** → **M9**.
- **Dynamic rescaling** (`rescale`, `rescale_threshold`, `rescale_factor`, `make_naive`/`make_nonnaive`) — no direct Starsim analog; M2 ships static `pop_scale` only, dynamic rescaling is **ported in a later milestone (kept, not dropped)**.
- **Health-system bed feedback as a *tested* feature** — the `no_hosp_factor`/`no_icu_factor` hooks are present in the branch-3/branch-4 probability expressions for parity, but `n_beds_* = None` defaults make them inert and the M2 anchor does not configure beds; their active validation is deferred (Open question D).
- **Variant-level severity scalers** (alpha `rel_severe_prob=1.64`, delta `rel_severe_prob=3.2`, parameters.py:386–416) — single-variant M2 uses the global scalers at 1.0; per-variant scalers are M3.

## Open questions for Cliff

- **A. `exposed` semantics:** switch `exposed` to v3's "infected-at-all" meaning (stays true through recovery/death, for `n_exposed` parity) — a behavioral change from M1's covid.py:93 which clears it at I-onset — or keep M1's "pre-infectious" meaning? (Matters only if `n_exposed` becomes a gated metric; it is not in the current pinned summary set.)
- **B. `beta_dist` draw site + sampler:** draw `rel_trans_base` for all agents at init (recommended, matches v3's once-per-agent semantics) vs. per-`uids` at infection? And which `ss.Dist` reproduces v3's `neg_binomial(par1=1.0, par2=0.45, step=0.01)` discretized sampler — a built-in `ss.nbinom`-style dist, or wrap `cvu.n_neg_binomial` in an `ss.Dist`? (The realized mean **must** be ≈ 1.0.)
- **C. Viral-load kernel storage:** precompute a per-agent cached kernel at infection (recommended; avoids per-step whole-population recompute) vs. recompute `compute_viral_load` each step? And confirm the kernel length / clip behavior for the longest infections (`high_cap=4` day high-phase cap).
- **D. Bed-capacity feedback:** ship the `no_hosp_factor`/`no_icu_factor` hooks present-but-untested at M2 (recommended; inert under default `n_beds_*=None`) with active validation deferred, or pull a bed-configured anchor into M2's gate?
- **E. Lognormal parametrization + rounding:** confirm `ss.lognorm_ex(mean=ss.days(par1), std=ss.days(par2))` reproduces v3's `lognormal_int` (mean/std of the lognormal, rounded to integer days) closely enough for the duration parity, or whether M2 needs hpvsim-style CRN-safe stochastic rounding of fractional `ti_*` to avoid a `np.ceil` bias (the hpvsim M02 finding).
- **F. `pop_scale`/`total_pop` plumbing:** forward `total_pop` straight to `ss.Sim` and let Starsim compute/apply `pop_scale` (recommended, the hpvsim shape), confirming the `pop_size → n_agents` mapping and that exactly one of `total_pop`/`pop_scale` is set (Starsim raises if both).
- **G. Re-convergence residual policy:** if the transmission metrics do not fully re-converge to `|z| < 3` after `viral_load + beta_dist` are verified mean-preserving, accept a documented residual with a loosened per-metric gate (`|z| < 5`, MIGRATION_PLAN.md escape hatch), or treat any residual as a blocking bug?

## Linked documents

- [`MIGRATION_PLAN.md`](../MIGRATION_PLAN.md) — overall plan; §M2 (Natural-history parity) is this milestone's source.
- [`specs/2026-05-29-covasim-m1-basic-transmission-design.md`](2026-05-29-covasim-m1-basic-transmission-design.md) — M1 design (the `cv.COVID`/`cv.Sim`/`cv.Network`/`cv.People` surface M2 extends; the documented ~15% transmission gap M2 closes).
- [`specs/2026-05-29-covasim-m0-foundation-design.md`](2026-05-29-covasim-m0-foundation-design.md) — M0 design (the regression harness, anchor pattern, `parity_gate` z-score gate that M2 reuses).
- hpvsim M02 design + shipped code (`/home/cliffk/idm/hpvsim/hpvsim/hpv.py`) — the branching multi-stage natural-history-on-`ss.Infection` template this spec mirrors (pre-scheduled trajectory, dedicated bernoullis, `request_death`/`step_die`, `scale=True` event counters, multi-seed z-score parity gate).
- [`covasim/covid.py`](../../covasim/covid.py) — the M1 `cv.COVID` this milestone extends.
- [`covasim/_v2_legacy/people.py`](../../covasim/_v2_legacy/people.py) — the quarantined v3 health-state machine (the authoritative natural-history reference: `set_prognoses` 139–161, `infect` decision tree 435–586, `check_*` 222–312, step order 164–186).
- [`covasim/parameters.py`](../../covasim/parameters.py) (active) — durations 83–95, scalers + bed pars 98–120, `get_prognoses` 230–282, `relative_prognoses` 285–294, `beta_dist`/`viral_dist`/`asymp_factor` 60–63, `pop_scale`/`scaled_pop` 47–52.
- [`covasim/_v2_legacy/sim.py`](../../covasim/_v2_legacy/sim.py) + [`covasim/utils.py`](../../covasim/utils.py) — `compute_viral_load` (utils.py:39–84), `compute_trans_sus` (87–95), `n_neg_binomial` (414–432), `rescale_vec` result scaling (sim.py:775–787).
- [`tests/regression/`](../../tests/regression/) — M0/M1 harness; M2 adds `anchor_m2.py` and extends `short_summary.py` + `multi_seed_v3.py --anchor m2`.
