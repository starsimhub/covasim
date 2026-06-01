# Covasim v4.0 — M2 Natural-History-Parity Implementation Plan

> **For agentic workers:** Implement this plan task-by-task; steps use checkbox (`- [ ]`) syntax for tracking. No special plugin or sub-skill is required.
>
> **CRITICAL — commit discipline.** This is a local effort with a **pause-for-review-and-commit** cadence. The assistant (Claude) **prepares and stages** each piece of work and then **pauses for Cliff Kerr (the Covasim author) to review and commit**. **The assistant NEVER runs `git commit` and NEVER runs `git push`.** Where this plan reaches a check-in boundary it says **PAUSE FOR CLIFF** and lists what to stage; it contains no `git commit` commands. There are **4 check-ins** across M2 (a 5th, tolerance/residual, is conditional).
>
> **VERIFY-AT-CODE-TIME discipline.** Several Starsim 3.3.x signatures are quoted from the design spec + API findings (starsim 3.3.4 at `/home/cliffk/idm/starsim/starsim/`). They are correct as of writing, but the implementer **MUST verify each one against the installed `starsim` before relying on it** — every such point is flagged inline with **[VERIFY]**. Do NOT guess an API name; read the real source (`python -c "import starsim as ss, inspect; print(inspect.getsource(ss.Infection.set_outcomes))"`) or grep the installed package. Confirmed present in starsim 3.3.4 at plan time: `ss.lognorm_ex`, `ss.bernoulli`, `ss.nbinom`, `ss.choice`, `ss.probperday`, `ss.People.request_death(self, uids)`.

**Goal:** Make `cv.COVID` epidemiologically complete for a single variant. Extend M1's asymptomatic-only S→E→I→R into Covasim's **full prognosis tree** — exposed → infectious → (asymptomatic / mild / severe / critical) → recovered / dead. Restoring the symptomatic disease course (with its longer mild/severe/critical recovery durations) is **the main driver that re-converges the M1 transmission metrics to v3.1.8** — the reported "~15% M1 gap" was found (M2-design diagnostic; see the design spec's problem statement) to be **mostly an artifact of the M1 v3 baseline accidentally carrying symptomatic agents**, not a v4 defect (v4's M1 durations are correct). Also add the per-agent time-varying `rel_trans` (`viral_load × beta_dist`) for transmission-shape fidelity (a smaller, genuine ~7% contributor), and absolute population scaling (`pop_scale`/`total_pop`) to `cv.Sim`. The user-visible demo: run a single-variant epidemic and plot the health-outcome curves (`cum_severe`, `cum_critical`, `cum_deaths`) alongside the re-converged infection curve.

**Architecture:** M2 is a pure **extension** of the M1 surface — it adds states, parameters, and methods to the existing `cv.COVID(ss.Infection)` (file `covasim/covid.py`) and adds `pop_scale`/`total_pop` forwarding to `cv.Sim(ss.Sim)` (file `covasim/sim.py`). **No new public class is created, and nothing is quarantined or removed.** The locked public names from M1 — `cv.COVID`, `cv.Sim`, `cv.People`, `cv.Network` — are unchanged. The whole prognosis trajectory (every branch outcome + every duration) is **pre-drawn once at infection** in `set_prognoses` (the v3 design, confirmed in the quarantined `covasim/_v2_legacy/people.py`; the hpvsim M02 trajectory-based pattern); `step_state` only flips booleans on `ti_<x> <= ti` thresholds and never re-draws a probability. Transmission stays the stock CRN-safe `ss.Infection.infect()`; M2 only writes the per-agent `rel_trans` state it consumes. The continuous-runnability invariant (`cv.Sim().run()` returns results) holds at every commit, since M2 only grows the disease's state machine.

**Tech Stack:** Python 3.9–3.13, pytest + pytest-xdist, covasim (the v4 port on `starsim-port`), starsim 3.3.x (3.3.4 at plan time), sciris.

**Authority:** the design spec `migration_plan/specs/2026-05-29-covasim-m2-natural-history-parity-design.md` (authoritative for every decision). The capability scope is `MIGRATION_PLAN.md` §M2. This plan implements exactly that scope.

**Class/file names — LOCKED by Cliff:** `cv.COVID` (file `covasim/covid.py`), `cv.Sim`/`cv.People`/`cv.Network` keep their names. No new public class at M2.

**Spec open questions A–G — default path adopted here** (each flagged for Cliff at the relevant check-in; if Cliff rules otherwise, adjust):
- **A** (`exposed` semantics) — default: **switch `exposed` to v3's "infected-at-all" meaning** (stays true through recovery/death) only if a gate metric needs it; since `n_exposed` is *not* in the pinned summary set, the recommendation is to **keep M1's behavior unless burden parity requires the change** — confirm at code-time (check-in 1).
- **B** (`beta_dist` draw site + sampler) — default: **draw `rel_trans_base` for all agents at init** in `init_post`, wrapping the active `cv.n_neg_binomial` (utils.py:414, still active) in an `ss.Dist`; the realized mean **must** be ≈ 1.0 (check-in 2).
- **C** (viral-load kernel storage) — default: **precompute a per-agent cached kernel at infection** in `set_prognoses`, indexed each step by `ti − ti_infectious` (check-in 2).
- **D** (bed-capacity feedback) — default: ship the `no_hosp_factor`/`no_icu_factor` hooks **present-but-inert** (default `n_beds_*=None`); active validation deferred (check-in 1).
- **E** (lognormal parametrization + rounding) — default: reuse M1's `ss.lognorm_ex(mean=ss.days(par1), std=ss.days(par2))`; add hpvsim-style CRN-safe stochastic rounding of fractional `ti_*` **only if** the duration/burden parity shows a `np.ceil`-style bias (check-in 1).
- **F** (`pop_scale`/`total_pop` plumbing) — default: forward `total_pop`/`pop_scale` straight to `ss.Sim`, let Starsim compute/apply `pop_scale`; set exactly one of the two (Starsim raises if both) (check-in 3).
- **G** (re-convergence residual policy) — default: if transmission metrics do not fully re-converge to `|z| < 3` after both factors are verified mean-preserving, document the residual and loosen the per-metric gate to `|z| < 5` **only with Cliff's sign-off** (check-in 4 / conditional check-in 5).

---

## Starting state (confirmed at plan time)

- Branch: `starsim-port` (M1 is landed and committed). The assistant never creates/switches branches.
- M1 shipped: `covasim/covid.py` (`cv.COVID(ss.Infection)`, minimal S→E→I→R — `define_pars` with `beta`/`init_prev`/`dur_exp2inf`/`dur_asym2rec`; `define_states` with `susceptible`/`infected`/`exposed`/`recovered` + `ti_infected`/`ti_exposed`/`ti_infectious`/`ti_recovered` + `rel_sus`/`rel_trans`, `reset=True`; `infectious` **property** = `infected & (ti_infectious <= ti)`; `set_prognoses` schedules `ti_infectious`/`ti_recovered`; `step_state` flips E→I and I→R; `init_results`/`update_results` add `n_infectious`; `init_post` exact-count `pop_infected` seeding). `covasim/sim.py` (`cv.Sim`, thin wrapper, per-layer beta dict). `covasim/people.py` (`cv.People`, v3 age distribution). `covasim/network.py` (`cv.Network` + `make_networks(pop_type, contacts)`).
- The active (non-quarantined) `covasim/parameters.py` provides `get_prognoses(by_age=True, version=None)` and `relative_prognoses(prognoses)`. **CRITICAL: `get_prognoses` already calls `relative_prognoses` internally** (it returns CONDITIONAL probabilities). Do **not** call `relative_prognoses` again on its output — that would double-divide. (This corrects the spec decision 1c snippet, which shows a redundant second `relative_prognoses` call.)
- The active `covasim/utils.py` still exports `cv.sample` and `cv.n_neg_binomial(rate, dispersion, n, step)` (utils.py:414) — the exact v3 `beta_dist` sampler is available to wrap.
- The regression harness `tests/regression/`: `anchor_m1.py` (dual-version: `_IS_V4 = hasattr(cv, 'COVID')`; v3 branch zeroes the prognoses; `make_sim`/`run_and_summarize`), `short_summary.py` (`build_summary` for M0; `build_summary_m1` + `METRIC_KEYS_M1 = ('cum_infections','peak_prevalence','peak_n_infectious')`, dual-version via `hasattr(sim,'diseases')`; `SKIP_KEYS`; `_series_max`), `parity.py` (`parity_gate(v4_seeds, v3_seeds, z_threshold=3.0, skip_keys=frozenset())`), `multi_seed_v3.py` (`--anchor m0|m1_random|m1_hybrid --n --start-seed --out`; `_anchor_runner` registry; per-seed runner), `compare.py` (`--anchor`, `--save-snapshot`, ±10% drift), `README.md`.
- `tests/test_m1_parity.py` (slow gate, `@pytest.mark.parametrize('pop_type',['random','hybrid'])`, `N_V4_SEEDS=10`, `M_V3_SEEDS=30`, baseline `v3_m1_{pop_type}_seeds_n30.json`).
- `.gitignore` already has `tests/regression/v3_m1_*seeds*.json`, `v4_m1_*seeds*.json`, `v3_m1_contacts*.json`, `anchor_m1_snapshot.json` (lines 150–154).
- The frozen v3.1.8 reference is generated **via git worktree** (`git worktree add /tmp/cov-v3 main`); running the harness with `cwd=/tmp/cov-v3` shadows the editable v4 install so `import covasim` is v3.1.8. (Documented in Task 5 Step 1.)

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `covasim/covid.py` | **Modify** (extend `cv.COVID`) | Add prognosis-tree pars/states/timers; four-branch `set_prognoses`; multi-stage `step_state`; `request_death` + `step_die`; `beta_dist` `rel_trans_base` + viral-load kernel + per-step `rel_trans` write; age-prognosis fill in `init_post`; burden Results (`new_*`/`cum_*`/`n_*`) |
| `covasim/sim.py` | **Modify** (extend `cv.Sim`) | Add `pop_scale=None`/`total_pop=None` to `__init__`; forward to `ss.Sim` |
| `tests/test_covid.py` | **Modify** (extend) | Prognosis-tree structural-invariant tests (timer-chain ordering, asymptomatic-cannot-die, `step_die` reset); the two mean-preserving normalization unit tests |
| `tests/test_sim.py` | **Modify** (extend) | `pop_scale`/`total_pop` scaling test (extensive results × pop_scale; intensive unscaled) |
| `tests/regression/anchor_m2.py` | **Create** | M2 single-variant full-natural-history anchor (random + hybrid), dual-version (v3.1.8 + v4) |
| `tests/regression/short_summary.py` | **Modify** | Add `build_summary_m2` + `METRIC_KEYS_M2` (burden + re-converged transmission metrics), dual-version extractor |
| `tests/regression/multi_seed_v3.py` | **Modify** | Add `--anchor m2_random`/`m2_hybrid` to the `_anchor_runner` registry |
| `tests/regression/compare.py` | **Modify** | Add `m2_random`/`m2_hybrid` to `_resolve_run` |
| `tests/regression/README.md` | **Modify** | Document the M2 anchor + the `git worktree` baseline-regeneration workflow |
| `tests/test_m2_parity.py` | **Create** | `@pytest.mark.slow` z-score parity gate vs v3.1.8 M2 baseline (burden + transmission), both backends; skips when baseline absent |
| `.gitignore` | **Modify** | Add `tests/regression/v3_m2_*seeds*.json`, `v4_m2_*seeds*.json`, `anchor_m2_snapshot.json` |

The generated M2 baselines (`tests/regression/v3_m2_*seeds*.json`) are gitignored and stay local-only.

---

## Task 1: Extend `cv.COVID` with the full prognosis tree (CHECK-IN 1)

**Files:**
- Modify: `covasim/covid.py` (pars, states, timers, `set_prognoses` branch tree, `step_state` transitions, `request_death`/`step_die`, age-prognosis fill in `init_post`)
- Modify: `tests/test_covid.py` (prognosis-tree structural invariants — no baseline needed)

Authority: spec decision 1 (1a–1f) + the natural-history reference (`_v2_legacy/people.py:139–586`) + Starsim findings §1–§2 + the hpvsim M02 trajectory pattern. **This task adds NO transmission-shape code** (`beta_dist`/viral_load are Task 2) — `rel_trans` stays M1's flat 1.0 here so the prognosis tree can be validated structurally in isolation.

- [ ] **Step 1: Read the authorities and confirm the API**

```bash
git -C /home/cliffk/idm/covasim branch --show-current   # expect: starsim-port
sed -n '139,161p;435,586p' covasim/_v2_legacy/people.py    # set_prognoses + infect decision tree
sed -n '230,300p' covasim/parameters.py                    # get_prognoses (already conditional!) + relative_prognoses
python -c "import starsim as ss, inspect; print(inspect.signature(ss.Infection.set_prognoses)); print(inspect.signature(ss.Infection.step_die)); print(inspect.signature(ss.People.request_death))"
python -c "import starsim as ss, inspect; print(inspect.getsource(ss.SIR.set_prognoses))"   # the branch template
```

Confirm: `get_prognoses(by_age=True)` returns CONDITIONAL probs (do not re-divide); `ss.Infection.set_prognoses(self, uids, sources=None)`; `ss.Disease.step_die(self, uids)` exists to `super()` into; `request_death(self, uids)`. **[VERIFY]** whether `ss.SIR.step_die` calls `super().step_die()` (it may not — confirm whether calling super is required or optional).

- [ ] **Step 2: Write failing prognosis-tree structural tests (TDD)**

Extend `tests/test_covid.py`. These need **no v3 baseline** — they assert the trajectory machinery is internally consistent (the hpvsim M02 lifecycle-invariant pattern). Build a minimal sim (the existing `_minimal_sim` helper with a stock `ss.RandomNet`, `copy_inputs=False` so the test keeps a live module reference), run it, and assert:

```python
def test_prognosis_states_present():
    _, covid = _minimal_sim()
    for s in ['symptomatic','severe','critical','dead']:
        assert hasattr(covid, s), f'missing state {s}'
    for t in ['ti_symptomatic','ti_severe','ti_critical','ti_dead']:
        assert hasattr(covid, t), f'missing timer {t}'

def test_timer_chain_ordered():
    # For all infected at the first step, ti_infectious <= ti_symptomatic <= ti_severe
    # <= ti_critical <= ti_dead, ignoring NaN (un-scheduled) entries (the trajectory invariant).
    ...

def test_asymptomatic_and_mild_cannot_die():
    # only agents with ti_critical set ever get ti_dead; agents without ti_symptomatic
    # never get ti_severe/ti_critical/ti_dead.
    ...

def test_recovery_xor_death():
    # No agent has BOTH ti_recovered and ti_dead non-NaN (exclusive branches).
    ...

def test_step_die_resets_states():
    # Manually flip symptomatic/severe/critical on some uids, call covid.step_die(uids),
    # assert all stage flags clear and dead==True for those uids.
    ...

def test_deaths_occur_in_long_run():
    # With the full prognosis tree + a high-mortality config, cum_deaths > 0 by end.
    ...
```

Fill the `...` as you implement Steps 3–7 (run red, implement, run green). Use `(covid.ti_dead.notnan).uids` / `np.isnan(...)` to materialize the scheduled subsets. **[VERIFY]** the `FloatArr.notnan` / `np.isnan(arr.values)` access form on the installed starsim.

- [ ] **Step 3: Add the new parameters (`define_pars`, extending covid.py:45–50)**

Add the remaining durations + scalers + bed pars (spec decision 1a; defaults from `parameters.py:83–120`). Lognormals reuse M1's `ss.lognorm_ex(mean=ss.days(par1), std=ss.days(par2))` form (Open question E):

```python
self.define_pars(
    beta         = ss.probperday(0.016),                                  # M1
    init_prev    = None,                                                  # M1
    dur_exp2inf  = ss.lognorm_ex(mean=ss.days(4.5),  std=ss.days(1.5)),   # M1
    dur_asym2rec = ss.lognorm_ex(mean=ss.days(8.0),  std=ss.days(2.0)),   # M1
    # --- M2 progression durations (parameters.py:85-88) ---
    dur_inf2sym  = ss.lognorm_ex(mean=ss.days(1.1),  std=ss.days(0.9)),   # I -> symptomatic
    dur_sym2sev  = ss.lognorm_ex(mean=ss.days(6.6),  std=ss.days(4.9)),   # symptomatic -> severe
    dur_sev2crit = ss.lognorm_ex(mean=ss.days(1.5),  std=ss.days(2.0)),   # severe -> critical
    dur_crit2die = ss.lognorm_ex(mean=ss.days(10.7), std=ss.days(4.8)),   # critical -> death
    # --- M2 recovery durations (parameters.py:91-95) ---
    dur_mild2rec = ss.lognorm_ex(mean=ss.days(8.0),  std=ss.days(2.0)),
    dur_sev2rec  = ss.lognorm_ex(mean=ss.days(18.1), std=ss.days(6.3)),
    dur_crit2rec = ss.lognorm_ex(mean=ss.days(18.1), std=ss.days(6.3)),
    # --- M2 global severity scalers (parameters.py:98-101), all 1.0 ---
    rel_symp_prob=1.0, rel_severe_prob=1.0, rel_crit_prob=1.0, rel_death_prob=1.0,
    # --- M2 health-system capacity hooks (parameters.py:116-120); inert at None (Open question D) ---
    n_beds_hosp=None, n_beds_icu=None, no_hosp_factor=2.0, no_icu_factor=2.0,
    # beta_dist / viral_dist / asymp_factor are added in Task 2 (transmission-shape closure).
)
```

- [ ] **Step 4: Add the new states + timers + base-prob FloatArrs (`define_states`, extending covid.py:54–66)**

Add four BoolStates, four timers, and the four per-agent base CONDITIONAL probability FloatArrs (spec decision 1b). Keep the M1 `reset=True` block and append:

```python
ss.BoolState('symptomatic', label='Symptomatic'),
ss.BoolState('severe',      label='Severe (needs hospitalization)'),
ss.BoolState('critical',    label='Critical (needs ICU)'),
ss.BoolState('dead',        label='Dead'),
ss.FloatArr('ti_symptomatic', label='Time of symptom onset'),
ss.FloatArr('ti_severe',      label='Time of severe onset'),
ss.FloatArr('ti_critical',    label='Time of critical onset'),
ss.FloatArr('ti_dead',        label='Time of death'),
ss.FloatArr('symp_prob',   default=0.0, label='P(symptomatic | infected)'),
ss.FloatArr('severe_prob', default=0.0, label='P(severe | symptomatic)'),
ss.FloatArr('crit_prob',   default=0.0, label='P(critical | severe)'),
ss.FloatArr('death_prob',  default=0.0, label='P(death | critical)'),
```

`ti_recovered` (already present) is reused for all four recovery paths; `ti_dead` is **mutually exclusive** with `ti_recovered`. `rel_trans_base` (the `beta_dist` draw) and the viral-load kernel are added in Task 2.

  - **[VERIFY] `exposed` semantics (Open question A).** M1's `step_state` clears `exposed` at I-onset (covid.py:93). v3's `exposed` means "infected at all". Decide whether to keep M1's behavior or stop clearing `exposed`. Recommendation: **keep M1's behavior unless burden parity needs `n_exposed`** — it is not in the M2 pinned summary set. If changed, it is a behavioral change from M1; flag for Cliff at check-in 1.

- [ ] **Step 5: Declare the per-branch scratch bernoullis (in `__init__`)**

Per the hpvsim M02 pattern (Starsim findings §5a): declare a dedicated reusable `ss.bernoulli(p=0.5)` per branch decision in `__init__` (after `define_states`), so each draw sits on a stable per-Dist CRN slot. Moving these later shifts the CRN stream — declare them once and do not reorder:

```python
self._symp_bern   = ss.bernoulli(p=0.5)
self._severe_bern = ss.bernoulli(p=0.5)
self._crit_bern   = ss.bernoulli(p=0.5)
self._death_bern  = ss.bernoulli(p=0.5)
```

**[VERIFY]** that scratch `ss.bernoulli` instances declared as plain attributes are auto-registered into the module's Dist collection (the hpvsim M02 code relied on this). If not, register them explicitly per the installed starsim convention.

- [ ] **Step 6: Fill the per-agent base probabilities by age-binning (`init_post`, extending covid.py:113–139)**

In `init_post`, **before** the existing exact-count seeding, fill the four `*_prob` FloatArrs by age-binning (spec decision 1c; mirrors people.py:152–157). **Use `get_prognoses` output directly — it is already conditional (do NOT call `relative_prognoses` again):**

```python
import covasim.parameters as cvpar
progs = cvpar.get_prognoses(by_age=True)        # ALREADY conditional probs (calls relative_prognoses internally)
age = np.asarray(self.sim.people.age)
inds = np.digitize(age, progs['age_cutoffs']) - 1
self.symp_prob[:]   = progs['symp_probs'][inds]
self.severe_prob[:] = progs['severe_probs'][inds] * progs['comorbidities'][inds]  # comorbidity folds into severe
self.crit_prob[:]   = progs['crit_probs'][inds]
self.death_prob[:]  = progs['death_probs'][inds]
```

  - **[VERIFY]** `import covasim.parameters as cvpar` does not create a circular import at module load (covid.py is imported before sim.py in `__init__.py`, after parameters.py — should be safe; import lazily inside `init_post` if it isn't). Confirm `self.symp_prob[:] = arr` assigns the full FloatArr (the M1 `self.results.n_infectious[self.ti] = ...` shows index assignment works; the `[:]` whole-array form mirrors people.py:152). `trans_ORs`/`sus_ORs` are all 1.0 → `rel_sus`/`rel_trans_base` are not driven by age (parameters.py:258).

- [ ] **Step 7: Rewrite `set_prognoses` as the four-branch pre-scheduled tree (replacing covid.py:74–86)**

Replace M1's asymptomatic-only scheduling with v3's full tree (spec decision 1d; people.py:522–579). All four branch draws happen here, once per infection (CRN-safe via `uids`). Keep the M1 entry, defensively NaN the downstream timers, then branch. Reference shape (verify every call):

```python
def set_prognoses(self, uids, sources=None):
    super().set_prognoses(uids, sources)   # logs the infection
    ti = self.ti
    p = self.pars
    # --- M1 entry ---
    self.susceptible[uids] = False
    self.infected[uids]    = True
    self.exposed[uids]     = True
    self.ti_infected[uids] = ti
    self.ti_exposed[uids]  = ti
    # Defensively NaN downstream timers (no waning in M2, but keeps the trajectory clean).
    for arr in (self.ti_symptomatic, self.ti_severe, self.ti_critical, self.ti_dead, self.ti_recovered):
        arr[uids] = np.nan
    self.ti_infectious[uids] = ti + p.dur_exp2inf.rvs(uids)

    # Branch 1: symptomatic? (people.py:523-532)
    self._symp_bern.set(p=p.rel_symp_prob * self.symp_prob[uids])
    is_symp = self._symp_bern.rvs(uids)
    asymp_uids = uids[~is_symp]
    symp_uids  = uids[is_symp]
    self.ti_recovered[asymp_uids] = self.ti_infectious[asymp_uids] + p.dur_asym2rec.rvs(asymp_uids)
    if len(symp_uids) == 0:
        return
    self.ti_symptomatic[symp_uids] = self.ti_infectious[symp_uids] + p.dur_inf2sym.rvs(symp_uids)

    # Branch 2: severe? (people.py:538-547)
    self._severe_bern.set(p=p.rel_severe_prob * self.severe_prob[symp_uids])
    is_sev = self._severe_bern.rvs(symp_uids)
    mild_uids = symp_uids[~is_sev]
    sev_uids  = symp_uids[is_sev]
    self.ti_recovered[mild_uids] = self.ti_symptomatic[mild_uids] + p.dur_mild2rec.rvs(mild_uids)
    if len(sev_uids) == 0:
        return
    self.ti_severe[sev_uids] = self.ti_symptomatic[sev_uids] + p.dur_sym2sev.rvs(sev_uids)

    # Branch 3: critical? (people.py:552-560) -- bed hook inert at default n_beds_hosp=None
    hosp_factor = p.no_hosp_factor if self._hosp_max() else 1.0
    self._crit_bern.set(p=p.rel_crit_prob * self.crit_prob[sev_uids] * hosp_factor)
    is_crit = self._crit_bern.rvs(sev_uids)
    noncrit_uids = sev_uids[~is_crit]
    crit_uids    = sev_uids[is_crit]
    self.ti_recovered[noncrit_uids] = self.ti_severe[noncrit_uids] + p.dur_sev2rec.rvs(noncrit_uids)
    if len(crit_uids) == 0:
        return
    self.ti_critical[crit_uids] = self.ti_severe[crit_uids] + p.dur_sev2crit.rvs(crit_uids)

    # Branch 4: die? (people.py:565-579) -- bed hook inert at default n_beds_icu=None
    icu_factor = p.no_icu_factor if self._icu_max() else 1.0
    self._death_bern.set(p=p.rel_death_prob * self.death_prob[crit_uids] * icu_factor)
    is_dead = self._death_bern.rvs(crit_uids)
    survive_uids = crit_uids[~is_dead]
    die_uids     = crit_uids[is_dead]
    self.ti_recovered[survive_uids] = self.ti_critical[survive_uids] + p.dur_crit2rec.rvs(survive_uids)
    self.ti_dead[die_uids] = self.ti_critical[die_uids] + p.dur_crit2die.rvs(die_uids)
    # die_uids keep ti_recovered = NaN (set above) -> exclusive with ti_dead.
    return
```

  - **[VERIFY]** the boolean-mask-then-index idiom `uids[is_symp]` where `is_symp` is the bernoulli `.rvs(uids)` result. In starsim the `.rvs(uids)` of an `ss.bernoulli` returns a numpy bool array aligned to `uids`; `uids[bool_array]` gives the sub-`uids`. Confirm `ss.uids` supports boolean indexing (it should — it is array-like); the hpvsim M02 code used exactly `cin_uids = uids[cin_mask]`.
  - **[VERIFY]** `self.symp_prob[uids]` returns a plain numpy float array (per-uid) suitable as the bernoulli `p` array — the spec's `.set(p=array)` pattern (hpvsim M02 `self._cin_bern.set(p=p_cin)`).
  - **Bed hooks (Open question D):** add small helpers `_hosp_max()`/`_icu_max()` that return `False` when `n_beds_hosp`/`n_beds_icu` is `None`, else `np.count_nonzero(self.severe) >= self.pars.n_beds_hosp` / `... self.critical ... n_beds_icu` (the within-step stock read at the infection timestep; people.py reads beds on the infection day). Inert at default `None`. **[VERIFY]** the live count access (`np.count_nonzero(self.severe)` mirrors the M1 `np.count_nonzero(self.infectious)` in `update_results`).

- [ ] **Step 8: Rewrite `step_state` for the full multi-stage transitions (replacing covid.py:88–98)**

Extend M1's two transitions to the full set in v3's `update_states_pre` order (spec decision 1e; people.py:164–186): infectious → symptomatic → severe → critical → recovery → death. Each block masks `infected & (ti_<x> <= ti)` and flips the boolean. **Capture each transitioned-uid count** to feed the `new_*` flow Results (Task 3). (The `rel_trans` viral-load write from Task 2 goes at the **top** of `step_state`, before these transitions — left out here, added in Task 2.)

```python
def step_state(self):
    ti = self.ti
    # (Task 2 will write self.rel_trans here, before the transitions.)

    # E semantics per Open question A (default: keep M1 -- clear `exposed` at I-onset).
    new_infectious = (self.exposed & (self.ti_infectious <= ti)).uids
    self.exposed[new_infectious] = False
    self._new_infectious = len(new_infectious)

    to_symp = (self.infected & (self.ti_symptomatic <= ti) & ~self.symptomatic).uids
    self.symptomatic[to_symp] = True
    self._new_symptomatic = len(to_symp)

    to_sev = (self.infected & (self.ti_severe <= ti) & ~self.severe).uids
    self.severe[to_sev] = True
    self._new_severe = len(to_sev)

    to_crit = (self.infected & (self.ti_critical <= ti) & ~self.critical).uids
    self.critical[to_crit] = True
    self._new_critical = len(to_crit)

    # Recovery (clears stage flags; people.py:271-276). Death never trips this (ti_recovered NaN for diers).
    rec = (self.infected & (self.ti_recovered <= ti)).uids
    self.infected[rec]     = False
    self.recovered[rec]    = True
    self.symptomatic[rec]  = False
    self.severe[rec]       = False
    self.critical[rec]     = False
    self._new_recoveries = len(rec)

    # Death (request it; people.step_die -> cv.COVID.step_die does the flag reset).
    to_dead = (self.infected & (self.ti_dead <= ti)).uids
    self._new_deaths = len(to_dead)
    if len(to_dead):
        self.sim.people.request_death(to_dead)
    return
```

  - **[VERIFY]** the `& ~self.symptomatic` guard idiom (BoolState negation in a mask). If `~BoolState` is not supported, drop the guard — `[to_symp] = True` is idempotent anyway; the guard is only for clean `new_*` flow counting. Match the hpvsim M02 `(self.cin & ~self.cancerous & ...)` form.
  - **[VERIFY]** `self.ti` vs `self.t.ti` — be consistent with M1 (covid.py uses `self.ti`).
  - The `self._new_*` scratch attributes are read by `update_results` (Task 3); initialize them to 0 in `init_post`/`init_results` so step 0 is defined.

- [ ] **Step 9: Add `step_die` to reset the stage flags (spec decision 1f)**

```python
def step_die(self, uids):
    super().step_die(uids)   # [VERIFY] confirm ss.Disease/ss.Infection provides a base step_die to super into
    self.infected[uids]     = False
    self.exposed[uids]      = False
    self.symptomatic[uids]  = False
    self.severe[uids]       = False
    self.critical[uids]     = False
    self.recovered[uids]    = False
    self.susceptible[uids]  = False
    self.dead[uids]         = True
    return
```

`people.step_die` (loop slot 9, after transmission) calls this with the **reconciled** death set. Setting `dead=True` here matches people.py:300–308 (the agent is confirmed dead this step). **[VERIFY]** whether `super().step_die(uids)` is required (SIR overrides without super — if calling super errors, drop it).

- [ ] **Step 10: Run the structural-invariant tests**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_covid.py -v
```

Expected: all PASS. Key: timer chain ordered (`ti_infectious ≤ ti_symptomatic ≤ ti_severe ≤ ti_critical ≤ ti_dead`); only the critical branch reaches `ti_dead`; no agent has both `ti_recovered` and `ti_dead`; `step_die` clears all stage flags + sets `dead`; deaths occur in a long high-mortality run. Also confirm the M1 functional tests (E→I→R, only-infectious-transmit, permanent-immunity) still pass — M2 must not regress them.

- [ ] **Step 11: Confirm the continuous-runnability invariant still holds**

```bash
python -c "import covasim as cv; s=cv.Sim(pop_size=2000, pop_infected=20, n_days=60); s.run(); print('OK; results present:', s.results is not None)"
```

- [ ] **PAUSE FOR CLIFF (check-in 1).** Report: the extended `covasim/covid.py` (new pars/states/timers, four-branch `set_prognoses`, multi-stage `step_state`, `request_death`/`step_die`, age-prognosis fill), the new structural-invariant tests in `tests/test_covid.py` (green), and confirmation the M1 functional tests still pass + the invariant holds. **Flag:** Open question A (`exposed` semantics decision + whether it was changed), Open question D (bed hooks present-but-inert), Open question E (lognormal reuse; whether stochastic rounding was needed), and the correction that `get_prognoses` is already conditional (no double-divide). Suggested staging: `git add covasim/covid.py tests/test_covid.py`. Do NOT commit. Wait for Cliff.

---

## Task 2: Per-agent time-varying `rel_trans` = `beta_dist × viral_load(t)` — the M1-gap closure (CHECK-IN 2)

**Files:**
- Modify: `covasim/covid.py` (`rel_trans_base` state + `beta_dist`/`viral_dist`/`asymp_factor` pars; the `beta_dist` draw in `init_post`; the per-agent viral-load kernel built in `set_prognoses`; the per-step `rel_trans` write at the top of `step_state`)
- Modify: `tests/test_covid.py` (the two mean-preserving normalization unit tests)

Authority: spec decision 2 (2a–2c) + the Transmission findings (`compute_viral_load` utils.py:39–84, `n_neg_binomial` utils.py:414–432, `compute_trans_sus` utils.py:87–95) + Starsim findings §4 (writing `rel_trans` each step). **This is the precise mechanism that re-converges the transmission metrics.**

- [ ] **Step 1: Write the two failing normalization unit tests (TDD)**

Add to `tests/test_covid.py`. These confirm both factors are mean-preserving *before* integration (spec decision 5b) — a biased mean in either would itself shift the epidemic and confound the gap closure:

```python
def test_beta_dist_mean_is_one():
    # Draw N rel_trans_base samples; assert mean ~ 1.0 (tol ~0.02), std ~1.5 (overdispersion).
    # Build the draw via the same path cv.COVID uses (the ss.Dist wrapping n_neg_binomial).
    ...

def test_viral_load_kernel_timeaverage_is_one():
    # For an agent with a known infect_days_total, the per-day kernel time-average ~ 1.0;
    # high phase ~1.538, low phase ~0.769 with viral_dist defaults (frac_time=0.3, load_ratio=2).
    ...
```

- [ ] **Step 2: Add the transmission-shape pars + the `rel_trans_base` state (covid.py)**

Add to `define_pars` (spec decision 1a tail / 2):

```python
beta_dist  = dict(dist='neg_binomial', par1=1.0, par2=0.45, step=0.01),  # per-agent overdispersion (parameters.py:60)
viral_dist = dict(frac_time=0.3, load_ratio=2, high_cap=4),              # viral-load kinetics (parameters.py:61)
asymp_factor = 1.0,                                                       # asymptomatic rel. transmission (parameters.py:63)
```

Add to `define_states`:

```python
ss.FloatArr('rel_trans_base', default=1.0, label='Per-agent constant transmissibility (beta_dist draw)'),
```

- [ ] **Step 3: Draw `rel_trans_base` for all agents at init (Open question B; spec decision 2a)**

In `init_post` (alongside the age-prognosis fill from Task 1 Step 6), draw the per-agent constant overdispersion factor once for the whole population (matches v3's "drawn once per agent at init, held constant"; people.py:159). `trans_OR` is 1.0 so `rel_trans_base` is the `beta_dist` draw alone. Wrap the **active** `cv.n_neg_binomial` (utils.py:414, kept active) so the discretization matches v3 exactly:

```python
import covasim.utils as cvu
bd = self.pars.beta_dist
n  = len(self.sim.people)
# v3's exact discretized neg-binomial sampler (mean=par1=1.0, k=par2=0.45, step=0.01).
draws = cvu.n_neg_binomial(rate=bd['par1'], dispersion=bd['par2'], n=n, step=bd['step'])
self.rel_trans_base[:] = draws
```

  - **[VERIFY] / Open question B:** using `cvu.n_neg_binomial` directly draws off numpy's global RNG, which is **not** CRN-safe and not seeded off the sim's per-Dist stream. Two options, pick at code-time:
    - (i) seed numpy from the sim seed right before the draw (`np.random.seed(self.sim.pars.rand_seed)` — simple, reproducible, but global), or
    - (ii) wrap the negative-binomial in an `ss.Dist` (e.g. `ss.nbinom` — confirmed present — with a post-`rvs` `*step` discretization, or an `ss.Dist` callback that calls `cvu.n_neg_binomial`) so it participates in deterministic per-Dist seeding (the spec's recommended-but-flagged path).
    - The numeric check (Step 1) **must** confirm the realized mean ≈ 1.0 regardless of which path is chosen. If `ss.nbinom`'s parametrization differs from v3's `(rate, dispersion)` mapping (`nbn_n=dispersion`, `nbn_p=dispersion/(rate/step+dispersion)`, utils.py:430–431), reproduce that mapping explicitly or wrap `cvu.n_neg_binomial`. Flag the chosen path for Cliff.

- [ ] **Step 4: Build the per-agent viral-load kernel at infection (Open question C; spec decision 2b)**

In `set_prognoses` (Task 1 Step 7), **after** `ti_infectious` and the recovery/death date are known for each branch, precompute a small per-agent kernel via the closed form (avoids per-step whole-population recompute; the hpvsim cached-kernel optimization). `compute_viral_load` (utils.py:39–84) is a two-level mean-preserving step function. With `viral_dist` defaults: normalizer `Z = 1 + frac_time*(load_ratio−1) = 1.3`; HIGH `= load_ratio/Z = 1.5385`; LOW `= 1/Z = 0.7692`; switch at `trans_point = min(frac_time, high_cap/infect_days_total)`; 0 outside `[0, infect_days_total)`.

Store one of:
- (a) a 2D per-agent kernel array `self.viral_load_kernel` (rows = uids, cols = day-since-infectious up to a max length = longest `infect_days_total` in the population), or
- (b) the three scalars per agent (`infect_days_total`, `trans_point*infect_days_total` switch day) and reconstruct the two-level value each step (cheaper memory).

**Recommendation: (b)** store per-agent `ti_recovered_or_dead` (already on the agent as `ti_recovered`/`ti_dead`) + nothing extra, and compute the two-level value inline each step (it is a couple of vector ops). The closed-form per-agent value on day `d = ti − ti_infectious`:

```python
# infect_days_total = (ti_dead if set else ti_recovered) - ti_infectious   (people.py:59-65)
# trans_point_days  = min(frac_time, high_cap/infect_days_total) * infect_days_total
# early = d < trans_point_days
# vl = (HIGH if early else LOW), 0 if d<0 or d>=infect_days_total
```

  - **[VERIFY] / Open question C:** confirm the kernel length / clip for the longest infections (`high_cap=4` caps the high phase at 4 days). The kernel must be defined for the whole infectious window including the day of recovery/death. Reproduce `compute_viral_load`'s `time_stop = time_dead if dead else time_recovered` (utils.py:59–62) and its `invalid` zeroing (utils.py:80–82) exactly.

- [ ] **Step 5: Write `self.rel_trans` each step at the top of `step_state` (spec decision 2c)**

At the **top** of `step_state` (before the stage transitions from Task 1 Step 8; `step_state` runs at loop slot ~4, before `diseases.step()`/`infect()`, so the write modulates this step's transmission — Starsim findings §0, §4):

```python
ti = self.ti
inf = self.infectious.uids                          # currently-infectious agents (the property)
if len(inf):
    days_since = (ti - self.ti_infectious[inf]).astype(int)
    vl = self._viral_load(inf, days_since)          # the closed-form two-level value (Step 4)
    f_asymp = np.where(self.symptomatic[inf], 1.0, self.pars.asymp_factor)  # default 1.0 -> neutral
    self.rel_trans[inf] = self.rel_trans_base[inf] * vl * f_asymp
```

Stock `ss.Infection.infect()` then gates `rel_trans` by `infectious` and computes per edge `p = rel_trans[src] × rel_sus[trg] × beta_per_dt` (diseases.py:228–290) — so the per-step write directly modulates this step's per-contact transmission. **No custom transmission code.**

  - **[VERIFY]** `self.infectious.uids` works (the `infectious` property returns `infected & (ti_infectious <= ti)`, a BoolArr-like; confirm `.uids` on the property result, else materialize `(self.infected & (self.ti_infectious <= self.ti)).uids` as in covid.py:92).
  - **[VERIFY]** `self.symptomatic[inf]` returns a per-`inf` numpy bool array for the `np.where`.

- [ ] **Step 6: Run the normalization tests + the prognosis-tree tests + the invariant**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_covid.py -v
python -c "import covasim as cv; s=cv.Sim(pop_size=2000, pop_infected=20, n_days=60); s.run(); print('OK; results:', s.results is not None)"
```

Expected: the two normalization tests pass (mean ≈ 1.0 for both factors); the prognosis-tree structural tests still pass; the sim runs. Sanity-check that the epidemic is now **larger** than M1's (the front-loading + overdispersion effect) by comparing peak `n_infectious` before/after Task 2 on a fixed seed (informational).

- [ ] **PAUSE FOR CLIFF (check-in 2).** Report: the `beta_dist`/`viral_dist`/`asymp_factor` pars + `rel_trans_base` state, the init-time `beta_dist` draw (with the chosen CRN path from Step 3 [VERIFY]), the per-agent viral-load kernel (storage choice from Step 4), and the per-step `rel_trans` write; the two mean-preserving normalization tests (green); and the observed before/after epidemic-size change on a fixed seed. **Flag:** Open question B (draw site + sampler/CRN path) and Open question C (kernel storage). Suggested staging: `git add covasim/covid.py tests/test_covid.py`. Do NOT commit. Wait for Cliff.

---

## Task 3: `pop_scale`/`total_pop` on `cv.Sim` + the burden Results (CHECK-IN 3)

**Files:**
- Modify: `covasim/sim.py` (add `pop_scale=None`/`total_pop=None` to `__init__`; forward to `ss.Sim`)
- Modify: `covasim/covid.py` (the burden Results: `new_*`/`cum_*`/`n_*` in `init_results`/`update_results`/`finalize_results`)
- Modify: `tests/test_sim.py` (scaling test)

Authority: spec decision 3 (scaling) + decision 5a (Results) + Starsim findings §3 (`SimPars.validate_total_pop`, `scale=True` auto-multiply at finalize) + §5 (results).

- [ ] **Step 1: Add the burden Results to `cv.COVID` (extending covid.py:100–111)**

Extend `init_results` with the new flow + stock + cumulative Results (spec decision 5a; all `scale=True` since they are agent counts subject to `pop_scale`). Mirror M1's `n_infectious` pattern (covid.py:103–104). **[VERIFY]** the `ss.Result(name, dtype=, scale=, label=)` signature against the installed starsim (M1 used `ss.Result('n_infectious', dtype=int, scale=True, label=...)`).

```python
def init_results(self):
    super().init_results()
    self.define_results(
        # Stocks (count of agents with the flag true on that day):
        ss.Result('n_infectious',  dtype=int, scale=True, label='Number infectious'),   # M1
        ss.Result('n_symptomatic', dtype=int, scale=True, label='Number symptomatic'),
        ss.Result('n_severe',      dtype=int, scale=True, label='Number severe'),
        ss.Result('n_critical',    dtype=int, scale=True, label='Number critical'),
        ss.Result('n_dead',        dtype=int, scale=True, label='Number dead (cumulative)'),
        ss.Result('n_recovered',   dtype=int, scale=True, label='Number recovered'),
        # New (daily) flows (written in step_state from the per-stage transitioned-uid counts):
        ss.Result('new_symptomatic', dtype=int, scale=True, label='New symptomatic'),
        ss.Result('new_severe',      dtype=int, scale=True, label='New severe'),
        ss.Result('new_critical',    dtype=int, scale=True, label='New critical'),
        ss.Result('new_recoveries',  dtype=int, scale=True, label='New recoveries'),
        ss.Result('new_deaths',      dtype=int, scale=True, label='New deaths'),
        # Cumulative flows (filled in finalize_results via np.cumsum):
        ss.Result('cum_symptomatic', dtype=int, scale=True, label='Cumulative symptomatic'),
        ss.Result('cum_severe',      dtype=int, scale=True, label='Cumulative severe'),
        ss.Result('cum_critical',    dtype=int, scale=True, label='Cumulative critical'),
        ss.Result('cum_recoveries',  dtype=int, scale=True, label='Cumulative recoveries'),
        ss.Result('cum_deaths',      dtype=int, scale=True, label='Cumulative deaths'),
    )
    return
```

  - **Note:** `cum_infections`/`new_infections`/`prevalence` are provided by the stock `ss.Infection.init_results` (diseases.py:169–178) — do not redefine them. M2 adds only the burden Results.

- [ ] **Step 2: Write the flows + stocks in `update_results`; cumulatives in `finalize_results`**

`update_results` writes the daily flows (from the `self._new_*` scratch counts captured in `step_state`, Task 1 Step 8) and the stocks (live `np.count_nonzero`):

```python
def update_results(self):
    super().update_results()
    ti = self.ti
    r = self.results
    # Stocks (the infectious-property pattern M1 established):
    r.n_infectious[ti]  = int(np.count_nonzero(self.infectious))
    r.n_symptomatic[ti] = int(np.count_nonzero(self.symptomatic))
    r.n_severe[ti]      = int(np.count_nonzero(self.severe))
    r.n_critical[ti]    = int(np.count_nonzero(self.critical))
    r.n_dead[ti]        = int(np.count_nonzero(self.dead))
    r.n_recovered[ti]   = int(np.count_nonzero(self.recovered))
    # Flows (captured in step_state; default 0 at step 0):
    r.new_symptomatic[ti] = getattr(self, '_new_symptomatic', 0)
    r.new_severe[ti]      = getattr(self, '_new_severe', 0)
    r.new_critical[ti]    = getattr(self, '_new_critical', 0)
    r.new_recoveries[ti]  = getattr(self, '_new_recoveries', 0)
    r.new_deaths[ti]      = getattr(self, '_new_deaths', 0)
    return

def finalize_results(self):
    super().finalize_results()
    r = self.results
    r.cum_symptomatic[:] = np.cumsum(r.new_symptomatic)
    r.cum_severe[:]      = np.cumsum(r.new_severe)
    r.cum_critical[:]    = np.cumsum(r.new_critical)
    r.cum_recoveries[:]  = np.cumsum(r.new_recoveries)
    r.cum_deaths[:]      = np.cumsum(r.new_deaths)
    return
```

  - **[VERIFY]** `finalize_results` is the right hook for cumsum (hpvsim M02 used it; diseases.py:406–427) and that scaling (`× pop_scale`) is applied **after** `finalize_results` so the cumsum is over raw counts (Starsim findings §3 — scaling happens once at `Module.finalize_results` / `Sim.finalize_results`; store unscaled counts during the run). If the order causes double-scaling, compute cumsum on the raw `new_*` before any scale multiply. **[VERIFY]** that defining `super().finalize_results()` does not already finalize before cumsum — call `super()` first, then cumsum.
  - **[VERIFY]** `self.ti_dead`-driven `n_dead` count: `dead` is set in `step_die` (loop slot 9) which runs after `update_results`? Check the loop order (Starsim findings §0: `update_results` is slot 10, `step_die` is slot 9) — `step_die` precedes `update_results`, so `n_dead` on the death day is correct. Confirm.

- [ ] **Step 3: Add `pop_scale`/`total_pop` to `cv.Sim.__init__` and forward (spec decision 3)**

Extend `cv.Sim.__init__` (sim.py:44) to accept `pop_scale=None`/`total_pop=None` and forward them to `ss.Sim`. Starsim computes `pop_scale = total_pop / n_agents` and auto-multiplies every `scale=True` Result at finalize (`validate_total_pop`, parameters.py:322; Starsim findings §3). Set **exactly one** of `total_pop`/`pop_scale` (Starsim raises if both):

```python
def __init__(self, pars=None, people=None, pop_size=20_000, pop_infected=20,
             pop_type='random', n_days=60, start_day='2020-03-01', rand_seed=1,
             beta=None, pop_scale=None, total_pop=None, **kwargs):
    ...
    super().__init__(pars=pars, people=people, networks=networks, diseases=diseases,
                     start=ss.date(start_day), dur=ss.days(n_days), dt=ss.days(1),
                     rand_seed=rand_seed, pop_scale=pop_scale, total_pop=total_pop, **kwargs)
```

  - **[VERIFY] / Open question F:** confirm `ss.Sim`/`SimPars` accepts `pop_scale`/`total_pop` via `**kwargs` (Starsim findings §3 says they are `SimPars` fields). Confirm the `pop_size → n_agents` mapping (M1's `cv.People(pop_size)` sets `n_agents`; `pop_size` is not a `SimPars` field). If passing both `pop_scale` and `total_pop` (both non-None) is requested, raise a clear error before forwarding (Starsim raises anyway). If only one of the two is ever set in v3 terms, default both to `None` and forward — Starsim then uses `pop_scale=1` (no scaling), preserving M1 behavior.

- [ ] **Step 4: Write the scaling test (`tests/test_sim.py`)**

```python
def test_pop_scale_multiplies_extensive_results():
    # Same seed + n_agents, one with pop_scale=1 and one with pop_scale=10.
    # Extensive (count) results -- cum_infections, cum_deaths, n_infectious peak -- scale by ~10x.
    # Intensive results -- prevalence -- are UNCHANGED (scale=False).
    ...

def test_total_pop_sets_pop_scale():
    # cv.Sim(pop_size=2000, total_pop=20000) -> pop_scale == 10 (Starsim computes it).
    ...
```

  - **[VERIFY]** which results object holds the scaled values (disease results vs sim results) and the key access (`build_summary_m1` reads `disease.results['prevalence']` and `disease.infected.sum()`). Confirm `prevalence` is `scale=False` (Starsim findings §5; `Infection.init_results` defines it `scale=False`).

- [ ] **Step 5: Run the sim + scaling tests**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_sim.py test_covid.py -v
python -c "import covasim as cv; s=cv.Sim(pop_size=2000, pop_infected=20, n_days=60); s.run(); d=list(s.diseases.values())[0]; print('cum_deaths key present:', 'cum_deaths' in d.results)"
```

Expected: scaling tests pass; the burden Results exist and are populated; the M1 + prognosis-tree tests still pass; the invariant holds.

- [ ] **PAUSE FOR CLIFF (check-in 3).** Report: the burden Results (`new_*`/`cum_*`/`n_*`) in `cv.COVID`, the `pop_scale`/`total_pop` forwarding on `cv.Sim`, and the scaling test (green). **Flag:** Open question F (the `pop_scale`/`total_pop` plumbing + the `pop_size → n_agents` mapping + the "exactly one of the two" constraint). Suggested staging: `git add covasim/covid.py covasim/sim.py tests/test_sim.py`. Do NOT commit. Wait for Cliff.

---

## Task 4: M2 anchor + short-summary + parity gate + v3.1.8 baseline (CHECK-IN 4)

**Files:**
- Create: `tests/regression/anchor_m2.py`
- Modify: `tests/regression/short_summary.py` (`build_summary_m2` + `METRIC_KEYS_M2`)
- Modify: `tests/regression/multi_seed_v3.py` (add `m2_random`/`m2_hybrid` to `_anchor_runner`)
- Modify: `tests/regression/compare.py` (add `m2_random`/`m2_hybrid` to `_resolve_run`)
- Modify: `tests/regression/README.md` (M2 anchor + worktree workflow)
- Create: `tests/test_m2_parity.py` (slow z-score gate, both backends)
- Modify: `.gitignore` (M2 baseline patterns)
- Generate (gitignored): the v3.1.8 M2 baseline via the worktree

Authority: spec "M2 anchor scenario + pinned metrics" + "Acceptance test" + "v3.1.8-baseline-via-worktree workflow". Reuse the harness (`parity.py` unchanged; `multi_seed_v3.py`/`compare.py` extended exactly as M1 extended them for `m1_*`).

- [ ] **Step 1: Create `tests/regression/anchor_m2.py` (dual-version, mirroring `anchor_m1.py`)**

Single-variant, full natural history, no interventions, no waning. Mirror `anchor_m1.py`'s `_IS_V4 = hasattr(cv, 'COVID')` dual-version structure, but **do NOT zero the prognoses** under v3 (the whole point of M2 is the full tree). Pinned pars from the spec: `pop_size=20_000`, `pop_infected=20`, `n_days=120` (long enough to accumulate severe/critical/deaths past the peak).

```python
"""M2 anchor: single-variant FULL natural-history sim (random + hybrid backends).

Exercises the M2 capability -- the full prognosis tree (symptomatic/severe/critical/
dead) + the viral_load x beta_dist transmissibility closure. Runs under BOTH v3.1.8
(to generate the gitignored baseline, from a frozen v3.1.8 worktree) and v4 (the
port), selected by duck-typing on cv.COVID. Unlike anchor_m1, the v3 branch keeps
the default age-based prognoses (the full tree is the point) -- it only forces
use_waning=False + n_variants=1 to isolate single-variant non-waning dynamics.

Run as a script:  python tests/regression/anchor_m2.py
"""
import sys
from pathlib import Path
import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary_m2  # noqa: E402

_IS_V4 = hasattr(cv, 'COVID')
POP_SIZE     = 20_000
POP_INFECTED = 20
N_DAYS       = 120

def make_sim(pop_type='random', rand_seed=0, **kwargs):
    if _IS_V4:
        pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type,
                    n_days=N_DAYS, rand_seed=rand_seed, verbose=0)
        return cv.Sim(**sc.mergedicts(pars, kwargs))
    # v3.1.8: single-variant, non-waning, full prognosis tree (defaults).
    pars = dict(pop_size=POP_SIZE, pop_infected=POP_INFECTED, pop_type=pop_type,
                n_days=N_DAYS, rand_seed=rand_seed, n_variants=1,
                use_waning=False, verbose=0)
    return cv.Sim(pars=sc.mergedicts(pars, kwargs))

def run_and_summarize(pop_type='random', rand_seed=0, **kwargs):
    sim = make_sim(pop_type=pop_type, rand_seed=rand_seed, **kwargs)
    sim.run()
    return build_summary_m2(sim)

if __name__ == '__main__':
    for pt in ('random', 'hybrid'):
        print(f'M2 anchor short summary ({pt}; covasim {cv.__version__}, v4={_IS_V4}):')
        for k, v in run_and_summarize(pop_type=pt).items():
            print(f'  {k:<24} {v:>14.4g}')
```

  - **[VERIFY in the v3.1.8 worktree]** that `cv.Sim(pars=dict(..., use_waning=False, n_variants=1))` with default prognoses produces the full symptomatic/severe/critical/death tree (it should — these are v3's defaults). Confirm `use_waning=False` does not zero the burden.

- [ ] **Step 2: Add `build_summary_m2` + `METRIC_KEYS_M2` to `short_summary.py`**

Add the M2 summary builder. **Two metric families, both gated** (spec): the new burden metrics + the re-converged M1 transmission metrics. Dual-version (v3.1.8 `sim.summary`/`sim.results` vs v4 disease results), mirroring `build_summary_m1`'s `hasattr(sim, 'diseases')` branch:

```python
METRIC_KEYS_M2 = (
    # Re-converged transmission metrics (the M1-gap-closure gate):
    'cum_infections', 'peak_prevalence', 'peak_n_infectious',
    # Burden metrics (the M2-specific gate):
    'cum_symptomatic', 'cum_severe', 'cum_critical', 'cum_deaths',
)

def build_summary_m2(sim):
    """M2 short summary: burden + re-converged transmission metrics, dual-version."""
    if hasattr(sim, 'diseases'):  # v4
        disease = list(sim.diseases.values())[0]
        res = disease.results
        cum_infections = float(int(disease.infected.sum()) + int(disease.recovered.sum()) + int(disease.dead.sum()))
        out = {
            'cum_infections':    cum_infections,
            'peak_prevalence':   float(np.asarray(res['prevalence']).max()),
            'peak_n_infectious': float(np.asarray(res['n_infectious']).max()),
            'cum_symptomatic':   float(np.asarray(res['cum_symptomatic'])[-1]),
            'cum_severe':        float(np.asarray(res['cum_severe'])[-1]),
            'cum_critical':      float(np.asarray(res['cum_critical'])[-1]),
            'cum_deaths':        float(np.asarray(res['cum_deaths'])[-1]),
        }
    else:  # v3.1.8
        summary = sim.summary
        out = {
            'cum_infections':    float(summary['cum_infections']),
            'peak_prevalence':   float(_series_max(sim, 'prevalence')),
            'peak_n_infectious': float(_series_max(sim, 'n_infectious')),
            'cum_symptomatic':   float(summary['cum_symptomatic']),
            'cum_severe':        float(summary['cum_severe']),
            'cum_critical':      float(summary['cum_critical']),
            'cum_deaths':        float(summary['cum_deaths']),
        }
    return out
```

  - **[VERIFY — critical for cross-version parity]** the v4 burden-cumulative access. With `pop_scale` applied at finalize, `cum_deaths` etc. are scaled — but the anchor uses default `pop_scale=1`, so unscaled. Confirm the v4 `cum_*` are the **scaled** end-of-run values matching v3's `summary['cum_*']` (also scaled by v3's `pop_scale=1`). The `cum_infections` v4 definition adds `dead` to the M1 (infected+recovered) sum since M2 has deaths — confirm this matches v3's `summary['cum_infections']` (which counts the seed + all ever-infected). **[VERIFY]** whether v4 has a native `cum_infections` result (from `ss.Infection.init_results`) that is cleaner to read than reconstructing from states — prefer the native result if it counts the seed correctly.

- [ ] **Step 3: Extend `multi_seed_v3.py` and `compare.py` with the M2 anchor**

In `multi_seed_v3.py`, add to `_anchor_runner` (mirroring the `m1_*` branch):

```python
if anchor.startswith('m2_'):
    pop_type = anchor.split('_', 1)[1]
    if pop_type not in ('random', 'hybrid'):
        raise ValueError(f"Unknown M2 anchor {anchor!r}; use m2_random or m2_hybrid.")
    return (lambda seed: _run_seed_m2(seed, pop_type), f'v3_m2_{pop_type}_seeds_n{{n}}.json')
```

with a `_run_seed_m2(seed, pop_type)` importing `from anchor_m2 import make_sim` + `from short_summary import build_summary_m2`. In `compare.py`, add `m2_*` to `_resolve_run` (importing `anchor_m2.run_and_summarize`). Keep the existing `m0`/`m1_*` behavior intact. **[VERIFY]** `_run_seed_m2` mirrors `_run_seed_m1` (anchor_m2 returns the same dict shape).

- [ ] **Step 4: Add the M2 gitignore patterns**

Append to `.gitignore` (after the M1 block at lines 150–154):

```
# M2 single-variant full-natural-history regression baselines (regenerate from a v3.1.8 worktree)
tests/regression/v3_m2_*seeds*.json
tests/regression/v4_m2_*seeds*.json
tests/regression/anchor_m2_snapshot.json
```

- [ ] **Step 5: Create the slow z-score parity gate `tests/test_m2_parity.py`**

Mirror `tests/test_m1_parity.py`, retargeted to `anchor_m2` + `build_summary_m2`, both backends. The gate covers **both** metric families (spec acceptance items 4 + 5):

```python
"""M2 acceptance gate: multi-seed parity vs the v3.1.8 M2 baseline.

Gates BOTH the new burden metrics (cum_symptomatic/cum_severe/cum_critical/cum_deaths)
AND the re-converged transmission metrics (cum_infections/peak_prevalence/peak_n_infectious)
on |z| < 3 vs the gitignored v3.1.8 baseline. Skips cleanly when the baseline is absent
(generated from a frozen v3.1.8 worktree -- see tests/regression/README.md). Marked slow.

    cd tests && pytest test_m2_parity.py -m slow -v
"""
import json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent))
from regression.anchor_m2 import make_sim                 # noqa: E402
from regression.short_summary import build_summary_m2     # noqa: E402
from regression.parity import parity_gate                 # noqa: E402

N_V4_SEEDS = 10
M_V3_SEEDS = 30
Z_THRESHOLD = 3.0

def _baseline_path(pop_type):
    return Path(__file__).parent / 'regression' / f'v3_m2_{pop_type}_seeds_n{M_V3_SEEDS}.json'

def _run_v4_seeds(pop_type, n):
    rows = []
    for seed in range(n):
        sim = make_sim(pop_type=pop_type, rand_seed=seed); sim.run()
        rows.append(build_summary_m2(sim))
    return rows

@pytest.mark.slow
@pytest.mark.parametrize('pop_type', ['random', 'hybrid'])
def test_m2_anchor_parity(pop_type):
    baseline = _baseline_path(pop_type)
    if not baseline.exists():
        pytest.skip(
            f'Missing v3.1.8 M2 baseline at {baseline}. Regenerate via '
            f'`cd /tmp/cov-v3 && python <repo>/tests/regression/multi_seed_v3.py '
            f'--anchor m2_{pop_type} --n {M_V3_SEEDS}` from a frozen v3.1.8 worktree.'
        )
    v3_rows = json.loads(baseline.read_text())
    v4_rows = _run_v4_seeds(pop_type, N_V4_SEEDS)
    failures = parity_gate(v4_rows, v3_rows, z_threshold=Z_THRESHOLD)
    if failures:
        details = '\n'.join(f'  {name:<22} z={z:+.2f}' for name, z in failures)
        pytest.fail(
            f'M2 {pop_type} parity drift exceeds |z|>={Z_THRESHOLD} on {len(failures)} '
            f'metrics (v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):\n{details}'
        )
    return v4_rows
```

  - **[VERIFY]** the `slow` marker is registered (M0/M1 registered it in `tests/pytest.ini`).

- [ ] **Step 6: Generate the v3.1.8 M2 baseline via the git worktree**

The current env's editable `covasim` is **v4**, so it cannot generate v3.1.8 baselines directly. Create a frozen v3.1.8 worktree whose cwd shadows the editable v4 install:

```bash
# One-time: add the worktree at the pre-port baseline commit (main is v3.1.8-equivalent).
git -C /home/cliffk/idm/covasim worktree add /tmp/cov-v3 main
# Confirm the worktree imports v3.1.8 (cwd shadows the editable v4 on sys.path):
cd /tmp/cov-v3 && python -c "import covasim as cv; print('worktree covasim:', cv.__version__, 'has COVID:', hasattr(cv,'COVID'))"   # expect 3.1.8, COVID False
# Generate both backends' 30-seed M2 baselines, writing into the WORKING tree's regression dir:
cd /tmp/cov-v3 && python /home/cliffk/idm/covasim/tests/regression/multi_seed_v3.py --anchor m2_random --n 30 \
    --out /home/cliffk/idm/covasim/tests/regression/v3_m2_random_seeds_n30.json
cd /tmp/cov-v3 && python /home/cliffk/idm/covasim/tests/regression/multi_seed_v3.py --anchor m2_hybrid --n 30 \
    --out /home/cliffk/idm/covasim/tests/regression/v3_m2_hybrid_seeds_n30.json
```

  - **[VERIFY]** the worktree `import covasim` resolves to the v3.1.8 source under `/tmp/cov-v3` (not the editable v4). Running `multi_seed_v3.py` from `cwd=/tmp/cov-v3` puts the worktree's `covasim/` first on `sys.path`. If the editable v4 install still shadows it (egg-link precedence), prepend the worktree to `PYTHONPATH` (`PYTHONPATH=/tmp/cov-v3 python ...`) or run from a v3.1.8 venv. Confirm `cv.__version__ == '3.1.8'` before trusting the output.
  - The `multi_seed_v3.py` script imports `anchor_m2` from its **own** directory (`/home/cliffk/idm/covasim/tests/regression/`, via the `sys.path.insert`), which contains the v4-aware dual-version anchor — `anchor_m2._IS_V4` will be `False` under the worktree's v3.1.8 covasim, so it takes the v3 branch. Confirm this resolves correctly (the script path is the v4 working tree; only `import covasim` comes from the worktree).
  - **[VERIFY]** the M2 anchor at `n_days=120`, `pop_size=20_000`, 30 seeds × 2 backends is not prohibitively slow under v3.1.8 (it was fine for M1 at `n_days=60`). If too slow, reduce `M_V3_SEEDS` (and update the filename) — flag for Cliff.
  - When done, the worktree can stay (for M3+) or be removed: `git -C /home/cliffk/idm/covasim worktree remove /tmp/cov-v3` (do this only if Cliff agrees; it is cheap to keep).

- [ ] **Step 7: Run the M2 gates + the demo plot + the full fast suite under strict warnings**

```bash
# Print the anchor summary (sanity): expect non-zero cum_severe/cum_critical/cum_deaths.
cd /home/cliffk/idm/covasim && python tests/regression/anchor_m2.py
# The slow parity gate (runs if the baseline exists; else skips):
cd /home/cliffk/idm/covasim/tests && pytest test_m2_parity.py -m slow -v
# The development drift gate (informational; needs a v3.1.8 snapshot via the worktree):
cd /tmp/cov-v3 && python /home/cliffk/idm/covasim/tests/regression/compare.py --anchor m2_random --save-snapshot \
    --baseline /home/cliffk/idm/covasim/tests/regression/anchor_m2_snapshot.json
cd /home/cliffk/idm/covasim && python tests/regression/compare.py --anchor m2_random \
    --baseline tests/regression/anchor_m2_snapshot.json
# Demo plot (acceptance item 6): health-outcome curves + re-converged infection curve.
python -c "import matplotlib; matplotlib.use('agg'); import covasim as cv; s=cv.Sim(pop_size=10000, pop_infected=50, n_days=120); s.run(); s.plot(); import matplotlib.pyplot as plt; plt.savefig('/tmp/m2_demo.png'); print('saved /tmp/m2_demo.png')"
# Full non-quarantined suite under the strict-warnings bar (matches run_tests):
cd /home/cliffk/idm/covasim/tests && COVASIM_INTERACTIVE=0 COVASIM_WARNINGS=error pytest test_*.py -n auto --durations=0 2>&1 | tail -30
```

Expected: the anchor prints non-zero burden metrics; `test_m2_parity` runs (if baseline present) and reports z-scores, or skips cleanly; the demo plot renders; the full root suite passes under `COVASIM_WARNINGS=error` (quarantine not collected; `test_baselines`/`test_regression` carry the documented v4-skips; `test_m1_parity`/`test_m2_parity` skip if baselines absent). **Confirm `test_m1_parity` still passes/skips — M2 must not regress the M1 transmission gate; if M1 now runs (baseline present), the transmission metrics should be ≥ M1's (the closure made the epidemic bigger), which is the intended re-convergence.**

- [ ] **Step 8: Update `tests/regression/README.md`**

Add an M2 section documenting: the M2 anchor (single-variant, full natural history, `n_days=120`), the `--anchor m2_random`/`m2_hybrid` sweep options, the **git-worktree** baseline workflow (the `git worktree add /tmp/cov-v3 main` + `cwd=/tmp/cov-v3` shadowing trick), and the two gated metric families (burden + re-converged transmission). Mirror the existing M1 README section's style.

- [ ] **PAUSE FOR CLIFF (check-in 4).** Report: `anchor_m2.py`, the `build_summary_m2`/`METRIC_KEYS_M2` additions, the `multi_seed_v3.py`/`compare.py` `m2_*` additions, `test_m2_parity.py`, the README + gitignore edits, the demo plot, the full-suite results under strict warnings, and **the observed parity z-scores** (or drift) for both metric families vs v3.1.8 — especially whether the transmission metrics re-converged to `|z| < 3` (the explicit M1-acceptance condition) and whether the burden metrics pass. **Flag:** any residual transmission gap (Open question G — whether to loosen to `|z| < 5` with a documented rationale) and any cross-version `cum_*` access mismatch. Suggested staging: `git add covasim/ tests/regression/anchor_m2.py tests/regression/short_summary.py tests/regression/multi_seed_v3.py tests/regression/compare.py tests/regression/README.md tests/test_m2_parity.py tests/test_covid.py tests/test_sim.py .gitignore`. Do NOT commit. Wait for Cliff.

---

## Task 5: End-to-end verification (no commits)

A manual verification pass against the spec's acceptance test. Contains no staging; the assistant never commits.

- [ ] **Step 1: Confirm branch + working tree + the worktree**

```bash
git -C /home/cliffk/idm/covasim branch --show-current   # expect starsim-port
git -C /home/cliffk/idm/covasim status --short
git -C /home/cliffk/idm/covasim worktree list           # /tmp/cov-v3 present if kept
```

- [ ] **Step 2: Continuous-runnability invariant (acceptance item 1)**

```bash
python -c "import covasim as cv; s=cv.Sim(); s.run(); print('cv.Sim().run() OK, results:', s.results is not None)"
python -c "import covasim as cv; cv.Sim(pop_type='hybrid', pop_size=2000, n_days=120).run(); print('hybrid OK')"
```

- [ ] **Step 3: Full non-quarantined suite under strict warnings (acceptance item 2)**

```bash
cd /home/cliffk/idm/covasim/tests && ./run_tests 2>&1 | tail -30
```

Expected: green under `COVASIM_WARNINGS=error`/`COVASIM_INTERACTIVE=0` — the M1 + M2 functional tests, the harness tests, the skipped slow gates, the documented v4-skips; quarantine not collected; the M1 anchor gate not regressed.

- [ ] **Step 4: Prognosis-tree structural invariants + burden parity + M1-gap closure (acceptance items 3, 4, 5) — requires the v3.1.8 baselines**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_covid.py -v                       # structural invariants
cd /home/cliffk/idm/covasim/tests && pytest test_m2_parity.py -m slow -v           # burden + transmission parity
cd /home/cliffk/idm/covasim/tests && pytest test_m1_parity.py -m slow -v           # M1 gate not regressed
```

Expected: structural invariants green; burden metrics + re-converged transmission metrics within `|z| < 3` (both backends), or skip cleanly if no v3.1.8 baseline — record which. If a residual transmission gap persists after both factors are verified mean-preserving (Step 5b checks in Task 2), document it with the front-loading/overdispersion diagnostics and escalate per Open question G.

- [ ] **Step 5: Demo plot (acceptance item 6)**

Confirm `/tmp/m2_demo.png` renders the health-outcome curves + the re-converged infection curve.

- [ ] **Step 6: Confirm `.gitignore` excludes the generated M2 baselines**

```bash
git -C /home/cliffk/idm/covasim check-ignore tests/regression/v3_m2_random_seeds_n30.json tests/regression/anchor_m2_snapshot.json
```

- [ ] **Step 7: Final report to Cliff**

Summarize: the four check-in boundaries and what was staged at each; the prognosis-tree additions to `cv.COVID` (states/timers/pars/branches/`step_die`); the transmission-shape closure (`beta_dist` draw path + viral-load kernel); the `pop_scale`/`total_pop` plumbing; the burden Results; how each Open question A–G was resolved; test results (structural invariants + full suite + the two parity-metric families with z-scores); **whether the M1 transmission metrics re-converged to `|z| < 3`** (the explicit M1-acceptance condition); any escalations (CRN path for `beta_dist`, residual transmission gap, cross-version access, strict-warnings issues); and confirmation that the assistant committed nothing. **Milestone completion is Cliff's call: acceptance test green locally AND Cliff has reviewed and committed.**

---

## (Conditional) Task 6: Tolerance / residual-gap pass (CHECK-IN 5, only if needed)

If, after Task 2's factors are verified mean-preserving and Task 4's gate is run, the transmission metrics do **not** re-converge to `|z| < 3` (or a burden metric fails), do a focused residual pass before involving Cliff on a tolerance change:
- **(a)** Confirm the viral-load high phase is placed **early** (front-loading; matters for susceptible depletion) — check the kernel's `early` boolean against `compute_viral_load` utils.py:76.
- **(b)** Confirm the overdispersion **variance** is preserved (not accidentally averaged across contacts) — the per-agent `rel_trans_base` must vary agent-to-agent with std ≈ 1.5, not collapse to its mean.
- **(c)** Confirm the bernoulli branch CRN slots did not shift (declaring/reordering the scratch bernoullis in Task 1 Step 5 changes the per-Dist CRN identifier — hpvsim M02 finding).
- **(d)** Confirm `get_prognoses` was used **once** (not double-divided — the plan's correction to spec 1c).

Only after these are ruled out, propose to Cliff a documented per-metric loosening to `|z| < 5` (the MIGRATION_PLAN.md escape hatch) with a written residual rationale. **PAUSE FOR CLIFF (check-in 5).** Do NOT commit.

---

## Self-review checklist

After all tasks, verify against `MIGRATION_PLAN.md` §M2 and the spec:

| Requirement | Implementing task |
|---|---|
| Work on `starsim-port`; assistant never branches/commits/pushes; 4 PAUSE-FOR-CLIFF check-ins (5th conditional) | Tasks 1–4 boundaries; Task 6 conditional |
| Continuous-runnability invariant (`cv.Sim().run()` returns results) at every commit | Task 1 Step 11; Task 2 Step 6; Task 3 Step 5; Task 5 Step 2 |
| M2 EXTENDS `cv.COVID`/`cv.Sim` — no new public class, nothing quarantined/removed | Tasks 1–3 |
| Full prognosis tree: symptomatic/severe/critical/dead states + timers + base-prob FloatArrs | Task 1 Steps 4, 6 |
| New pars: dur_inf2sym/sym2sev/sev2crit/crit2die/mild2rec/sev2rec/crit2rec, rel_*_prob, n_beds_*/no_*_factor | Task 1 Step 3 |
| Four-branch pre-scheduled `set_prognoses` (symptomatic? severe? critical? die?), conditional probs, dedicated bernoullis | Task 1 Steps 5, 7 |
| Multi-stage threshold-only `step_state` (no re-draws), v3 order, flow counts captured | Task 1 Step 8 |
| Death via `sim.people.request_death` + `step_die` reset of all stage flags | Task 1 Steps 8, 9 |
| Age prognoses via `get_prognoses` (ALREADY conditional — no double-divide); comorbidity folds into severe | Task 1 Step 6 |
| Per-agent `rel_trans_base` = beta_dist draw (mean ≈ 1.0), drawn once at init | Task 2 Steps 2, 3 |
| Per-agent viral-load kernel (two-level, mean-preserving) precomputed at infection | Task 2 Step 4 |
| Per-step `rel_trans` write at top of `step_state` (before transmission); asymp_factor gate | Task 2 Step 5 |
| Two mean-preserving normalization unit tests (beta_dist mean ≈ 1, viral-load avg ≈ 1) | Task 2 Step 1 |
| `pop_scale`/`total_pop` forwarded to `ss.Sim`; static scaling only (dynamic rescaling deferred) | Task 3 Step 3 |
| Burden Results (`new_*`/`cum_*`/`n_*`, all `scale=True`); cum via cumsum in finalize_results | Task 3 Steps 1, 2 |
| M2 anchor (single-variant, full natural history, random+hybrid, n_days=120), dual-version | Task 4 Step 1 |
| `build_summary_m2` + `METRIC_KEYS_M2` (burden + re-converged transmission), dual-version | Task 4 Step 2 |
| `multi_seed_v3.py`/`compare.py` `--anchor m2_*` additions | Task 4 Step 3 |
| Slow z-score parity gate `test_m2_parity.py` (both metric families, `|z|<3`, both backends, skips when absent) | Task 4 Step 5 |
| v3.1.8 M2 baseline via the git worktree (`/tmp/cov-v3`, cwd-shadow) | Task 4 Step 6 |
| Prognosis-tree structural invariants (timer chain, asymptomatic-cannot-die, recovery-xor-death, step_die reset) | Task 1 Step 2 |
| Burden parity (`cum_symptomatic`/`severe`/`critical`/`deaths`) within `|z|<3` | Task 4 Steps 5,7; Task 5 Step 4 |
| M1-gap closure: `cum_infections`/`peak_prevalence`/`peak_n_infectious` re-converge to `|z|<3` | Task 4 Steps 5,7; Task 5 Step 4 |
| M1 gate not regressed | Task 4 Step 7; Task 5 Step 4 |
| Demo plots health-outcome curves + re-converged infection curve | Task 4 Step 7; Task 5 Step 5 |
| Full non-quarantined suite passes under `COVASIM_WARNINGS=error` | Task 4 Step 7; Task 5 Step 3 |

## Linked documents

- [`../specs/2026-05-29-covasim-m2-natural-history-parity-design.md`](../specs/2026-05-29-covasim-m2-natural-history-parity-design.md) — the authoritative M2 design spec (decisions 1–5, Open questions A–G, acceptance test, worktree workflow).
- [`2026-05-29-covasim-m1-basic-transmission.md`](2026-05-29-covasim-m1-basic-transmission.md) — the M1 plan (the format this mirrors; the `cv.COVID`/`cv.Sim`/harness surface M2 extends; the documented ~15% transmission gap M2 closes).
- [`../MIGRATION_PLAN.md`](../MIGRATION_PLAN.md) — overall plan; §M2 is this milestone's capability source; the scope-items table defers dynamic rescaling.
- `covasim/covid.py` — the M1 `cv.COVID` this milestone extends.
- `covasim/_v2_legacy/people.py` — the quarantined v3 health-state machine (authoritative natural-history reference: `set_prognoses` 139–161, `infect` decision tree 435–586, `check_*` 222–312, step order 164–186).
- `covasim/parameters.py` (active) — `get_prognoses` 230–282 (ALREADY conditional), `relative_prognoses` 285–294, durations 83–95, scalers + bed pars 98–120, `beta_dist`/`viral_dist`/`asymp_factor` 60–63.
- `covasim/utils.py` (active) — `compute_viral_load` 39–84, `compute_trans_sus` 87–95, `n_neg_binomial` 414–432 (the exact `beta_dist` sampler, still callable as `cv.n_neg_binomial`).
- hpvsim M02 shipped code (`/home/cliffk/idm/hpvsim/hpvsim/hpv.py`) — the branching multi-stage natural-history-on-`ss.Infection` template (pre-scheduled trajectory, dedicated bernoullis, `request_death`/`step_die`, `scale=True` event counters, multi-seed z-score parity gate).
- Starsim 3.3.x source (verify signatures at code-time): `/home/cliffk/idm/starsim/starsim/{diseases.py,people.py,parameters.py,distributions.py}`.

## Open questions carried from the spec (flag at the noted check-in; default path taken in this plan)

- **A** (`exposed` semantics) — default: keep M1's "pre-infectious" meaning unless burden parity needs `n_exposed` (not in the pinned set). Flag at check-in 1.
- **B** (`beta_dist` draw site + sampler/CRN) — default: draw all agents at init via the active `cv.n_neg_binomial`; pick the CRN path (seed numpy vs wrap in `ss.Dist`) so realized mean ≈ 1.0. Flag at check-in 2.
- **C** (viral-load kernel storage) — default: per-agent closed-form two-level value computed each step from `ti_infectious` + recovery/death date. Flag at check-in 2.
- **D** (bed feedback) — default: hooks present-but-inert at default `n_beds_*=None`; active validation deferred. Flag at check-in 1.
- **E** (lognormal parametrization + rounding) — default: reuse M1's `ss.lognorm_ex(mean=ss.days, std=ss.days)`; add CRN-safe stochastic rounding only if a `np.ceil` bias shows in parity. Flag at check-in 1.
- **F** (`pop_scale`/`total_pop` plumbing) — default: forward both to `ss.Sim`; set exactly one (Starsim raises if both); confirm `pop_size → n_agents`. Flag at check-in 3.
- **G** (re-convergence residual policy) — default: if transmission metrics do not re-converge to `|z| < 3`, document the residual and loosen to `|z| < 5` only with Cliff's sign-off. Flag at check-in 4 / conditional check-in 5.
