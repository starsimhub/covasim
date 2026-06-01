# Covasim v4.0 — M3 Multi-Variant + Cross-Immunity: design spec

> **Status:** drafted 2026-05-29, pending Cliff's review/commit. Authoritative for every M3 decision; the
> implementation plan (`../plans/2026-05-29-covasim-m3-multivariant-cross-immunity.md`) implements exactly this.
> This spec was produced from a deep parallel code analysis (v3 engine, hpvsim precedent, starsim 3.4.x
> primitives, current v4 seams) and a scored, adversarially-stress-tested design panel (3 candidate
> architectures; the chosen one survived three independent adversaries).

## Goal

Make `cv.COVID` support **multiple co-circulating variants** (e.g. wild + alpha + delta) with **cross-immunity**,
reproducing v3.1.8's per-variant and aggregate trajectories. The user-visible demo: a multi-variant epidemic with
staggered variant introduction, plotting per-variant infection curves (`cum_infections_by_variant`,
`n_infectious_by_variant`) and the aggregate burden. **Waning immunity / neutralizing-antibody (NAb) time-kinetics
are explicitly deferred to M4** — M3 ships a *static, NAb-free* cross-immunity.

## Problem statement

Three findings from the code analysis frame every M3 decision:

1. **A COVID host has exactly ONE SEIR chain ("host exclusivity").** v3 `People` carries scalar per-agent
   `exposed_variant`/`infectious_variant`/`recovered_variant` tags (NaN-or-int) and **one** set of `date_*`
   schedules. An agent physically cannot run two variants' clocks. The 2D `by_variant` boolean arrays exist
   **only for result counting**, not parallel disease clocks (`_v2_legacy/people.py:76-83,514-579`;
   `defaults.py:78-88`). Exclusivity is enforced *implicitly* by the susceptible-gate: `infect()` drops
   non-susceptible agents, infection flips `susceptible=False`, recovery restores it (`people.py:469-471,495`).
   This is the **inverse of hpvsim**, whose genotypes genuinely co-infect — hpvsim's only host exclusivity is
   the terminal cancer-cancellation, so the hpvsim "one module per genotype" template provides **no precedent**
   for COVID's from-infection-onward, released-on-recovery exclusivity.

2. **The M3/M4 boundary is entangled in v3.** v3 routes *all* cross-immunity through the NAb machinery:
   `effective_nabs = people.nab * immunity[variant, recovered_variant]`, then the logistic `calc_VE`
   (`_v2_legacy/immunity.py:303-350`). And both reinfection and cross-immunity are gated by `use_waning` — which
   *also* turns on NAb time-decay. **At `use_waning=False`, v3 has ZERO cross-immunity and ZERO reinfection**
   (`sus_imm` stays 0; recovered are never made susceptible again). Consequence: M3 cannot literally replicate
   any v3 code path. M3 must **synthesize** a NAb-free static cross-immunity — apply the matrix directly to a
   per-agent per-variant immunity reduction — and **split** the `use_waning` entanglement so reinfection + static
   cross-immunity work without NAb decay. The M3 regression baseline is therefore **hand-derived from the
   cross-immunity matrix + variant pars**, NOT taken from a v3 `use_waning=False` run (which shows no effect).

3. **The public results contract is fixed and quirky.** `sim.results['variant']` is a sub-dict of exactly 12
   2D `(n_variants, npts)` Result objects (enumerated in §7); aggregates are computed *independently* of the
   per-variant arrays (not by summing them) except a shared seed-offset; `prevalence_by_variant` is defined as
   `new_infections_by_variant / n_alive` (a long-standing misnomer to copy verbatim). The variant *parameter*
   surface (`get_variant_pars`/`get_variant_choices`/`get_cross_immunity`, `n_variants`/`variant_map`/
   `variant_pars`/`variants`, `imm_states`) already exists in the kept-from-v3 `parameters.py`/`defaults.py`;
   the variant *logic* and the `cv.variant` class still live in `_v2_legacy/immunity.py` and must be restored.

## Architecture decision — Design B (single module, internal variant axis)

Three candidate architectures were designed and scored (1–5 on seven criteria; max 35):

| Design | Total | Exclusivity | Bwd-compat | Fwd-compat M4/M6 | Parity | Idiomatic | Clarity | Impl-risk⁻¹ |
|---|---|---|---|---|---|---|---|---|
| **B — single module + variant axis** | **30** | 5 | 5 | 4 | 5 | 4 | 4 | 3 |
| C — hybrid (chain owner + modifiers) | 29 | 5 | 5 | 5 | 4 | 3 | 4 | 3 |
| A — N modules per variant (hpvsim-style) | 23 | 3 | 4 | 3 | 4 | 4 | 3 | 2 |

**Chosen: Design B** — keep **exactly one `cv.COVID(ss.Infection)`** in `sim.diseases`, add the variant dimension
*inside* it. This is faithful to v3's actual data model and makes the #1 risk (host exclusivity) **structural and
free**: there is one set of host BoolStates and one set of `ti_*` dates, so an agent cannot run two SEIR clocks.

**Why not Design A (N modules, the plan's tentative wording).** A mirrors hpvsim, but COVID is hpvsim's inverse:
A must *actively enforce* host exclusivity every step as a net-new, order-dependent invariant (infect-by-one
must set `susceptible=False` on all other variant modules; release on recovery; prevent two `ti_dead`) — with no
hpvsim precedent and a single point of failure if starsim's loop ordering ever changes (judge: exclusivity 3/5).
Worse for the arc: M4's NAb state (`peak_nab`/`nab`/`t_nab_event`) and M6's vaccine state are **host-level**, and
A has no natural home for them — they end up awkwardly accreting on the connector (judge: fwd-compat 3/5). A also
needs the most results-reconstruction glue and per-variant CRN-offset bookkeeping.

**Why not Design C (hybrid).** C scored within one point of B, but its own author concluded the faithful version
of C *is* B (its element-by-element comparison marked 7 of 8 components "identical to B"). C is B with the
variant pars factored into separate "modifier" objects — an extra indirection that buys nothing here and scores
worse on starsim-idiomatic (3/5). We adopt B and borrow C's one good idea where useful (treating per-variant
pars as a clean data bundle on the single module).

**This diverges from the MIGRATION_PLAN's tentative "one disease instance per variant" wording (M3 sub-task 1).**
The plan explicitly deferred the mechanism to "the M3 design spec" (M3 sub-task 2) — this spec exercises that
delegation and selects B. `MIGRATION_PLAN.md` §M3 is updated to match. **`cv.CrossImmunity(ss.Connector)` (the one
pre-approved new class) is retained** — in Design B it operates on the single module (reads `recovered_variant` +
the matrix, writes the module's 2D immunity arrays), which also makes it the natural home for M4's NAb engine
("fold into `cv.CrossImmunity`" per the plan's M4).

## Design decisions

### 1. State layout — what is 1D host vs 2D variant-indexed

The crux of Design B. **Unchanged from M2 (1D host states ⇒ exclusivity is structural):**

- **Host BoolStates** (auto-generate aggregate `n_*` results): `susceptible, infected, exposed, symptomatic,
  severe, critical, recovered, dead`. Exactly one per agent.
- **Host `ti_*` FloatArrs** (one clock per host): `ti_infected, ti_exposed, ti_infectious, ti_symptomatic,
  ti_severe, ti_critical, ti_recovered, ti_dead, ti_vl_switch`. One `ti_dead` ⇒ "two death schedules" is
  impossible by construction.
- **Host per-agent FloatArrs**: `rel_trans, rel_trans_base, rel_sus, symp_prob, severe_prob, crit_prob,
  death_prob`. `rel_sus` keeps the age-OR baseline (covid.py:144); the connector does **not** overwrite it (§5).

**NEW — scalar per-agent variant tags** (the v3 `*_variant` scalars; `ss.FloatArr`, NaN = "none"):
`exposed_variant`, `infectious_variant`, `recovered_variant`. `infectious_variant` is set in `step_state` the
moment an agent becomes infectious, using the **same `infected & (ti_infectious <= ti)` gate as the `infectious`
property**, so for `nv==1` the source mask `infectious & (infectious_variant==0)` equals the full infectious set
(byte-identity — adversary-required). `recovered_variant` is set at recovery and drives cross-immunity.

**NEW — 2D variant-indexed immunity arrays**, shape `(nv, n_agents)` (the v3 `imm_states`; plain ndarrays, not
`ss.Arr`, allocated in `init_post` once `nv` and `n_agents` are known): `sus_imm`, `symp_imm`, `sev_imm`.
Because they are not `ss.Arr`, they do **not** auto-grow/reorder on UID churn — **M3 asserts a constant
population (no births) in `init_post`** and documents that M4/M6 must convert these to a growth-aware 2D
abstraction (adversary-required).

**NEW — transient 2D flow accumulators**, shape `(nv,)`, reset each step (mirrors v3 `flows_variant`):
`_flow_variant = {new_infections, new_symptomatic, new_severe, new_infectious}`.

### 2. The `infect()` override — per-variant transmission

`ss.Infection.infect()` (starsim `diseases.py:237-290`) is single-beta: it snapshots
`rel_trans = infectious*rel_trans` and `rel_sus = susceptible*rel_sus` once, then loops networks with one
betamap. M3 **overrides `infect()`** to wrap the network loop in an outer `for vi in range(nv)` (mirroring v3
`sim.py:622-649`), changing only: (a) the source snapshot masks by `infectious_variant==vi`; (b) `beta` is
scaled by `variant_pars[label].rel_beta`; (c) `rel_sus` folds in this variant's cross-immunity
`(1 - sus_imm[vi])`. **The CRN-load-bearing `self.trans_rng.rvs(src, trg)` call is preserved verbatim.**

**Exclusivity at dedup (the subtle part).** v3 mutates `susceptible=False` *inside* the sequential variant loop,
so a later variant never re-selects an agent an earlier one took. The override computes all variants' candidate
targets before mutating state, so it reproduces that priority **at dedup**: candidates are appended in **ascending
`vi` order**, then `ss.uids.unique(return_index=True)` keeps the first occurrence ⇒ lowest-variant-index wins,
matching v3's sequential-loop outcome. The surviving per-target variant is recorded and passed to `set_prognoses`
via `self._new_case_variant` (since `ss.Infection.step()` calls `set_outcomes(new_cases, sources)` with no variant
arg). **Key the per-target variant by UID, not array position,** so it survives `set_outcomes`' congenital/age
split (harmless in M3 — no births — but required before any congenital path; adversary-required).

**CRN honesty (adversary-required).** Do **not** claim byte-identity for `nv>1`. `trans_rng` is an
`ss.multi_random` that auto-jumps per `rvs(src,trg)` call, so calling it once per variant per step gives variants
**independent** draws (correct vs v3, which also drew per-variant) — but it is a *different consumption pattern*
than M2's single call. Byte-identity is asserted only for **`nv==1`** (a unit test vs stock
`ss.Infection.infect()` on a fixed network+seed); multi-variant gets a no-error + independent-draws test.

**Risk mitigation.** The override forks an upstream method; mitigated by (a) the `nv==1` byte-equality test, and
(b) a dedicated **exclusivity test** seeding two variants into overlapping contacts in one step and asserting
every new case has exactly one finite `infectious_variant` and at most one finite `ti_dead`.

### 3. `set_prognoses` — variant-aware branch probabilities

Extends M2's pre-drawn-tree `set_prognoses`. For each new case it reads the per-target variant
(`self._new_case_variant`, or an explicit arg for seeds/imports), sets `exposed_variant`, and scales the four
branch bernoullis by **both** the global `rel_*_prob`, the **per-variant** `variant_pars[label].rel_*_prob`, **and**
the cross-immunity reductions (v3 `people.py:523,538`):

```python
symp_p  = p.rel_symp_prob   * vp['rel_symp_prob']   * symp_prob[uids]  * (1 - symp_imm[v, uids])
sev_p   = p.rel_severe_prob * vp['rel_severe_prob'] * severe_prob[sub] * (1 - sev_imm[v, sub])
crit_p  = p.rel_crit_prob   * vp['rel_crit_prob']   * crit_prob[sub]
death_p = p.rel_death_prob  * vp['rel_death_prob']  * death_prob[sub]
```

When new cases of a step carry mixed variants, loop `for v in unique(variant_of_uids)` and slice (a tiny loop over
*this step's* new cases). The four scratch bernoullis are reused with per-subset `set(p=...)`; CRN is uid-keyed.

**Reinfection (adversary-required fix):** `set_prognoses` must **clear `recovered=True`** (and the recovered tag)
for reinfected agents, mirroring v3 `people.py:497`, or `n_recovered` double-counts on same-step reinfection.

### 4. `cv.variant` — restored

`cv.variant` is **restored verbatim** from `_v2_legacy/immunity.py` into a new active `covasim/immunity.py`, with
its v3 public signature `variant(variant, days, label=None, n_imports=1, rescale=True)`. In Design B it is **not**
an `ss.Infection` — it is a lightweight registration+seeding descriptor (as in v3):

- `parse()` — string aliases via `get_variant_choices`, or a pars dict (unchanged).
- `initialize()` — registers `label` into the **single COVID module's** `variant_map`(int→label) and
  `variant_pars`(label-keyed 5-key dict), assigns `self.index`, grows `nv`, and (re)allocates the 2D immunity
  arrays. wild = index 0 always.
- `apply()` — mid-run introduction on matched `days`: selects susceptibles, applies `rescale`, and calls
  `covid.import_variant(uids, variant=self.index)` (which routes to `set_prognoses(uids, variant=index)`),
  bumping the `n_imports` result.

`cv.Sim(variants=[...])` (and the single-string / single-object sugar) registers each into the one module before
state allocation. **When `variants` is empty, `nv==1`, no connector is attached, and behavior is byte-identical
to M2** (the single-module path is structurally untouched).

### 5. `cv.CrossImmunity(ss.Connector)` — static, NAb-free

Runs at loop slot 6 (after all `step_state`, before transmission at slot 9 — exactly where v3 ran
`check_immunity`). It resolves the sibling via `self.sim.diseases['covid']` (post-deepcopy), and each step writes
the module's 2D `sus_imm`/`symp_imm`/`sev_imm` for agents who have **ever recovered**:

```python
def step(self):
    covid = self.sim.diseases['covid']
    rec = covid.ever_recovered.uids            # ever-recovered, NOT the transient `recovered` BoolState
    if not len(rec): return
    src_v = covid.recovered_variant[rec].astype(int)
    for v in range(covid.nv):                  # target variant
        covid.sus_imm[v, rec]  = self.matrix[v, src_v]
        covid.symp_imm[v, rec] = self.matrix[v, src_v]
        covid.sev_imm[v, rec]  = self.matrix[v, src_v]
```

**Ever-recovered keying (adversary-required).** Cross-immunity is keyed off an **ever-recovered / finite-
`ti_recovered`** tag (v3's `was_inf = t >= date_recovered`, `immunity.py:326`), not the transient `recovered`
BoolState — so protection persists across the reinfection window and stays correct once M4 adds time-since-
recovery kinetics. (Implementation: a `ti_recovered`-finite mask, or an `ever_recovered` BoolState set at first
recovery and never cleared.)

`build_immunity_matrix(variant_map, override=None)` replicates `_v2_legacy/immunity.py:284-295` exactly:
`np.ones((nv,nv))`, then `matrix[target_v, source_v] = get_cross_immunity()[label_target][label_source]`
(asymmetric; diagonal 1.0). The matrix lives on the connector (`self.pars.immunity`, overridable), mirrored to
`sim['immunity']` for backwards-compat reads.

**Why the connector writes `sus_imm` (not `rel_sus`).** `infect()` reads `sus_imm` directly and combines it with
the `rel_sus` age-OR baseline (§2), so the connector never touches `rel_sus` — sidestepping both the
"`rel_trans` is overwritten by `step_state`" trap and the "connector must re-apply the age-OR baseline" trap. The
age structure in `rel_sus` is preserved untouched.

### 6. Reinfection / `use_waning` split

v3 gates reinfection + cross-immunity behind `use_waning` (which also = NAb decay = M4). M3 introduces an internal
switch **`cross_immunity_active`** (auto-True when `nv>1` / a `CrossImmunity` connector is attached). When active:
recovery sets `susceptible=True` again (so the agent is challengeable), and the connector's `sus_imm`/`symp_imm`/
`sev_imm` gate the next infection. **`use_waning` is left reserved for M4 NAb kinetics.** When inactive (single
variant, no connector — the M2 path), recovered agents stay `susceptible=False` ⇒ **M2 behavior is preserved
byte-identically.** (Recommended default; the alternative — redefining `use_waning=False` to mean "static
cross-immunity, no decay" — is a public-flag behavior change and is **not** recommended.)

**Documented divergence (adversary-required).** Because M3 applies the matrix directly (no `people.nab`, no
`calc_VE`), `sus_imm` for a same-variant challenge = diagonal = 1.0 ⇒ **same-variant reinfection is exactly 0**.
v3's literal `use_waning` numerics (`calc_VE(nab·1.0) < 1`) permit a tiny same-variant reinfection. This divergence
is intended and is **recorded in the hand-derived baseline** so it is not mistaken for a parity miss.

### 7. Results — 2D `by_variant` + the bridge, scaling, offset, denominators

**Aggregate results (free, byte-identical to M2):** the 1D host BoolStates auto-produce `n_*`; M2's manual
`n_infectious`/`new_*`/`cum_*` flows are counted off 1D host transitions, **independently** of the variant arrays
(matching v3). The `cum_infections` seed-offset is the existing M2 behavior.

**By_variant results (manual, 2D):** in `init_results`, define the **exact 12-key** sub-dict as 2D
`(nv, npts)` Results attached as a nested `ss.Results` (natively supported — `ss.ndict` `flatten` descends nested,
`_iter/asdict` skip them):

```
prevalence_by_variant, incidence_by_variant            (scale=False)
{new_,cum_}{infections,symptomatic,severe,infectious}_by_variant   (8, scale=True)
n_exposed_by_variant, n_infectious_by_variant          (2, scale=True)
```

`update_results` writes `_flow_variant` into `new_*_by_variant[:, ti]` and the two stocks via
`count(exposed & exposed_variant==v)` / `count(infectious & infectious_variant==v)`. `finalize_results`:
`cum_*_by_variant = cumsum(new_*_by_variant, axis=1)`, then the v3 quirks **verbatim**:
`incidence_by_variant = new_infections_by_variant / n_susceptible`,
`prevalence_by_variant = new_infections_by_variant / n_alive`.

**The adversary-required corrections (the "results for free" claim was wrong):**

- **Flat-vs-nested bridge.** v3 exposes `sim.results['variant'][key]` at the **sim root**; starsim namespaces
  module results under `sim.results['covid']`. M3 **bridges** the `['variant']` sub-dict to the v3 top-level path
  (a reference attached in `cv.Sim.finalize`), and likewise surfaces the `n_imports` result. The *full* flat
  aggregate-results / `sim.summary` compat is a broader port-wide concern — see Open Question E.
- **Manual `pop_scale` scaling.** starsim's auto-scaler (`modules.py:727`) scales only top-level `ss.Result`
  objects and does **not** descend into the nested sub-dict. M3 scales the 8 `scale=True` by_variant Results by
  `pop_scale` explicitly in `finalize_results` (mirroring v3's einsum, `sim.py:778`).
- **Seed-offset.** `+= pop_infected * rescale_vec[0]` on **both** `cum_infections` and the wild
  `cum_infections_by_variant` (v3 `sim.py:786-787`). Not inherited from M2 — baked into the hand-derived baseline.
- **Recomputed denominators.** `incidence/prevalence_by_variant` use v3's scaled-population denominators
  (`n_alive = scaled_pop − cum_deaths`; `n_susceptible = n_alive − n_exposed − (1−use_waning)·cum_recoveries`),
  reached at the sim level in finalize — not the module's raw BoolState auto-counts.
- **Float dtype.** All by_variant Results are `result_float`, not int, matching v3 and avoiding truncation.

### 8. `step_die` + infection log

- **`step_die` (adversary-required):** clears the scalar `*_variant` tags → NaN (mirrors v3 `people.py:309-311`),
  for clean state + M4/M6 safety.
- **Infection log (adversary-required):** when the log exists, transmissions carry `variant = variant_map[v]`
  (the **string** label) — TransTree depends on this. Note starsim's networkx `InfectionLog` only materializes
  with an analyzer and lacks v3's `dict(source,target,date,layer,variant)` format; **full TransTree parity is
  deferred to M9**, but M3 attaches the variant label at the transmission site so the data is captured.

## M3 anchor scenario + pinned metrics

A multi-variant anchor (`tests/regression/anchor_m3.py`), dual-version (v3.1.8 + v4), `random` and `hybrid`
backends. Default proposal: `pop_size=20_000`, `pop_infected=100` (wild at t0), `n_days=120`, introduce
**alpha at day 10** and **delta at day 30** (`cv.variant('alpha', days=10, n_imports=20)`,
`cv.variant('delta', days=30, n_imports=20)`), `use_waning` set so the v3 baseline exercises cross-immunity
(see Open Question D — the v3 baseline **must** run with cross-immunity active, i.e. v3 `use_waning=True` with NAb
present, and M3 parity targets the *shape* of per-variant displacement, accepting the documented NAb-vs-static
divergence). Pinned metrics (`build_summary_m3`, `METRIC_KEYS_M3`): aggregate `cum_infections`, `peak_prevalence`,
`peak_n_infectious`, `cum_deaths`; per-variant `cum_infections_by_variant[v]` and `peak_n_infectious_by_variant[v]`
for each of wild/alpha/delta; and the variant *displacement ordering* (which variant dominates, and when).

## Acceptance test

Multi-variant prevalence and per-variant trajectories overlap a v3.1.8 multi-variant baseline within tolerance
(slow z-score gate `tests/test_m3_parity.py`, both backends, skips when the baseline is absent). Because M3's
cross-immunity is NAb-free (a deliberate divergence), the gate targets per-variant trajectory **shape** and
ordering with the documented residual rationale rather than bit-level NAb agreement; the per-metric threshold may
be set to `|z| < 5` with a written rationale, exactly as M2 (signed off per Open Question D).

## Hand-derived baseline note

Per problem-statement finding (2), there is **no faithful v3 `use_waning=False` cross-immunity path**. The M3
baseline is generated from a frozen v3.1.8 worktree run **with cross-immunity active** (the realistic multi-variant
regime), and the *expected* M3-vs-v3 divergence (static matrix vs NAb·calc_VE; same-variant reinfection = 0) is
documented alongside the baseline so reviewers can distinguish intended divergence from regression.

## Adversary punch-list (must all be addressed in the plan)

1. `infect()` override: `nv==1` byte-equality test vs stock; multi-variant no-error + independent-draws test; do
   not claim `nv>1` byte-identity.
2. `set_prognoses` clears `recovered` (+ tag) on reinfection (no `n_recovered` double-count).
3. `step_die` clears scalar `*_variant` tags → NaN.
4. Cross-immunity keyed off **ever-recovered** (finite `ti_recovered`), not the transient `recovered` BoolState.
5. Dedicated **exclusivity test** (two variants, overlapping contacts, one step ⇒ one `infectious_variant`, ≤1
   `ti_dead`).
6. Document same-variant-reinfection = 0 divergence in the baseline.
7. Results: bridge `sim.results['variant']` to the v3 top-level path; **manually** scale 2D `scale=True` results by
   `pop_scale`; add the seed-offset to `cum_infections` + wild `cum_infections_by_variant`; recompute
   `incidence/prevalence` denominators at sim level; float dtype for all by_variant; create + route the
   `n_imports` Result.
8. Source-tagging invariant: set `infectious_variant` in `step_state` with the exact `infected & (ti_infectious
   <= ti)` gate so `nv==1` mask = full infectious set.
9. Key `_new_case_variant` by UID; survive `set_outcomes` congenital split (defensive for M3).
10. Assert constant population (no births) in M3; document M4/M6 must make `sus_imm/symp_imm/sev_imm` growth-aware.

## Out of scope for M3 (deferred)

- **NAb time-kinetics / waning** (`peak_nab`/`nab`/`t_nab_event`/`nab_kin`/`update_peak_nab`/`calc_VE`/
  `precompute_waning`) — **M4**. M3 ships the static matrix only; the dormant NAb functions may be copied into
  `immunity.py` (imported, not wired) as M4's landing zone.
- **Vaccination** (incl. per-variant efficacy) — **M6**.
- **Full TransTree parity** — **M9** (M3 only attaches the variant label at the transmission site).
- **Full flat aggregate-results / `sim.summary` compat** — see Open Question E.
- Dynamic rescaling beyond the static `pop_scale`/`total_pop` from M2.

## Open questions for Cliff

- **A — Architecture (needs sign-off; diverges from the plan's tentative wording).** Adopt **Design B** (single
  module + internal variant axis) instead of the plan's tentative "one disease instance per variant" (Design A)?
  Recommendation: **yes** (structural exclusivity; natural M4/M6 host-level state; direct results mapping; judge
  30 vs 23; survived all adversaries). `cv.CrossImmunity(ss.Connector)` is retained.
- **B — Reinfection / `use_waning` split (genuine semantic fork).** Introduce an internal `cross_immunity_active`
  switch (auto-on when `nv>1`), leaving `use_waning` for M4 — preserving M2 single-variant behavior exactly?
  Recommendation: **yes** (the alternative redefines a public flag's meaning).
- **C — Cross-immunity axes in M3.** Implement all three axes (`sus_imm`/`symp_imm`/`sev_imm`) directly from the
  matrix, or susceptibility only (deferring symp/sev to M4)? Recommendation: **all three** (v3 derives all three
  from the one matrix; M4 later layers NAb weighting on top).
- **D — Baseline regime + gate threshold.** Generate the v3 baseline with cross-immunity active (`use_waning=True`)
  and target per-variant trajectory shape with the documented static-vs-NAb divergence + `|z| < 5` rationale (as
  M2)? Recommendation: **yes**.
- **E — Results bridge scope.** Bridge the `['variant']` sub-dict + `n_imports` to the v3 top-level path now, and
  **defer** the full flat aggregate-results / `sim.summary` compatibility (`sim.results['cum_infections']` at the
  root) to a later milestone (M9 plotting/results pass)? Recommendation: **yes** (the parity harness already reads
  results version-aware, so the M3 gate does not depend on the full flat bridge).
- **F — Variant introduction timing.** In-scope for M3: both t0 seeding (`days=0`) and mid-run import
  (`apply()` on matched days). Recommendation: **in scope** (core to the staggered-introduction demo).

## Linked documents

- `../plans/2026-05-29-covasim-m3-multivariant-cross-immunity.md` — the task-by-task implementation plan.
- `MIGRATION_PLAN.md` §M3 — capability scope (updated to reflect Design B).
