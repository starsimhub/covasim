# Covasim v4.0 — M3 Multi-Variant + Cross-Immunity Implementation Plan

> **For agentic workers:** Implement this plan task-by-task. No special plugin or sub-skill is required.
>
> **CRITICAL — commit discipline.** Local effort, **pause-for-review-and-commit** cadence. The assistant
> **prepares and stages** each piece of work and **pauses for Cliff Kerr to review and commit**. **The assistant
> NEVER runs `git commit` and NEVER runs `git push`.** Check-in boundaries say **PAUSE FOR CLIFF**. There are
> **4 check-ins** across M3.
>
> **VERIFY-AT-CODE-TIME discipline.** Starsim 3.4.x signatures are quoted from the design spec + the M3 code
> analysis (installed starsim at `/home/cliffk/idm/starsim/starsim/`). They are correct as of writing, but the
> implementer **MUST verify each one against the installed `starsim` before relying on it** — flagged inline with
> **[VERIFY]**. Read the real source (`python -c "import starsim as ss, inspect; print(inspect.getsource(ss.Infection.infect))"`).
> Confirmed present at plan time: `ss.Connector`, `ss.Infection.infect` (single-beta, `diseases.py:237-290`),
> `ss.multi_random`, loop slots 5/6/9 (`loop.py:24-42`), nested `ss.Results` (`results.py`), `ss.Result(shape=...)`.
> **NOTE the version mismatch:** the plan prose says "starsim v4.0" but the **installed package is 3.4.x** — design
> to the actual installed API.

**Goal:** Make `cv.COVID` support multiple co-circulating variants with **static, NAb-free cross-immunity**
(waning/NAbs are M4). Demo: a multi-variant epidemic with staggered variant introduction (wild + alpha + delta),
plotting per-variant infection curves + aggregate burden.

**Architecture — Design B (single module + internal variant axis), authoritative in the design spec.** Keep
**one** `cv.COVID(ss.Infection)`; add the variant dimension inside it (scalar `*_variant` tags + 2D
`sus_imm/symp_imm/sev_imm`), so **host exclusivity is structural and free** (one SEIR chain / one `ti_dead` per
host). Override `infect()` to loop variants with per-variant beta (mirroring v3 `sim.py:622-649`). Add the one
pre-approved new class `cv.CrossImmunity(ss.Connector)` operating on the single module, and restore `cv.variant`.
**When `variants` is empty (`nv==1`), the single-module path is byte-identical to M2** — the continuous-runnability
invariant holds at every commit. This **diverges** from the MIGRATION_PLAN's tentative "one disease instance per
variant" (Design A); the divergence is decided in the design spec (§Architecture) and gated on Open Question A.

**Tech Stack:** Python 3.9–3.13, pytest + pytest-xdist, covasim (v4 port on `starsim-port`), starsim 3.4.x, sciris.

**Authority:** the design spec `migration_plan/specs/2026-05-29-covasim-m3-multivariant-cross-immunity-design.md`
(authoritative). Capability scope: `MIGRATION_PLAN.md` §M3.

**Class/file names — LOCKED:** `cv.COVID` (covid.py), `cv.Sim`/`cv.People`/`cv.Network` keep names. **New:**
`cv.CrossImmunity(ss.Connector)` (covasim/connectors.py); **restored:** `cv.variant` (covasim/immunity.py).

**Spec Open Questions A–F — default path adopted here** (each flagged for Cliff; if Cliff rules otherwise, adjust):
- **A** (architecture) — **Design B** (single module + variant axis). *Surfaced for explicit sign-off at the
  plan/spec check-in, since it diverges from the written plan.*
- **B** (reinfection / `use_waning` split) — internal `cross_immunity_active` flag (auto-on when `nv>1`); leave
  `use_waning` for M4; preserve M2 single-variant behavior (check-in 3).
- **C** (cross-immunity axes) — all three (`sus_imm`/`symp_imm`/`sev_imm`) directly from the matrix (check-in 3).
- **D** (baseline regime + threshold) — v3 baseline with cross-immunity active; target trajectory shape; `|z| < 5`
  with written rationale, as M2 (check-in 4).
- **E** (results bridge scope) — bridge `['variant']` + `n_imports` to v3 top-level path now; defer full flat
  aggregate-results / `sim.summary` compat to a later milestone (check-in 4).
- **F** (introduction timing) — both t0 and mid-run import in scope (check-ins 2 + 4).

---

## Starting state (confirmed at plan time)

- Branch `starsim-port`, M2 landed and committed. The assistant never creates/switches branches.
- `covasim/covid.py` — single-variant `cv.COVID(ss.Infection)` with the full natural-history tree, `_dur`
  integer rounding, per-agent `rel_trans` = `rel_trans_base × viral_load(t) × f_asymp`, burden Results, exact-count
  `init_post` seeding. `rel_sus` (FloatArr) written once as age ORs in `_fill_prognoses` (covid.py:144); `rel_trans`
  rewritten every `step_state`. Four scratch bernoullis + seeding rng (offset 50) + beta_dist rng (offset 60).
- `covasim/sim.py` — `cv.Sim(ss.Sim)`: `_BETA_LAYER`/`_BASE_BETA`, assembles `cv.People` + `make_networks` +
  one `cv.COVID(beta=per-layer dict, init_prev=int)`; `pop_scale`/`total_pop` forwarded. **No `variants=`/
  `connectors=` yet** (the refactor seam, sim.py:44-76).
- `covasim/parameters.py` (kept-from-v3) — `get_variant_pars`/`get_variant_choices`/`get_cross_immunity` already
  present; `make_pars` builds `variant_pars['wild']`. **Add back** `n_variants`/`variant_map`/`variant_pars`/
  `variants` to the public pars where missing.
- `covasim/defaults.py` (kept-from-v3) — variant_states, by_variant_states, `result_*_by_variant`, `variant_pars`
  list, imm_states (`sus_imm`/`symp_imm`/`sev_imm`), variant colors, `overview_variant_plots` already defined.
- `covasim/_v2_legacy/immunity.py` — the v3 `cv.variant` class + `init_immunity`/`check_immunity`/matrix build
  (porting reference). NAb kinetics here are **M4**, not M3.
- Regression harness `tests/regression/`: `anchor_m{0,1,2}.py`, `short_summary.py` (`build_summary{,_m1,_m2}`,
  `METRIC_KEYS_M{1,2}`, dual-version), `parity.py` (`parity_gate(..., z_threshold=...)`), `multi_seed_v3.py`
  (`--anchor` registry), `compare.py`, `README.md` (M1 + M2 sections). `tests/test_m{1,2}_parity.py` (slow gates).
- Frozen v3.1.8 reference via `git worktree add /tmp/cov-v3 main`; run harness with `cwd=/tmp/cov-v3`.

---

## Task 1: Variant scaffolding on `cv.COVID` — `nv==1` byte-identical (CHECK-IN 1)

Make the single module variant-aware **without changing single-variant behavior**.

- [ ] Add `nv=1`, `variant_map={0:'wild'}`, `variant_pars={'wild':{...5 keys all 1.0...}}` to `cv.COVID`
      (default; grown by `cv.variant` later). Read defaults from `parameters.get_variant_pars`.
- [ ] Add scalar per-agent tags `exposed_variant`, `infectious_variant`, `recovered_variant` (`ss.FloatArr`,
      default NaN). [VERIFY] `ss.FloatArr` NaN default.
- [ ] In `step_state`, when an agent becomes infectious, set `infectious_variant` using the **exact**
      `infected & (ti_infectious <= ti)` gate that defines the `infectious` property (spec §1 source-tagging
      invariant). For seeds, `exposed_variant`/`infectious_variant` = 0 (wild).
- [ ] Allocate 2D immunity arrays `sus_imm`/`symp_imm`/`sev_imm` = `np.zeros((nv, n_agents))` in `init_post`
      (all-zero ⇒ no effect at nv=1). Add `_flow_variant` transient accumulators (zeros(nv), reset each step).
- [ ] **Assert constant population (no births) in `init_post`** (the 2D arrays are not growth-aware; spec §1).
- [ ] Define the 12-key `results['variant']` sub-dict (2D `(nv, npts)`, **float dtype**) as a nested `ss.Results`.
      Fill `new_*_by_variant`/stocks in `update_results`; `cum_*`/quirk-denominators in `finalize_results`
      (spec §7). At `nv==1`, assert `by_variant[0,:] == aggregate` for the matching keys.
- [ ] `step_die`: clear the scalar `*_variant` tags → NaN (spec §8).
- [ ] **Tests:** `test_baselines.py` UNCHANGED (the byte-identity guard). New: `by_variant[0]==aggregate` at
      `nv==1`; the scalar-tag lifecycle (set on infectious, cleared on death).
- [ ] **PAUSE FOR CLIFF.** Stage: `covasim/covid.py`, any `parameters.py` pars additions, the new unit tests.

## Task 2: `infect()` override + variant-aware `set_prognoses` + restore `cv.variant` (CHECK-IN 2)

Per-variant transmission + variant introduction (no cross-immunity yet).

- [ ] Override `cv.COVID.infect()` (spec §2): copy stock `ss.Infection.infect()` [VERIFY current source], wrap the
      network loop in `for vi in range(nv)`; mask sources by `infectious_variant==vi`; scale beta by
      `variant_pars[label].rel_beta`; fold `(1 - sus_imm[vi])` into `rel_sus`; **preserve `trans_rng.rvs(src,trg)`
      verbatim**. Concatenate candidates in ascending `vi` order; dedup via `ss.uids.unique(return_index=True)`
      (lowest-index wins = v3 tie-break). Record surviving per-target variant **by UID** in `self._new_case_variant`.
- [ ] Variant-aware `set_prognoses` (spec §3): read `_new_case_variant` (or explicit arg for seeds/imports); set
      `exposed_variant`; scale the four bernoullis by global × per-variant `rel_*_prob` × `(1 - {symp,sev}_imm)`;
      loop over `unique(variant)` of this step's new cases. **Clear `recovered`(+tag) for reinfected agents.**
- [ ] Add `cv.COVID.import_variant(uids, variant)` → routes to `set_prognoses(uids, variant=index)`; bumps the
      `n_imports` result.
- [ ] Restore `cv.variant` in **new `covasim/immunity.py`** (spec §4): signature
      `variant(variant, days, label=None, n_imports=1, rescale=True)`; `parse()`/`initialize()`/`apply()`.
      `initialize()` registers into the module's `variant_map`/`variant_pars`, assigns `index`, grows `nv`,
      reallocates the 2D arrays. `apply()` selects susceptibles on matched `days`, applies `rescale`, calls
      `import_variant`. [VERIFY] how `apply` is invoked each step — as an `ss.Intervention`-like object iterated
      in the loop, or a hook in `cv.Sim`; choose the least-magic option (likely a tiny intervention wrapper).
- [ ] `cv.Sim(variants=...)` (sim.py:44-76): accept a `cv.variant` / list / string sugar; register each into the
      one module before state allocation; attach the variant-introduction mechanism. **`variants` empty ⇒
      byte-identical M2 path** (assert via `test_baselines.py`).
- [ ] Export `cv.variant` in `__init__.py`.
- [ ] **Tests:** `nv==1` byte-equality of the overridden `infect()` vs stock `ss.Infection.infect()` on a fixed
      network+seed; a 2-variant run produces independent per-variant draws with no `DistNotReadyError`; per-variant
      `rel_beta`/`rel_*_prob` actually scale the right module arithmetic; mid-run `apply()` seeds `n_imports` on the
      right day.
- [ ] **PAUSE FOR CLIFF.** Stage: `covasim/covid.py`, `covasim/immunity.py` (new), `covasim/sim.py`,
      `covasim/__init__.py`, the new tests.

## Task 3: `cv.CrossImmunity(ss.Connector)` + reinfection (CHECK-IN 3)

Make multi-variant epidemiology correct.

- [ ] New `covasim/connectors.py`: `cv.CrossImmunity(ss.Connector)` (spec §5). `init_post` resolves
      `self.sim.diseases['covid']` (post-deepcopy) and builds the matrix via `build_immunity_matrix(variant_map,
      override)`. `step()` writes `sus_imm`/`symp_imm`/`sev_imm` for **ever-recovered** agents (finite
      `ti_recovered`) from `matrix[v, recovered_variant]`. [VERIFY] `ss.Connector` base + slot-6 timing.
- [ ] `build_immunity_matrix` in `immunity.py`: `np.ones((nv,nv))` then `matrix[target,source] =
      get_cross_immunity()[label_target][label_source]` (asymmetric, diagonal 1.0). Mirror to `sim['immunity']`.
- [ ] Add the internal `cross_immunity_active` flag (auto-True when `nv>1` / a `CrossImmunity` is attached, Open
      Q B). When active, recovery sets `susceptible=True` again (reinfection enabled); add an **ever-recovered**
      tag (or finite-`ti_recovered` mask). When inactive (M2 path), recovered stay `susceptible=False`.
- [ ] Auto-attach `cv.CrossImmunity` in `cv.Sim` when `nv>1`; allow user override via `connectors=`.
      Export `cv.CrossImmunity` in `__init__.py`.
- [ ] **Tests:** the **exclusivity test** (two variants, overlapping contacts, one step ⇒ each new case has one
      finite `infectious_variant` and ≤1 finite `ti_dead`); cross-immunity reduces heterologous reinfection by the
      matrix factor; **same-variant reinfection == 0** (documented divergence); single-variant runs still
      byte-identical (`cross_immunity_active=False`).
- [ ] **PAUSE FOR CLIFF.** Stage: `covasim/connectors.py` (new), `covasim/immunity.py`, `covasim/covid.py`,
      `covasim/sim.py`, `covasim/__init__.py`, the new tests.

## Task 4: Results bridge + M3 anchor + parity gate + v3.1.8 baseline (CHECK-IN 4)

- [ ] Results bridge (spec §7, Open Q E): in `cv.Sim.finalize`, attach `self.results['variant']` referencing the
      module's sub-dict (v3 top-level path); surface `n_imports`. **Manually scale** the 8 `scale=True` by_variant
      Results by `pop_scale`; add the seed-offset to `cum_infections` + wild `cum_infections_by_variant`; recompute
      `incidence/prevalence_by_variant` denominators at sim level. Defer full flat aggregate-results / `sim.summary`
      compat.
- [ ] Attach the variant **label** at the transmission site for the infection log (spec §8); full TransTree → M9.
- [ ] `tests/regression/anchor_m3.py` (dual-version): multi-variant (wild + alpha@day10 + delta@day30),
      `random`/`hybrid`, the spec's anchor pars. v3 branch runs **with cross-immunity active** (Open Q D).
- [ ] `short_summary.py`: `build_summary_m3` + `METRIC_KEYS_M3` (aggregate `cum_infections`/`peak_prevalence`/
      `peak_n_infectious`/`cum_deaths` + per-variant `cum_infections_by_variant`/`peak_n_infectious_by_variant` for
      wild/alpha/delta + displacement ordering). Dual-version.
- [ ] `multi_seed_v3.py` + `compare.py`: add `--anchor m3_random`/`m3_hybrid`.
- [ ] `tests/test_m3_parity.py` (slow, both backends, skip when baseline absent). Threshold per Open Q D
      (`|z| < 5` with written rationale block, mirroring `test_m2_parity.py`).
- [ ] `.gitignore`: add M3 baseline patterns (`v3_m3_*seeds*.json`, `v4_m3_*seeds*.json`, `anchor_m3_snapshot.json`).
- [ ] `tests/regression/README.md`: add an M3 section (anchor, `build_summary_m3`, baseline-generation command,
      the cross-immunity-active baseline regime + the documented static-vs-NAb divergence).
- [ ] Generate the gitignored v3.1.8 M3 baseline via the worktree (`cd /tmp/cov-v3 && python -c "...sys.path.append
      ...; multi_seed_v3.main(['--anchor','m3_random','--n','30']); ...m3_hybrid..."`).
- [ ] **Demo plot** (`/tmp/m3_demo.png`): per-variant infection curves + aggregate burden (the acceptance demo).
- [ ] **Full suite** under the strict-warnings bar (`COVASIM_INTERACTIVE=0 COVASIM_WARNINGS=error`, `SCIRIS_BACKEND=agg`).
- [ ] **PAUSE FOR CLIFF.** Stage: the results-bridge changes (`sim.py`, `covid.py`), `anchor_m3.py` (new),
      `short_summary.py`, `multi_seed_v3.py`, `compare.py`, `test_m3_parity.py` (new), `.gitignore`, `README.md`.
      Surface the parity result table + the documented divergence + the `|z|` threshold decision (as M2).

## Task 5: End-to-end verification (no commits)

- [ ] `cv.Sim().run()` returns results (single-variant invariant) at every check-in's tree state.
- [ ] `test_baselines.py` unchanged through Tasks 1–3 (byte-identity guard); regenerate only if Task 4's results
      bridge intentionally changes a v4-internal number (with a note).
- [ ] Full non-quarantined suite green; the new M3 tests pass; the slow M3 gate skips cleanly without a baseline.

## Self-review checklist

- [ ] No `git commit`/`git push` issued by the assistant; work left staged at each PAUSE.
- [ ] Every `[VERIFY]` starsim signature checked against the installed package before use.
- [ ] Single-variant (`nv==1`) behavior byte-identical to M2 through Tasks 1–3 (`test_baselines.py`).
- [ ] All 10 adversary punch-list items (spec §"Adversary punch-list") addressed with a test or an explicit note.
- [ ] No NAb/`calc_VE`/waning machinery wired (M4 boundary respected); only the static matrix.
- [ ] Backwards-compat names preserved: `cv.variant`, `cv.get_variant_pars`/`get_variant_choices`/
      `get_cross_immunity`; pars `n_variants`/`variant_map`/`variant_pars`/`variants`; the 12-key `['variant']`
      results contract; `n_imports`.
- [ ] The static-vs-NAb cross-immunity divergence (same-variant reinfection = 0) documented in the baseline note.

## Linked documents

- `../specs/2026-05-29-covasim-m3-multivariant-cross-immunity-design.md` — authoritative design spec.
- `MIGRATION_PLAN.md` §M3 — capability scope (updated to Design B).

## Open questions carried from the spec (flagged at the noted check-in; default path taken in this plan)

- **A** architecture (Design B) — *plan/spec check-in (now)*; **B** reinfection flag — check-in 3; **C** axes —
  check-in 3; **D** baseline regime + threshold — check-in 4; **E** results bridge scope — check-in 4; **F**
  introduction timing — check-ins 2 + 4.
