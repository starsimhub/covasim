# Covasim v4.0 — M4 Waning immunity + NAbs: design spec

> **Status:** drafted 2026-05-29 (autonomous session, after M3 landed). Authoritative for M4.
> Grounded in the v3 NAb engine (`_v2_legacy/immunity.py`), the kept-from-v3 NAb parameters
> (`parameters.py:69-78`), and the M3 landing zone (`cv.CrossImmunity` + the per-variant
> `sus_imm/symp_imm/sev_imm` arrays + host-level state on `cv.COVID`).

## Goal

Port Covasim's neutralizing-antibody (NAb) waning-immunity engine — the single largest net-new
piece of the migration (MIGRATION_PLAN §"Explicit divergence"). NAb levels rise to a peak at
infection then decay along a kinetic curve; protection against (re)infection, symptoms, and severe
disease is a logistic function of the current NAb level weighted by cross-immunity. Demo:
reinfection + immune waning over a long horizon, with NAb-vs-no-waning dynamics matching v3.

**M4 makes the M3 divergence converge.** M3 shipped a *static* cross-immunity (`sus_imm = matrix`
value); v3 weights it by the per-agent NAb titre (`sus_imm = calc_VE(nab × matrix)`). The M3 v3.1.8
baseline was generated with `use_waning=True`, so turning NAbs on in v4 should shrink the M3
per-variant divergence (esp. the late escape variant delta) toward that baseline.

## Architecture decision — `use_waning` gates static (M3) vs NAb-weighted (M4) cross-immunity

The NAb engine is **additive and gated behind `use_waning`** (a public Covasim flag, default in v3
`True`; in the v4 port M2/M3 ran the static path, so v4's `cv.Sim` defaults `use_waning=False` to
preserve M2/M3 byte-identity, and M4 turns it on explicitly):

- **`use_waning=False` (M2/M3 path, unchanged):** no NAb state evolves; `cv.CrossImmunity` writes
  the raw static matrix; `nv==1` attaches no connector ⇒ permanent immunity. **Byte-identical to M3.**
- **`use_waning=True` (M4 path):** NAb state evolves; `cv.CrossImmunity` advances NAb kinetics each
  step and writes `sus_imm/symp_imm/sev_imm = calc_VE(nab × matrix, axis)`. A connector is attached
  whenever `use_waning OR nv>1`, so even single-variant `use_waning=True` gets reinfection (v3:
  `calc_VE(nab × 1.0) < 1` permits same-variant reinfection — the M3 same-variant=0 divergence
  closes here).

This reuses the M3 connector as "the natural home for M4's NAb engine" (MIGRATION_PLAN §M4) and
keeps everything host-level on `cv.COVID` for M6 (vaccines share the NAb pipeline).

## Design decisions

### 1. NAb state — host-level on `cv.COVID`

Three `ss.FloatArr` on `cv.COVID` (the v3 `nab_states`): `peak_nab` (default 0), `nab` (default 0),
`t_nab_event` (default NaN = "no NAb event yet"). One set per host (a host has one NAb pool from all
prior infections/vaccinations), so they are 1D, not variant-indexed (cross-immunity weights them
per-variant in the connector). M6 vaccines write the same arrays.

### 2. NAb acquisition + boosting — at infection, in `set_prognoses` (the v3 `update_peak_nab`)

When `use_waning`, after the symptom branch is drawn, set each new case's peak NAb:
- **No prior NAb** (`peak_nab==0`): draw `init_nab ~ nab_init` (normal, par1=0, par2=2), then
  `peak = 2**init_nab × rel_imm_symp[level] × (1 + nab_eff['alpha_inf_diff'])`, where `level ∈
  {asymp:0.85, mild:1.0, severe:1.5}` is the agent's symptom outcome (v3 `update_peak_nab` symp
  scaling + the `norm_factor` that normalises natural NAbs onto the vaccine scale).
- **Prior NAb** (`peak_nab>0`, i.e. reinfection/boosting): `peak_nab *= nab_boost` (1.5).
- Set `t_nab_event = ti`.
Drawn with an `ss.normal` Dist (CRN), one stable slot. "severe" = reached the severe/critical branch;
"mild" = symptomatic non-severe; "asymp" = asymptomatic.

### 3. Waning kinetics — precomputed kernel, advanced per step (the v3 `update_nab`)

Precompute `nab_kin = precompute_waning(npts, nab_decay)` **once** at init (the linear-growth →
two-phase-exponential-decay `nab_growth_decay`, ported verbatim). Each step in the connector, for
agents with a NAb event (`t_nab_event` finite):
`nab += nab_kin[ti − t_nab_event] × peak_nab`, then clamp to `[0, peak_nab]`.

### 4. NAb → protection — `calc_VE` + `check_immunity`, in `cv.CrossImmunity.step()`

Ported `calc_VE(nab, ax, nab_eff)` = inverse-logit `expit(alpha + beta·log? )` — precisely v3's
`exp(alpha)·nab^beta / (1 + exp(alpha)·nab^beta)`, with `(alpha,beta)` per axis from `nab_eff`.
Each step, after advancing NAb, for each target variant `v` (the v3 `check_immunity`):
```
natural_imm[was_inf] = matrix[v, recovered_variant]      # M3's cross-immunity matrix, reused
effective_nabs = nab × natural_imm                       # vaccine_imm via max() is M6
sus_imm[v]  = calc_VE(effective_nabs, 'sus')
symp_imm[v] = calc_VE(effective_nabs, 'symp')
sev_imm[v]  = calc_VE(effective_nabs, 'sev')
```
`was_inf` = finite `ti_recovered ≤ ti` (M3's ever-recovered mask). `calc_VE(0)=0` for non-was_inf,
so the whole array is well-defined.

### 5. Breakthrough transmissibility — `trans_redux`, in `set_prognoses`

At (re)infection of an agent with prior NAbs (`peak_nab>0` *before* this infection's boost), reduce
its `rel_trans_base` by `trans_redux` (0.59), once per agent (the v3 first-breakthrough rule).

### 6. Reinfection regime

`use_waning ⇒ cross_immunity_active` (recovery restores `susceptible=True`). Same mechanism as M3;
M4 just makes the protection NAb-weighted and time-decaying, so reinfection probability rises as
NAbs wane (the whole point).

### 7. Results — `pop_nabs`, `pop_protection`

Two M4 aggregate results (v3 `result_imm`): `pop_nabs` = mean `nab` over alive agents per step;
`pop_protection` = mean wild-axis `sus_imm` over alive agents. Used by the directional `test_waning`
acceptance and `cv.nab_histogram` (M9).

## Acceptance test

Directional (v3 `test_immunity::test_waning`): with `use_waning=True` vs `False`, `cum_infections`,
`cum_reinfections`, `pop_nabs`, `pop_protection` are all strictly higher with waning. Plus: NAb
trajectories rise-then-decay (kernel shape); and the **M3 anchor re-convergence** — re-running the
M3 multi-variant anchor with `use_waning=True` shrinks the per-variant divergence (esp. delta)
toward the v3.1.8 `use_waning=True` baseline (target: the informational metrics move from |z|~25-46
toward the gate band; documented residual is the remaining CRN/discretisation offset).

## Out of scope for M4 (deferred)

- **Vaccination + vaccine NAbs / per-variant efficacy** — M6 (the `max(natural, vaccine)` branch in
  `check_immunity`, `vaccine_pars`/`vaccine_map`, `target_eff`).
- **`historical_*` / `prior_immunity`** pre-t0 NAb imprinting — M6.
- **`nab_histogram` analyzer** — M9 (M4 only emits `pop_nabs`/`pop_protection`).
- Diagnosed-state clearing on reinfection — M5 (no diagnosis state until then).

## Adversary punch-list (address in the plan)

1. `use_waning=False` stays byte-identical to M3/M2 (no NAb state evolves, static matrix). Gate it.
2. NAb arrays are 1D host state, not variant-indexed (the cross-immunity weighting is per-variant).
3. `nab_kin` index `ti − t_nab_event` must be clamped to the kernel length (long horizons).
4. `calc_VE(0)` must be exactly 0 (so non-recovered agents get no spurious protection); verify `nab^beta` at nab=0.
5. Boosting reads `peak_nab>0` BEFORE writing the new peak (don't boost a just-set peak).
6. Breakthrough `trans_redux` applies once per agent and to `rel_trans_base` (so it persists through the per-step viral-load recompute).
7. `was_inf` keyed off finite `ti_recovered ≤ ti` (reuse M3), not the transient `recovered` BoolState.
8. M3 static path and M4 NAb path share ONE connector; branch on `use_waning` cleanly.

## Linked documents

- `../plans/2026-05-29-covasim-m4-waning-nabs.md` — the task-by-task plan.
- `MIGRATION_PLAN.md` §M4 — capability scope.
- v3 reference: `covasim/_v2_legacy/immunity.py` (the engine being ported).
