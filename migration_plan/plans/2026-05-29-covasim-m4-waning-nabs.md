# Covasim v4.0 — M4 Waning immunity + NAbs Implementation Plan

> Implements the M4 design spec (`../specs/2026-05-29-covasim-m4-waning-nabs-design.md`, authoritative).
> Additive + gated behind `use_waning`: `use_waning=False` stays byte-identical to M3/M2; `use_waning=True`
> turns on the NAb engine. Commit at each working increment (autonomous session).

**Architecture:** NAb state is host-level on `cv.COVID`; the NAb kinetics + `calc_VE` mapping live in
`cv.CrossImmunity.step()` (the M3 connector, reused — "fold into cv.CrossImmunity"). Vaccines are M6.

## Task 1: Port the dormant NAb engine functions into `covasim/immunity.py` (CHECK-IN 1)

- [ ] Port `precompute_waning` + `nab_growth_decay` + `nab_decay` + `exp_decay` from `_v2_legacy/immunity.py`
      verbatim (numpy, no global state) → produce the `nab_kin` kernel array of length npts.
- [ ] Port `calc_VE(nab, ax, nab_eff)` verbatim (inverse-logit; `(alpha,beta)` per axis).
- [ ] Unit tests: kernel rises over `growth_time` then decays (two phases); `calc_VE(0)==0`,
      `calc_VE` monotonic increasing in nab, in [0,1).
- [ ] Imported but NOT wired yet (the M3-sanctioned landing zone). **Commit.**

## Task 2: NAb state + acquisition/boosting on `cv.COVID`, `use_waning` wiring (CHECK-IN 2)

- [ ] Add `peak_nab` (default 0), `nab` (default 0), `t_nab_event` (default NaN) `ss.FloatArr` states.
- [ ] Add `use_waning` to `cv.COVID` pars (default False) + NAb pars (`nab_init`/`nab_decay`/`nab_boost`/
      `nab_eff`/`rel_imm_symp`/`trans_redux`) from `parameters.py`. Add an `ss.normal` init_nab Dist.
- [ ] In `set_prognoses`, when `use_waning`: classify each new case's symptom level (asymp/mild/severe
      from the branch masks), call an `_update_peak_nab(uids, symp_level)` helper: boost prior-NAb agents
      (`peak_nab>0`) by `nab_boost`; draw `init_nab` + scale for the rest; set `t_nab_event=ti`. Apply the
      breakthrough `trans_redux` to `rel_trans_base` for first-breakthrough agents.
- [ ] `cv.Sim(use_waning=False)` param → forwarded to `cv.COVID`; attach `cv.CrossImmunity` when
      `use_waning OR nv>1`. Precompute `nab_kin` at init (needs npts).
- [ ] Tests: `use_waning=False` byte-identical to M3 (the M2/M3 guard); `use_waning=True` single-variant
      run evolves NAbs (peak set at infection, nonzero). **Commit.**

## Task 3: NAb-weighted cross-immunity in the connector + results + acceptance (CHECK-IN 3)

- [ ] `cv.CrossImmunity.step()` branches on `covid.pars.use_waning`:
      - static (M3): `sus_imm = matrix[v, src]` (unchanged).
      - NAb (M4): advance `nab += nab_kin[clamp(ti − t_nab_event)] × peak_nab` (clamp [0,peak]); then
        `effective_nabs = nab × matrix[v, src]`; `sus_imm/symp_imm/sev_imm[v] = calc_VE(effective_nabs, axis)`.
- [ ] `pop_nabs` (mean nab) + `pop_protection` (mean wild sus_imm) results, written in `update_results`.
- [ ] Tests: directional `test_waning` (waning ⇒ higher cum_infections/cum_reinfections/pop_nabs/
      pop_protection than no-waning); NAb rise-then-decay; M3-anchor re-convergence with `use_waning=True`
      (delta divergence shrinks toward the v3 baseline). **Commit.**

## Self-review checklist

- [ ] `use_waning=False` byte-identical to M3 (test_baselines-style guard + M3 anchor unchanged).
- [ ] `nab_kin` index clamped; `calc_VE(0)==0`; boost reads peak BEFORE writing; trans_redux once per agent.
- [ ] No vaccine machinery wired (M6 boundary); natural NAbs only.
- [ ] Re-convergence quantified vs the v3 `use_waning=True` baseline; residual documented.
