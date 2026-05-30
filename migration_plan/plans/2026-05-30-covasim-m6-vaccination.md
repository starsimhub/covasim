# Covasim v4.0 — M6 Vaccination Implementation Plan

> Implements the M6 design spec (`../specs/2026-05-30-covasim-m6-vaccination-design.md`). Additive +
> gated: no vaccination intervention => M1-M5 byte-identical. Vaccines reuse the M4 NAb pipeline.
> Commit at each working increment.

## Task 1: vaccination state + NAb boost generalisation + connector vaccine branch (CHECK-IN 1)

- [ ] Add `vaccinated` (BoolState), `doses`/`vaccine_source`/`date_vaccinated` (FloatArr) states to
      cv.COVID; `vaccine_pars={}`/`vaccine_map={}` registry attributes; a separate `_vacc_nab_init`
      ss.normal Dist (created last). step_die clears `vaccinated`? No -- vaccination persists through
      death-removal; leave it (dead agents removed from active arrays).
- [ ] Generalise `_update_peak_nab(uids, nab_pars=None, symp_scale=None)`: vaccine path uses the
      vaccine's nab_init/nab_boost via `_vacc_nab_init.set(loc, scale)`, no symp scaling/normalisation;
      natural path unchanged (self._nab_init, symp_scale). Verify M4 byte-identity.
- [ ] `covid.vaccinate_agents(uids, vaccine_label, vaccine_index)` action: set vaccinated/vaccine_source
      /doses/date_vaccinated, call `_update_peak_nab(uids, vaccine_pars[label])`, bump new_doses/new_vaccinated.
- [ ] Connector (use_waning): `imm = max(natural_cross_imm, vaccine_eff[v])` over the union of
      ever-recovered + vaccinated; effective_nabs = nab×imm; calc_VE per axis. M4 byte-identical with
      no vaccine. new_doses/cum_doses/new_vaccinated/cum_vaccinated results.
- [ ] Port `calc_VE_symp` into immunity.py (for target_eff). **Commit.**

## Task 2: BaseVaccination + vaccinate_prob/num + vaccinate wrapper + products (CHECK-IN 2)

- [ ] `cv.BaseVaccination(cv.Intervention)`: `_parse_vaccine_pars` (string product via
      get_vaccine_choices/variant_pars/dose_pars, or dict), register vaccine_pars/vaccine_map +
      per-variant efficacy at init, `target_eff` back-calc, `check_doses`, the `vaccinate()` dose action,
      `select_people` (abstract).
- [ ] `cv.vaccinate_prob` (prob on days; 2nd-dose scheduling; booster), `cv.vaccinate_num`
      (num_doses/day; priority sequence; second-dose prioritisation; pop_scale), `cv.vaccinate` wrapper.
      Selection via ss.bernoulli (CRN); num ordering deterministic per (seed, ti).
- [ ] Export cv.BaseVaccination/vaccinate/vaccinate_prob/vaccinate_num. cv.Sim(interventions=[...]) already forwards.
- [ ] Tests: byte-identical w/o vaccine; vaccinate_prob('pfizer') reduces infections/severe/deaths +
      cum_doses/cum_vaccinated; per-variant efficacy; 2-dose scheduling. **Commit.**

## Task 3: simple_vaccine + anchor/gate/baseline + demo (CHECK-IN 3)

- [ ] `cv.simple_vaccine` (non-NAb, use_waning=False): direct rel_sus×rel_sus, symp_prob×rel_symp on
      applied days, cumulative dose handling; stores doses/vaccinated. Export.
- [ ] anchor_m6 (single variant, use_waning=True, vaccinate_prob('pfizer')) dual-version, build_summary_m6
      (cum_infections/cum_severe/cum_deaths/cum_doses/cum_vaccinated + shape), m6 anchors in
      multi_seed_v3/compare, test_m6_parity (|z|<5), .gitignore, README. Generate v3.1.8 baseline.
- [ ] Demo /tmp/m6_demo.png (vaccination reduces the epidemic). Full suite strict-warnings. **Commit.**

## Self-review checklist

- [ ] No vaccination => M1-M5 byte-identical; vaccine NAb draw on a separate Dist (M4 unchanged).
- [ ] imm = max(natural, vaccine); union of ever-recovered + vaccinated; calc_VE(0)=0 elsewhere.
- [ ] Dose scheduling correct (skip fully-dosed, 2nd dose at +interval, boosters target vaccinated).
- [ ] target_eff back-calc matches v3 (calc_VE_symp); vaccine_source indexed by vaccine_map.
- [ ] Parity vs a v3.1.8 vaccination baseline; directional + per-variant-efficacy tests.
