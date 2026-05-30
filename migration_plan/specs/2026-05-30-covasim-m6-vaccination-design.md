# Covasim v4.0 — M6 Vaccination: design spec

> **Status:** drafted 2026-05-30 (autonomous session, after M3/M4/M5 landed). Grounded in the v3
> vaccination code (`_v2_legacy/interventions.py`: `BaseVaccination`/`vaccinate_prob`/`vaccinate_num`/
> `simple_vaccine`), the kept-from-v3 vaccine parameters (`parameters.py`: `get_vaccine_choices`/
> `get_vaccine_variant_pars`/`get_vaccine_dose_pars`), and the M4 NAb engine (the connector's
> `check_immunity`, which already has the `max(natural, vaccine)` slot ported-but-natural-only).

## Goal

Port vaccination: campaigns (`vaccinate_prob`/`vaccinate_num`/`vaccinate`), NAb-based vaccine products
with per-variant efficacy (pfizer/moderna/az/jj/novavax/sinovac/sinopharm), dose scheduling, boosters,
`target_eff` back-calculation, and the non-NAb `simple_vaccine`. Demo: a vaccination campaign reduces
infections/severe/deaths, including per-variant efficacy. Acceptance: the vaccination-impact
trajectory matches v3.1.8.

**Vaccines are the last consumer of the M4 NAb pipeline.** A vaccine dose is a NAb-conferring event
(exactly like a natural infection): it calls the M4 `update_peak_nab` with the *vaccine's*
`nab_init`/`nab_boost` (no symptom scaling, no natural normalisation), and the connector's
`check_immunity` already combines natural + vaccine immunity as `imm = max(natural_cross, vaccine_eff)`
then `effective_nabs = nab × imm`. So M6 is mostly: (a) host vaccination state, (b) wiring the vaccine
branch in the connector, (c) the vaccination interventions, (d) `simple_vaccine`.

## Design decisions

### 1. Host vaccination state on `cv.COVID` (gated/inert; v3 vacc_states)

`vaccinated` (`ss.BoolState` -> auto `n_vaccinated`), `doses` (`ss.FloatArr`, default 0),
`vaccine_source` (`ss.FloatArr`, default NaN = index of the vaccine an agent received), `date_vaccinated`
(`ss.FloatArr`). All default empty, so M1-M5 are byte-identical with no vaccination intervention.

### 2. Vaccine registry on `cv.COVID`

`vaccine_pars` (dict label -> per-vaccine pars incl. per-variant efficacy + `nab_init`/`nab_boost`/
`doses`/`interval`) and `vaccine_map` (dict index -> label), mirroring `variant_pars`/`variant_map`.
Populated by each vaccination intervention at init (each vaccine gets a stable `index`).

### 3. NAb boost generalised — `_update_peak_nab(uids, nab_pars=None, symp_scale=None)`

The M4 method becomes parameterised:
- `nab_pars=None` (natural infection): use `self.pars` `nab_init`/`nab_boost`, scale by `symp_scale`
  (rel_imm_symp) and the natural-normalisation `1 + alpha_inf_diff` (unchanged M4 behaviour).
- `nab_pars=<vaccine pars>` (a dose): use the vaccine's `nab_init`/`nab_boost`, **no** symptom scaling,
  **no** natural normalisation (v3 `update_peak_nab` with `symp=None`).
The initial draw uses a **separate** `ss.normal` Dist `self._vacc_nab_init` (`.set(loc, scale)` per
vaccine), distinct from the natural `self._nab_init`, so a vaccination dose drawn at the intervention
slot does not perturb the natural-infection NAb draw at the transmission slot -> **M4 stays
byte-identical when no vaccine is present**.

### 4. Connector vaccine-immunity branch (the M4 `check_immunity` `max`)

`cv.CrossImmunity.step()` (under `use_waning`) currently writes, for every ever-recovered agent,
`sus_imm = calc_VE(nab × matrix[v, recovered_variant])`. M6 extends the per-agent immunity weight to
the **union of ever-recovered and vaccinated** agents: for each target variant `v`,
`imm = max(natural_cross_imm, vaccine_efficacy[v])`, where `vaccine_efficacy[v] =
vaccine_pars[vaccine_map[vaccine_source]][variant_label]`; then `effective_nabs = nab × imm` and
`sus_imm/symp_imm/sev_imm = calc_VE(effective_nabs, axis)`. With **no** vaccine registered the vaccine
term is 0 and `max(natural, 0) = natural` over exactly the ever-recovered set -> **M4 byte-identical**.

### 5. Vaccination interventions (on `cv.Intervention`)

- `cv.BaseVaccination(cv.Intervention)`: parse vaccine (string product or dict), register into
  `vaccine_pars`/`vaccine_map` at init, `target_eff` back-calculation (via `calc_VE_symp`, ported into
  `immunity.py`), and the dose action (`vaccinate()`): skip dead / already-fully-dosed, set
  `vaccinated`/`vaccine_source`/`doses`/`date_vaccinated`, call `_update_peak_nab(uids, vaccine_pars)`,
  bump `new_doses`/`new_vaccinated`.
- `cv.vaccinate_prob(BaseVaccination)`: per-day probability `prob` on matched `days`; schedules the
  2nd dose at `+interval`; booster targets the vaccinated.
- `cv.vaccinate_num(BaseVaccination)`: fixed `num_doses`/day in a priority `sequence`, second doses
  prioritised; `pop_scale` dose accounting.
- `cv.vaccinate(*a, **kw)`: wrapper -> `vaccinate_num` if `num_doses` given, else `vaccinate_prob`.
Selection draws use `ss.bernoulli` (CRN); `vaccinate_num` ordering uses a deterministic per-(seed, ti)
stream (as M5 `test_num`).

### 6. `cv.simple_vaccine` — the non-NAb path (`use_waning=False`)

Direct modification of `rel_sus` (× `rel_sus` factor) and the symptomatic probability (× `rel_symp`)
on the day(s) applied, with `cumulative` dose handling. Does **not** go through the NAb pipeline
(preserves the v3 `use_waning=False` behaviour and public API). Stores `doses`/`vaccinated`.

### 7. Results

`new_doses`/`cum_doses`, `new_vaccinated`/`cum_vaccinated` flows (bumped in `vaccinate()`); the
`n_vaccinated` stock is auto from the BoolState. `pop_protection` (M4) now reflects vaccine immunity too.

## Acceptance test

Multi-seed parity: an M6 anchor (single-variant, `use_waning=True`, `vaccinate_prob('pfizer', ...)`)
vs a v3.1.8 baseline, gating `cum_infections`/`cum_severe`/`cum_deaths`/`cum_doses`/`cum_vaccinated` +
shape at `|z|<5`. Plus directional tests (vaccination reduces infections/severe/deaths; per-variant
efficacy: a vaccine protects more against wild than an escape variant; `simple_vaccine` reduces
susceptibility without waning).

## Out of scope for M6 (deferred)

- `historical_vaccinate_prob` / `historical_wave` / `prior_immunity` (pre-t0 NAb imprinting) — these
  depend on running NAb kinetics before t=0; defer (note for a follow-up).
- `nab_histogram` analyzer — M9.
- Full per-layer quar/iso factors (M5 Open Q A) — unrelated.

## Adversary punch-list

1. No vaccination intervention => M1-M5 byte-identical (no NAb draw, no vaccine term in the connector).
2. The vaccine NAb draw uses a SEPARATE Dist from the natural draw (no CRN cross-talk) -> M4 byte-identical.
3. `_update_peak_nab` vaccine path: no symp scaling / no natural normalisation; boost reads peak_nab>0 before writing.
4. Connector: `imm = max(natural, vaccine)`; union of ever-recovered + vaccinated; `calc_VE(0)=0` for the rest.
5. Dose scheduling: skip already-fully-dosed (per *this* vaccine), 2nd dose at `+interval`; boosters target vaccinated.
6. `target_eff` back-calculation matches v3 (calc_VE_symp on a NAb grid).
7. `vaccine_source` indexing in the connector is by the vaccine index (vaccine_map), not array position.

## Linked documents

- `../plans/2026-05-30-covasim-m6-vaccination.md` — task-by-task plan.
- `MIGRATION_PLAN.md` §M6. v3 reference: `_v2_legacy/interventions.py` + `parameters.py` vaccine fns.
