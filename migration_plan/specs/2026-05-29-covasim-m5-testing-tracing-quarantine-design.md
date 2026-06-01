# Covasim v4.0 — M5 Testing / tracing / quarantine: design spec

> **Status:** drafted 2026-05-29 (autonomous session, after M3+M4 landed). A starting-point spec
> grounded in the v3 reference; the implementer should confirm the open questions before/at coding.
> Unlike M0–M4 this was NOT produced via a full adversarial design panel — treat the design as a
> well-reasoned proposal, not a locked decision.

## Goal

Port Covasim's testing, contact-tracing, and quarantine/isolation subsystem (MIGRATION_PLAN §M5).
Demo: run testing + contact tracing + quarantine/isolation and show diagnoses and quarantine counts
over time. Acceptance: testing / diagnoses / quarantine counts match v3.1.8 with these interventions.

## v3 reference surface (mapped)

- `cv.test_num` (`_v2_legacy/interventions.py:713`) — fixed tests/day, symptomatic weighting,
  quarantine policy, ILI prevalence, subtargets, rescaling correction.
- `cv.test_prob` (`:858`) — per-person test probability by symptom + quarantine state
  (`symp_prob`/`asymp_prob`/`symp_quar_prob`/`asymp_quar_prob`/`quar_policy`, `ili_prev`, `swab_delay`,
  `sensitivity`/`loss_prob`/`test_delay`).
- `cv.contact_tracing` (`:984`) — on diagnosis, trace per-layer contacts (`trace_probs`/`trace_time`),
  set `known_contact`, schedule quarantine; capacity limit; no-testing⇒no-effect.
- `People.test` (`people.py:589`) — the testing action → `date_pos_test`/`date_diagnosed`.
- `People.check_diagnosed/check_quar/check_enter_iso/check_exit_iso` (`:315/335/361/368`) — the
  diagnosis + quarantine + isolation state machines.
- `People.schedule_quarantine` (`:620`).
- Pars: `quar_factor`/`iso_factor` (per-layer beta multipliers), `quar_period`.

## Proposed design (Design B continuation)

### 1. Host states (on `cv.COVID`, gated/inert until interventions present)

Add the v3 testing/quarantine states as `ss.BoolState`/`ss.FloatArr` on `cv.COVID` (host-level,
like the disease states): `tested`, `diagnosed`, `known_contact`, `quarantined`, `isolated`
(BoolStates → auto `n_*` results), and the date FloatArrs `date_tested`, `date_pos_test`,
`date_diagnosed`, `date_end_quarantine`, `date_end_isolation`. **All default False/NaN ⇒ inert when no
testing/tracing intervention is present, so M1–M4 are byte-identical.** (Alternative: a separate
host-state module; but keeping them on `cv.COVID` matches M4's host-level NAb/vaccine state and keeps
one place for the per-agent epidemic state — recommended.)

The diagnosis/quarantine/isolation **state machines** (`check_diagnosed`/`check_quar`/
`check_enter_iso`/`check_exit_iso`) run in `cv.COVID.step_state` (date-threshold flips, mirroring the
existing natural-history transitions), gated on whether any testing intervention is attached.

### 2. Interventions infrastructure — restore `cv.Intervention` + `interventions.py`

Create an active `covasim/interventions.py` with `cv.Intervention(ss.Intervention)` (the base, same
public name) and the concrete `cv.test_num`/`cv.test_prob`/`cv.contact_tracing` as `ss.Intervention`
subclasses. They run at the **intervention loop slot (7)** — after the connector (NAb/cross-immunity,
slot 5) and before disease transmission (slot 8) — matching where v3 applied interventions. The
testing interventions select UIDs and call a `covid.test(uids, ...)` action (ported `People.test`);
`contact_tracing` reads the infection-log / network edges and schedules quarantine.

### 3. `quar_factor` / `iso_factor` — beta modifiers

v3 multiplies a quarantined/isolated agent's per-layer transmissibility + susceptibility by
`quar_factor[layer]` / `iso_factor[layer]`. In v4 the cleanest hook is the disease's `infect()`
(which already folds per-agent `rel_trans`/`rel_sus`): multiply `rel_trans`/`rel_sus` for
quarantined/isolated agents by the (layer-aware) factor. **Open question A:** v3's factors are
per-layer, but `infect()` loops layers — applying a per-layer quar factor needs the factor inside the
network loop (per (layer, agent)). Simplest M5: a single scalar quar/iso reduction on `rel_trans`/
`rel_sus` (layer-averaged); full per-layer fidelity may need the factor threaded into the network
loop. Decide vs the v3 `compute_trans_sus` per-layer application.

### 4. CRN / sampling

Testing draws (who gets tested, sensitivity, loss) use `ss.bernoulli`/`ss.Dist` (CRN), one stable
slot per intervention, replacing v3's `cvu.n_binomial`/global RNG.

## Acceptance test

Multi-seed parity (the established harness): an M5 anchor (the M2/M3 scenario + `test_prob` +
`contact_tracing`) vs a v3.1.8 baseline, gating `cum_tests`/`cum_diagnoses`/`n_quarantined`/
`n_isolated` (+ the burden metrics) at `|z| < 5`. Plus directional unit tests (testing ⇒ diagnoses;
tracing ⇒ quarantine; quarantine reduces transmission).

## Out of scope for M5 (deferred)

- Vaccination (M6); calibration/Fit (M7); analyzers/TransTree (M9).
- `sequence` intervention scheduler (M5 or opportunistic — small).

## Open questions for Cliff

- **A — per-layer quar/iso factor application** (see §3): scalar M5 approximation vs full per-layer
  threading into `infect()`. Recommendation: start scalar, note the residual, refine if the parity
  gate demands it.
- **B — where the testing/quarantine states live**: on `cv.COVID` (recommended, matches M4) vs a
  separate host-state module.
- **C — `test_num` rescaling correction** under the absolute-agent-count model (no dynamic rescaling
  yet): confirm the tests/day accounting vs `pop_scale`.

## Linked documents

- `../plans/2026-05-29-covasim-m5-testing-tracing-quarantine.md` — task-by-task plan.
- `MIGRATION_PLAN.md` §M5 — capability scope. v3 reference: `_v2_legacy/{interventions,people}.py`.
