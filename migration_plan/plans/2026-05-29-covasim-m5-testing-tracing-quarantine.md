# Covasim v4.0 — M5 Testing / tracing / quarantine Implementation Plan

> Implements the M5 design spec (`../specs/2026-05-29-covasim-m5-testing-tracing-quarantine-design.md`).
> Additive: the testing/quarantine states default inert, so M1–M4 stay byte-identical until a testing
> or tracing intervention is attached. Commit at each working increment.

**Architecture:** host testing/quarantine states + state machines on `cv.COVID`; a restored active
`covasim/interventions.py` with `cv.Intervention(ss.Intervention)` and `cv.test_num`/`cv.test_prob`/
`cv.contact_tracing` running at loop slot 7; `quar_factor`/`iso_factor` fold into `infect()`.

## Task 1: Host states + diagnosis/quarantine/isolation state machines (CHECK-IN 1)

- [ ] Add `tested`/`diagnosed`/`known_contact`/`quarantined`/`isolated` BoolStates + the date FloatArrs
      (`date_tested`/`date_pos_test`/`date_diagnosed`/`date_end_quarantine`/`date_end_isolation`) to
      `cv.COVID` (default False/NaN ⇒ inert). Add `quar_factor`/`iso_factor`/`quar_period` pars.
- [ ] Port `check_diagnosed`/`check_quar`/`check_enter_iso`/`check_exit_iso` into `step_state`
      (date-threshold flips), gated so they no-op when no testing intervention is present.
- [ ] Port the `covid.test(uids, sensitivity, loss_prob, test_delay)` action (the v3 `People.test`):
      true-positive draw (ss.bernoulli), `date_pos_test`/`date_diagnosed`.
- [ ] Tests: states default inert ⇒ M2/M3/M4 byte-identical; direct `test()` ⇒ diagnoses. **Commit.**

## Task 2: `cv.Intervention` base + `cv.test_num` / `cv.test_prob` (CHECK-IN 2)

- [ ] New active `covasim/interventions.py`: `cv.Intervention(ss.Intervention)`; export.
- [ ] Port `cv.test_prob` (per-person prob by symptom/quarantine state, ili_prev, swab_delay) and
      `cv.test_num` (fixed tests/day, symp weighting, quar policy, subtargets, rescaling) as
      `ss.Intervention` subclasses running at slot 7; they call `covid.test(...)`.
- [ ] Wire `cv.Sim(interventions=[...])` (forward to ss.Sim). Confirm slot ordering (after connector,
      before transmission).
- [ ] Tests: `test_prob`/`test_num` produce diagnoses; cum_tests/cum_diagnoses results. **Commit.**

## Task 3: `cv.contact_tracing` + quar/iso beta modifiers + anchor/gate (CHECK-IN 3)

- [ ] Port `cv.contact_tracing` (trace per-layer contacts of the newly-diagnosed via the network edges,
      `trace_probs`/`trace_time`, set `known_contact`, `schedule_quarantine`; capacity limit).
- [ ] Apply `quar_factor`/`iso_factor` in `infect()` (reduce quarantined/isolated agents' rel_trans/
      rel_sus; see spec Open Q A re per-layer fidelity).
- [ ] M5 anchor (M2/M3 scenario + test_prob + contact_tracing) + build_summary_m5 (cum_tests/
      cum_diagnoses/n_quarantined/n_isolated + burden); test_m5_parity.py; v3.1.8 baseline; README.
- [ ] Tests: diagnoses/quarantine counts match v3.1.8; quarantine reduces transmission. **Commit.**

## Self-review checklist

- [ ] States inert ⇒ M1–M4 byte-identical (verify against the M2/M3/M4 anchors).
- [ ] Testing draws use ss.bernoulli (CRN), one stable slot per intervention.
- [ ] Intervention slot ordering: after connector (slot 5), before transmission (slot 8).
- [ ] Parity vs a v3.1.8 baseline with the same interventions wired.
