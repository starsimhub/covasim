# Covasim v4.0 — M1 Basic-Transmission Implementation Plan

> **For agentic workers:** Implement this plan task-by-task; steps use checkbox (`- [ ]`) syntax for tracking. No special plugin or sub-skill is required.
>
> **CRITICAL — commit discipline.** This is a local effort with a **pause-for-review-and-commit** cadence. The assistant (Claude) **prepares and stages** each piece of work and then **pauses for Cliff Kerr (the Covasim author) to review and commit**. **The assistant NEVER runs `git commit` and NEVER runs `git push`.** Where this plan reaches a check-in boundary it says **PAUSE FOR CLIFF** and lists what to stage; it does not contain `git commit` commands. Check-ins happen 4 times across M1 (the four boundaries below).
>
> **VERIFY-AT-CODE-TIME discipline.** Several Starsim 3.3.x signatures are quoted from the API findings (starsim 3.3.4 at `/home/cliffk/idm/starsim/starsim/`). They are correct as of writing, but the implementer **MUST verify each one against the installed `starsim` before relying on it** — every such point is flagged inline with **[VERIFY]**. Do NOT guess an API name; read the real source (`python -c "import starsim as ss, inspect; print(inspect.getsource(ss.Network.append))"`) or grep the installed package.

**Goal:** Land the first real port code in `covasim/`: a thin `cv.Sim(ss.Sim)` + `cv.People(ss.People)`, a `cv.Network(ss.Network)` lift-and-shift of `population.py` (random single-layer + hybrid h/s/w/c, with per-layer beta and age restrictions), and a minimal single-variant `cv.COVID(ss.Infection)` doing S→E→I→R transmission + recovery via stock CRN-safe `ss.Infection.infect()`. Quarantine the v3 modules M1 replaces into `covasim/_v2_legacy/` and their tests into `tests/_legacy/`, retire the M0 `cv.v4` stub, and reuse the M0 regression harness for a new M1 anchor + contact-structure-equivalence + infection-trajectory parity gates. The only hard invariant at every commit remains the **continuous-runnability invariant**: `cv.Sim().run()` returns results.

**Architecture:** Starsim composes a sim from modules. `cv.Sim(ss.Sim)` assembles a `cv.People`, one or more `cv.Network` instances (one per contact layer — the hpvsim M01 multi-instance pattern), and one `cv.COVID` disease, then forwards to `ss.Sim`. `cv.Network` subclasses **`ss.Network`** (static, not `ss.DynamicNetwork`) and builds its edgelist in `add_pairs()` from the ported `population.py` builders, ending in `self.append(p1=, p2=, beta=)`. `cv.COVID` subclasses `ss.Infection`, overriding only `__init__`, `set_prognoses`, and `step_state`; transmission is the **inherited** stock `ss.Infection.infect()` (the key M1 insight — no custom transmission code). Per-layer beta lives on the disease's `beta` dict keyed by network label (the `betamap`), not on the network. The M0 harness (`tests/regression/{anchor.py,short_summary.py,parity.py,multi_seed_v3.py,compare.py}`) is reused; M1 adds `anchor_m1.py` and a contact-structure metrics helper.

**Tech Stack:** Python 3.9–3.13, pytest + pytest-xdist, covasim (v3.1.8 engine being replaced on `starsim-port`), starsim 3.3.x, sciris.

**Authority:** the design spec `migration_plan/specs/2026-05-29-covasim-m1-basic-transmission-design.md` (it is authoritative for every decision). The capability scope is `MIGRATION_PLAN.md` §M1. This plan implements exactly that scope.

**Class names — LOCKED by Cliff (2026-05-29):** `cv.COVID` (disease module, file `covasim/covid.py`) and `cv.Network` (the per-layer network); `cv.People`/`cv.Sim` keep their names. Spec Open question H is resolved. Open questions A–G proceed on the recommended defaults (each still flagged at its check-in).

**Spec open questions adopted as the default path here** (each flagged for Cliff at the relevant check-in; if Cliff rules otherwise, adjust): A=(a) keep `population.py` builders as free functions feeding `cv.Network.add_pairs`; B=build with `ss.Dist` draws, defer `ss.RandomSafeNet`; C=add `exposed`/`infectious`/`recovered` fresh with `reset=True` + `infectious` property override; D=keep `rel_trans=rel_sus=1.0` at M1; E=exact-count `pop_infected` seed; F=keep `base.py` active with dormant `BaseSim`/`BasePeople` and name-preserved `cv.Layer`/`cv.Contacts`/`cv.Result`/`cv.ParsObj`; G=skip (not quarantine) the v3 bit-for-bit assertions in `test_baselines.py`/`test_regression.py`.

---

## Starting state (confirmed at plan time)

- Branch: `starsim-port` (checked out; the assistant never creates/switches branches).
- M0 is landed: `covasim/_v4.py` (the `cv.v4.Sim` stub), `covasim/_v2_legacy/__init__.py` (empty quarantine), `tests/_legacy/__init__.py` (empty quarantine), the `tests/regression/` harness package (`anchor.py`, `short_summary.py`, `parity.py`, `multi_seed_v3.py`, `multi_seed_v4.py`, `compare.py`, `README.md`), `tests/test_m0_parity.py` (slow gate), `tests/test_regression_harness.py`, `tests/test_v4_stub.py`. There is a local gitignored `tests/regression/v3_seeds_n30.json` (the M0 baseline).
- `covasim/__init__.py` imports the full v3 stack flat (`from .sim import *`, …) and ends with `from . import _v4 as v4`.
- `covasim/` modules: `analysis, base, defaults, immunity, interventions, misc, parameters, people, plotting, population, requirements, run, settings, sim, utils, version` + `data/`, `regression/`.
- `base.py` defines `ParsObj` (63), `Result` (117), `BaseSim` (203), `BasePeople` (877), `Contacts` (1509), `Layer` (1610).
- `population.py` `__all__` = `make_people, make_randpop, make_random_contacts, make_microstructured_contacts, make_hybrid_contacts, make_synthpop`.
- There is **no** `tests/conftest.py` yet. CI runs `pytest -v test_*.py unittests/test_*.py -n auto` from `tests/`; `run_tests` runs `pytest test_*.py -n auto` with `COVASIM_WARNINGS=error` and `COVASIM_INTERACTIVE=0`.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `covasim/sim.py` | **git mv → `_v2_legacy/sim.py`**, then Create new | New thin `cv.Sim(ss.Sim)` assembling the M1 stack; v3 `Sim`/`BaseSim` engine quarantined |
| `covasim/people.py` | **git mv → `_v2_legacy/people.py`**, then Create new | New thin `cv.People(ss.People)` (keeps name, supplies v3 age distribution); v3 health-state machine quarantined |
| `covasim/_v2_legacy/sim.py` | Move (git mv) | Quarantined v3 `Sim`/`BaseSim`/integration loop |
| `covasim/_v2_legacy/people.py` | Move (git mv) | Quarantined v3 `People` health-state machine |
| `covasim/_v2_legacy/analysis.py` | Move (git mv) | Analyzers / `Fit` / `Calibration` / `TransTree` — M7/M9 |
| `covasim/_v2_legacy/immunity.py` | Move (git mv) | Variants / waning / NAbs / cross-immunity — M3/M4 |
| `covasim/_v2_legacy/interventions.py` | Move (git mv) | Testing / tracing / quarantine / vaccination — M5/M6 |
| `covasim/_v2_legacy/run.py` | Move (git mv) | `MultiSim` / `Scenarios` / parallel — M8 |
| `covasim/_v2_legacy/plotting.py` | Move (git mv) | v3 plotting — M9 (M1 uses Starsim `sim.plot()`) |
| `covasim/network.py` | Create | `cv.Network(ss.Network)` — per-layer static network (random + hybrid backends) |
| `covasim/covid.py` | Create | `cv.COVID(ss.Infection)` — minimal single-variant S→E→I→R |
| `covasim/population.py` | Modify (kept active) | Port builders to feed `cv.Network`; keep `cv.make_*` names; guard `make_synthpop` (M9) |
| `covasim/base.py` | Keep active (no move) | `cv.Layer`/`cv.Contacts`/`cv.Result`/`cv.ParsObj` exported for name-preservation; `BaseSim`/`BasePeople` dormant (Open question F) |
| `covasim/__init__.py` | Modify | Trim to the M1 surface; add `network`/`people`/`covid`/`sim` imports; retire `cv.v4` stub |
| `covasim/_v4.py` | **Delete** | M0 stub retired (real `cv.Sim` now is the Starsim-based one) |
| `tests/conftest.py` | Create | `collect_ignore_glob = ['_legacy/*', 'devtests/*']` |
| `tests/_legacy/test_analysis.py` | Move (git mv from `tests/`) | Quarantined v3 test |
| `tests/_legacy/test_immunity.py` | Move (git mv) | Quarantined v3 test |
| `tests/_legacy/test_interventions.py` | Move (git mv) | Quarantined v3 test |
| `tests/_legacy/test_run.py` | Move (git mv) | Quarantined v3 test |
| `tests/_legacy/test_resume.py` | Move (git mv) | Quarantined v3 test |
| `tests/_legacy/test_other.py` | Move (git mv) | Quarantined v3 test |
| `tests/_legacy/test_sim.py` | Move (git mv) | Quarantined v3 `Sim`/`People` test; new M1 `test_sim.py` replaces it |
| `tests/_legacy/test_parameters.py` | Move (git mv) | Quarantined (drives v3 `Sim`); a thin M1 `test_parameters.py` may be added later |
| `tests/_legacy/test_utils.py` | Move (git mv) | Quarantined (drives removed v3 numba kernels) |
| `tests/test_v4_stub.py` | **Delete** | Guards the retired stub |
| `tests/test_network.py` | Create | `cv.Network` contact-structure equivalence (per-layer degree + age-mixing) |
| `tests/test_covid.py` | Create | `cv.COVID` functional tests (minimal `ss.Sim` + stock `ss.RandomNet`) |
| `tests/test_sim.py` | Create (new M1 file) | M1 `cv.Sim`/`cv.People` assembly + continuous-runnability |
| `tests/regression/anchor_m1.py` | Create | M1 single-variant anchor (`PARS_RANDOM`/`PARS_HYBRID`, `make_sim`, `run_and_summarize`) |
| `tests/regression/contact_stats.py` | Create | `degree_by_layer` + `age_mixing_matrix` helpers + comparison fns |
| `tests/test_m1_parity.py` | Create | `@pytest.mark.slow` z-score trajectory parity gate vs v3.1.8 M1 baseline |
| `tests/test_baselines.py` | Modify | Skip v3 bit-for-bit assertions (Open question G) |
| `tests/test_regression.py` | Modify | Skip v3 bit-for-bit assertions (Open question G) |
| `.gitignore` | Modify | Add `tests/regression/v3_m1_seeds_n*.json` + `v4_m1_seeds*.json` |

The generated M1 baselines (`tests/regression/v3_m1_seeds_n*.json`) are gitignored and stay local-only.

---

## Task 0: Quarantine + in-place replacement scaffold (CHECK-IN 1)

This is the hpvsim-M01 quarantine move: a single reviewable unit of `git mv`s (preserving history) + the new-empty-shells so the tree still imports, the `cv.v4` retirement, the `__init__.py` trim, and `tests/conftest.py`. **No new behavioral code lands here** beyond the smallest possible `cv.Sim`/`cv.People`/`cv.Network`/`cv.COVID` shells needed to keep `import covasim` working — those shells are fleshed out in Tasks 1–3. (If preferred, write the full shells here so `import covasim` works at this check-in; the design lets the quarantine be reviewed first.)

- [ ] **Step 1: Confirm the starting point**

```bash
git -C /home/cliffk/idm/covasim status --short
git -C /home/cliffk/idm/covasim branch --show-current   # expect: starsim-port
git -C /home/cliffk/idm/covasim rev-parse --short HEAD
```

Record the HEAD short SHA for the report. If not on `starsim-port`, stop and ask Cliff.

- [ ] **Step 2: `git mv` the v3 modules M1 does not touch into `covasim/_v2_legacy/`**

These are untouched-by-M1 subsystems (spec decision 6 "RECOMMENDED — quarantine these"):

```bash
cd /home/cliffk/idm/covasim
git mv covasim/analysis.py      covasim/_v2_legacy/analysis.py
git mv covasim/immunity.py      covasim/_v2_legacy/immunity.py
git mv covasim/interventions.py covasim/_v2_legacy/interventions.py
git mv covasim/run.py           covasim/_v2_legacy/run.py
git mv covasim/plotting.py      covasim/_v2_legacy/plotting.py
```

- [ ] **Step 3: `git mv` the two replaced engine modules into `_v2_legacy/`**

`sim.py` and `people.py` are **replaced in place** (spec decision 6 "THE THREE CONSEQUENTIAL CASES"). Move the v3 originals out before writing the new thin versions:

```bash
cd /home/cliffk/idm/covasim
git mv covasim/sim.py    covasim/_v2_legacy/sim.py
git mv covasim/people.py covasim/_v2_legacy/people.py
```

Note: **`base.py`, `parameters.py`, `defaults.py`, `settings.py`, `misc.py`, `utils.py`, `version.py`, `requirements.py`, `population.py`, `data/`, `regression/` stay active** (spec decision 6 "RECOMMENDED — keep active" + the `base.py` Open question F decision). `population.py` is kept active and rewritten in Task 1.

- [ ] **Step 4: `git mv` the v3 tests that exercise removed/replaced APIs into `tests/_legacy/`**

Per spec decision 6 "v3 tests → `tests/_legacy/`": move whole files (no partial-file surgery; fresh M1 test files are written in Tasks 1–4).

```bash
cd /home/cliffk/idm/covasim
git mv tests/test_analysis.py      tests/_legacy/test_analysis.py
git mv tests/test_immunity.py      tests/_legacy/test_immunity.py
git mv tests/test_interventions.py tests/_legacy/test_interventions.py
git mv tests/test_run.py           tests/_legacy/test_run.py
git mv tests/test_resume.py        tests/_legacy/test_resume.py
git mv tests/test_other.py         tests/_legacy/test_other.py
git mv tests/test_sim.py           tests/_legacy/test_sim.py
git mv tests/test_parameters.py    tests/_legacy/test_parameters.py
git mv tests/test_utils.py         tests/_legacy/test_utils.py
```

Keep active at the `tests/` root: `test_baselines.py`, `test_regression.py` (with v3 assertions skipped — Step 8), `test_m0_parity.py`, `test_regression_harness.py`, and the new M1 tests (added Tasks 1–4). **`tests/unittests/`** is a separate suite — read its contents first (`ls tests/unittests/`) and quarantine any file there that drives the removed v3 `Sim`/`People`/intervention APIs; flag the list for Cliff. [VERIFY: inspect `tests/unittests/` at code-time.]

- [ ] **Step 5: Delete the M0 `cv.v4` stub and its test (spec decision 7)**

```bash
cd /home/cliffk/idm/covasim
git rm covasim/_v4.py
git rm tests/test_v4_stub.py
```

- [ ] **Step 6: Create `tests/conftest.py` to suppress collection of the quarantines**

```python
"""Pytest configuration for the Covasim test suite.

During the v4.0 Starsim port, v3 tests that exercise removed/replaced APIs are
quarantined under tests/_legacy/. This stops pytest from collecting (and erroring
on) them when a bare `pytest` or `pytest .` is run. devtests/ are developer
scratch tests and are likewise not part of the suite.
"""
collect_ignore_glob = ['_legacy/*', 'devtests/*']
```

- [ ] **Step 7: Trim `covasim/__init__.py` to the M1 surface**

Read the current `covasim/__init__.py` (the 34-line file ending in `from . import _v4 as v4`). Rewrite the model-import block so it (a) keeps the stable/active modules, (b) drops the quarantined ones, (c) adds the new M1 modules, (d) removes the `cv.v4` line. Target import block (preserve the `settings`/`version`/license preamble lines 1–15 verbatim):

```python
# Import the actual model
from .defaults      import * # Depends on settings
from .misc          import * # Depends on version
from .parameters    import * # Depends on settings, misc
from .utils         import * # Depends on defaults
from .base          import * # Depends on version, misc, defaults, parameters, utils
                             # (keeps cv.Layer / cv.Contacts / cv.Result / cv.ParsObj; BaseSim/BasePeople dormant)
from .population    import * # Population builders (cv.make_randpop / cv.make_hybrid_contacts / ...)
from .network       import * # NEW: cv.Network(ss.Network)
from .covid         import * # NEW: cv.COVID(ss.Infection)
from .people        import * # NEW thin cv.People(ss.People)
from .sim           import * # NEW thin cv.Sim(ss.Sim)
```

Removed vs M0: `plotting`, `interventions`, `immunity`, `analysis`, `run`, and the `from . import _v4 as v4` line. **Order matters** — `base`/`population` before `network`/`covid`/`people`/`sim`. Document the milestone-staged API expansion in the module docstring (hpvsim convention), e.g. "M1: cv.Sim, cv.People, cv.Network, cv.COVID + population builders. M2+: natural history, variants, immunity, interventions, analyzers restored from `_v2_legacy/`."

  - **[VERIFY] import-time hazard:** `base.py`/`population.py`/`people.py` may `import` the now-moved modules (e.g. `population.py` imports `from . import people as cvppl`; `base.py` may reference plotting). Grep for cross-imports of the quarantined modules and the removed engine before trimming: `grep -rn "from \. import\|from \.\(sim\|people\|analysis\|immunity\|interventions\|run\|plotting\)" covasim/*.py`. Any active module that imports a quarantined one must be fixed (the new thin `people.py`/`sim.py` will not import the v3 chain; `population.py`'s `import people` is repointed in Task 1; `base.py`'s dormant `BaseSim`/`BasePeople` may need their plotting/sim references neutralized or guarded). Resolve every such reference so `import covasim` succeeds.

- [ ] **Step 8: Skip the v3 bit-for-bit assertions in `test_baselines.py` / `test_regression.py` (Open question G)**

Read both files. Add `@pytest.mark.skip(reason='v3.1.8 bit-for-bit baselines; regenerated for v4 at M10')` to the test functions that assert exact-number equality against `baseline.json` / `covasim/regression/`. Do NOT delete them — they document the v4-internal gate restored at M10. (If a file is a single top-level script-style test, skip at the function granularity; verify the exact function names by reading the files.) These two files stay at the `tests/` root (not quarantined), per Open question G.

- [ ] **Step 9: Add the M1 baseline gitignore entries**

Append to `.gitignore` (after the M0 `v3_seeds_n*.json` block):

```
# M1 single-variant regression baselines (regenerate from a v3.1.8 env via multi_seed_v3.py against anchor_m1)
tests/regression/v3_m1_seeds_n*.json
tests/regression/v4_m1_seeds*.json
```

- [ ] **Step 10: (If writing full shells here) ensure `import covasim` works; otherwise create minimal shells**

To preserve the continuous-runnability invariant at this check-in, `covasim/network.py`, `covasim/covid.py`, `covasim/people.py`, `covasim/sim.py` must at least *exist and import cleanly*. Two options:
  - (a) Write the full implementations now (Tasks 1–3 bodies) and review everything together at check-in 1 — simpler tree, larger review.
  - (b) Write minimal shells now (`class Sim(ss.Sim): pass` etc.) so `import covasim` works, then flesh out in Tasks 1–3 with separate check-ins.

**Recommended: (b)** — it matches the spec's 4-check-in cadence (quarantine reviewed first as a clean history-preserving move). Minimal shells:

```python
# covasim/people.py
import starsim as ss
__all__ = ['People']
class People(ss.People):
    pass
```
```python
# covasim/network.py
import starsim as ss
__all__ = ['Network']
class Network(ss.Network):
    pass
```
```python
# covasim/covid.py
import starsim as ss
__all__ = ['COVID']
class COVID(ss.Infection):
    pass
```
```python
# covasim/sim.py
import starsim as ss
__all__ = ['Sim']
class Sim(ss.Sim):
    pass
```

(These are placeholders; Tasks 1–3 replace the bodies. With shells, `cv.Sim().run()` runs a disease-free Starsim sim — the invariant holds even before the real engine lands.)

- [ ] **Step 11: Verify the tree imports and collection ignores the quarantine**

```bash
python -c "import covasim as cv; print('covasim imports OK; version', cv.__version__)"
python -c "import covasim as cv; print('cv.Sim base:', cv.Sim.__mro__[1].__module__ + '.' + cv.Sim.__mro__[1].__name__)"  # expect starsim...Sim
python -c "import covasim as cv; print('cv.Layer present:', hasattr(cv,'Layer'), 'cv.Contacts:', hasattr(cv,'Contacts'), 'cv.Result:', hasattr(cv,'Result'))"
python -c "import covasim as cv; print('cv.v4 gone:', not hasattr(cv,'v4'))"
cd /home/cliffk/idm/covasim/tests && pytest --collect-only -q 2>&1 | grep -E "_legacy|test_sim|test_immunity" | head
```

Expected: `import covasim` succeeds; `cv.Sim` subclasses `ss.Sim`; `cv.Layer`/`cv.Contacts`/`cv.Result` still present (Open question F); `cv.v4` removed; no `_legacy/*` test collected.

- [ ] **Step 12: Confirm the invariant and that the non-quarantined suite collects**

```bash
python -c "import covasim as cv; s=cv.Sim(); s.run(); print('continuous-runnability OK; results present:', s.results is not None)"
cd /home/cliffk/idm/covasim/tests && pytest test_regression_harness.py -q 2>&1 | tail -5
```

Expected: `cv.Sim().run()` returns results (with shells, a disease-free run; that still satisfies the invariant). The harness unit tests still pass (they import only the public harness API). [NOTE: `test_regression_harness.py::test_anchor_runs` runs the M0 anchor which uses `pop_type='hybrid'`/`use_waning=True` on the *new* `cv.Sim` — with shells it will not produce `cum_infections>0`. Expect this M0 anchor test to FAIL or need updating once the real engine lands; record it and address in Task 4 (the M0 anchor is superseded by `anchor_m1.py`). If it blocks collection, mark it `@pytest.mark.skip(reason='M0 anchor uses M2+ features; superseded by anchor_m1 at M1')` and flag for Cliff.]

- [ ] **PAUSE FOR CLIFF (check-in 1).** Report: the `git mv` quarantine moves (modules + tests), the `cv.v4`/`test_v4_stub` deletions, `tests/conftest.py`, the `__init__.py` trim, the `test_baselines`/`test_regression` skips, the `.gitignore` edit, the four new shells, and any cross-import fixes from Step 7. Confirm `import covasim` + `cv.Sim().run()` work and the quarantine is not collected. Suggested staging: `git add -A covasim/ tests/conftest.py tests/test_baselines.py tests/test_regression.py .gitignore` (the `git mv`/`git rm` are already staged). Do NOT commit. Wait for Cliff.

---

## Task 1: `cv.Network(ss.Network)` + ported `population.py` builders (CHECK-IN 2)

**Files:**
- Modify: `covasim/population.py` (port builders to feed `cv.Network`; keep `cv.make_*` names; guard `make_synthpop`)
- Create/flesh: `covasim/network.py` (`cv.Network(ss.Network)`)
- Create: `tests/regression/contact_stats.py` (degree + age-mixing helpers)
- Create: `tests/test_network.py` (contact-structure equivalence)

Authority: spec decision 1 + Network findings §1–§5. Acceptance metrics: per-layer **degree distribution** + **age-mixing matrix**, both vs v3.1.8.

- [ ] **Step 1: Write failing network unit tests (TDD scaffold)**

Create `tests/test_network.py`. Start with construction + structural assertions that don't need a v3 baseline (they fail until `cv.Network` is implemented):

```python
"""Contact-structure tests for cv.Network (the M1 acceptance gate, structural half).

Builds cv.Network instances directly (random single-layer 'a' and the four hybrid
layers h/s/w/c) and asserts per-layer degree and age-mixing structure. The v3.1.8
EQUIVALENCE half (vs a gitignored baseline) lives in test_network_equivalence_* and
skips cleanly when the baseline is absent.
"""
import numpy as np
import pytest
import covasim as cv

POP = 5000  # small enough to be fast; large enough for stable degree stats

# --- structural tests (no baseline needed) ---

def _make_people(n=POP, seed=0):
    # Build a cv.People with the v3 age distribution (see cv.People in Task 2);
    # if Task 2 not yet done, build ss.People with default age_data.
    ...  # implementer: use cv.People(n) once Task 2 lands; for Task 1, an ss.People is fine

def test_random_single_layer_mean_degree():
    # random backend layer 'a' should have mean degree ~20 (parameters.py:178)
    ...

def test_hybrid_layer_mean_degrees():
    # h~poisson-cluster, s~20 over [6,22), w~16 over [22,65), c~20 over all (parameters.py:185)
    ...

def test_school_work_age_windows():
    # every member of layer 's' has age in [6,22); layer 'w' in [22,65)
    ...
```

Fill the `...` with concrete assertions as you implement Steps 2–4 (TDD: write the assertion, run it red, implement, run it green). Use `np.add.at` over `edges.p1`/`edges.p2` for the per-agent degree histogram (the hpvsim concurrency-histogram idiom; Network findings §5).

- [ ] **Step 2: Port `population.py` builders in place (keep names; Open question A option (a))**

Read `covasim/population.py` fully first. The builders to keep (names + signatures preserved, per `__all__`): `make_people`, `make_randpop`, `make_random_contacts`, `make_microstructured_contacts`, `make_hybrid_contacts`. Port them so they still return **edgelist dicts** (`dict(p1=array, p2=array)`), reproducing the *exact* v3 generation procedure (Network findings §1, §5 — these are the acceptance-critical details):
  - **Ages** (`make_randpop`, population.py:198-204): multinomial over `cvd.default_age_data` bins (`cvu.n_multinomial`) + uniform-within-bin. Drives the age-mixing metric — reproduce exactly.
  - **`make_random_contacts`** (population.py:241-284): pool pre-draw `choose_r(max_n=pop_size, n=int(pop_size*n*1.2))`; per-person Poisson count `n_poisson(n, pop_size)`; **`p_count = round(p_count/2.0)`** (the halving — essential for mean degree n on undirected singly-stored edges; Network findings hazards); sequential pool slicing; `mapping` remap to global UIDs for subpopulations.
  - **`make_microstructured_contacts`** (population.py:287-329): Poisson-sized disjoint **fully-connected** clusters (households). **Hazard: household degree ≠ `contacts['h']`** — the value is a Poisson cluster-size parameter (Network findings §2 caveat).
  - **`make_hybrid_contacts`** (population.py:332-364): `h`=microstructured(contacts['h']=2.0); `c`=random over all (contacts['c']=20); `s`=random over `ages∈[6,22)` (contacts['s']=20) with `mapping`; `w`=random over `ages∈[22,65)` (contacts['w']=16) with `mapping`.
  - **RNG (Open question B):** replace the global-numpy helpers with `ss.Dist` draws (`ss.poisson`, `ss.choice`) so the network participates in Starsim's deterministic per-distribution seeding. **[VERIFY] `ss.poisson`/`ss.choice` signatures and how to draw a fixed-`n` sample** (not slot-keyed) at code-time — `cvu.choose_r`/`cvu.n_poisson` are population-level draws, not per-agent; confirm the `ss.Dist.rvs(n)` form gives the array you need. Do NOT invest in `ss.RandomSafeNet`-style single-agent stability (deferred to M8). If matching the v3 procedure with `ss.Dist` proves to perturb the degree distribution beyond tolerance, fall back to the v3 `cvu` helpers seeded off `sim.pars.rand_seed` and flag for Cliff.
  - **`make_synthpop`** (population.py:367-434): guard it — raise a clear `NotImplementedError('synthpops backend not yet ported (M9)')` so `pop_type='synthpops'` fails cleanly (spec decision 6; continuous-runnability + good error hygiene). Keep the name.
  - **[VERIFY]** `population.py` imports `from . import people as cvppl` and constructs `cvppl.People(...)` (population.py:96). Repoint this to the new `cv.People` (Task 2) or, better, restructure so `make_people` returns the popdict/networks and `cv.Sim` does the People construction (see Step 3). Resolve so `population.py` does not import the quarantined v3 `people.py`.

- [ ] **Step 3: Implement `cv.Network(ss.Network)` in `covasim/network.py`**

Subclass **`ss.Network`** (static — both M1 backends have `dynam_layer=0`; spec decision 1). One instance per layer (hpvsim M01 multi-instance pattern). Build edges in `add_pairs()` from the ported builders, ending in `self.append(...)`. Reference shape (verify every Starsim call):

```python
import numpy as np
import starsim as ss
import covasim as cv

__all__ = ['Network', 'make_networks']

_KNOWN_LAYERS = ('a', 'h', 's', 'w', 'c')  # 'a'=random/all; h/s/w/c=hybrid

class Network(ss.Network):
    """A single Covasim contact layer on the Starsim base (random or hybrid).

    Args:
        layer (str): layer key, one of 'a','h','s','w','c'.
        edges (dict): optional precomputed dict(p1=, p2=) to inject (else built in add_pairs).
    """
    def __init__(self, layer='a', edges=None, **kwargs):
        if layer not in _KNOWN_LAYERS:
            raise ValueError(f'Unknown layer {layer!r}; known: {list(_KNOWN_LAYERS)}.')
        self.layer = layer
        kwargs.setdefault('name', layer)
        super().__init__(**kwargs)   # [VERIFY] ss.Network.__init__ signature (name/label/**kwargs)
        self._prebuilt_edges = edges

    def add_pairs(self):
        # [VERIFY] in stock ss.Network, edges are built in init_post -> add_pairs; confirm the hook.
        if self._prebuilt_edges is not None:
            p1, p2 = self._prebuilt_edges['p1'], self._prebuilt_edges['p2']
            beta = np.ones(len(p1))
            self.append(p1=ss.uids(p1), p2=ss.uids(p2), beta=beta)  # [VERIFY] append kwargs/dtypes
        return
```

Decision detail (Open question A): the per-layer edgelist is computed by the `population.py` builders (Step 2) and **injected** via `edges=` so `cv.make_randpop`/`cv.make_hybrid_contacts` stay the single readable place for generation. `make_networks(pop_type, people)` (a free function in `network.py`, exported) calls the right builder for the backend and returns `[Network('a', edges=...)]` (random) or `[Network(k, edges=...) for k in 'hswc']` (hybrid). It needs the people's ages (for hybrid age windows) and pop_size.
  - **[VERIFY]** Stock `ss.Network` builds edges in `init_post(add_pairs=True) → add_pairs()` (API findings §1). Confirm: does `init_post` run after People exist so UIDs are valid? If the builder needs ages, `make_networks` must run *after* People are built — call it from `cv.Sim.__init__` passing the already-built `cv.People` (Task 2 / Task 3), or build edges lazily in `add_pairs` reading `self.sim.people`. Pick whichever the verified `init_post` ordering supports; the hpvsim M01 `add_pairs` read `self.sim.people` directly.
  - **`net_beta` NOT overridden** (spec decision 1; hpvsim M01). Per-layer scalar β is carried on the disease (Task 2 / decision 5). Per-edge `beta` stays 1.0.
  - **`step` NOT overridden** → static layer (no per-step rewiring; spec defers dynamic layers).

- [ ] **Step 4: Create `tests/regression/contact_stats.py`**

Helpers used by both the equivalence test and the M1 anchor:

```python
"""Contact-structure metrics for the M1 acceptance gate.

degree_by_layer(networks)   -> {layer: np.ndarray of per-agent degree}
age_mixing_matrix(network, ages, bin_edges) -> 2D ndarray (source-age x target-age)
cosine_similarity(a, b)     -> float in [-1,1]
"""
import numpy as np

def degree_by_layer(networks):
    # For each cv.Network, count per-agent edge endpoints via np.add.at over p1 and p2.
    ...

def age_mixing_matrix(network, ages, bin_edges):
    # 5-year-binned source-age x target-age contact matrix (count both edge directions).
    ...

def cosine_similarity(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a.dot(b) / denom) if denom else 0.0
```

[VERIFY] how to read edge endpoints off a built `cv.Network`: `net.edges.p1` / `net.edges.p2` (API findings §1: `self.edges` is an `sc.objdict` with `p1`/`p2`/`beta`). Confirm `net.edges.p1` returns a usable integer array at code-time.

- [ ] **Step 5: Run the structural network tests**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_network.py -v
```

Expected: structural tests PASS — random `a` mean degree ≈20; hybrid `s`≈20, `w`≈16, `c`≈20; `h` realized degree consistent with a Poisson(2.0) fully-connected-cluster construction (NOT assumed ==2.0); school members all in `[6,22)`, work in `[22,65)`. Pin exact tolerances on first run (hpvsim-style); record chosen tolerances in the test.

- [ ] **Step 6: Add the v3.1.8 equivalence test (skips when baseline absent)**

Add to `tests/test_network.py` a `test_contact_structure_equivalence` that loads a gitignored baseline (per-layer degree histogram + age-mixing matrix, generated from a v3.1.8 env — see Task 4 Step 3 for the generator) and compares: mean degree per layer within tolerance, degree-distribution shape close (coarse-binned relative diff or KS), and age-mixing **cosine similarity > threshold** (spec §M1 anchor metrics; hpvsim used >0.85 — pin on first run). `@pytest.mark.skipif` when the baseline file is absent so CI stays green.

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_network.py -v   # equivalence test should SKIP (no baseline yet)
```

- [ ] **PAUSE FOR CLIFF (check-in 2).** Report: ported `population.py` builders (names preserved; `make_synthpop` guarded), `covasim/network.py` (`cv.Network` + `make_networks`), `tests/regression/contact_stats.py`, `tests/test_network.py` (structural tests green; equivalence test skipping). Flag the provisional name `cv.Network` (Open question H) and the RNG decision (Open question B). Suggested staging: `git add covasim/population.py covasim/network.py tests/regression/contact_stats.py tests/test_network.py`. Do NOT commit. Wait for Cliff.

---

## Task 2: `cv.COVID(ss.Infection)` minimal S→E→I→R (CHECK-IN 3)

**Files:**
- Create/flesh: `covasim/covid.py` (`cv.COVID(ss.Infection)`)
- Create/flesh: `covasim/people.py` (`cv.People(ss.People)`)
- Create: `tests/test_covid.py` (functional tests, hpvsim-style)

Authority: spec decision 2 (the full reference class is quoted in the spec) + Transmission findings §1–§5 + Starsim findings §2. Override **only** `__init__`, `set_prognoses`, `step_state`; inherit transmission.

- [ ] **Step 1: Write failing `cv.COVID` functional tests (TDD)**

Create `tests/test_covid.py`. Mirror hpvsim M01's `test_hpv.py`: build a **minimal `ss.Sim` with a stock `ss.RandomNet`** (not `cv.Network`), `copy_inputs=False` so the test keeps a live reference to the module instance, run a few steps:

```python
"""Functional tests for cv.COVID (minimal single-variant S->E->I->R).

Uses a minimal ss.Sim with a stock ss.RandomNet so the disease is exercised in
isolation from the cv.Network port.
"""
import numpy as np
import pytest
import starsim as ss
import covasim as cv

def _minimal_sim(n_agents=1000, beta=0.05, init_prev=0.02, n_days=20):
    covid = cv.COVID(beta=beta, init_prev=ss.bernoulli(p=init_prev))  # [VERIFY] beta/init_prev kwargs
    sim = ss.Sim(diseases=covid, networks='random', n_agents=n_agents,
                 start=ss.date('2020-03-01'), dur=ss.days(n_days), dt=ss.days(1),
                 verbose=0, copy_inputs=False)   # [VERIFY] ss.Sim kwargs (start/dur/dt forms)
    return sim, covid

def test_states_present():
    _, covid = _minimal_sim()
    for s in ['susceptible','exposed','infectious','recovered']:
        assert hasattr(covid, s), f'missing state {s}'

def test_seir_progression():
    sim, covid = _minimal_sim()
    sim.run()
    # someone progressed E->I->R; recovered count > 0 by end; no deaths (asymptomatic-only path)
    assert (covid.recovered.sum() > 0)

def test_only_infectious_transmit():
    # exposed-but-not-yet-infectious agents must NOT transmit (infectious property override)
    ...

def test_permanent_immunity():
    # recovered agents never return to susceptible (use_waning=False semantics)
    ...
```

- [ ] **Step 2: Implement `cv.COVID` in `covasim/covid.py`**

Port the spec's reference class verbatim-ish (spec decision 2), grounding each value in v3 pars (Transmission findings §4). Override `__init__`/`set_prognoses`/`step_state`; do NOT override `infect`/`net_beta`.

```python
import starsim as ss
__all__ = ['COVID']

class COVID(ss.Infection):
    """Minimal single-variant COVID-19 disease: S->E->I->R transmission + recovery.

    M1 collapses the natural-history tree to the asymptomatic branch (recovery via
    asym2rec, no symptoms/severity/death, permanent immunity). Symptomatic/severe/
    critical/death are added in M2. Transmission is the stock ss.Infection.infect()
    (CRN-safe per-edge draw); no custom transmission code.

    Args:
        beta: per-contact transmission prob (scalar or dict keyed by layer); v3 pars['beta']=0.016.
        init_prev: ss.bernoulli for seeding (or None if cv.Sim seeds an exact pop_infected count).
        dur_exp2inf: E->I latency (v3 dur.exp2inf, lognormal mean 4.5 std 1.5).
        dur_asym2rec: I->R recovery (v3 dur.asym2rec, lognormal mean 8.0 std 2.0).
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.probperday(0.016),  # [VERIFY] ss.probperday exists & is accepted by validate_beta
            init_prev = None,
            dur_exp2inf  = ss.lognorm_ex(mean=ss.days(4.5), std=ss.days(1.5)),   # [VERIFY] lognorm_ex(mean=,std=) signature
            dur_asym2rec = ss.lognorm_ex(mean=ss.days(8.0), std=ss.days(2.0)),
        )
        self.update_pars(pars=pars, **kwargs)
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('exposed',     label='Exposed'),
            ss.BoolState('infectious',  label='Infectious'),
            ss.BoolState('recovered',   label='Recovered'),
            ss.FloatArr('ti_exposed',    label='Time of exposure'),
            ss.FloatArr('ti_infectious', label='Time of becoming infectious'),
            ss.FloatArr('ti_recovered',  label='Time of recovery'),
            ss.FloatArr('rel_sus',   default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
            ss.FloatArr('ti_infected',   label='Time of infection'),
            reset = True,  # drop inherited Infection states; use ours (Open question C). [VERIFY] reset semantics
        )

    # Only infectious (not merely exposed) agents transmit: stock infect() zeroes
    # rel_trans for non-`infectious` agents. [VERIFY] whether 'infectious' as a BoolState
    # already satisfies infect()'s use of self.infectious, or a property override is needed.
    # (ss.Infection.infectious is a PROPERTY aliased to `infected` by default; with a BoolState
    #  named 'infectious' present, confirm infect() picks up the state, else override the property.)

    def set_prognoses(self, uids, sources=None):
        super().set_prognoses(uids, sources)   # logs the infection
        ti = self.ti
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = ti
        self.ti_infected[uids] = ti
        self.ti_infectious[uids] = ti + self.pars.dur_exp2inf.rvs(uids)
        self.ti_recovered[uids]  = self.ti_infectious[uids] + self.pars.dur_asym2rec.rvs(uids)

    def step_state(self):
        ti = self.ti
        new_inf = (self.exposed & (self.ti_infectious <= ti)).uids
        self.exposed[new_inf] = False
        self.infectious[new_inf] = True
        rec = (self.infectious & (self.ti_recovered <= ti)).uids
        self.infectious[rec] = False
        self.recovered[rec] = True   # permanent immunity: never back to susceptible (use_waning=False)
```

  - **[VERIFY — critical]** the `infectious` mechanism. API findings §2: `ss.Infection.infectious` is a *property* returning `self.infected`, and `infect()` zeroes `rel_trans` for non-`infectious` agents. The spec (decision 2, Open question C) adds `infectious` as a `BoolState` with `reset=True`. Confirm at code-time whether a `BoolState('infectious')` shadows the inherited property correctly so `infect()` reads the right mask, OR whether you must keep the property and name the state differently. Resolve this so **exposed agents do not transmit** (`test_only_infectious_transmit`). This is the single most error-prone point in `cv.COVID`.
  - **[VERIFY]** `self.ti` vs `self.t.ti` — use whichever the installed `ss.Module` exposes (API findings: `Module.ti`/`now`/`dt` props at modules.py:192-214; the SIR example uses both `self.ti` and `self.t.ti`). Be consistent.
  - **rel_trans/rel_sus = 1.0** (Open question D): no `beta_dist` overdispersion at M1; M2 reintroduces it.

- [ ] **Step 3: Implement the thin `cv.People(ss.People)` in `covasim/people.py`**

Keep the name (locked); supply the v3 age distribution as default `age_data` so the age-mixing metric matches (spec decision 3). `ss.People.__init__(self, n_agents, age_data=None, extra_states=None, mock=False)` (verified at plan time).

```python
import numpy as np
import starsim as ss
import covasim.defaults as cvd
__all__ = ['People']

def _default_age_data():
    """Covasim's default Seattle-2018 age pyramid as an [age, value] dataframe for ss.People."""
    # cvd.default_age_data is a 19-row [age_min, age_max, fraction] table (defaults.py:223-243).
    # [VERIFY] ss.People age_data expected shape: API findings say [age, value]; the hpvsim
    # adapter reshaped to [age, value]. Convert the v3 binned table accordingly.
    ...

class People(ss.People):
    """Covasim People on the Starsim base. Defaults to Covasim's age distribution.

    Args:
        n_agents (int): number of agents.
        age_data (dataframe): optional override; defaults to Covasim's default_age_data.
    """
    def __init__(self, n_agents, age_data=None, **kwargs):
        if age_data is None:
            age_data = _default_age_data()
        super().__init__(n_agents, age_data=age_data, **kwargs)
```

  - **[VERIFY]** the exact `age_data` dataframe shape `ss.People` expects (`[age, value]`) and that supplying the binned v3 table reproduces the v3 multinomial-bin + uniform-within-bin sample closely enough for the age-mixing metric. If `ss.People`'s age sampling differs materially from v3's, the age-mixing cosine-similarity gate (Task 1 Step 6) will catch it — flag for Cliff and consider sampling ages the v3 way and passing explicit per-agent ages instead. (API findings: `ss.People(n_agents, age_data=df)`.)

- [ ] **Step 4: Run the `cv.COVID` functional tests**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_covid.py -v
```

Expected: all PASS. Key assertion: exposed agents do not transmit (the `infectious` mechanism is correct); recovered agents stay recovered; recovery happens via `asym2rec`; no deaths.

- [ ] **PAUSE FOR CLIFF (check-in 3).** Report: `covasim/covid.py` (`cv.COVID`), `covasim/people.py` (`cv.People`), `tests/test_covid.py` (green). Flag: provisional name `cv.COVID` + file name `covid.py` (Open question H); the SEIR state mapping decision (Open question C) and how the `infectious` mechanism was resolved; rel_trans=1.0 (Open question D); the `age_data` shape decision (Open question E-adjacent). Suggested staging: `git add covasim/covid.py covasim/people.py tests/test_covid.py`. Do NOT commit. Wait for Cliff.

---

## Task 3: `cv.Sim(ss.Sim)` assembly + per-layer beta wiring

**Files:**
- Flesh: `covasim/sim.py` (the real thin `cv.Sim(ss.Sim)`)
- Create: `tests/test_sim.py` (new M1 file — assembly + continuous-runnability)

Authority: spec decisions 3, 4, 5 (the reference `cv.Sim` is quoted in the spec) + Starsim findings §3, §4.

- [ ] **Step 1: Write failing `cv.Sim` assembly tests (TDD)**

Create `tests/test_sim.py`:

```python
"""M1 cv.Sim / cv.People assembly tests + continuous-runnability invariant."""
import numpy as np
import pytest
import covasim as cv

def test_sim_runs_random():
    sim = cv.Sim(pop_size=2000, pop_infected=20, pop_type='random', n_days=40, rand_seed=1)
    sim.run()
    assert sim.results is not None

def test_sim_runs_hybrid():
    sim = cv.Sim(pop_size=2000, pop_infected=20, pop_type='hybrid', n_days=40, rand_seed=1)
    sim.run()
    assert sim.results is not None

def test_default_sim_runs():
    # continuous-runnability invariant: bare cv.Sim().run() returns results
    sim = cv.Sim(pop_size=1000, n_days=20)
    sim.run()
    assert sim.results is not None

def test_epidemic_grows():
    sim = cv.Sim(pop_size=5000, pop_infected=50, pop_type='random', n_days=60, rand_seed=1)
    sim.run()
    # with beta=0.016 and 20 contacts/day the epidemic should grow beyond the seed
    ...  # assert cumulative infections > pop_infected

def test_pop_infected_seed_count():
    # exactly pop_infected agents are infected at t==0 (Open question E: exact-count seed)
    ...

def test_override_kwargs():
    # passing networks=/diseases=/people= short-circuits the default assembly
    ...
```

- [ ] **Step 2: Implement the thin `cv.Sim(ss.Sim)` in `covasim/sim.py`**

Port the spec's reference `Sim` (spec decision 3). Use the `kwargs.pop('x', None)`-then-default idiom (hpvsim pattern) so tests can inject networks/disease/people. Wire per-layer beta on the disease (decision 5).

```python
import starsim as ss
import covasim as cv
from . import network as cvnet
__all__ = ['Sim']

# v3 beta_layer scalars (parameters.py:174-189). random 'a'=1.0; hybrid h=3.0,s=0.6,w=0.6,c=0.3.
_BETA_LAYER = {
    'random': {'a': 1.0},
    'hybrid': {'h': 3.0, 's': 0.6, 'w': 0.6, 'c': 0.3},
}
_BASE_BETA = 0.016  # v3 pars['beta'] (parameters.py:62)

class Sim(ss.Sim):
    """Covasim Sim on the Starsim base (M1: basic transmission).

    Args mirror v3 where they map cleanly:
        pop_size, pop_infected, pop_type ('random'|'hybrid'), n_days, start_day, rand_seed.
    Passing networks=/diseases=/people= overrides the default assembly.
    """
    def __init__(self, pars=None, people=None, pop_size=20_000, pop_infected=20,
                 pop_type='random', n_days=60, start_day='2020-03-01',
                 rand_seed=1, **kwargs):
        if pop_type not in _BETA_LAYER:
            raise ValueError(f"pop_type {pop_type!r} not supported in M1 (random|hybrid).")
        if people is None:
            people = cv.People(pop_size)
        networks = kwargs.pop('networks', None)
        if networks is None:
            networks = cvnet.make_networks(pop_type, people)   # [cv.Network('a')] or [h,s,w,c]
        diseases = kwargs.pop('diseases', None)
        if diseases is None:
            beta = {lk: ss.probperday(_BASE_BETA * bl) for lk, bl in _BETA_LAYER[pop_type].items()}
            diseases = cv.COVID(beta=beta)   # [VERIFY] validate_beta accepts a per-layer dict keyed by network label
        super().__init__(pars=pars, people=people, networks=networks, diseases=diseases,
                         start=ss.date(start_day), dur=ss.days(n_days), dt=ss.days(1),
                         rand_seed=rand_seed, **kwargs)  # [VERIFY] ss.Sim start/dur/dt/rand_seed kwargs
        # seed exactly pop_infected at t==0 (Open question E) -- see Step 3.
        self._pop_infected = pop_infected
```

  - **[VERIFY]** `ss.Sim.__init__` kwargs: `pars, label, people, demographics, connectors, networks, diseases, interventions, analyzers, custom, modules, copy_inputs, data, **kwargs` (API findings §3). Confirm `start`/`dur`/`dt`/`rand_seed` pass through `**kwargs` into `SimPars`, and that `start=ss.date(...)`, `dur=ss.days(...)`, `dt=ss.days(1)` is the accepted daily-step form (API findings §4). The 60-day run = `start=ss.date('2020-03-01'), dur=ss.days(60), dt=ss.days(1)`.
  - **[VERIFY]** the per-layer **beta dict keys must match the network labels** exactly (`ss.standardize_netkey`; `validate_beta` asserts the betamap key-set == network key-set, API findings §1, §2). The `cv.Network` `name`/`label` (Task 1) sets the network key — confirm `make_networks` labels are `'a'`/`'h'`/`'s'`/`'w'`/`'c'` and the beta dict uses the same. A mismatch raises in `validate_beta` at init.

- [ ] **Step 3: Wire the exact-count `pop_infected` seed (Open question E)**

v3 seeds an **exact count** (`cvu.choose(pop_size, pop_infected)`), not a per-agent probability. The Starsim idiom is `init_prev` as `ss.bernoulli` in `Infection.init_post` (diseases.py:150). Choose ONE (verify at code-time which is cleaner):
  - (a) Pass `init_prev` to `cv.COVID` as a callback/`ss.choice` that selects exactly `pop_infected` uids, OR
  - (b) Override a small hook on `cv.Sim` (e.g. after `super().__init__`, set the disease's `init_prev` to a selector) so exactly `pop_infected` agents are seeded.

  **[VERIFY]** how `Infection.init_post` consumes `init_prev` (it calls `self.pars.init_prev.filter()` → set_prognoses with `sources=-1`, API findings §2). The cleanest exact-count approach: set `cv.COVID(init_prev=ss.bernoulli(p=pop_infected/pop_size))` for an approximate count (small variance at 20k), OR implement an exact selector. The spec recommends **exact-count** so `cum_infections` (which adds `pop_infected` back, sim.py:786-787) aligns with v3. If an exact selector is awkward in the installed Starsim, fall back to the Bernoulli and flag for Cliff (note the variance is small at pop_size=20k).

- [ ] **Step 4: Run the assembly tests + the M0 anchor smoke**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_sim.py -v
python -c "import covasim as cv; s=cv.Sim(pop_size=2000, n_days=30); s.run(); print('cum infections:', float(s.results... ))"  # [VERIFY] how to read cum_infections off the ss.Sim results
```

Expected: random + hybrid sims run; the epidemic grows beyond the seed; exactly (or ~) `pop_infected` seeded at t0; override kwargs short-circuit assembly. [VERIFY] the results-access path for `cum_infections`/`prevalence`/`n_infectious` on the `ss.Sim` results object (API findings §2: `Infection.init_results` defines `prevalence`, `new_infections`, `cum_infections`; `BoolState`s auto-create `n_<state>`). The M1 `short_summary` (Task 4) depends on getting these keys right.

- [ ] **Step 5: Plot the demo infection curve (acceptance test item 5)**

```bash
python -c "import matplotlib; matplotlib.use('agg'); import covasim as cv; s=cv.Sim(pop_size=5000, pop_infected=50, n_days=80); s.run(); s.plot(); import matplotlib.pyplot as plt; plt.savefig('/tmp/m1_demo.png'); print('saved /tmp/m1_demo.png')"
```

Expected: a saved infection-curve plot via the Starsim `sim.plot()` (spec acceptance item 5). [VERIFY] `ss.Sim.plot()` exists and renders without error; if the default plot is unhelpful, plot `n_infectious`/`prevalence` explicitly.

---

## Task 4: M1 anchor + short-summary + trajectory parity gate (CHECK-IN 4)

**Files:**
- Create: `tests/regression/anchor_m1.py`
- Modify: `tests/regression/short_summary.py` (or add an M1 metric set)
- Create: `tests/test_m1_parity.py` (slow z-score gate)
- Generate (gitignored): the v3.1.8 M1 baseline + the contact-structure baseline

Authority: spec "M1 anchor scenario + pinned metrics" + "Acceptance test". Reuse the M0 harness (`parity.py`, `multi_seed_v3.py`, `compare.py`) unchanged.

- [ ] **Step 1: Create `tests/regression/anchor_m1.py`**

Two anchor variants (random + hybrid), single-variant, no interventions, no waning — isolates basic transmission (spec):

```python
"""M1 anchor: single-variant basic-transmission sim (random + hybrid backends).

Isolates the M1 capability (S->E->I->R transmission + recovery only). The v3.1.8
baseline is generated locally via multi_seed_v3-style sweeps and gitignored.

Run as a script to print the summary:
    python tests/regression/anchor_m1.py
"""
import sys
from pathlib import Path
import sciris as sc
import covasim as cv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_summary import build_summary_m1   # see Step 2

PARS_RANDOM = dict(pop_size=20_000, pop_infected=20, pop_type='random',
                   n_days=60, rand_seed=0, verbose=0)
PARS_HYBRID = dict(PARS_RANDOM, pop_type='hybrid')

def make_sim(pop_type='random', **kwargs):
    base = PARS_RANDOM if pop_type == 'random' else PARS_HYBRID
    pars = sc.mergedicts(sc.dcp(base), kwargs)
    return cv.Sim(**pars)

def run_and_summarize(pop_type='random', **kwargs):
    sim = make_sim(pop_type=pop_type, **kwargs)
    sim.run()
    return build_summary_m1(sim)

if __name__ == '__main__':
    for pt in ('random', 'hybrid'):
        print(f'M1 anchor short summary ({pt}):')
        for k, v in run_and_summarize(pop_type=pt).items():
            print(f'  {k:<24} {v:>14.4g}')
```

  - **IMPORTANT for the v3.1.8 baseline:** `anchor_m1.py` runs under **both** v3.1.8 and v4 (the M0 harness pattern). But v3.1.8's `cv.Sim` takes a **pars dict** (`cv.Sim(pars)` / `cv.Sim(pop_size=..., use_waning=False, ...)`) and needs `use_waning=False`, `n_variants=1`, and the asymptomatic-only config (`symp_prob→0`) to match M1's capability (Transmission findings §5 "Minimal config"). The v4 `cv.Sim` takes the M1 kwargs above. **Reconcile**: write the anchor so the *same* call works in both, OR provide a `_v3_pars()` / `_v4_pars()` split keyed on `cv.__version__`. The hpvsim approach was a single anchor whose pars are valid in both; here the asymptomatic-only/use_waning=False config is v3-only noise (v4 has no such pars). **Recommended:** make `anchor_m1.py` import-compatible by branching on `cv.__version__.startswith('3')` to add the v3-only pars (`use_waning=False, n_variants=1`, prognoses with `symp_prob=0`) when running under v3.1.8. [VERIFY] the exact v3 config that forces the asymptomatic-only S→E→I→R path — Transmission findings §5 gives `use_waning=False` + `symp_prob=0` via `get_prognoses(by_age=False)` with `symp_probs→0`. Test the v3 branch in the frozen v3.1.8 env.

- [ ] **Step 2: Add the M1 metric set to `short_summary.py`**

The M2+ burden metrics (`cum_symptomatic`/`cum_severe`/`cum_critical`/`cum_deaths`) are identically 0 in M1 and excluded from the M1 gate (spec). Add an `build_summary_m1(sim)` + `METRIC_KEYS_M1`:

```python
METRIC_KEYS_M1 = ('cum_infections', 'peak_prevalence', 'peak_n_infectious', 'prevalence', 'incidence')

def build_summary_m1(sim):
    """M1 short summary: basic-transmission metrics only (no burden metrics)."""
    # [VERIFY] result-access paths for both v3.1.8 (sim.summary / sim.results) AND v4 (ss.Sim results).
    # v3.1.8: sim.summary['cum_infections'], sim.results['prevalence'].values.max(), etc.
    # v4: confirm the ss.Sim results keys (cum_infections, prevalence, n_infectious) and access form.
    ...
```

  - **[VERIFY — critical for cross-version parity]** `build_summary_m1` must extract the same metric from both implementations' results objects. v3.1.8 uses `sim.summary` (objdict) + `sim.results['key'].values`; v4 `ss.Sim` results may differ (likely `sim.results.diseasename.cum_infections` or similar). Read both at code-time and write a version-branching extractor if needed. The parity gate compares dict values by key, so the keys + semantics must align across versions. This is where most cross-version parity bugs hide.

- [ ] **Step 3: Generate the gitignored v3.1.8 M1 baselines**

In a **frozen v3.1.8 env** (the current environment IS covasim 3.1.8 on `main`; use a separate checkout/venv pinned to v3.1.8, NOT the `starsim-port` working tree which now has the v4 engine):

```bash
# Trajectory baseline (30-seed sweep, random + hybrid):
<v3.1.8-python> tests/regression/multi_seed_v3.py --n 30 --out tests/regression/v3_m1_seeds_n30.json
# (multi_seed_v3.py imports PARS+build_summary; point it at anchor_m1 -- see note below)
```

  - The M0 `multi_seed_v3.py` imports `from anchor import PARS`. For M1 either (a) add a `--anchor anchor_m1` option to `multi_seed_v3.py`, or (b) write a tiny `multi_seed_v3_m1.py` wrapper that imports from `anchor_m1`. **Recommended (a)** — minimal change, keeps one sweep script. [VERIFY] the M0 `multi_seed_v3.py` structure before editing; keep its behavior for the M0 anchor intact.
  - **Contact-structure baseline:** also generate, from the v3.1.8 env, the per-layer degree histograms + age-mixing matrices for `random` and `hybrid` at the anchor pars/seed (a small script that builds a v3.1.8 `cv.Sim`, runs `make_people`, and dumps `sim.people.contacts[lkey]` degree counts + an age-mixing matrix). Save gitignored (e.g. `tests/regression/v3_m1_contacts.json`). `tests/test_network.py::test_contact_structure_equivalence` (Task 1 Step 6) consumes it. [VERIFY] v3.1.8 layer-degree access: `sim.people.contacts['a'].members` / iterate `p1`/`p2` (base.py:1610-1805).

- [ ] **Step 4: Create the slow z-score trajectory parity gate `tests/test_m1_parity.py`**

Mirror `tests/test_m0_parity.py` (read it first), retargeted to `anchor_m1` + `METRIC_KEYS_M1`, for both backends:

```python
"""M1 acceptance gate: multi-seed trajectory parity vs the v3.1.8 M1 baseline.

Runs N v4 seeds of the M1 anchor (random and hybrid), gates each metric on
|z| < 3 vs the gitignored v3.1.8 baseline. Skips cleanly when the baseline is absent.
Marked slow: the 5-minute PR job skips it; run locally or nightly.
"""
import json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent))
from regression.anchor_m1 import make_sim
from regression.short_summary import build_summary_m1, SKIP_KEYS
from regression.parity import parity_gate

N_V4_SEEDS = 10
M_V3_SEEDS = 30
Z_THRESHOLD = 3.0

def _baseline(pop_type):
    return Path(__file__).parent / 'regression' / f'v3_m1_{pop_type}_seeds_n{M_V3_SEEDS}.json'

def _run_v4(pop_type, n):
    rows = []
    for seed in range(n):
        sim = make_sim(pop_type=pop_type, rand_seed=seed); sim.run()
        rows.append(build_summary_m1(sim))
    return rows

@pytest.mark.slow
@pytest.mark.parametrize('pop_type', ['random', 'hybrid'])
def test_m1_anchor_parity(pop_type):
    bp = _baseline(pop_type)
    if not bp.exists():
        pytest.skip(f'Missing v3.1.8 M1 baseline at {bp}; regenerate from a frozen v3.1.8 env.')
    v3_rows = json.loads(bp.read_text())
    v4_rows = _run_v4(pop_type, N_V4_SEEDS)
    failures = parity_gate(v4_rows, v3_rows, z_threshold=Z_THRESHOLD, skip_keys=SKIP_KEYS)
    if failures:
        details = '\n'.join(f'  {n:<22} z={z:+.2f}' for n, z in failures)
        pytest.fail(f'M1 {pop_type} parity drift |z|>={Z_THRESHOLD}:\n{details}')
```

  - [VERIFY] the `slow` marker is registered (M0 registered it in `tests/pytest.ini`). Confirm `parity_gate` and `SKIP_KEYS` import paths match the M0 harness.
  - The per-backend baseline filename (`v3_m1_random_seeds_n30.json` / `v3_m1_hybrid_...`) implies the sweep writes per-backend files. Adjust the `multi_seed_v3.py --anchor` option to emit per-backend names, or run it twice with `--out`. Keep the gitignore pattern (`v3_m1_seeds_n*.json`) broad enough — update it to `v3_m1_*seeds*.json` if needed.

- [ ] **Step 5: Run the M1 gates + the full fast suite under strict warnings**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_m1_parity.py -v          # SKIPS if baseline absent
cd /home/cliffk/idm/covasim/tests && pytest test_network.py test_covid.py test_sim.py -v
# Full non-quarantined suite under the strict-warnings bar (matches run_tests):
cd /home/cliffk/idm/covasim/tests && COVASIM_INTERACTIVE=0 COVASIM_WARNINGS=error pytest test_*.py -n auto --durations=0 2>&1 | tail -30
```

Expected: M1 functional tests green; `test_m1_parity` skips cleanly (or, if the local v3.1.8 baseline was generated, runs and reports z-scores); the full root suite passes under `COVASIM_WARNINGS=error` (the quarantined `_legacy/*` are not collected; `test_baselines`/`test_regression` carry the documented v4-skips; the M0 anchor smoke is handled per Task 0 Step 12). If `import starsim` or the port emits a new warning that the strict bar turns into a failure, fix at the source (do not silence globally) — escalate if non-trivial.

- [ ] **Step 6: Run the development drift gate (informational) if a v3.1.8 snapshot exists**

```bash
# From a v3.1.8 env: python tests/regression/compare.py --save-snapshot --baseline tests/regression/anchor_m1_snapshot.json  (with anchor_m1)
# From the v4 env: python tests/regression/compare.py --baseline tests/regression/anchor_m1_snapshot.json
```

[VERIFY] `compare.py` imports `from anchor import run_and_summarize` — point it at `anchor_m1` (an `--anchor` option, same as `multi_seed_v3.py`). Expected: a ±10% drift table (informational; exit 0). Record the observed drift in the report; classify any >10% drift as expected feature-misalignment or a real bug (per MIGRATION_PLAN implementation conventions item 2).

- [ ] **PAUSE FOR CLIFF (check-in 4).** Report: `anchor_m1.py`, the `build_summary_m1`/`METRIC_KEYS_M1` additions, `test_m1_parity.py`, the `multi_seed_v3.py`/`compare.py` `--anchor` additions, the demo plot, full-suite results under strict warnings, and the observed contact-structure + trajectory drift vs v3.1.8 (with the cosine-similarity / degree tolerances pinned on first run). Flag any open question that surfaced (e.g. age-sampling mismatch, results-key access). Suggested staging: `git add tests/regression/anchor_m1.py tests/regression/short_summary.py tests/regression/multi_seed_v3.py tests/regression/compare.py tests/test_m1_parity.py tests/test_network.py`. Do NOT commit. Wait for Cliff.

---

## Task 5: End-to-end verification (no commits)

A manual verification pass against the spec's acceptance test. Contains no staging; the assistant never commits.

- [ ] **Step 1: Confirm branch + working tree**

```bash
git -C /home/cliffk/idm/covasim branch --show-current   # expect starsim-port
git -C /home/cliffk/idm/covasim status --short
```

- [ ] **Step 2: Continuous-runnability invariant (acceptance item 1)**

```bash
python -c "import covasim as cv; s=cv.Sim(); s.run(); print('cv.Sim().run() OK, results:', s.results is not None)"
python -c "import covasim as cv; cv.Sim(pop_type='hybrid', pop_size=2000, n_days=30).run(); print('hybrid OK')"
```

- [ ] **Step 3: Full non-quarantined suite under strict warnings (acceptance item 2)**

```bash
cd /home/cliffk/idm/covasim/tests && ./run_tests 2>&1 | tail -30
```

Expected: green under `COVASIM_WARNINGS=error`/`COVASIM_INTERACTIVE=0` — the M1 tests (`test_network`, `test_covid`, `test_sim`), the harness tests, the skipped slow gates, the documented v4-skips in `test_baselines`/`test_regression`; quarantine not collected.

- [ ] **Step 4: Contact-structure + trajectory equivalence (acceptance items 3, 4) — requires the v3.1.8 baselines**

```bash
cd /home/cliffk/idm/covasim/tests && pytest test_network.py -v -k equivalence   # vs v3 contact baseline
cd /home/cliffk/idm/covasim/tests && pytest test_m1_parity.py -m slow -v        # vs v3 trajectory baseline
```

Expected: per-layer degree + age-mixing match within the pinned tolerances (both backends); trajectory `|z| < 3` (or both skip cleanly if no v3.1.8 baseline is available — record which).

- [ ] **Step 5: Demo plot (acceptance item 5)**

Confirm `/tmp/m1_demo.png` (Task 3 Step 5) renders the infection curve.

- [ ] **Step 6: Confirm `.gitignore` excludes the generated M1 baselines**

```bash
git -C /home/cliffk/idm/covasim check-ignore tests/regression/v3_m1_seeds_n30.json tests/regression/v3_m1_contacts.json
```

- [ ] **Step 7: Final report to Cliff**

Summarize: the four check-in boundaries and what was staged at each; the quarantine inventory (modules + tests moved); the provisional names used (`cv.COVID`, `cv.Network`, file `covid.py`) and which Open questions A–H were resolved which way; test results (functional + full-suite + parity/drift); observed contact-structure + trajectory drift vs v3.1.8 with pinned tolerances; any escalations (strict-warnings issues, age-sampling/results-key mismatches, RNG decisions, the M0-anchor-smoke fate); and confirmation that the assistant committed nothing. **Milestone completion is Cliff's call: acceptance test green locally AND Cliff has reviewed and committed.**

---

## Self-review checklist

After all tasks, verify against `MIGRATION_PLAN.md` §M1 and the spec:

| Requirement | Implementing task |
|---|---|
| Work on `starsim-port`; assistant never branches/commits/pushes; 4 PAUSE-FOR-CLIFF check-ins | Tasks 0,1,2,4 boundaries |
| Continuous-runnability invariant (`cv.Sim().run()` returns results) at every commit | Task 0 Step 12; Task 3; Task 5 Step 2 |
| Quarantine v3 modules untouched by M1 → `covasim/_v2_legacy/` (analysis, immunity, interventions, run, plotting) | Task 0 Step 2 |
| Replace-in-place `sim.py`/`people.py` (v3 → `_v2_legacy/`; new thin classes) | Task 0 Step 3; Tasks 2,3 |
| Keep active: parameters/defaults/settings/misc/utils/version/requirements/data/regression + `base.py` (Open question F) | Task 0 Steps 3,7 |
| Quarantine v3 tests exercising removed APIs → `tests/_legacy/` | Task 0 Step 4 |
| `tests/conftest.py` with `collect_ignore_glob` | Task 0 Step 6 |
| Retire the M0 `cv.v4` stub (delete `_v4.py` + `test_v4_stub.py`; remove import) | Task 0 Steps 5,7 |
| Trim `covasim/__init__.py` to the M1 surface (keep `cv.Layer`/`cv.Contacts`/`cv.Result`; add network/covid/people/sim) | Task 0 Step 7 |
| Skip v3 bit-for-bit assertions in `test_baselines.py`/`test_regression.py` (Open question G) | Task 0 Step 8 |
| `cv.Network(ss.Network)` — random single-layer 'a' + hybrid h/s/w/c, per-layer beta, age windows (school 6–22, work 22–65) | Task 1 |
| Ported `population.py` builders keep `cv.make_*` names; `make_synthpop` guarded (M9); halving + Poisson-cluster reproduced | Task 1 Step 2 |
| `cv.COVID(ss.Infection)` minimal S→E→I→R; override only `__init__`/`set_prognoses`/`step_state`; inherit `infect()` | Task 2 |
| Only infectious (not exposed) agents transmit; permanent immunity (use_waning=False semantics) | Task 2 Step 2 |
| `cv.People(ss.People)` keeps name; supplies v3 age distribution | Task 2 Step 3 |
| Thin `cv.Sim(ss.Sim)` assembles people+networks+disease; per-layer beta on the disease (decision 5); exact-count seed (Open question E) | Task 3 |
| Daily step `dt=ss.days(1)`, `start=ss.date(...)`, `dur=ss.days(n_days)` | Task 3 Step 2 |
| Reuse M0 harness; add `anchor_m1.py` (random+hybrid) + `build_summary_m1` + `contact_stats.py` | Tasks 1,4 |
| Contact-structure equivalence test (per-layer degree + age-mixing cosine) vs v3.1.8 (skips when baseline absent) | Task 1 Step 6; Task 4 Step 3 |
| Trajectory parity gate `test_m1_parity.py` (`@pytest.mark.slow`, `|z|<3`, skips when baseline absent) | Task 4 Step 4 |
| Demo plots the infection curve via Starsim `sim.plot()` | Task 3 Step 5 |
| Full non-quarantined suite passes under `COVASIM_WARNINGS=error` | Task 4 Step 5; Task 5 Step 3 |
| Provisional names (`cv.COVID`, `cv.Network`, file `covid.py`) flagged for Cliff (Open question H) | check-ins 2,3 |

## Linked documents

- [`../specs/2026-05-29-covasim-m1-basic-transmission-design.md`](../specs/2026-05-29-covasim-m1-basic-transmission-design.md) — the authoritative M1 design spec (decisions 1–7, Open questions A–H, acceptance test).
- [`2026-05-29-covasim-m0-foundation.md`](2026-05-29-covasim-m0-foundation.md) — M0 plan (the harness + quarantine scaffold this plan reuses and extends).
- [`../MIGRATION_PLAN.md`](../MIGRATION_PLAN.md) — overall plan; §M1 is this milestone's capability source; Implementation conventions govern the invariant + validation gates.
- hpvsim M01 plan/code (the in-place-replacement + quarantine template): `/home/cliffk/idm/hpvsim/docs/superpowers/plans/2026-04-29-hpvsim-m1-basic-transmission.md`.
- Starsim 3.3.x source (verify signatures at code-time): `/home/cliffk/idm/starsim/starsim/{networks.py,diseases.py,sim.py,people.py}`.

## Open questions carried from the spec (flag at the noted check-in; default path taken in this plan)

- **A** (`population.py` shape) — default (a): free-function builders feeding `cv.Network`. Flag at check-in 2.
- **B** (network RNG) — default: `ss.Dist` draws; defer `ss.RandomSafeNet` to M8. Flag at check-in 2.
- **C** (SEIR state mapping) — default: fresh states + `reset=True` + `infectious` mechanism (verify property-vs-BoolState). Flag at check-in 3.
- **D** (`beta_dist` heterogeneity) — default: `rel_trans=rel_sus=1.0` at M1; M2 reintroduces. Flag at check-in 3.
- **E** (seeding) — default: exact-count `pop_infected` seed (fall back to Bernoulli if awkward). Flag at check-in 3.
- **F** (`base.py` quarantine) — default: keep `base.py` active, `BaseSim`/`BasePeople` dormant, `cv.Layer`/`cv.Contacts`/`cv.Result`/`cv.ParsObj` exported. Flag at check-in 1.
- **G** (`test_baselines`/`test_regression`) — default: skip the v3 bit-for-bit assertions with a documented "regenerated for v4 at M10" reason. Flag at check-in 1.
- **H** (class/file names) — `cv.COVID`, `cv.Network`, file `covasim/covid.py`. Flag at check-ins 2 (Network) and 3 (COVID/file).
