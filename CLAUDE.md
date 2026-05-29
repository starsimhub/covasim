# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Covasim is a stochastic agent-based simulator for COVID-19. It is a mature, production-stable library (v3.1.8) whose audience is **scientists, not software developers** — this drives nearly every design decision (see Conventions below). Covasim is now part of the [Starsim](https://starsim.org) framework, but this repo is its own standalone package.

## Commands

Install for development (editable, with test + docs extras):

```
pip install -e .[dev]      # or .[test] for just test deps
```

All test commands run **from the `tests/` directory**:

```
cd tests
./run_tests                # all integration tests, parallel, with timings
./check_coverage           # tests + HTML coverage report (htmlcov/index.html)
./check_style              # pylint over ../covasim
./check_everything         # integration + unit tests + coverage + docs
pytest test_sim.py         # run one test file
pytest test_sim.py::test_microsim   # run one test
```

`run_tests` sets `COVASIM_INTERACTIVE=0` (no plots) and `COVASIM_WARNINGS=error` (warnings become failures) — match this when reproducing test behavior. `pytest.ini` forces `SCIRIS_BACKEND=agg`.

### Baselines / regression

`test_baselines.py` and `test_regression.py` guard against unintended numerical changes. If you **intentionally** change model results, regenerate the saved values:

```
cd tests
./update_baseline          # rewrites baseline.json, benchmark.json, and ../covasim/regression/
```

Do not edit `baseline.json` by hand. The `covasim/regression/` folder stores default parameter snapshots per version.

## Architecture

Two core objects: **`People`** (per-agent health state) and **`Sim`** (runs the model, computes results, plots). A `Sim` holds a `People`, a parameters dict, and lists of interventions/analyzers.

Modules in `covasim/` are imported in dependency order (see `__init__.py`), fundamental → complex:

- `settings.py` / `defaults.py` — user options, default colors/plots/parameters
- `parameters.py` — builds the parameters dict and loads input data
- `base.py` — `ParsObj` (fundamental base class), `BaseSim`, `BasePeople`
- `people.py` / `population.py` — `People` class; population creation (ages, contacts)
- `interventions.py` — `Intervention` base class + concrete interventions (testing, tracing, vaccination, etc.); these dynamically modify parameters during a run
- `immunity.py` — variants/strains, waning immunity, neutralizing antibodies
- `sim.py` — `Sim`: initialize → run → plot (the heavy lifting)
- `run.py` — `MultiSim`, `Scenarios`, parallel runs
- `analysis.py` — `Analyzer` (runs during a sim), `Fit` (model-vs-data), `TransTree`
- `utils.py` — Numba-accelerated random number / hot-loop helpers

Convention: everything is used flat as `cv.Sim()`, `cv.test_prob()`, never `cv.sim.Sim()`. Heavily relies on **sciris** (`import sciris as sc`) for I/O, containers (`odict`), and helpers.

The `bin/` folder is a CLI wrapper (`covasim --pars "{...}"`); not Windows-compatible. The `covasim/data/` folder holds loading scripts + demographic data; the root `data/` folder holds epi data (largely not committed).

## Conventions (from style_guide.md)

These are deliberate and differ from typical Python practice — follow them:

- **No type annotations in code.** Functions accept flexible inputs (a date may be an int, string, or `datetime`; a quantity may be scalar/list/array). Coerce inputs instead of constraining them — e.g. use `sc.toarray()` to accept a list where an array is needed. Type info goes in **docstrings only**.
- **Sensible defaults everywhere.** Default to `None` and fill in a sensible value inside the function. Prefer keyword arguments with defaults over hard-coded locals so users can override.
- **Optimize for the reader, not the writer.** Avoid lambdas, dunder overriding, and deep class inheritance unless necessary — they raise ramp-up cost for non-developer users. Favor clarity, even line comments. If a number comes from a paper, link the paper in a comment.
- Clear scientific logic wins over "clean" code style.
- Style is enforced by pylint (`tests/check_style`, config in `tests/pylintrc`); base is Google's style guide with documented exceptions in `style_guide.md`.

Python 3.9–3.13. Update `covasim/version.py` and `CHANGELOG.md` together when bumping versions.
