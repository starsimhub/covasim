# Integration tests

This folder contains the core tests for Covasim. Recommended usage is `./check_coverage` or `./run_tests`. You can also use `pytest` to run all the tests in the folder. Description of other scripts included for convenience are below.

## check_coverage

Run all tests with parallelization, determine code coverage, and create an HTML report.

## check_everything

Run integration tests, unit tests, coverage, and build docs.

## run_tests

Run all tests, with parallelization, and showing how long each test took.

## update_baseline

The test `test_baselines.py` checks to see if results changed unintentionally. If you *intended* to change them, run this script to update the saved results. It also writes default parameter values to the `../covasim/regression` folder.

## migrate3to4 (v4.0 migration harness)

There is also a self-contained regression harness under `tests/migrate3to4/` used for the v3.1.8 -> v4.0 Starsim port. It is documented in [`tests/migrate3to4/README.md`](migrate3to4/README.md). It compares a v4 run of a pinned anchor scenario against a locally-generated, gitignored v3.1.8 baseline: a fast informational `+/-10%` drift CLI (`compare.py`) and multi-seed z-score parity gates, one per feature area (`migrate3to4/test_parity_*.py`). It layers on top of `baseline.json` / `test_baselines.py` rather than replacing them.
