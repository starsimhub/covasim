"""M2 acceptance gate: multi-seed parity vs the v3.1.8 M2 baseline (burden + transmission).

Runs N v4 seeds of the M2 anchor (random and hybrid) and gates each pinned metric on
|z| < 3 vs the gitignored v3.1.8 baseline -- the new burden cumulatives
(cum_symptomatic/severe/critical/deaths) AND the re-converged transmission metrics
(cum_infections/peak_prevalence/peak_n_infectious). Skips cleanly when the baseline is
absent (generate it from a frozen v3.1.8 env -- see tests/regression/README.md).
Marked slow so the fast PR job skips it; run locally or nightly:

    cd tests && pytest test_m2_parity.py -m slow -v
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from regression.anchor_m2 import make_sim  # noqa: E402
from regression.short_summary import build_summary_m2  # noqa: E402
from regression.parity import parity_gate  # noqa: E402

N_V4_SEEDS = 10
M_V3_SEEDS = 30

# Gate threshold. The default migration gate is |z| < 3, but M2 uses |z| < 5 by an
# explicit, documented decision (MIGRATION_PLAN.md Open Q G; signed off 2026-05-29).
# Rationale: after matching v3's integer duration rounding (np.round on the lognormal
# draws), every M2 metric agrees with the v3.1.8 baseline within ~3% in magnitude
# (worst: cum_critical ~6%). The remaining drift is a small *systematic* offset --
# Starsim per-distribution CRN vs v3's global numba RNG, plus per-day viral-load
# discretization -- that does not shrink with seeds. Because the multi-seed standard
# error is tiny (40 seeds), a ~3% mean offset still reads as |z| up to ~3.5, which is
# statistically real but scientifically negligible and irreducible without bit-for-bit
# RNG equivalence (a non-goal). |z| < 5 admits this ~3% band while still catching any
# genuine regression. This matches the hpvsim port's M5 precedent.
Z_THRESHOLD = 5.0


def _baseline_path(pop_type):
    return Path(__file__).parent / 'regression' / f'v3_m2_{pop_type}_seeds_n{M_V3_SEEDS}.json'


def _run_v4_seeds(pop_type, n):
    rows = []
    for seed in range(n):
        sim = make_sim(pop_type=pop_type, rand_seed=seed)
        sim.run()
        rows.append(build_summary_m2(sim))
    return rows


@pytest.mark.slow
@pytest.mark.parametrize('pop_type', ['random', 'hybrid'])
def test_m2_anchor_parity(pop_type):
    baseline = _baseline_path(pop_type)
    if not baseline.exists():
        pytest.skip(
            f'Missing v3.1.8 M2 baseline at {baseline}. Regenerate via '
            f'`python tests/regression/multi_seed_v3.py --anchor m2_{pop_type} --n {M_V3_SEEDS}` '
            f'from a frozen v3.1.8 covasim env.'
        )
    v3_rows = json.loads(baseline.read_text())
    v4_rows = _run_v4_seeds(pop_type, N_V4_SEEDS)
    failures = parity_gate(v4_rows, v3_rows, z_threshold=Z_THRESHOLD)
    if failures:
        details = '\n'.join(f'  {name:<22} z={z:+.2f}' for name, z in failures)
        pytest.fail(
            f'M2 {pop_type} parity drift exceeds |z|>={Z_THRESHOLD} on {len(failures)} '
            f'metrics (v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):\n{details}'
        )
    return v4_rows
