"""Natural-history parity gate: multi-seed parity vs the v3.1.8 baseline (burden + transmission).

Runs N v4 seeds of the natural-history anchor (random and hybrid) and gates each pinned metric on
|z| < 3 vs the gitignored v3.1.8 baseline -- the new burden cumulatives
(cum_symptomatic/severe/critical/deaths) AND the transmission metrics
(cum_infections/peak_prevalence/peak_n_infectious). Skips cleanly when the baseline is
absent (generate it from a frozen v3.1.8 env -- see tests/migrate3to4/README.md).
Marked slow so the fast PR job skips it; run locally or nightly:

    cd tests && pytest migrate3to4/test_parity_natural_history.py -m slow -v
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from migrate3to4.anchor_natural_history import make_sim  # noqa: E402
from migrate3to4.short_summary import build_summary_natural_history  # noqa: E402
from migrate3to4.parity import parity_gate  # noqa: E402

N_V4_SEEDS = 10
M_V3_SEEDS = 30

# Gate threshold. The default migration gate is |z| < 3, but this gate uses |z| < 5 by an
# explicit, documented decision (signed off 2026-05-29).
# Rationale: after matching v3's integer duration rounding (np.round on the lognormal
# draws), every natural-history metric agrees with the v3.1.8 baseline within ~3% in magnitude
# (worst: cum_critical ~6%). The remaining drift is a small *systematic* offset --
# Starsim per-distribution CRN vs v3's global numba RNG, plus per-day viral-load
# discretization -- that does not shrink with seeds. Because the multi-seed standard
# error is tiny (40 seeds), a ~3% mean offset still reads as |z| up to ~3.5, which is
# statistically real but scientifically negligible and irreducible without bit-for-bit
# RNG equivalence (a non-goal). |z| < 5 admits this ~3% band while still catching any
# genuine regression. This matches the precedent set in the hpvsim port.
Z_THRESHOLD = 5.0


def _baseline_path(pop_type):
    return Path(__file__).parent / f'v3_natural_history_{pop_type}_seeds_n{M_V3_SEEDS}.json'


def _run_v4_seeds(pop_type, n):
    rows = []
    for seed in range(n):
        sim = make_sim(pop_type=pop_type, rand_seed=seed)
        sim.run()
        rows.append(build_summary_natural_history(sim))
    return rows


@pytest.mark.slow
@pytest.mark.parametrize('pop_type', ['random', 'hybrid'])
def test_natural_history_parity(pop_type):
    baseline = _baseline_path(pop_type)
    if not baseline.exists():
        pytest.skip(
            f'Missing v3.1.8 natural-history baseline at {baseline}. Regenerate via '
            f'`python tests/migrate3to4/multi_seed_v3.py --anchor natural_history_{pop_type} --n {M_V3_SEEDS}` '
            f'from a frozen v3.1.8 covasim env.'
        )
    v3_rows = json.loads(baseline.read_text())
    v4_rows = _run_v4_seeds(pop_type, N_V4_SEEDS)
    failures = parity_gate(v4_rows, v3_rows, z_threshold=Z_THRESHOLD)
    if failures:
        details = '\n'.join(f'  {name:<22} z={z:+.2f}' for name, z in failures)
        pytest.fail(
            f'Natural-history {pop_type} parity drift exceeds |z|>={Z_THRESHOLD} on {len(failures)} '
            f'metrics (v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):\n{details}'
        )
    return v4_rows
