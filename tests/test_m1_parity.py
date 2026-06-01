"""M1 acceptance gate: multi-seed trajectory parity vs the v3.1.8 M1 baseline.

Runs N v4 seeds of the M1 anchor (random and hybrid), and gates each pinned metric
on |z| < 3 vs the gitignored v3.1.8 baseline. Skips cleanly when the baseline is
absent (it is generated from a frozen v3.1.8 env -- see tests/regression/README.md).
Marked slow so the fast PR job skips it; run locally or nightly:

    cd tests && pytest test_m1_parity.py -m slow -v
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from regression.anchor_m1 import make_sim  # noqa: E402
from regression.short_summary import build_summary_m1  # noqa: E402
from regression.parity import parity_gate  # noqa: E402

N_V4_SEEDS = 10
M_V3_SEEDS = 30
Z_THRESHOLD = 3.0


def _baseline_path(pop_type):
    return Path(__file__).parent / 'regression' / f'v3_m1_{pop_type}_seeds_n{M_V3_SEEDS}.json'


def _run_v4_seeds(pop_type, n):
    rows = []
    for seed in range(n):
        sim = make_sim(pop_type=pop_type, rand_seed=seed)
        sim.run()
        rows.append(build_summary_m1(sim))
    return rows


@pytest.mark.slow
@pytest.mark.parametrize('pop_type', ['random', 'hybrid'])
def test_m1_anchor_parity(pop_type):
    baseline = _baseline_path(pop_type)
    if not baseline.exists():
        pytest.skip(
            f'Missing v3.1.8 M1 baseline at {baseline}. Regenerate via '
            f'`python tests/regression/multi_seed_v3.py --anchor m1_{pop_type} --n {M_V3_SEEDS}` '
            f'from a frozen v3.1.8 covasim env.'
        )
    v3_rows = json.loads(baseline.read_text())
    v4_rows = _run_v4_seeds(pop_type, N_V4_SEEDS)
    failures = parity_gate(v4_rows, v3_rows, z_threshold=Z_THRESHOLD)
    if failures:
        details = '\n'.join(f'  {name:<22} z={z:+.2f}' for name, z in failures)
        pytest.fail(
            f'M1 {pop_type} parity drift exceeds |z|>={Z_THRESHOLD} on {len(failures)} '
            f'metrics (v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):\n{details}'
        )
    return v4_rows
