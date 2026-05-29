"""M0 acceptance gate: multi-seed mean parity vs. the v3.1.8 anchor baseline.

Runs N_V4_SEEDS v4 seeds of the M0 anchor, then gates each pinned metric on

    z = (v4_mean - v3_mean) / sqrt(v3_SE^2 + v4_SE^2)

failing any metric with |z| >= Z_THRESHOLD. The v3.1.8 baseline is the gitignored
multi-seed sweep at tests/regression/v3_seeds_n{M}.json, regenerated via
`python tests/regression/multi_seed_v3.py --n 30` from a FROZEN v3.1.8 env.

Marked slow so the fast PR job skips it; run it locally or in the nightly job:
    cd tests && pytest test_m0_parity.py -m slow -v
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from regression.anchor import make_sim  # noqa: E402
from regression.short_summary import build_summary, SKIP_KEYS  # noqa: E402
from regression.parity import parity_gate  # noqa: E402

N_V4_SEEDS = 10                 # 10 v4 seeds vs 30 v3 seeds (hpvsim's committed ratio)
M_V3_SEEDS = 30
Z_THRESHOLD = 3.0
BASELINE_PATH = Path(__file__).parent / 'regression' / f'v3_seeds_n{M_V3_SEEDS}.json'


def _run_v4_seeds(n, start_seed=0):
    rows = []
    for seed in range(start_seed, start_seed + n):
        sim = make_sim(rand_seed=int(seed))
        sim.run()
        rows.append(build_summary(sim))
    return rows


@pytest.mark.slow
@pytest.mark.skip(reason='M0 anchor (hybrid + waning) needs M2+ features; superseded by anchor_m1 in M1.')
def test_m0_anchor_parity():
    if not BASELINE_PATH.exists():
        pytest.skip(
            f'Missing v3.1.8 M0 baseline at {BASELINE_PATH}. Regenerate via '
            f'`python tests/regression/multi_seed_v3.py --n {M_V3_SEEDS}` from a '
            f'frozen v3.1.8 covasim env.'
        )
    v3_rows = json.loads(BASELINE_PATH.read_text())
    v4_rows = _run_v4_seeds(N_V4_SEEDS)

    failures = parity_gate(v4_rows, v3_rows, z_threshold=Z_THRESHOLD, skip_keys=SKIP_KEYS)
    if failures:
        details = '\n'.join(f'  {name:<24} z={z:+.2f}' for name, z in failures)
        pytest.fail(
            f'M0 anchor parity drift exceeds |z|>={Z_THRESHOLD} on {len(failures)} '
            f'metrics (v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):\n{details}'
        )
    return v4_rows
