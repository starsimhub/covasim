"""M4 acceptance gate: multi-seed parity vs the v3.1.8 use_waning=True baseline.

Runs N v4 seeds of the M4 anchor (the M3 multi-variant scenario WITH NAbs, use_waning=True) and
gates the FULL pinned metric set on |z| < 5 vs the v3.1.8 baseline. The v3 baseline regime is
already use_waning=True (the M3 anchor's v3 branch), so M4 reuses `v3_m3_<pt>_seeds_n*.json` --
no separate baseline.

Unlike M3 (whose static cross-immunity diverged from v3 on the per-variant escape dynamics, so only
a convergent subset was gated), M4's NAb engine reproduces v3's NAb-weighted protection: every
pinned metric -- aggregate burden AND per-variant wild/alpha/delta counts -- re-converges to within
|z|<3.5 of v3. So M4 gates the WHOLE set. Skips cleanly when the baseline is absent.

    cd tests && pytest test_m4_parity.py -m slow -v
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from regression.anchor_m4 import make_sim  # noqa: E402
from regression.short_summary import build_summary_m3, METRIC_KEYS_M3  # noqa: E402
from regression.parity import parity_gate, _mean_se  # noqa: E402

N_V4_SEEDS = 10
M_V3_SEEDS = 30
Z_THRESHOLD = 5.0  # documented as M2/M3 (Starsim CRN vs v3 numba RNG residual); M4 metrics land < 3.5.


def _baseline_path(pop_type):
    # M4 reuses the M3 v3.1.8 baseline (both are the use_waning=True regime).
    return Path(__file__).parent / 'regression' / f'v3_m3_{pop_type}_seeds_n{M_V3_SEEDS}.json'


def _run_v4_seeds(pop_type, n):
    rows = []
    for seed in range(n):
        sim = make_sim(pop_type=pop_type, rand_seed=seed)
        sim.run()
        rows.append(build_summary_m3(sim))
    return rows


@pytest.mark.slow
@pytest.mark.parametrize('pop_type', ['random', 'hybrid'])
def test_m4_anchor_parity(pop_type):
    baseline = _baseline_path(pop_type)
    if not baseline.exists():
        pytest.skip(
            f'Missing v3.1.8 baseline at {baseline}. Regenerate via '
            f'`python tests/regression/multi_seed_v3.py --anchor m3_{pop_type} --n {M_V3_SEEDS}` '
            f'from a frozen v3.1.8 covasim env (the use_waning=True regime, shared by M3 and M4).'
        )
    v3_rows = json.loads(baseline.read_text())
    v4_rows = _run_v4_seeds(pop_type, N_V4_SEEDS)

    # Diagnostic table (full metric set; M4 gates ALL of them).
    print(f'\nM4 {pop_type} parity (use_waning=True; v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):')
    for key in METRIC_KEYS_M3:
        m3 = _mean_se(v3_rows, key); m4 = _mean_se(v4_rows, key)
        if m3 is None or m4 is None:
            continue
        se = math.sqrt(m3[1] ** 2 + m4[1] ** 2)
        z = (m4[0] - m3[0]) / se if se > 0 else float('inf')
        print(f'  {key:<28} v3={m3[0]:>10.1f} v4={m4[0]:>10.1f} z={z:+7.2f}')

    failures = parity_gate(v4_rows, v3_rows, z_threshold=Z_THRESHOLD,
                           skip_keys={'_seed', '_total_pop', 'n_alive'})
    if failures:
        details = '\n'.join(f'  {name:<28} z={z:+.2f}' for name, z in failures)
        pytest.fail(
            f'M4 {pop_type} parity drift exceeds |z|>={Z_THRESHOLD} on {len(failures)} '
            f'metrics (v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):\n{details}'
        )
    return v4_rows
