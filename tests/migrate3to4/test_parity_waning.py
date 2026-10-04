"""Waning-immunity parity gate: multi-seed parity vs the v3.1.8 use_waning=True baseline.

Runs N v4 seeds of the waning anchor (the multi-variant scenario WITH NAbs, use_waning=True) and
gates the FULL pinned metric set on |z| < 5 vs the v3.1.8 baseline. The v3 baseline regime is
already use_waning=True (the variants anchor's v3 branch), so this gate reuses
`v3_variants_<pt>_seeds_n*.json` -- no separate baseline.

Unlike the variants gate (whose static cross-immunity diverges from v3 on the per-variant escape
dynamics, so only a convergent subset is gated), the NAb engine reproduces v3's NAb-weighted
protection: every pinned metric -- aggregate burden AND per-variant wild/alpha/delta counts --
converges to within |z|<3.5 of v3. So this gate covers the WHOLE set. Skips cleanly when the
baseline is absent.

    cd tests && pytest migrate3to4/test_parity_waning.py -m slow -v
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from migrate3to4.anchor_waning import make_sim  # noqa: E402
from migrate3to4.short_summary import build_summary_variants, METRIC_KEYS_VARIANTS  # noqa: E402
from migrate3to4.parity import parity_gate, _mean_se  # noqa: E402

N_V4_SEEDS = 10
M_V3_SEEDS = 30
Z_THRESHOLD = 5.0  # as for the other gates (Starsim CRN vs v3 numba RNG residual); metrics land < 3.5.


def _baseline_path(pop_type):
    # Reuses the variants v3.1.8 baseline (both are the use_waning=True regime).
    return Path(__file__).parent / f'v3_variants_{pop_type}_seeds_n{M_V3_SEEDS}.json'


def _run_v4_seeds(pop_type, n):
    rows = []
    for seed in range(n):
        sim = make_sim(pop_type=pop_type, rand_seed=seed)
        sim.run()
        rows.append(build_summary_variants(sim))
    return rows


@pytest.mark.slow
@pytest.mark.parametrize('pop_type', ['random', 'hybrid'])
def test_waning_parity(pop_type):
    baseline = _baseline_path(pop_type)
    if not baseline.exists():
        pytest.skip(
            f'Missing v3.1.8 baseline at {baseline}. Regenerate via '
            f'`python tests/migrate3to4/multi_seed_v3.py --anchor variants_{pop_type} --n {M_V3_SEEDS}` '
            f'from a frozen v3.1.8 covasim env (the use_waning=True regime, shared with the variants gate).'
        )
    v3_rows = json.loads(baseline.read_text())
    v4_rows = _run_v4_seeds(pop_type, N_V4_SEEDS)

    # Diagnostic table (full metric set; all of them are gated).
    print(f'\nWaning {pop_type} parity (use_waning=True; v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):')
    for key in METRIC_KEYS_VARIANTS:
        s3 = _mean_se(v3_rows, key); s4 = _mean_se(v4_rows, key)
        if s3 is None or s4 is None:
            continue
        se = math.sqrt(s3[1] ** 2 + s4[1] ** 2)
        z = (s4[0] - s3[0]) / se if se > 0 else float('inf')
        print(f'  {key:<28} v3={s3[0]:>10.1f} v4={s4[0]:>10.1f} z={z:+7.2f}')

    failures = parity_gate(v4_rows, v3_rows, z_threshold=Z_THRESHOLD,
                           skip_keys={'_seed', '_total_pop', 'n_alive'})
    if failures:
        details = '\n'.join(f'  {name:<28} z={z:+.2f}' for name, z in failures)
        pytest.fail(
            f'Waning {pop_type} parity drift exceeds |z|>={Z_THRESHOLD} on {len(failures)} '
            f'metrics (v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):\n{details}'
        )
    return v4_rows
