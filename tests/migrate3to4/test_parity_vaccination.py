"""Vaccination parity gate: multi-seed parity vs the v3.1.8 vaccination baseline.

Runs N v4 seeds of the vaccination anchor (single-variant, use_waning=True, vaccinate_prob('pfizer')) on
random + hybrid and gates the pinned metrics on |z| < 5 vs the gitignored v3.1.8 baseline -- the
burden/shape (cum_infections/cum_severe/cum_deaths/peak_n_infectious) AND the vaccination outcomes
(cum_doses/cum_vaccinated). Skips when the baseline is absent.

    cd tests && pytest migrate3to4/test_parity_vaccination.py -m slow -v

|z| < 5 as for the other parity gates (the documented Starsim-CRN-vs-v3-RNG residual). Vaccine
immunity shares the waning-immunity NAb pipeline (which converges to v3 at |z|<3.5), so the
vaccinated trajectory tracks v3.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from migrate3to4.anchor_vaccination import make_sim  # noqa: E402
from migrate3to4.short_summary import build_summary_vaccination, METRIC_KEYS_VACCINATION  # noqa: E402
from migrate3to4.parity import parity_gate, _mean_se  # noqa: E402

N_V4_SEEDS = 10
M_V3_SEEDS = 30
Z_THRESHOLD = 5.0


def _baseline_path(pop_type):
    return Path(__file__).parent / f'v3_vaccination_{pop_type}_seeds_n{M_V3_SEEDS}.json'


def _run_v4_seeds(pop_type, n):
    rows = []
    for seed in range(n):
        sim = make_sim(pop_type=pop_type, rand_seed=seed)
        sim.run()
        rows.append(build_summary_vaccination(sim))
    return rows


@pytest.mark.slow
@pytest.mark.parametrize('pop_type', ['random', 'hybrid'])
def test_vaccination_parity(pop_type):
    baseline = _baseline_path(pop_type)
    if not baseline.exists():
        pytest.skip(
            f'Missing v3.1.8 vaccination baseline at {baseline}. Regenerate via '
            f'`python tests/migrate3to4/multi_seed_v3.py --anchor vaccination_{pop_type} --n {M_V3_SEEDS}` '
            f'from a frozen v3.1.8 covasim env.'
        )
    v3_rows = json.loads(baseline.read_text())
    v4_rows = _run_v4_seeds(pop_type, N_V4_SEEDS)
    print(f'\nVaccination {pop_type} parity (v3 n={len(v3_rows)}, v4 n={len(v4_rows)}):')
    for key in METRIC_KEYS_VACCINATION:
        s3 = _mean_se(v3_rows, key); s4 = _mean_se(v4_rows, key)
        if s3 is None or s4 is None:
            continue
        se = math.sqrt(s3[1] ** 2 + s4[1] ** 2)
        z = (s4[0] - s3[0]) / se if se > 0 else float('inf')
        print(f'  {key:<22} v3={s3[0]:>10.1f} v4={s4[0]:>10.1f} z={z:+7.2f}')
    failures = parity_gate(v4_rows, v3_rows, z_threshold=Z_THRESHOLD,
                           skip_keys={'_seed', '_total_pop', 'n_alive'})
    if failures:
        details = '\n'.join(f'  {name:<22} z={z:+.2f}' for name, z in failures)
        pytest.fail(f'Vaccination {pop_type} parity drift exceeds |z|>={Z_THRESHOLD}:\n{details}')
    return v4_rows
