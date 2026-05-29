"""Multi-seed sweep of the M0 anchor in the CURRENT (v4) env, for ad-hoc diffing.

Identical sweep to multi_seed_v3.py but defaults to writing v4_seeds_n{N}.json.
The pytest parity gate (tests/test_m0_parity.py) generates v4 seeds in-process
and does not require this file; it exists for manual local comparison.

    python tests/regression/multi_seed_v4.py --n 10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_seed_v3 import main  # noqa: E402


if __name__ == '__main__':
    # Reuse the v3 sweep machinery; just change the default output name.
    argv = sys.argv[1:]
    if not any(a.startswith('--out') for a in argv):
        # Derive N from --n (default 30) for the default v4 filename.
        n = 30
        if '--n' in argv:
            n = int(argv[argv.index('--n') + 1])
        argv = argv + ['--out', str(Path(__file__).resolve().parent / f'v4_seeds_n{n}.json')]
    sys.exit(main(argv))
