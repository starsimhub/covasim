"""
Contact networks for Covasim on the Starsim base.

M1 placeholder: the real ``cv.Network(ss.Network)`` (random + hybrid backends,
lift-and-shifted from ``population.py``) is implemented in M1 Task 1. This shell
exists so ``import covasim`` and the continuous-runnability invariant hold at the
M1 quarantine check-in.
"""
import starsim as ss

__all__ = ['Network']


class Network(ss.Network):
    """Covasim contact layer on the Starsim base (placeholder; see M1 Task 1)."""
    pass
