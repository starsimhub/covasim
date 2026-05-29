"""
The COVID-19 disease module for Covasim on the Starsim base.

M1 placeholder: the real ``cv.COVID(ss.Infection)`` (minimal single-variant
S->E->I->R transmission) is implemented in M1 Task 2. This shell exists so
``import covasim`` and the continuous-runnability invariant hold at the M1
quarantine check-in.
"""
import starsim as ss

__all__ = ['COVID']


class COVID(ss.Infection):
    """Single-variant COVID-19 disease (placeholder; see M1 Task 2)."""
    pass
