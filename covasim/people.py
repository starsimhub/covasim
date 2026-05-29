"""
Defines the People class for Covasim on the Starsim base.

M1 placeholder: the real thin ``cv.People(ss.People)`` (keeps the public name,
supplies Covasim's default age distribution) is implemented in M1 Task 2. This
shell exists so ``import covasim`` and the continuous-runnability invariant hold
at the M1 quarantine check-in. The v3 ``People`` health-state machine is
quarantined at ``covasim/_v2_legacy/people.py``.
"""
import starsim as ss

__all__ = ['People']


class People(ss.People):
    """Covasim People on the Starsim base (placeholder; see M1 Task 2)."""
    pass
