"""
Defines the Sim class for Covasim on the Starsim base.

M1 placeholder: the real thin ``cv.Sim(ss.Sim)`` (assembles people + networks +
the COVID disease) is implemented in M1 Task 3. This shell subclasses ``ss.Sim``
with a small default population so ``cv.Sim().run()`` returns results (the
continuous-runnability invariant) at the M1 quarantine check-in. The v3 ``Sim``
engine is quarantined at ``covasim/_v2_legacy/sim.py``.
"""
import starsim as ss

__all__ = ['Sim']


class Sim(ss.Sim):
    """Covasim Sim on the Starsim base (placeholder; see M1 Task 3).

    Args:
        n_agents (int): number of agents (default 2000; replaced by ``pop_size`` in Task 3).
        kwargs: forwarded to ``ss.Sim``.
    """

    def __init__(self, n_agents=2000, **kwargs):
        super().__init__(n_agents=n_agents, **kwargs)
        return
