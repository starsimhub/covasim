"""Stub v4.0 Sim on the Starsim base (continuous-runnability invariant).

M0 ships NO real port code. This module exists solely to prove, from day one,
that the Starsim base imports and a Covasim-namespaced Sim subclass constructs
and runs to completion (Implementation conventions item 1: cv.Sim().run() must
return results at every commit on starsim-port). It is NOT the real port: it runs
a degenerate, disease-free sim. The real cv.Sim is built out starting in M1.

Exposed as cv.v4.Sim to avoid disturbing the existing v3.1.8 cv.Sim that the
current test suite depends on. The two coexist until the port lands in cv.Sim.

    import covasim as cv
    sim = cv.v4.Sim(n_agents=100).run()   # returns a run ss.Sim
"""
import starsim as ss


class Sim(ss.Sim):
    """Minimal v4 Sim stub: a disease-free Starsim sim that runs.

    Args:
        n_agents (int): number of agents (default 100, small for a fast smoke run).
        kwargs: forwarded to ss.Sim (e.g. start, stop, dt).

    **Example**::

        sim = cv.v4.Sim(n_agents=100).run()
    """

    def __init__(self, n_agents=100, **kwargs):
        super().__init__(n_agents=n_agents, **kwargs)
        return
