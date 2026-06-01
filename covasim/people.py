"""
Defines the People class for Covasim on the Starsim base.

``cv.People(ss.People)`` keeps Covasim's public class name and defaults to
Covasim's age distribution (the 2018 Seattle pyramid in ``defaults.default_age_data``)
so realized ages -- and hence the age-mixing structure of the contact network --
match v3.1.8. All per-agent disease state lives on ``cv.COVID`` (via define_states),
not on People; Starsim auto-aggregates module states onto People. The v3 ``People``
health-state machine is quarantined at ``covasim/_v2_legacy/people.py``.
"""
import numpy as np
import starsim as ss

from . import defaults as cvd

__all__ = ['People']


def _default_age_data():
    """Covasim's default age pyramid as the Nx2 [age_lower_edge, value] array ss.People expects.

    ``defaults.default_age_data`` is an Nx3 [age_min, age_max, fraction] table; ss.People's
    ``get_age_dist`` reads ages as lower bin edges + a value column, so drop age_max.
    """
    d = np.asarray(cvd.default_age_data, dtype=float)
    return np.c_[d[:, 0], d[:, 2]]


class People(ss.People):
    """Covasim People on the Starsim base; defaults to Covasim's age distribution.

    Args:
        n_agents (int): number of agents.
        age_data (array/df): optional age distribution override; defaults to Covasim's.
        kwargs: forwarded to ``ss.People`` (e.g. ``extra_states``).
    """

    # v3 stored per-agent disease state on People; in v4 it lives on the COVID disease module. These
    # names are proxied through to the disease so v3-style ``sim.people.exposed`` (etc.) keeps working
    # in custom interventions/analyzers. (Reads return the live disease array, so index-assignment
    # like ``sim.people.rel_sus[inds] = 0`` writes through to the disease.)
    _COVID_STATE_PROXY = frozenset([
        'susceptible', 'exposed', 'infectious', 'symptomatic', 'severe', 'critical', 'recovered',
        'dead', 'infected', 'diagnosed', 'tested', 'known_contact', 'quarantined', 'isolated',
        'vaccinated', 'rel_sus', 'rel_trans', 'doses', 'peak_nab', 'nab',
    ])

    def __init__(self, n_agents, age_data=None, **kwargs):
        if age_data is None:
            age_data = _default_age_data()
        super().__init__(n_agents, age_data=age_data, **kwargs)
        return

    def __getattr__(self, key):
        """Proxy v3 per-agent disease-state reads through to the COVID module (see _COVID_STATE_PROXY).

        ``__getattr__`` is only consulted when normal attribute lookup fails, so it cannot shadow real
        ``People`` attributes; it falls back to the standard AttributeError for everything else.
        """
        if key in People._COVID_STATE_PROXY:
            sim = self.__dict__.get('sim', None)  # __dict__ access avoids re-triggering __getattr__
            if sim is not None and hasattr(sim, 'diseases') and 'covid' in sim.diseases:
                return getattr(sim.diseases['covid'], key)
        raise AttributeError(f"'People' object has no attribute '{key}'")
