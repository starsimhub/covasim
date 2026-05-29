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

    def __init__(self, n_agents, age_data=None, **kwargs):
        if age_data is None:
            age_data = _default_age_data()
        super().__init__(n_agents, age_data=age_data, **kwargs)
        return
