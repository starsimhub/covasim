"""
Defines the People class for Covasim on the Starsim base.

``cv.People(ss.People)`` keeps Covasim's public class name and defaults to
Covasim's age distribution (the 2018 Seattle pyramid in ``defaults.default_age_data``)
so realized ages -- and hence the age-mixing structure of the contact network --
match v3.1.8. All per-agent disease state lives on ``cv.COVID`` (via define_states),
not on People; Starsim auto-aggregates module states onto People.
"""
import numpy as np
import starsim as ss

from . import defaults as cvd
from . import compat as cvc
from . import plotting as cvplt

__all__ = ['People']


def convert_age_data(age_data):
    """Convert a v3 Nx3 [age_min, age_max, fraction] age table to the Nx2 [age_lower_edge, value] array ss.People expects.

    ss.People reads the ages as lower bin edges, so the upper edge of the last bin is added as
    a final row with a value of 0. As in v3, ages are uniform within [age_min, age_max+1).
    """
    d = np.asarray(age_data, dtype=float)
    return np.vstack([d[:, [0, 2]], [d[-1, 1] + 1, 0]])


class People(cvc.V3People, ss.People):
    """Covasim People on the Starsim base; defaults to Covasim's age distribution.

    Args:
        n_agents (int): number of agents.
        age_data (array/df): optional age distribution override; defaults to Covasim's.
        kwargs: forwarded to ``ss.People`` (e.g. ``extra_states``).
    """

    def __init__(self, n_agents, age_data=None, **kwargs):
        if age_data is None:
            age_data = convert_age_data(cvd.default_age_data)
        super().__init__(n_agents, age_data=age_data, **kwargs)
        return

    def plot(self, *args, **kwargs):
        """Plot statistics of the population -- age distribution, numbers of contacts, and overall weight of contacts (number of contacts multiplied by beta per layer); see cv.plot_people()"""
        return cvplt.plot_people(people=self, *args, **kwargs)
