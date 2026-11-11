"""
Defines the People class.
"""
import numpy as np
import sciris as sc
import starsim as ss

from . import defaults as cvd
from . import geography as cvgeo
from . import compat as cvc
from . import plotting as cvplt

__all__ = ['People']


def convert_age_data(age_data):
    """
    Convert Covasim's age data, a table of (minimum age, maximum age, fraction) rows (e.g. from
    ``cv.data.get_age_distribution()``), to the (lower edge, value) rows that ``ss.People`` expects.

    ss.People reads the ages as lower bin edges, so the upper edge of the last bin is added as
    a final row with a value of 0. As in v3, ages are uniform within [age_min, age_max+1).
    """
    d = np.asarray(age_data, dtype=float)
    return np.vstack([d[:, [0, 2]], [d[-1, 1] + 1, 0]])


class People(cvc.V3People, ss.People):
    """
    The people in the simulation: their ages and sexes, and (once in a sim) their disease states.

    The disease states (e.g. ``people.exposed``) are stored on the COVID module
    (``sim.diseases.covid``), but can be read from the people as in v3. Dates such as
    ``people.date_infectious`` are the time indices (e.g. ``ti_infectious``), since the timestep
    is one day. The per-agent arrays only include agents who are alive, except for
    ``people.dead`` and ``people.date_dead``, which cover every agent ever created.

    Each person also has a home location (``x`` and ``y``, in degrees of longitude and latitude; NaN if
    there is no geography), a ``region``, and a ``sewershed`` (see geography.py), which are set from the
    sim's geography parameters (e.g. ``boundary_shapefile``) when the people are initialized.

    Args:
        n_agents (int): the number of agents
        age_data (array/dataframe): the age distribution, either as Covasim's (minimum age, maximum age, fraction) rows (e.g. from ``cv.data.get_age_distribution()``) or as Starsim's (lower edge, value) rows; defaults to Covasim's default age distribution
        geo_pars (dict): the geography parameters (e.g. ``boundary_shapefile``; see ``cv.make_pars()``); set by ``cv.Sim`` if not supplied
        pop_type (str): the population type, which the geography depends on (with 'rank', everyone is given a location); set by ``cv.Sim`` if not supplied
        kwargs (dict): passed to ``ss.People`` (e.g. ``extra_states``)

    **Example**::

        people = cv.People(1000, age_data=cv.data.get_age_distribution('nigeria'))
        sim = cv.Sim(people=people, pop_size=1000)
    """

    def __init__(self, n_agents, age_data=None, geo_pars=None, pop_type=None, extra_states=None, **kwargs):
        if age_data is None:
            age_data = cvd.default_age_data
        if (np.ndim(age_data) == 2) and (np.shape(age_data)[1] == 3): # Covasim's format, so convert it to Starsim's
            age_data = convert_age_data(age_data)
        geo_states = [
            ss.Arr('x', dtype=np.float64, nan=np.nan, label='Home longitude'), # Float64, not Starsim's float32, which is only accurate to about a metre here, so people near the edge of a polygon could end up outside it
            ss.Arr('y', dtype=np.float64, nan=np.nan, label='Home latitude'),
            ss.IntArr('region', default=1, label='Region'), # The integer ID of the region (see sim.region_labels)
            ss.IntArr('sewershed', default=0, label='Sewershed'), # The integer ID of the sewershed (see sim.sewershed_labels): 0 if there is no sewershed file, -1 outside all sewersheds
        ]
        super().__init__(n_agents, age_data=age_data, extra_states=geo_states + sc.tolist(extra_states), **kwargs)
        self.geo_pars = geo_pars
        self.pop_type = pop_type
        self.region_labels = {} # {integer region ID: region_code}; set when initialized
        self.sewershed_labels = {} # {integer sewershed ID: original sid}; set when initialized
        return

    def init_vals(self):
        """ Give each state its initial values, then give people their locations, regions, and sewersheds (see geography.py) """
        super().init_vals()
        if self.geo_pars is not None:
            self.region_labels, self.sewershed_labels = cvgeo.assign_geography(self, self.geo_pars, pop_type=self.pop_type)
        return

    @property
    def sequence_tracker(self):
        """ The sequence evolution tracker (None unless evo_pars['enable'] is True); see ``cv.LineageSequenceTracker`` """
        covid = self._v3_covid()
        return covid.sequence_tracker if covid is not None else None

    def plot(self, *args, **kwargs):
        """Plot statistics of the population -- age distribution, numbers of contacts, and overall weight of contacts (number of contacts multiplied by beta per layer); see cv.plot_people()"""
        return cvplt.plot_people(self, *args, **kwargs)
