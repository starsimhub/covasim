"""
Set the defaults across each of the different files.

The precision (32 bit by default, or 64 bit) is shared with Starsim; to change
it, use::

    cv.options.set(precision=64)
"""

import numpy as np
import numba as nb
import sciris as sc
import starsim as ss
from .settings import options as cvo # To set options

# Specify all externally visible functions this file defines
__all__ = ['default_float', 'default_int', 'get_default_colors', 'get_default_plots']


#%% Specify what data types to use

# Floats are the same as Starsim's, which follow ss.options.precision (set by cv.options.set(precision=...))
default_float = ss.dtypes.float
nbfloat       = ss.dtypes.nbfloat

# Integers for the contact arrays (population.py) and the Numba functions (utils.py); Starsim always uses int64
if cvo.precision == 32:
    default_int = np.int32
    nbint       = nb.int32
elif cvo.precision == 64: # pragma: no cover
    default_int = np.int64
    nbint       = nb.int64
else:
    raise NotImplementedError(f'Precision must be either 32 bit or 64 bit, not {cvo.precision}')


#%% Define other defaults

# Parameters that can vary by variant
variant_pars = [
    'rel_beta',
    'rel_symp_prob',
    'rel_severe_prob',
    'rel_crit_prob',
    'rel_death_prob',
]

# Default age data, based on Seattle 2018 census data
default_age_data = np.array([
    [ 0,  4, 0.0605],
    [ 5,  9, 0.0607],
    [10, 14, 0.0566],
    [15, 19, 0.0557],
    [20, 24, 0.0612],
    [25, 29, 0.0843],
    [30, 34, 0.0848],
    [35, 39, 0.0764],
    [40, 44, 0.0697],
    [45, 49, 0.0701],
    [50, 54, 0.0681],
    [55, 59, 0.0653],
    [60, 64, 0.0591],
    [65, 69, 0.0453],
    [70, 74, 0.0312],
    [75, 79, 0.02016], # Calculated based on 0.0504 total for >=75
    [80, 84, 0.01344],
    [85, 89, 0.01008],
    [90, 99, 0.00672],
])


def get_default_colors():
    """
    Specify the plot colors of the results, by result name without its prefix
    (e.g. "infections" for "cum_infections"). The plotting functions use
    ``cv.defaults.default_colors``, which is built once from this.

    NB, includes duplicates since stocks and flows are named differently.
    """
    c = sc.objdict()
    c.susceptible           = '#4d771e'
    c.exposed               = '#c78f65'
    c.exposed_by_variant    = '#c75649'
    c.infectious            = '#e45226'
    c.infectious_by_variant = c.infectious
    c.infections            = '#b62413'
    c.reinfections          = '#732e26'
    c.infections_by_variant = '#b62413'
    c.tests                 = '#aaa8ff'
    c.diagnoses             = '#5f5cd2'
    c.diagnosed             = c.diagnoses
    c.quarantined           = '#5c399c'
    c.isolated              = '#9756ff'
    c.doses                 = c.quarantined
    c.vaccinated            = c.quarantined
    c.recoveries            = '#9e1149'
    c.recovered             = c.recoveries
    c.symptomatic           = '#c1ad71'
    c.symptomatic_by_variant= c.symptomatic
    c.severe                = '#c1981d'
    c.severe_by_variant     = c.severe
    c.critical              = '#b86113'
    c.deaths                = '#000000'
    c.dead                  = c.deaths
    c.known_dead            = c.deaths
    c.known_deaths          = c.deaths
    c.default               = '#000000'
    c.pop_nabs              = '#32733d'
    c.pop_protection        = '#9e1149'
    c.pop_symp_protection   = '#b86113'
    return c

# Build the colors once, since they are looked up for every line plotted
default_colors = get_default_colors()


# Define the 'overview plots', i.e. the most useful set of plots to explore different aspects of a simulation
overview_plots = [
    'cum_infections',
    'cum_severe',
    'cum_critical',
    'cum_deaths',
    'cum_known_deaths',
    'cum_diagnoses',
    'new_infections',
    'new_severe',
    'new_critical',
    'new_deaths',
    'new_diagnoses',
    'n_infectious',
    'n_severe',
    'n_critical',
    'n_susceptible',
    'new_tests',
    'n_symptomatic',
    'new_quarantined',
    'n_quarantined',
    'new_doses',
    'new_vaccinated',
    'cum_vaccinated',
    'cum_doses',
    'test_yield',
    'r_eff',
]

overview_variant_plots = [
    'cum_infections_by_variant',
    'new_infections_by_variant',
    'n_infectious_by_variant',
    'cum_reinfections',
    'new_reinfections',
    'pop_nabs',
    'pop_protection',
    'pop_symp_protection',
]

def get_default_plots(which='default', kind='sim', sim=None):
    """
    Specify which quantities to plot.

    Args:
        which (str): 'default', 'overview', 'variant', 'overview-variant', 'seir', or 'all'
        kind (str): 'sim' for a single sim, or 'scens' for scenarios and multisims (which have different default plots)
        sim (Sim): the sim; only needed for which='all', which plots every result key of the sim

    Returns:
        A dict of lists of result keys (keyed by plot title), or a list of result keys (one plot each)
    """
    if which is None:
        which = 'default'
    which = str(which).lower() # To make comparisons easier

    # Check that kind makes sense
    sim_kind   = 'sim'
    scens_kind = 'scens'
    kindmap = {
        None:        sim_kind,
        'sim':       sim_kind,
        'default':   sim_kind,
        'msim':      scens_kind,
        'scen':      scens_kind,
        'scens':     scens_kind,
        'scenarios': scens_kind,
    }
    if kind not in kindmap.keys():
        errormsg = f'Expecting "sim" or "scens", not "{kind}"'
        raise ValueError(errormsg)
    else:
        is_sim = kindmap[kind] == sim_kind

    # Default plots -- different for sims and scenarios
    if which == 'default':

        if is_sim:
            plots = sc.odict({
                'Total counts': [
                    'cum_infections',
                    'n_infectious',
                    'cum_diagnoses',
                ],
                'Daily counts': [
                    'new_infections',
                    'new_diagnoses',
                ],
                'Health outcomes': [
                    'cum_severe',
                    'cum_critical',
                    'cum_deaths',
                    'cum_known_deaths',
                ],
            })

        else: # pragma: no cover
            plots = sc.odict({
                'Cumulative infections': [
                    'cum_infections',
                ],
                'New infections per day': [
                    'new_infections',
                ],
                'Cumulative deaths': [
                    'cum_deaths',
                ],
            })

    # Show an overview
    elif which == 'overview': # pragma: no cover
        plots = sc.dcp(overview_plots)

    # Plot absolutely everything
    elif which == 'all': # pragma: no cover
        if sim is None:
            errormsg = 'To get all the result keys with which="all", the sim must be supplied'
            raise ValueError(errormsg)
        plots = sim.result_keys('all')

    # Show an overview plus variants
    elif 'overview' in which and 'variant' in which: # pragma: no cover
        plots = sc.dcp(overview_plots) + sc.dcp(overview_variant_plots)

    # Show default but with variants
    elif which.startswith('variant'): # pragma: no cover
        if is_sim:
            plots = sc.odict({
                'Cumulative infections by variant': [
                    'cum_infections_by_variant',
                ],
                'New infections by variant': [
                    'new_infections_by_variant',
                ],
                'Health outcomes': [
                    'cum_severe',
                    'cum_critical',
                    'cum_deaths',
                ],
            })

        else: # pragma: no cover
            plots = sc.odict({
                    'Cumulative infections by variant': [
                        'cum_infections_by_variant',
                    ],
                    'New infections by variant': [
                        'new_infections_by_variant',
                    ],
                    'New diagnoses': [
                        'new_diagnoses',
                    ],
                    'Cumulative deaths': [
                        'cum_deaths',
                    ],
            })

    # Plot SEIR compartments
    elif which == 'seir': # pragma: no cover
        plots = [
            'n_susceptible',
            'n_preinfectious',
            'n_infectious',
            'n_removed',
        ]

    else: # pragma: no cover
        errormsg = f'The choice which="{which}" is not supported: choices are "default", "overview", "all", "variant", "overview-variant", or "seir", along with any result key (see sim.result_keys(\'all\') for options)'
        raise ValueError(errormsg)

    return plots
