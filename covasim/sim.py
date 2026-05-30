"""
Defines the Sim class for Covasim on the Starsim base.

``cv.Sim(ss.Sim)`` is a thin wrapper that assembles the M1 module stack -- a
``cv.People``, one ``cv.Network`` per contact layer, and a ``cv.COVID`` disease --
and forwards to ``ss.Sim`` with a daily timestep. Per-layer transmissibility
(Covasim's ``beta * beta_layer``) is carried on the disease's ``beta`` dict keyed
by network layer. ``pop_infected`` agents are seeded exactly at t=0.

Passing ``people=`` / ``networks=`` / ``diseases=`` overrides the corresponding
default, so tests (and later milestones) can inject their own components.
"""
import starsim as ss

from . import people as cvppl
from . import network as cvnet
from . import covid as cvcov

__all__ = ['Sim']

# Covasim's per-layer beta weights (parameters.reset_layer_pars: beta_layer), keyed by pop_type.
_BETA_LAYER = {
    'random': {'a': 1.0},
    'hybrid': {'h': 3.0, 's': 0.6, 'w': 0.6, 'c': 0.3},
}
_BASE_BETA = 0.016  # Covasim pars['beta'] (parameters.py:62)


class Sim(ss.Sim):
    """Covasim Sim on the Starsim base (M1: basic transmission).

    Args:
        pop_size (int): number of agents.
        pop_infected (int): number of agents infected at t=0 (an exact count).
        pop_type (str): 'random' (single layer) or 'hybrid' (household/school/work/community).
        n_days (int): number of days to simulate.
        start_day (str/date): simulation start date.
        rand_seed (int): random seed.
        beta (float): base per-contact daily transmissibility (default Covasim's 0.016).
        people/networks/diseases: optionally inject these to override the default assembly.
        kwargs: forwarded to ``ss.Sim``.
    """

    def __init__(self, pars=None, people=None, pop_size=20_000, pop_infected=20,
                 pop_type='random', n_days=60, start_day='2020-03-01', rand_seed=1,
                 beta=None, pop_scale=None, total_pop=None, variants=None, **kwargs):
        if pop_type not in _BETA_LAYER:
            raise ValueError(f"pop_type {pop_type!r} not supported in M1 (choices: 'random', 'hybrid').")
        base_beta = _BASE_BETA if beta is None else beta

        if people is None:
            people = cvppl.People(pop_size)

        networks = kwargs.pop('networks', None)
        if networks is None:
            networks = cvnet.make_networks(pop_type)

        # Additional co-circulating variants (M3): a cv.variant, a list, or string/dict sugar. They are
        # registered into the single cv.COVID module (growing its variant axis) before state allocation.
        # variants empty => nv==1 => byte-identical to M2.
        diseases = kwargs.pop('diseases', None)
        if diseases is None:
            betadict = {lk: ss.probperday(base_beta*bl) for lk, bl in _BETA_LAYER[pop_type].items()}
            diseases = cvcov.COVID(beta=betadict, init_prev=int(pop_infected), variants=variants)

        # Absolute population scaling: each agent represents pop_scale real people. Starsim
        # auto-multiplies every scale=True Result by pop_scale at finalize. Pass at most one of
        # total_pop / pop_scale (Starsim derives the other; setting both raises). Dynamic
        # rescaling (v3 rescale/make_naive) is deferred to a later milestone.
        scale_kw = {}
        if total_pop is not None:
            scale_kw['total_pop'] = total_pop
        if pop_scale is not None:
            scale_kw['pop_scale'] = pop_scale

        super().__init__(pars=pars, people=people, networks=networks, diseases=diseases,
                         start=ss.date(start_day), dur=ss.days(n_days), dt=ss.days(1),
                         rand_seed=rand_seed, **scale_kw, **kwargs)
        return
