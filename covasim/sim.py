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
import numpy as np
import starsim as ss

from . import people as cvppl
from . import network as cvnet
from . import covid as cvcov
from . import connectors as cvconn

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
                 beta=None, pop_scale=None, total_pop=None, variants=None, use_waning=False, **kwargs):
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
            diseases = cvcov.COVID(beta=betadict, init_prev=int(pop_infected), variants=variants,
                                   use_waning=use_waning)

        # Auto-attach the cross-immunity connector when waning immunity is on (M4) OR more than one
        # variant circulates (M3): it applies cross-immunity each step (NAb-weighted under use_waning,
        # else the static matrix) and enables reinfection. Users can pass connectors=... to override.
        connectors = kwargs.pop('connectors', None)
        if connectors is None and (getattr(diseases, 'nv', 1) > 1 or getattr(diseases.pars, 'use_waning', False)):
            connectors = cvconn.CrossImmunity()
        if connectors is not None:
            kwargs['connectors'] = connectors

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

    # By_variant Result keys that scale with population (counts); the rest (prevalence/incidence) are rates.
    _BY_VARIANT_SCALE_KEYS = (
        'new_infections_by_variant', 'cum_infections_by_variant',
        'new_symptomatic_by_variant', 'cum_symptomatic_by_variant',
        'new_severe_by_variant', 'cum_severe_by_variant',
        'new_infectious_by_variant', 'cum_infectious_by_variant',
        'n_exposed_by_variant', 'n_infectious_by_variant',
    )

    def finalize(self):
        """Finalize, then bridge the multi-variant results to the v3 top-level path (M3, Open Q E).

        Starsim namespaces module results under ``sim.results['covid']`` and its auto-scaler does not
        descend into the nested ``['variant']`` sub-dict, so M3 here (mirroring v3 ``sim.finalize``):
          - scales the count-type by_variant Results by ``pop_scale``;
          - adds the initial-wild seed-offset to ``cum_infections_by_variant[0]`` (v3 sim.py:786-787);
          - recomputes the ``prevalence``/``incidence`` by_variant rates against scaled denominators;
          - references ``sim.results['variant']`` and ``sim.results['n_imports']`` at the v3 top-level path.
        Full flat aggregate-results / ``sim.summary`` compat is deferred (Open Q E).
        """
        super().finalize()
        covid = self.diseases.get('covid')
        if covid is None or 'variant' not in covid.results:
            return
        vres = covid.results['variant']
        pop_scale = float(self.pars.pop_scale)

        # Manually scale the count-type by_variant Results (the auto-scaler skips the nested sub-dict).
        if pop_scale != 1.0:
            for key in self._BY_VARIANT_SCALE_KEYS:
                vres[key].values *= pop_scale

        # Seed-offset: the initial wild seeds enter cum_infections_by_variant[0] (v3 sim.py:786-787).
        n_seed = int(getattr(covid.pars, '_n_initial_cases', 0) or 0)
        if n_seed:
            vres['cum_infections_by_variant'].values[0, :] += n_seed * pop_scale

        # Recompute the by_variant rate denominators against the scaled population (v3-style; the
        # prevalence_by_variant misnomer = new_infections_by_variant / n_alive is copied verbatim).
        n_raw = len(covid.rel_sus.raw)                              # initial agent count (no births in M3)
        cum_deaths = np.asarray(covid.results['cum_deaths'], dtype=float)
        n_alive = n_raw * pop_scale - cum_deaths
        n_susc = np.asarray(covid.results['n_susceptible'], dtype=float)
        new_inf = np.asarray(vres['new_infections_by_variant'], dtype=float)
        vres['incidence_by_variant'].values[:]  = np.divide(new_inf, n_susc,  out=np.zeros_like(new_inf), where=n_susc > 0)
        vres['prevalence_by_variant'].values[:] = np.divide(new_inf, n_alive, out=np.zeros_like(new_inf), where=n_alive > 0)

        # Bridge to the v3 top-level path so sim.results['variant'][key] / sim.results['n_imports'] work.
        self.results['variant'] = vres
        if 'n_imports' in covid.results:
            self.results['n_imports'] = covid.results['n_imports']

        # Flat aggregate-results bridge (Open Q E): reference every top-level Result of the covid
        # module at the sim root, so v3-style sim.results['cum_deaths'] etc. resolve (used by cv.Fit /
        # cv.Calibration). Additive -- references, no dynamics change. The nested 'variant' sub-dict is
        # already bridged above; skip it here.
        for key, res in covid.results.items():
            if isinstance(res, ss.Result) and key not in self.results:
                self.results[key] = res
        return
