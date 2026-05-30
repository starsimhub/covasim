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
import sciris as sc
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

    def __init__(self, pars=None, people=None, pop_size=None, pop_infected=None,
                 pop_type=None, n_days=None, start_day=None, end_day=None, rand_seed=None,
                 beta=None, pop_scale=None, total_pop=None, variants=None, use_waning=None,
                 datafile=None, location=None, **kwargs):
        # Covasim accepts its parameters either as a dict (the v3 ``cv.Sim(pars)`` form) or as keyword
        # arguments; an explicit keyword overrides the same key in the dict. Pull the Covasim sim-level
        # keys out of the pars dict here -- whatever remains (verbose, interventions, ...) is forwarded
        # to ss.Sim, which understands it.
        pars = sc.dcp(pars) if isinstance(pars, dict) else {}
        def _pick(key, kwarg, default):
            if kwarg is not None:
                pars.pop(key, None)        # explicit keyword wins; drop any dict duplicate
                return kwarg
            return pars.pop(key, default)  # else the dict value, else the Covasim default
        if pop_size is None:
            pop_size = pars.pop('n_agents', None)  # n_agents is the Starsim alias for pop_size
        pop_size     = int(_pick('pop_size',     pop_size,     20_000))
        pop_infected = int(_pick('pop_infected', pop_infected, 20))
        pop_type     = _pick('pop_type',     pop_type,     'random')
        start_day    = _pick('start_day',    start_day,    '2020-03-01')
        rand_seed    = _pick('rand_seed',    rand_seed,    1)
        beta         = _pick('beta',         beta,         None)
        pop_scale    = _pick('pop_scale',    pop_scale,    None)
        total_pop    = _pick('total_pop',    total_pop,    None)
        use_waning   = _pick('use_waning',   use_waning,   False)
        datafile     = _pick('datafile',     datafile,     None)
        location     = _pick('location',     location,     None)
        if variants is None:
            variants = pars.pop('variants', None)

        # Duration: as in v3, an ``end_day`` (a date) takes precedence over ``n_days`` if both are given;
        # otherwise use ``n_days``, defaulting to 60.
        _n_days = _pick('n_days', n_days, None)
        end_day = _pick('end_day', end_day, None)
        if end_day is not None:
            n_days = int(sc.daydiff(start_day, end_day))
        else:
            n_days = 60 if _n_days is None else _n_days

        if pop_type not in _BETA_LAYER:
            raise ValueError(f"pop_type {pop_type!r} not supported in M1 (choices: 'random', 'hybrid').")
        base_beta = _BASE_BETA if beta is None else beta

        if people is None:
            age_data = None
            if location is not None:
                # Use the country/region age pyramid (v3 ``location=``). get_age_distribution returns an
                # Nx3 [age_min, age_max, fraction] table; ss.People wants Nx2 [age_lower_edge, value].
                from . import data as cvdata
                raw = np.asarray(cvdata.get_age_distribution(location), dtype=float)
                age_data = raw[:, [0, 2]]
            people = cvppl.People(pop_size, age_data=age_data)

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
            # Route any remaining recognised COVID disease parameters out of the pars dict into the
            # module (the v3 form ``cv.Sim(dict(rel_death_prob=2, nab_decay=...))``); the rest stays
            # for ss.Sim. Skip keys ss.Sim also defines so sim-level pars aren't hijacked.
            covid_keys = set(diseases.pars.keys())
            overrides = {k: pars.pop(k) for k in list(pars) if k in covid_keys}
            overrides.update({k: kwargs.pop(k) for k in list(kwargs) if k in covid_keys})  # also from kwargs
            if overrides:
                diseases.pars.update(overrides)

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

        # Default verbosity from the Covasim option (v3: pars['verbose'] = cv.options.verbose) when the
        # user did not set it, so cv.options(verbose=0) silences cv.Sim().run() as it did in v3.
        if 'verbose' not in pars and 'verbose' not in kwargs:
            from . import settings as cvset
            kwargs['verbose'] = cvset.options.verbose

        super().__init__(pars=pars, people=people, networks=networks, diseases=diseases,
                         start=ss.date(start_day), dur=ss.days(n_days), dt=ss.days(1),
                         rand_seed=rand_seed, **scale_kw, **kwargs)

        # Report Covasim's version + git info (v3 sim.version / sim.git_info), not Starsim's inherited ones.
        from . import version as cvv
        from . import misc as cvm
        self.version = cvv.__version__
        try:
            self.git_info = cvm.git_info(verbose=False)
        except Exception:
            self.git_info = None

        # Snapshot the resolved Covasim sim-level config + the COVID module (for export_pars /
        # introspection). _cv_covid references the disease object as built here, so its pars are
        # readable even before the sim is initialized.
        self._cv_config = dict(pop_size=pop_size, pop_infected=pop_infected, pop_type=pop_type,
                               n_days=n_days, start_day=str(start_day), end_day=end_day, rand_seed=rand_seed,
                               beta=base_beta, use_waning=bool(use_waning))
        self._cv_covid = diseases

        # Optional epi data to fit against (v3 ``cv.Sim(datafile=...)``), read into ``self.data`` for
        # ``cv.Fit`` / ``sim.compute_fit``.
        self.data = None
        if datafile is not None:
            from . import misc as cvm
            self.data = datafile if hasattr(datafile, 'columns') else cvm.load_data(datafile)
        return

    def __getattr__(self, key):
        """v3 compat: expose the Covasim sim-level config + COVID parameters as attributes.

        So ``sim.beta`` / ``sim.start_day`` / ``sim.rel_death_prob`` (and, since ``ss.Sim.__getitem__``
        delegates to ``getattr``, ``sim['beta']`` etc.) resolve. ``__getattr__`` is only consulted when
        normal attribute lookup fails, so it cannot shadow real attributes. (Reads only; to *set* a
        disease parameter, build a fresh ``cv.Sim(dict(...))`` or use ``cv.dynamic_pars``.)
        """
        cfg = self.__dict__.get('_cv_config', None)  # __dict__ access avoids re-triggering __getattr__
        if cfg is not None and key in cfg:
            return cfg[key]
        covid = self.__dict__.get('_cv_covid', None)
        if covid is not None and hasattr(covid, 'pars') and key in covid.pars:
            return covid.pars[key]
        raise AttributeError(f"'Sim' object has no attribute '{key}'")

    def initialize(self, *args, reset=False, **kwargs):
        """v3-compatibility alias for ``init`` (Starsim renamed ``initialize`` -> ``init``).

        ``reset`` is accepted for v3 compatibility. Note: cleanly re-initialising an already-run sim is
        limited under the Starsim object model -- prefer building a fresh ``cv.Sim`` to change parameters.
        """
        if 'people' not in self.pars:
            self.pars.people = None  # Starsim pops this on init; restore it so a re-init won't KeyError
        return self.init(*args, **kwargs)

    def day(self, day, *args):
        """Convert date(s) to integer day index/indices relative to ``start_day`` (v3 ``Sim.day``).

        Numbers (and numeric arrays) are treated as day offsets and passed through; only genuine dates
        (strings / datetimes) are converted. Accepts a scalar, list, or array.
        """
        start = self._cv_config['start_day']
        def _one(d):
            if isinstance(d, (int, np.integer)):
                return int(d)
            if isinstance(d, (float, np.floating)):
                return int(round(d))
            return int(sc.daydiff(start, d))  # a date string / datetime
        inputs = list(day) if isinstance(day, (list, tuple, np.ndarray)) else [day]
        inputs += list(args)
        out = [_one(d) for d in inputs]
        return out[0] if len(out) == 1 else out

    def date(self, *args, **kwargs):
        """Convert day index/indices to date string(s) relative to ``start_day`` (v3 ``Sim.date``)."""
        return sc.date(*args, start_date=self._cv_config['start_day'], **kwargs)

    def get_analyzers(self, label=None):
        """Return the list of analyzers matching ``label`` (or all of them); v3 ``Sim.get_analyzers``."""
        analyzers = list(self.analyzers.values()) if hasattr(self, 'analyzers') else []
        if label is None:
            return analyzers
        return [a for a in analyzers if getattr(a, 'label', None) == label or a.name == label]

    def get_analyzer(self, label=None, die=True):
        """Return a single analyzer matching ``label`` (or the sole analyzer); v3 ``Sim.get_analyzer``."""
        matches = self.get_analyzers(label)
        if matches:
            return matches[-1]
        if die:
            raise ValueError(f'No analyzer found matching {label!r}.')
        return None

    def get_interventions(self, label=None):
        """Return the list of interventions matching ``label``/class (or all); v3 ``Sim.get_interventions``."""
        ivs = list(self.interventions.values()) if hasattr(self, 'interventions') else []
        if label is None:
            return ivs
        if isinstance(label, type):
            return [iv for iv in ivs if isinstance(iv, label)]
        return [iv for iv in ivs if getattr(iv, 'label', None) == label or iv.name == label]

    def get_intervention(self, label=None, die=True):
        """Return a single intervention matching ``label``/class (or the sole one); v3 ``Sim.get_intervention``."""
        matches = self.get_interventions(label)
        if matches:
            return matches[-1]
        if die:
            raise ValueError(f'No intervention found matching {label!r}.')
        return None

    def compute_fit(self, *args, **kwargs):
        """Compute the goodness-of-fit against ``self.data`` (v3 ``Sim.compute_fit``); returns a ``cv.Fit``."""
        from . import analysis as cva
        self.fit = cva.Fit(self, *args, **kwargs)
        return self.fit

    def make_transtree(self, *args, **kwargs):
        """Return the transmission tree (v3 ``Sim.make_transtree``).

        Unlike v3, v4 only records the transmission log when a ``cv.TransTree`` analyzer is attached
        (the log is gated so unrelated runs stay byte-identical). If one was attached, this returns it;
        otherwise it raises with instructions.
        """
        covid = list(self.diseases.values())[0]
        if getattr(covid, '_record_transmissions', False) and getattr(covid, 'infection_events', None):
            tt = self.get_analyzer(label=None, die=False)
            from . import analysis as cva
            for a in self.get_analyzers():
                if isinstance(a, cva.TransTree):
                    return a
        raise RuntimeError("make_transtree requires a transmission log: attach "
                           "analyzers=cv.TransTree() before running (v4 only logs transmissions when "
                           "a cv.TransTree analyzer is present), then read sim.get_analyzer('transtree').")

    def brief(self, output=False):
        """Print (or return) a one-line summary of the sim (v3 ``Sim.brief``)."""
        covid = self.diseases.get('covid') if hasattr(self, 'diseases') else None
        if covid is not None and 'cum_infections' in covid.results:
            ci = float(np.asarray(covid.results['cum_infections']).max())
            string = f'Sim({self.label!r}; {self._cv_config["n_days"]} days; {self._cv_config["pop_size"]} agents; {ci:n} cumulative infections)'
        else:
            string = f'Sim({self.label!r}; {self._cv_config["n_days"]} days; {self._cv_config["pop_size"]} agents; not run)'
        if output:
            return string
        print(string)
        return

    def export_pars(self, filename=None, indent=2, **kwargs):
        """Export the sim's parameters to a JSON-compatible dict (the v3 ``BaseSim.export_pars``).

        Returns the resolved Covasim sim-level config plus the COVID module's scalar parameters; if
        ``filename`` is given, also writes the dict to JSON. Per-distribution / array parameters are
        summarised rather than dumped.

        Args:
            filename (str): if given, write the JSON here.
            indent (int): JSON indent.
        """
        pars = dict(self._cv_config)
        # Prefer the live in-sim disease (post-init); else the construction-time reference.
        covid = self.diseases.get('covid') if hasattr(self, 'diseases') else None
        if covid is None:
            covid = getattr(self, '_cv_covid', None)
        if covid is not None:
            covid_pars = {}
            for key, val in dict(covid.pars).items():
                try:
                    covid_pars[key] = sc.jsonify(val, die=False)  # scalars/lists/dicts; skip the rest
                except Exception:
                    covid_pars[key] = str(type(val).__name__)
            pars['covid'] = covid_pars
        pars = sc.jsonify(pars, die=False)
        if filename is not None:
            sc.savejson(filename, pars, indent=indent)
        return pars

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

        self._finalize_variant_bridge(covid, vres)
        return

    def save(self, filename=None, shrink=False, **kwargs):
        """Save the sim to disk (M10).

        Covasim's ``cv.COVID`` module legitimately carries large per-agent + by-variant state, so
        unlike the stock ``ss.Sim.save`` (which shrinks a run sim and trips Starsim's size check),
        ``cv.Sim`` saves the *full* sim by default -- matching v3 ``sim.save()`` semantics.

        Args:
            filename (str): path to save to (defaults to the sim's ``simfile``).
            shrink (bool): drop people/distributions before saving (default False; see above).
            kwargs: passed through to ``ss.Sim.save`` / ``sc.makefilepath``.
        """
        return super().save(filename=filename, shrink=shrink, **kwargs)

    @staticmethod
    def load(filename, *args, **kwargs):
        """Load a saved sim from disk (M10); the v3 ``cv.Sim.load`` classmethod, via ``cv.load``."""
        from . import misc as cvm
        return cvm.load(filename, *args, **kwargs)

    def plot(self, keys=None, fig=None, to_plot=None, **kwargs):
        """Plot the headline COVID result panels (a Covasim-specific view; M9).

        The stock ``ss.Sim.plot`` does not handle Covasim's 2D by-variant results, so ``cv.Sim`` plots
        the key 1D burden/shape series from the disease module directly.

        Args:
            keys (list/str): result key(s) to plot (default: the standard burden/shape panels). The
                special values ``'variant'`` (per-variant cumulative infections) and ``'overview'`` (a
                multi-panel burden/testing overview) are also accepted.
            to_plot (list/str): v3 alias for ``keys``.
            fig: an existing figure to plot into.
        """
        import matplotlib.pyplot as plt  # local import (plotting is an optional dependency)
        from . import settings as cvset
        covid = list(self.diseases.values())[0]
        res = covid.results
        keys = keys if keys is not None else to_plot  # accept the v3 ``to_plot`` alias
        if isinstance(keys, str):
            keys = [keys]

        # Special 'variant' view: one line per variant of cumulative infections.
        if keys is not None and list(keys) == ['variant']:
            vres = self.results.get('variant') if hasattr(self.results, 'get') else None
            if vres is None or 'cum_infections_by_variant' not in vres:
                raise ValueError("No by-variant results to plot (run a multi-variant sim).")
            arr = np.asarray(vres['cum_infections_by_variant'])  # (npts, nv)
            t = np.arange(arr.shape[0])
            if fig is None:
                fig, ax = plt.subplots(figsize=(7, 4.5))
            else:
                ax = np.atleast_1d(fig.axes)[0]
            for v in range(arr.shape[1]):
                ax.plot(t, arr[:, v], lw=2, label=f'variant {v}')
            ax.set_title('cum_infections_by_variant'); ax.set_xlabel('Day'); ax.legend()
            fig.tight_layout()
            return fig if cvset.options.returnfig else None

        # Special 'overview' view: the v3 multi-panel burden/testing overview (available keys only).
        if keys is not None and list(keys) == ['overview']:
            keys = [k for k in ['cum_infections', 'new_infections', 'n_infectious', 'cum_symptomatic',
                                'new_severe', 'cum_severe', 'cum_critical', 'cum_deaths', 'cum_tests',
                                'cum_diagnoses', 'n_susceptible', 'n_exposed'] if k in res]

        default = ['cum_infections', 'n_infectious', 'cum_symptomatic', 'cum_severe',
                   'cum_critical', 'cum_deaths']
        requested = list(keys) if keys is not None else default
        keys = [k for k in requested if k in res]
        missing = [k for k in requested if k not in res]
        if not keys:
            raise ValueError(f'None of the requested result keys are available: {missing}. '
                             f'Available 1D keys include: {[k for k in res.keys()][:12]} ...')
        t = np.arange(covid.t.npts)
        if fig is None:
            ncol = min(3, len(keys))
            nrow = int(np.ceil(len(keys) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.5 * nrow), squeeze=False)
            axes = axes.flatten()
        else:
            axes = np.atleast_1d(fig.axes)
        for ax, k in zip(axes, keys):
            ax.plot(t, np.asarray(res[k]), lw=2)
            ax.set_title(k)
            ax.set_xlabel('Day')
        for ax in axes[len(keys):]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig if cvset.options.returnfig else None

    def plot_result(self, key, fig=None, **kwargs):
        """Plot a single result series (the v3 ``Sim.plot_result``)."""
        return self.plot(keys=[key], fig=fig, **kwargs)

    def to_excel(self, filename=None, skip_pars=None):
        """Export results + parameters to an Excel workbook (the v3 ``Sim.to_excel``).

        Writes a 'Results' sheet (the time series via ``to_df``) and a 'Parameters' sheet (the flattened
        Covasim config from ``export_pars``). Returns the ``sc.Spreadsheet``.
        """
        import pandas as pd
        # Build the results sheet from the 1D covid result series only -- the nested 2D by-variant
        # sub-dict breaks a flat DataFrame (the same reason cv.Sim.plot avoids stock ss plotting).
        covid = list(self.diseases.values())[0]
        res = covid.results
        data = {k: np.asarray(res[k]) for k in res.keys()
                if isinstance(res[k], ss.Result) and np.ndim(np.asarray(res[k])) == 1}
        result_df = pd.DataFrame(data)
        result_df.insert(0, 'date', np.asarray(self.t.timevec))
        flat = sc.flattendict(self.export_pars(), sep='_')
        par_df = pd.DataFrame.from_dict({k: [v] for k, v in flat.items()}, orient='index', columns=['Value'])
        spreadsheet = sc.Spreadsheet()
        spreadsheet.freshbytes()
        with pd.ExcelWriter(spreadsheet.bytes, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, sheet_name='Results')
            par_df.to_excel(writer, sheet_name='Parameters')
        spreadsheet.load()
        if filename is not None:
            spreadsheet.save(filename)
        return spreadsheet

    def calibrate(self, calib_pars, **kwargs):
        """Calibrate the sim against ``self.data`` (v3 ``Sim.calibrate``); returns a ``cv.Calibration``.

        Thin wrapper over ``cv.Calibration``; ``calib_pars`` format is ``{par: [best, low, high]}``.
        The data to fit defaults to ``self.data`` (from ``cv.Sim(datafile=...)``).
        """
        from . import analysis as cva
        data = kwargs.pop('data', None)
        if data is None:
            data = getattr(self, 'data', None)
        calib = cva.Calibration(self, calib_pars=calib_pars, data=data, **kwargs)
        return calib.calibrate()

    def _finalize_variant_bridge(self, covid, vres):
        """Attach the variant + flat result bridges at the sim top level (helper for finalize)."""
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

        # v3 exposed time keys ``date`` and ``t`` (Starsim only provides ``timevec``); aliases for
        # plotting against dates / day indices. References / derived arrays -- no dynamics change.
        if 'timevec' in self.results and 'date' not in self.results:
            self.results['date'] = self.results['timevec']
        if 't' not in self.results:
            self.results['t'] = np.arange(self.t.npts)
        return
