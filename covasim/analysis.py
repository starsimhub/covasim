"""
Analysis tools for Covasim on the Starsim base.

M7 restores ``cv.Fit`` -- the model-vs-data goodness-of-fit class (v3 ``analysis.Fit``). It is used
post-run as ``cv.Fit(sim, data=...)`` and reuses the kept-from-v3 ``cv.compute_gof`` (misc.py).
Terminology (as v3): *difference* = sim - data per matched point; *goodness-of-fit (gof)* = the
difference through ``compute_gof``; *loss* = gof x weight; *mismatch* = sum of losses (the scalar
minimised during calibration).

M9 adds the analyzers (``cv.Analyzer`` base + ``snapshot``/``age_histogram``/``daily_age_stats``/
``nab_histogram``), ``cv.TransTree`` (over a gated transmission log on ``cv.COVID``), and the
Covasim-specific plots (``cv.Sim.plot``, ``Fit.plot``, ``TransTree.plot``). The synthpops population
backend (optional dependency, not installed here) is the only remaining M9 piece.
"""
import numpy as np
import pandas as pd
import sciris as sc

from . import misc as cvm

__all__ = ['Fit', 'Calibration']


def _to_daykey(d):
    """Normalise a date/day to a hashable key for matching data to sim time points.

    Integers (and integer-like) are treated as day offsets; everything else (date strings,
    datetimes, ss.date) is normalised to an ISO 'YYYY-MM-DD' string.
    """
    if isinstance(d, (int, np.integer)):
        return int(d)
    try:
        return pd.Timestamp(d).strftime('%Y-%m-%d')
    except Exception:
        return str(d)


class Fit(sc.prettyobj):
    """
    Calculate the fit (mismatch) between a model run and data (the v3 ``cv.Fit``).

    Args:
        sim (cv.Sim): a run sim (results ready). Its bridged top-level results are read.
        data (DataFrame): data to fit, indexed by date (or integer day offset), with columns named
            like the sim result keys (e.g. ``cum_deaths``). Falls back to ``sim.data`` if not given.
        weights (dict): relative weight per result (default cum_deaths:10, cum_diagnoses:5, else 1).
        keys (list): which result keys to fit (default: the cumulative keys present in both).
        custom (dict): extra series to fit, ``{name: {'data': [...], 'sim': [...], 'weight': w}}``.
        compute (bool): compute the mismatch immediately.
        die (bool): raise (vs warn) if no data are supplied / no points match.

    Attributes (after compute): ``diffs``/``gofs``/``losses``/``mismatches`` (per key) and the scalar
    ``mismatch``.
    """

    def __init__(self, sim, data=None, weights=None, keys=None, custom=None, compute=True, die=True,
                 **gof_kwargs):
        self.weights    = sc.mergedicts({'cum_deaths': 10, 'cum_diagnoses': 5}, weights)
        self.user_keys  = keys
        self.custom     = sc.mergedicts(custom)
        self.die        = die
        self.gof_kwargs = gof_kwargs

        # Data: a DataFrame indexed by date/day, columns = result keys.
        data = data if data is not None else getattr(sim, 'data', None)
        if data is None or (hasattr(data, '__len__') and len(data) == 0):
            if self.die and not self.custom:
                raise RuntimeError('cv.Fit requires data (a DataFrame) or custom series.')
            data = pd.DataFrame()
        self.data = data

        # Sim results (the bridged flat top-level Results) + the sim date vector.
        self.sim_results = sc.objdict()
        for key in sim.results.keys():
            res = sim.results[key]
            arr = getattr(res, 'values', None)
            if arr is not None and np.ndim(arr) == 1:  # 1D top-level Results only (skip nested 'variant')
                self.sim_results[key] = np.asarray(arr)
        self.sim_daykeys = [_to_daykey(d) for d in np.asarray(sim.t.timevec)]
        self.sim_npts = len(self.sim_daykeys)

        # Populated during compute.
        self.keys = None
        self.custom_keys = list(self.custom.keys())
        self.inds   = sc.objdict(sim=sc.objdict(), data=sc.objdict())
        self.pair   = sc.objdict()
        self.diffs  = sc.objdict()
        self.gofs   = sc.objdict()
        self.losses = sc.objdict()
        self.mismatches = sc.objdict()
        self.mismatch = None

        if compute:
            self.compute()
        return

    def compute(self):
        """Run the full pipeline: reconcile -> diffs -> gofs -> losses -> mismatch."""
        self.reconcile_inputs()
        self.compute_diffs()
        self.compute_gofs()
        self.compute_losses()
        self.compute_mismatch()
        return self.mismatch

    def reconcile_inputs(self):
        """Pair sim and data points by matching dates/days (v3 reconcile_inputs)."""
        data_cols = list(self.data.columns) if len(self.data) else []
        if self.user_keys is None:
            sim_cum = [k for k in self.sim_results.keys() if k.startswith('cum_')]
            self.keys = [k for k in sim_cum if k in data_cols]  # cumulative keys present in both
        else:
            self.keys = list(self.user_keys)
            missing = [k for k in self.keys if k not in data_cols]
            if missing and self.die:
                raise sc.KeyNotFoundError(f'Requested keys not in data: {missing}')

        # Map each data row (by date/day key) to a sim time index.
        daykey_to_simind = {dk: i for i, dk in enumerate(self.sim_daykeys)}
        matches = 0
        for key in self.keys:
            sim_inds, data_inds = [], []
            series = self.data[key]
            for pos, (idx, datum) in enumerate(series.items()):
                if np.isfinite(datum):
                    dk = _to_daykey(idx)
                    if dk in daykey_to_simind:
                        sim_inds.append(daykey_to_simind[dk])
                        data_inds.append(pos)
            self.inds.sim[key] = np.array(sim_inds, dtype=int)
            self.inds.data[key] = np.array(data_inds, dtype=int)
            self.pair[key] = sc.objdict(
                sim=np.array([self.sim_results[key][i] for i in sim_inds], dtype=float),
                data=np.array([series.values[j] for j in data_inds], dtype=float),
            )
            matches += len(sim_inds)

        # Custom series: paired directly (no date matching).
        for key, custom in self.custom.items():
            if 'sim' not in custom or 'data' not in custom:
                raise sc.KeyNotFoundError(f'Custom input {key!r} must have "sim" and "data" keys.')
            c_sim, c_data = np.asarray(custom['sim'], dtype=float), np.asarray(custom['data'], dtype=float)
            if len(c_sim) != len(c_data):
                raise ValueError(f'Custom {key!r}: sim and data must be the same length.')
            self.pair[key] = sc.objdict(sim=c_sim, data=c_data)
            self.weights[key] = custom.get('weights', custom.get('weight', 1.0))
            matches += len(c_sim)

        if matches == 0:
            msg = 'No paired data points found between data and sim; check the dates/keys.'
            if self.die:
                raise ValueError(msg)
            cvm.warn(msg)
        return

    def compute_diffs(self, absolute=False):
        """sim - data per matched point."""
        for key in self.pair.keys():
            d = self.pair[key].sim - self.pair[key].data
            self.diffs[key] = np.abs(d) if absolute else d
        return

    def compute_gofs(self, **kwargs):
        """Goodness-of-fit per key via cv.compute_gof."""
        kwargs = sc.mergedicts(self.gof_kwargs, kwargs)
        for key in self.pair.keys():
            self.gofs[key] = cvm.compute_gof(self.pair[key].data, self.pair[key].sim, **kwargs)
        return

    def compute_losses(self):
        """Weighted goodness-of-fit per key."""
        for key in self.gofs.keys():
            weight = self.weights.get(key, 1.0)
            if sc.isiterable(weight):
                weight = np.asarray(weight)
                if len(weight) == self.sim_npts:           # weight given over the full sim -> trim to matches
                    weight = weight[self.inds.sim[key]]
            self.losses[key] = self.gofs[key] * weight
        return

    def compute_mismatch(self, use_median=False):
        """Sum the losses into per-key mismatches and the scalar total mismatch."""
        for key in self.losses.keys():
            self.mismatches[key] = np.median(self.losses[key]) if use_median else np.sum(self.losses[key])
        self.mismatch = float(np.sum([v for v in self.mismatches.values()]))
        return self.mismatch

    def plot(self, fig=None, **kwargs):
        """Plot, per fitted key, the sim-vs-data paired points and the per-point loss."""
        import matplotlib.pyplot as plt
        keys = list(self.pair.keys())
        if not keys:
            return None
        if fig is None:
            fig, axes = plt.subplots(2, len(keys), figsize=(4.5 * len(keys), 7), squeeze=False)
        else:
            axes = np.array(fig.axes).reshape(2, len(keys))
        for j, k in enumerate(keys):
            x = np.arange(len(self.pair[k].sim))
            axes[0, j].plot(x, self.pair[k].data, 'o-', label='data', alpha=0.7)
            axes[0, j].plot(x, self.pair[k].sim, 's-', label='sim', alpha=0.7)
            axes[0, j].set_title(k); axes[0, j].legend()
            axes[1, j].bar(x, self.losses.get(k, np.zeros_like(x)))
            axes[1, j].set_title(f'{k} loss (mismatch={self.mismatches.get(k, 0):.2f})')
        fig.suptitle(f'Fit: total mismatch = {self.mismatch:.3f}')
        fig.tight_layout()
        return fig

    def summarize(self):
        """Print the per-key mismatches and the total."""
        if self.mismatch is not None:
            print('Mismatch values by key:')
            print(self.mismatches)
            print(f'\nTotal mismatch: {self.mismatch}')
        else:
            print('Mismatch not yet computed.')
        return


class Calibration(sc.prettyobj):
    """
    Calibrate a cv.Sim to data by minimising the cv.Fit mismatch (the v3 ``cv.Calibration``), wrapping
    Starsim's Optuna-based ``ss.Calibration``.

    Args:
        sim (cv.Sim): a base (un-run) sim; each trial deep-copies it, applies the trial's parameters,
            runs it, and scores the fit.
        calib_pars (dict): parameters to calibrate, ``{key: [best, low, high]}`` (v3 form) -- each is
            sampled in ``[low, high]`` (initial guess ``best``) and applied to the COVID module's pars
            (or the sim pars), resolving an optional dotted ``path``.
        data (DataFrame): the target data passed to ``cv.Fit`` (date/day-indexed, result-key columns).
        total_trials (int): total Optuna trials (default 30).
        n_workers (int): parallel workers (default 1).
        weights (dict): per-key fit weights (passed to cv.Fit).
        reseed (bool): whether to vary rand_seed across trials (default False -> cleaner convergence).
        fit_kw (dict): extra kwargs for cv.Fit.
        kwargs: forwarded to ss.Calibration (e.g. verbose, die).

    After ``calibrate()``: ``best_pars`` (objdict) and ``df`` (per-trial results).
    """

    def __init__(self, sim, calib_pars, data, total_trials=30, n_workers=1, weights=None,
                 reseed=False, fit_kw=None, **kwargs):
        import starsim as ss
        self.sim = sim
        self.data = data
        self.weights = weights
        self.fit_kw = sc.mergedicts(fit_kw)
        self.best_pars = None
        self.df = None

        # Translate v3-style [best, low, high] into ss.Calibration sampler specs (keeping the path).
        ss_calib_pars = {}
        for key, spec in calib_pars.items():
            if isinstance(spec, (list, tuple)) and len(spec) == 3:
                best, low, high = spec
                ss_calib_pars[key] = dict(low=float(low), high=float(high), guess=float(best), path=key)
            else:  # already an ss-style spec dict
                d = dict(spec)
                d.setdefault('path', key)
                ss_calib_pars[key] = d

        kwargs.setdefault('verbose', False)
        kwargs.setdefault('die', False)
        self._calib = ss.Calibration(
            sim, ss_calib_pars, build_fn=self._build_fn, eval_fn=self._eval_fn,
            total_trials=int(total_trials), n_workers=int(n_workers), reseed=reseed, **kwargs)
        return

    @staticmethod
    def _apply_par(sim, path, value):
        """Apply a calibrated ``value`` to the sim at ``path`` (a COVID/sim par; dotted paths allowed)."""
        covid = sim.diseases['covid']
        parts = str(path).split('.')
        if len(parts) == 1:
            key = parts[0]
            if key in covid.pars:
                covid.pars[key] = value
            elif key in sim.pars:
                sim.pars[key] = value
            else:
                covid.pars[key] = value  # default to the disease module
        else:
            obj = sim
            for p in parts[:-1]:
                if p in ('sim',):
                    obj = sim
                elif hasattr(obj, 'diseases') and p in obj.diseases:
                    obj = obj.diseases[p]
                else:
                    obj = getattr(obj, p)
            tgt = parts[-1]
            if hasattr(obj, 'pars') and tgt in obj.pars:
                obj.pars[tgt] = value
            else:
                setattr(obj, tgt, value)
        return

    def _build_fn(self, sim, calib_pars=None):
        """Apply the trial's sampled parameters to a fresh sim copy (ss.Calibration build_fn).

        The base sim copy may be un-initialized; init it so ``sim.diseases`` exists. The calibratable
        severity scalers (rel_*_prob) are read at set_prognoses runtime, so applying them after init
        still takes effect. (Init-time pars would need a pre-init hook; out of scope for M7.)
        """
        if not sim.initialized:
            sim.init()
        if calib_pars:
            for parname, spec in calib_pars.items():
                if isinstance(spec, dict) and 'value' in spec:
                    self._apply_par(sim, spec.get('path', parname), spec['value'])
                elif parname == 'rand_seed':  # reseed adds a raw int
                    sim.pars.rand_seed = int(spec)
        return sim

    def _eval_fn(self, sim, **kwargs):
        """Score a run sim: the cv.Fit mismatch (minimised by Optuna). None sim -> inf."""
        if sim is None:
            return np.inf
        fit = Fit(sim, data=self.data, weights=self.weights, die=False, **self.fit_kw)
        return fit.mismatch if fit.mismatch is not None else np.inf

    def calibrate(self, **kwargs):
        """Run the Optuna calibration; populate ``best_pars`` and ``df``."""
        self._calib.calibrate(**kwargs)
        self.best_pars = self._calib.best_pars
        self.df = getattr(self._calib, 'df', None)
        return self


# %% Analyzers (M9) ---------------------------------------------------------------------------------

import starsim as ss  # noqa: E402  (kept here so Fit/Calibration above stay starsim-light)

__all__ += ['Analyzer', 'snapshot', 'age_histogram', 'nab_histogram']

_SNAPSHOT_STATES = ('susceptible', 'exposed', 'infectious', 'symptomatic', 'severe', 'critical',
                    'recovered', 'dead', 'diagnosed', 'quarantined', 'isolated', 'vaccinated')


class Analyzer(ss.Analyzer):
    """Base class for Covasim analyzers (same public name as v3; runs at the analyzer loop slot)."""

    def _covid(self):
        return list(self.sim.diseases.values())[0]

    def _resolve_days(self, days):
        """Convert a list of int day-offsets / date-strings into a set of int day indices."""
        tv = [str(d)[:10] for d in np.asarray(self.sim.t.timevec)]
        out = set()
        for d in sc.tolist(days):
            if isinstance(d, (int, np.integer)):
                out.add(int(d))
            else:
                ds = str(d)[:10]
                if ds in tv:
                    out.add(tv.index(ds))
        return out

    def _datekey(self, ti):
        return str(np.asarray(self.sim.t.timevec)[ti])[:10]


class snapshot(Analyzer):
    """
    Snapshot the per-agent disease state on specified days (the v3 ``cv.snapshot``).

    ``snapshots[date]`` is an objdict with the per-(active-)agent boolean state arrays, ``age``, and
    (under waning) ``nab``. ``get(day_or_date)`` retrieves one.
    """

    def __init__(self, days, *args, **kwargs):
        super().__init__(**kwargs)
        self.days = sc.tolist(days) + list(args)
        self._dayset = None
        self.snapshots = sc.odict()
        return

    def init_post(self):
        super().init_post()
        self._dayset = self._resolve_days(self.days)
        return

    def step(self):
        ti = self.ti
        if ti not in self._dayset:
            return
        covid = self._covid()
        snap = sc.objdict()
        snap['age'] = np.asarray(self.sim.people.age).copy()
        for state in _SNAPSHOT_STATES:
            if hasattr(covid, state):
                snap[state] = np.asarray(getattr(covid, state)).copy()
        if covid.pars.use_waning:
            snap['nab'] = np.asarray(covid.nab).copy()
        self.snapshots[self._datekey(ti)] = snap
        return

    def get(self, key=None):
        """Retrieve a snapshot by date string, or the first if no key."""
        if key is None:
            return self.snapshots[list(self.snapshots.keys())[0]]
        if isinstance(key, (int, np.integer)):
            key = str(np.asarray(self.sim.t.timevec)[int(key)])[:10]
        return self.snapshots[str(key)[:10]]


class age_histogram(Analyzer):
    """
    Age histograms of disease states on specified days (the v3 ``cv.age_histogram``).

    ``hists[date][state]`` is the per-age-bin count of agents in ``state``; ``bins`` are the age-bin edges.
    """

    def __init__(self, days, states=None, bins=None, **kwargs):
        super().__init__(**kwargs)
        self.days = sc.tolist(days)
        self.states = states or ['exposed', 'infectious', 'severe', 'critical', 'dead']
        self.bins = np.array(bins) if bins is not None else np.arange(0, 101, 10)
        self._dayset = None
        self.hists = sc.odict()
        return

    def init_post(self):
        super().init_post()
        self._dayset = self._resolve_days(self.days)
        return

    def step(self):
        ti = self.ti
        if ti not in self._dayset:
            return
        covid = self._covid()
        ages = np.asarray(self.sim.people.age)
        hist = sc.objdict(bins=self.bins)
        for state in self.states:
            if not hasattr(covid, state):
                continue
            mask = np.asarray(getattr(covid, state))
            counts, _ = np.histogram(ages[mask.astype(bool)], bins=self.bins)
            hist[state] = counts
        self.hists[self._datekey(ti)] = hist
        return


class nab_histogram(Analyzer):
    """
    Histogram of neutralizing-antibody (NAb) levels on specified days (the v3 ``cv.nab_histogram``).

    Requires ``use_waning=True``. ``hists[date]`` = (counts, bin_edges) over agents with NAb > 0.
    """

    def __init__(self, days, bins=None, **kwargs):
        super().__init__(**kwargs)
        self.days = sc.tolist(days)
        self.bins = np.array(bins) if bins is not None else np.linspace(0, 20, 41)
        self._dayset = None
        self.hists = sc.odict()
        return

    def init_post(self):
        super().init_post()
        self._dayset = self._resolve_days(self.days)
        return

    def step(self):
        ti = self.ti
        if ti not in self._dayset:
            return
        covid = self._covid()
        nab = np.asarray(covid.nab)
        nab = nab[nab > 0]
        counts, edges = np.histogram(nab, bins=self.bins)
        self.hists[self._datekey(ti)] = sc.objdict(counts=counts, bins=edges)
        return


__all__ += ['TransTree']


class TransTree(Analyzer):
    """
    Reconstruct the transmission tree (the v3 ``cv.TransTree``) from the disease's transmission log.

    Attaching this analyzer switches on transmission logging in ``cv.COVID`` (off otherwise, so the
    core sim is unaffected). After the run it exposes:

      - ``infection_events``: list of ``(source_uid, target_uid, ti, variant_index)`` (source < 0 = a
        seed/import, i.e. a tree root with no infector);
      - ``n_targets``: ``{source_uid: number of secondary infections}``;
      - ``r0``: the mean number of secondary infections per infector (a transmission-tree R estimate);
      - ``make_detailed()``: a DataFrame of the events.

    The full networkx graph + tree plotting are part of the M9 plotting pass (deferred).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.infection_events = None
        self.n_targets = None
        self.r0 = None
        return

    def init_post(self):
        super().init_post()
        covid = self._covid()
        covid._record_transmissions = True   # switch on transmission logging
        covid.infection_events = []
        return

    def step(self):
        """No per-step work: TransTree records via cv.COVID.infect() and assembles in finalize()."""
        pass

    def finalize(self):
        super().finalize()
        covid = self._covid()
        self.infection_events = list(covid.infection_events)
        sources = np.array([e[0] for e in self.infection_events], dtype=int) if self.infection_events else np.array([], dtype=int)
        infectors = sources[sources >= 0]                  # exclude seeds/imports (no infector)
        uniq, counts = (np.unique(infectors, return_counts=True) if len(infectors) else (np.array([]), np.array([])))
        self.n_targets = {int(u): int(c) for u, c in zip(uniq, counts)}
        # R estimate: mean secondary infections per distinct infector.
        self.r0 = float(len(infectors) / len(uniq)) if len(uniq) else 0.0
        return

    def make_detailed(self):
        """Return a DataFrame of the transmission events (source, target, day, variant)."""
        rows = [dict(source=s, target=t, day=ti, variant=v) for s, t, ti, v in (self.infection_events or [])]
        return pd.DataFrame(rows, columns=['source', 'target', 'day', 'variant'])

    def plot(self, fig=None, **kwargs):
        """Plot the offspring distribution (secondary infections per infector) + infections over time."""
        import matplotlib.pyplot as plt
        if fig is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        else:
            axes = fig.axes
        offspring = np.array(list(self.n_targets.values())) if self.n_targets else np.array([0])
        axes[0].hist(offspring, bins=np.arange(0, offspring.max() + 2) - 0.5)
        axes[0].set_title(f'Offspring per infector (R0={self.r0:.2f})')
        axes[0].set_xlabel('Secondary infections'); axes[0].set_ylabel('Number of infectors')
        days = np.array([e[2] for e in self.infection_events]) if self.infection_events else np.array([])
        if len(days):
            axes[1].hist(days, bins=np.arange(days.min(), days.max() + 2))
        axes[1].set_title('Transmission events over time'); axes[1].set_xlabel('Day')
        fig.tight_layout()
        return fig


__all__ += ['daily_age_stats']


class daily_age_stats(Analyzer):
    """
    Record daily counts of disease states by age bin (the v3 ``cv.daily_age_stats``).

    ``age_results[state]`` is an ``(npts, n_bins)`` array of the per-day count of agents in ``state`` by
    age bin; ``bins`` are the (lower) bin edges. Useful for age-stratified time series.
    """

    def __init__(self, states=None, edges=None, **kwargs):
        super().__init__(**kwargs)
        self.states = states or ['exposed', 'severe', 'critical', 'dead', 'diagnosed']
        self.edges = np.array(edges) if edges is not None else np.linspace(0, 100, 11)
        self.bins = self.edges[:-1]
        self.age_results = sc.objdict()  # not "results" -- that attr is locked on ss.Module
        return

    def init_post(self):
        super().init_post()
        npts = len(np.asarray(self.sim.t.timevec))
        for state in self.states:
            self.age_results[state] = np.zeros((npts, len(self.bins)))
        return

    def step(self):
        ti = self.ti
        covid = self._covid()
        ages = np.asarray(self.sim.people.age)
        for state in self.states:
            if not hasattr(covid, state):
                continue
            mask = np.asarray(getattr(covid, state)).astype(bool)
            counts, _ = np.histogram(ages[mask], bins=self.edges)
            self.age_results[state][ti, :] = counts
        return
