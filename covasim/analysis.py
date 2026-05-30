"""
Analysis tools for Covasim on the Starsim base.

M7 restores ``cv.Fit`` -- the model-vs-data goodness-of-fit class (v3 ``analysis.Fit``). It is used
post-run as ``cv.Fit(sim, data=...)`` and reuses the kept-from-v3 ``cv.compute_gof`` (misc.py).
Terminology (as v3): *difference* = sim - data per matched point; *goodness-of-fit (gof)* = the
difference through ``compute_gof``; *loss* = gof x weight; *mismatch* = sum of losses (the scalar
minimised during calibration).

The other analyzers (``snapshot``/``age_histogram``/``daily_age_stats``/``nab_histogram``) and the
``cv.TransTree`` + ``cv.Fit.plot`` come in M9; ``cv.Calibration`` (Optuna) is the rest of M7.
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
