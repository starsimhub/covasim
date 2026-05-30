"""
Multi-run + scenario tools for Covasim on the Starsim base.

M8 restores ``cv.MultiSim`` (multi-seed runs + median/quantile uncertainty bands),
``cv.parallel``/``cv.multi_run``/``cv.single_run``, and ``cv.Scenarios`` (named parameter-set
comparison) -- all wrapping Starsim's ``ss.MultiSim``/``ss.parallel``.

``ss.MultiSim`` runs ``cv.Sim`` multi-seed correctly, but its ``reduce()`` does not handle Covasim's
bridged top-level + nested ``['variant']`` results, so ``cv.MultiSim`` uses ``ss.MultiSim`` to RUN the
seeds and does its OWN reduction over the per-seed COVID-module time-series Results (the v3 UQ pattern:
median trajectory + low/high quantile bands).
"""
import numpy as np
import sciris as sc
import starsim as ss

from . import sim as cvsim

__all__ = ['MultiSim', 'parallel', 'multi_run', 'single_run', 'Scenarios']


def single_run(sim, **kwargs):
    """Run one copy of ``sim`` and return it (the v3 ``cv.single_run``)."""
    s = sc.dcp(sim)
    s.run(**kwargs)
    return s


class MultiSim(sc.prettyobj):
    """
    Run a sim across multiple seeds and combine into uncertainty intervals (the v3 ``cv.MultiSim``).

    Args:
        sim (cv.Sim): the base sim to run across seeds (omit if passing ``sims``).
        sims (list): an explicit list of (built) sims to run, instead of a base sim + n_runs.
        n_runs (int): number of seeds to run from the base sim (default 4).
        label (str): an optional label.

    After ``run()``: ``sims`` (the per-seed run sims). After ``reduce()``/``mean()``/``median()``:
    ``results[key] = objdict(best, low, high)`` (median or mean trajectory + quantile bands).
    """

    def __init__(self, sim=None, sims=None, n_runs=4, label=None, **kwargs):
        if sims is not None:
            self.base_sim = None
            self.sims = list(sims)
            self.n_runs = len(self.sims)
        else:
            self.base_sim = sim
            self.sims = None
            self.n_runs = int(n_runs)
        self.label = label
        self.kwargs = kwargs
        self.results = None
        return

    def run(self, **kwargs):
        """Run the seeds (or the explicit sims) via ``ss.MultiSim``."""
        if self.base_sim is not None:
            msim = ss.MultiSim(base_sim=self.base_sim, n_runs=self.n_runs, **self.kwargs)
        else:
            msim = ss.MultiSim(sims=self.sims, **self.kwargs)
        msim.run(**kwargs)
        self.sims = list(msim.sims)
        return self

    @staticmethod
    def _covid_results(s):
        return list(s.diseases.values())[0].results

    def reduce(self, quantiles=None, use_mean=False, keys=None):
        """Combine the per-seed COVID time series into a median/mean + low/high quantile band."""
        if self.sims is None:
            raise RuntimeError('Run the MultiSim before reducing.')
        q = quantiles if quantiles is not None else (0.1, 0.9)
        r0 = self._covid_results(self.sims[0])
        if keys is None:
            keys = [k for k in r0.keys() if isinstance(r0[k], ss.Result) and np.ndim(np.asarray(r0[k])) == 1]
        red = sc.objdict()
        for k in keys:
            stack = np.array([np.asarray(self._covid_results(s)[k]) for s in self.sims])  # (n_runs, npts)
            best = stack.mean(axis=0) if use_mean else np.median(stack, axis=0)
            red[k] = sc.objdict(best=best, low=np.quantile(stack, q[0], axis=0),
                                high=np.quantile(stack, q[1], axis=0))
        self.results = red
        return self

    def mean(self, **kwargs):
        """Reduce using the mean trajectory."""
        return self.reduce(use_mean=True, **kwargs)

    def median(self, **kwargs):
        """Reduce using the median trajectory."""
        return self.reduce(use_mean=False, **kwargs)

    def plot(self, keys=None, fig=None, **kwargs):
        """Plot the median trajectory + quantile band for each key."""
        import matplotlib.pyplot as plt
        if self.results is None:
            self.reduce()
        keys = keys or [k for k in ('n_infectious', 'cum_infections', 'cum_severe', 'cum_deaths')
                        if k in self.results]
        t = np.arange(len(self.results[keys[0]].best))
        if fig is None:
            fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4))
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, keys):
            r = self.results[k]
            ax.plot(t, r.best, lw=2, label='median')
            ax.fill_between(t, r.low, r.high, alpha=0.25, label='10-90%')
            ax.set_title(k); ax.set_xlabel('Day'); ax.legend()
        fig.tight_layout()
        return fig


def multi_run(sim, n_runs=4, **kwargs):
    """Run ``sim`` across ``n_runs`` seeds and return the list of run sims (the v3 ``cv.multi_run``)."""
    return MultiSim(sim, n_runs=n_runs).run(**kwargs).sims


def parallel(*args, **kwargs):
    """Run multiple distinct sims in parallel and return a ``cv.MultiSim`` (the v3 ``cv.parallel``)."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        sims = list(args[0])
    else:
        sims = list(args)
    return MultiSim(sims=sims).run(**kwargs)


class Scenarios(sc.prettyobj):
    """
    Compare named scenarios, each a parameter override over a common base (the v3 ``cv.Scenarios``),
    built on ``cv.MultiSim`` for per-scenario multi-seed uncertainty.

    Args:
        basepars (dict): cv.Sim kwargs shared by every scenario.
        scenarios (dict): ``{name: {'name': label, 'pars': {cv.Sim kwargs override}}}``.
        n_runs (int): seeds per scenario (default 3).

    After ``run()``: ``results[name] = objdict(label, results=<reduced UQ>, msim=<cv.MultiSim>)``.
    """

    def __init__(self, basepars=None, scenarios=None, n_runs=3, label=None, **kwargs):
        self.basepars = sc.mergedicts(basepars)
        self.scenarios = scenarios or {}
        self.n_runs = int(n_runs)
        self.label = label
        self.kwargs = kwargs
        self.results = None
        return

    def run(self, **kwargs):
        """Run each scenario as a multi-seed cv.MultiSim and store its reduced UQ results."""
        self.results = sc.objdict()
        for name, scen in self.scenarios.items():
            pars = sc.mergedicts(self.basepars, scen.get('pars', {}))
            base = cvsim.Sim(**pars)
            msim = MultiSim(base, n_runs=self.n_runs, **self.kwargs).run(**kwargs).reduce()
            self.results[name] = sc.objdict(label=scen.get('name', name), results=msim.results, msim=msim)
        return self

    def plot(self, key='cum_infections', fig=None, **kwargs):
        """Overlay one result key (median + band) across scenarios."""
        import matplotlib.pyplot as plt
        if self.results is None:
            self.run()
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            ax = fig.gca()
        for name, scen in self.results.items():
            r = scen.results[key]
            t = np.arange(len(r.best))
            line, = ax.plot(t, r.best, lw=2, label=scen.label)
            ax.fill_between(t, r.low, r.high, alpha=0.2, color=line.get_color())
        ax.set_title(key); ax.set_xlabel('Day'); ax.legend()
        return fig
