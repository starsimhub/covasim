"""
Functions and classes for running multiple Covasim runs.

``cv.MultiSim`` is built on ``ss.MultiSim``, and ``cv.Scenarios`` is built on ``cv.MultiSim``.
Both keep the v3 interface: e.g. after ``msim.reduce()``, ``msim.results['cum_infections']``
is a result with ``.values``, ``.low``, and ``.high``; and after ``scens.run()``,
``scens.results['cum_infections']['baseline']['best']`` is the median of the baseline scenario.
"""

#%% Imports
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from . import misc as cvm
from . import compat as cvc
from . import plotting as cvplt
from . import sim as cvs
from .settings import options as cvo


# Specify all externally visible functions this file defines
__all__ = ['make_metapars', 'MultiSim', 'Scenarios', 'single_run', 'multi_run', 'parallel']


def make_metapars():
    """ Create default metaparameters for a Scenarios run """
    metapars = sc.objdict(
        n_runs    = 3, # Number of runs per scenario
        noise     = 0.0, # Use noise, optionally
        noisepar  = 'beta',
        quantiles = {'low':0.1, 'high':0.9},
        verbose   = cvo.verbose,
    )
    return metapars


#%% MultiSim

class MultiSim(cvc.V3MultiSim, ss.MultiSim):
    """
    Class for running multiple copies of a simulation. The parameter n_runs
    controls how many copies of the simulation there will be, if a list of sims
    is not provided. This is the main class that's used to run multiple versions
    of a simulation (e.g., with different random seeds).

    This is ``ss.MultiSim`` with Covasim's options for each run (``noise``, ``noisepar``,
    and ``keep_people``) and the v3 ways of summarizing the runs (``reduce()``, ``mean()``,
    ``median()``, and ``combine()``), which replace ``msim.results`` (every result, with a
    run axis first, as in Starsim) with the results of the reduced or combined sim.

    Args:
        sims       (Sim/list) : a single sim or a list of sims
        base_sim   (Sim)      : the sim used for shared properties; if not supplied, the first of the sims provided
        label      (str)      : the name of the multisim
        n_runs     (int)      : if a single sim is provided, the number of replicates (default 4)
        initialize (bool)     : whether or not to initialize the sims (otherwise, initialize them during run)
        inplace    (bool)     : whether to update the supplied sims with the results (default False, as in v3; Starsim's default is True)
        kwargs     (dict)     : how to run the sims, e.g. noise=0.1, keep_people=True, or parallel=False (see ``ss.MultiSim``)

    Returns:
        msim: a MultiSim object

    MultiSims can be merged with ``cv.MultiSim.merge()`` and split with ``msim.split()``
    (see ``ss.MultiSim``).

    **Examples**::

        sim = cv.Sim() # Create the sim
        msim = cv.MultiSim(sim, n_runs=5) # Create the multisim
        msim.run() # Run them in parallel
        msim.combine() # Combine into one sim
        msim.plot() # Plot results

        sim = cv.Sim() # Create the sim
        msim = cv.MultiSim(sim, n_runs=11, noise=0.1, keep_people=True) # Set up a multisim with noise
        msim.run() # Run
        msim.reduce() # Compute statistics
        msim.plot() # Plot

        sims = [cv.Sim(beta=0.015*(1+0.02*i)) for i in range(5)] # Create sims
        for sim in sims: sim.run() # Run sims in serial
        msim = cv.MultiSim(sims) # Convert to multisim
        msim.plot() # Plot as single sim
    """
    run_options = ss.MultiSim.run_options + ['noise', 'noisepar'] # Passed to make_run()

    def __init__(self, sims=None, base_sim=None, label=None, n_runs=None, initialize=False, inplace=False, **kwargs):
        """ As for ss.MultiSim, except that by default the original sims are not modified when they are run (the v3 behavior) """
        super().__init__(sims=sims, base_sim=base_sim, label=label, n_runs=n_runs, initialize=initialize, inplace=inplace, **keep_people(kwargs))
        self.which = None # Whether the runs have been reduced or combined
        return

    def make_run(self, sim, values, seed, noise=0.0, noisepar=None):
        """
        The sim of one run (see ``ss.MultiSim.make_run()``): as in v3, the parameters given are set with
        ``sim[key] = value``, and with noise, the parameter ``noisepar`` (default beta) is multiplied by a
        random factor, 1 + noise*z for z > 0 and 1/(1 - noise*z) for z < 0, where z is a standard normal draw
        """
        values = dict(values or {})
        seed = values.pop('rand_seed', seed)
        new = super().make_run(sim, None, seed)
        for key,val in values.items():
            new[key] = val
        if noise:
            noisepar = sc.ifelse(noisepar, 'beta')
            noiseval = noise*np.random.default_rng(seed).normal()
            new[noisepar] *= (1 + noiseval) if noiseval > 0 else 1/(1 - noiseval)
        return new

    def run(self, reduce=False, combine=False, **kwargs):
        """
        Run the sims (see ``ss.MultiSim.run()``), and optionally reduce or combine them.

        Args:
            reduce  (bool): whether or not to reduce after running (see reduce())
            combine (bool): whether or not to combine after running (see combine(), not compatible with reduce)
            kwargs  (dict): e.g. n_runs, noise, keep_people, parallel; use run_args to pass arguments to sim.run()

        Returns:
            msim (MultiSim): the MultiSim, modified in place

        **Examples**::

            msim.run()
            msim.run(run_args=dict(until='2020-06-01'))
        """
        super().run(**keep_people(kwargs))
        self.which = None
        if reduce:
            self.reduce()
        elif combine:
            self.combine()
        return self

    def reduce(self, quantiles=None, use_mean=False, bounds=None, output=False):
        """
        Combine multiple sims into a single sim statistically: by default, use
        the median value and the 10th and 90th percentiles for the lower and upper
        bounds. If use_mean=True, then use the mean and ±2 standard deviations
        for lower and upper bounds.

        The reduced sim becomes the base sim, so e.g. ``msim.results['cum_infections']``
        (the same as ``msim.base_sim.results['cum_infections']``) has ``.values``, ``.low``,
        and ``.high``.

        Args:
            quantiles (dict): the quantiles to use, e.g. [0.1, 0.9] or {'low : '0.1, 'high' : 0.9}
            use_mean (bool): whether to use the mean instead of the median
            bounds (float): if use_mean=True, the multiplier on the standard deviation for upper and lower bounds (default 2)
            output (bool): whether to return the "reduced" sim (otherwise, return the MultiSim)

        **Example**::

            msim = cv.MultiSim(cv.Sim())
            msim.run()
            msim.reduce()
            msim.summarize()
        """
        self.check_run('reduce')
        bounds = sc.ifelse(bounds, 2)
        quantiles = sc.ifelse(quantiles, dict(low=0.1, high=0.9))
        if not isinstance(quantiles, dict):
            quantiles = dict(low=float(quantiles[0]), high=float(quantiles[1]))
        reduced = sc.dcp(self.sims[0])
        reduced.metadata = dict(parallelized=True, combined=False, n_runs=len(self.sims), quantiles=quantiles, use_mean=use_mean, bounds=bounds)
        flats = [sim.results.flatten() for sim in self.sims]
        for key,res in reduced.results.flatten().items(): # The results of the sim and of each module, e.g. the COVID results
            raw = np.array([flat[key].values for flat in flats], dtype=float) # Shape (n_runs, npts), or (n_runs, npts, ncols) for results by variant
            if use_mean:
                mean, std = raw.mean(axis=0), raw.std(axis=0)
                res.values, res.low, res.high = mean, mean - bounds*std, mean + bounds*std
            else:
                res.values = np.quantile(raw, 0.5, axis=0)
                res.low, res.high = np.quantile(raw, quantiles['low'], axis=0), np.quantile(raw, quantiles['high'], axis=0)
        reduced.compute_summary()
        if self.which is None:
            self.orig_base_sim = self.base_sim
        self.base_sim = reduced
        self.results = reduced.results # As in v3, the results are those of the reduced sim, rather than with a run axis
        self.which = 'reduced'
        return reduced if output else self

    def mean(self, bounds=None, **kwargs):
        """
        Alias for reduce(use_mean=True): the mean and ±2 standard deviations.

        Args:
            bounds (float): the multiplier on the standard deviation for the upper and lower bounds (default 2)
            kwargs (dict): passed to reduce()
        """
        return self.reduce(use_mean=True, bounds=bounds, **kwargs)

    def median(self, quantiles=None, **kwargs):
        """
        Alias for reduce(use_mean=False): the median and the 10th and 90th percentiles.

        Args:
            quantiles (list or dict): upper and lower quantiles (default 0.1 and 0.9)
            kwargs (dict): passed to reduce()
        """
        return self.reduce(use_mean=False, quantiles=quantiles, **kwargs)

    def combine(self, output=False):
        """
        Combine multiple sims into a single sim with scaled results, e.g. to
        treat several smaller sims as one larger population (see ``ss.MultiSim.combine()``).
        Count results are summed, and other results (e.g. prevalence) are averaged, weighted
        by each sim's total population.

        Note: unlike v3, the people are not combined; the combined sim has the
        people of the first sim.

        Args:
            output (bool): whether to return the combined sim (otherwise, return the MultiSim)

        **Example**::

            msim = cv.MultiSim(cv.Sim())
            msim.run()
            msim.combine()
            msim.summarize()
        """
        combined = super().combine()
        if self.which is None:
            self.orig_base_sim = self.base_sim
        self.base_sim = combined
        self.results = combined.results # As in v3, the results are those of the combined sim, rather than with a run axis
        self.which = 'combined'
        return combined if output else self

    def reset(self):
        """ Undo reduce() or combine(): the base sim is the one the MultiSim was made with, and the results are those of the runs """
        if self.which is not None:
            self.base_sim = self.orig_base_sim
            self.results = ss.run.MultiResults.from_sims(self.sims, self.runs)
            self.which = None
        return self

    @property
    def summary(self):
        """ v3: the summary of the reduced or combined sim (that of the base sim) """
        return self.base_sim.summary

    @summary.setter
    def summary(self, summary):
        pass # Set by ss.MultiSim, but in v3 it is the base sim's

    @classmethod
    def merge(cls, *args, base=False):
        """
        Merge several MultiSims into one (see ``ss.MultiSim.merge()``).

        Args:
            args (MultiSim): the MultiSims to merge (either a list, or separate arguments)
            base (bool): if True (as in v3), the new MultiSim has the base sims of each (e.g. reduced) as its sims, rather than all of their sims

        **Examples**::

            msim = cv.MultiSim.merge(msim1, msim2) # All the runs of each
            msim = cv.MultiSim.merge(msim1, msim2, base=True) # The two reduced sims, e.g. to plot them as scenarios
        """
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        if base:
            return cls(sims=[sc.dcp(msim.base_sim) for msim in args])
        args = [cls(sims=list(msim.sims), label=msim.label) if msim.which else msim for msim in args] # The runs of a reduced or combined MultiSim, rather than its reduced or combined sim
        return super().merge(*args)

    def compare(self, t=None, sim_inds=None, output=False, do_plot=False, **kwargs):
        """
        Create a dataframe compare sims at a single point in time.

        Args:
            t        (int/str) : the day (or date) to do the comparison; default, the end
            sim_inds (list)    : list of integers of which sims to include (default: all)
            output   (bool)    : whether or not to return the comparison as a dataframe
            do_plot  (bool)    : whether or not to plot the comparison (see also plot_compare())
            kwargs   (dict)    : passed to plot_compare()

        Returns:
            df (dataframe): a dataframe comparison
        """
        # Handle time
        if t is None:
            t = -1
            daystr = 'the last day'
        else:
            daystr = f'day {t}'

        # Handle the indices
        if sim_inds is None:
            sim_inds = list(range(len(self.sims)))

        # Move the results to a dictionary
        resdict = sc.ddict(dict)
        for i,s in enumerate(sim_inds):
            sim = self.sims[s]
            day = sim.day(t) # Unlikely, but different sims might have different start days
            label = sim.label
            if not label: # Give it a label if it doesn't have one
                label = f'Sim {s}'
            if label in resdict: # Avoid duplicates
                label += f' ({i})'
            for reskey in sim.result_keys():
                res = sim.results[reskey]
                val = res.values[day]
                if res.scale and (val == np.round(val)): # Show whole-number counts as ints (but not e.g. 4.5 from scaling)
                    val = int(val)
                resdict[label][reskey] = val

        if do_plot:
            self.plot_compare(t=t, sim_inds=sim_inds, **kwargs)

        df = pd.DataFrame.from_dict(resdict).astype(object) # astype is necessary to prevent type coercion
        if not output:
            print(f'Results for {daystr} in each sim:')
            print(df)
        else:
            return df

    def plot(self, to_plot=None, inds=None, plot_sims=False, color_by_sim=None, max_sims=5, colors=None, labels=None, alpha_range=None, plot_args=None, show_args=None, **kwargs):
        """
        Plot all the sims  -- arguments passed to Sim.plot(). The
        behavior depends on whether or not combine() or reduce() has been called.
        If so, this function by default plots only the combined/reduced sim (which
        you can override with plot_sims=True). Otherwise, it plots a separate line
        for each sim.

        Note that this function is complex because it aims to capture the flexibility
        of both sim.plot() and scens.plot(). By default, if combine() or reduce()
        has been used, it will resemble sim.plot(); otherwise, it will resemble
        scens.plot(). This can be changed via color_by_sim, together with the
        other options.

        Args:
            to_plot      (list) : list or dict of which results to plot; see cv.get_default_plots() for structure
            inds         (list) : if not combined or reduced, the indices of the simulations to plot (if None, plot all)
            plot_sims    (bool) : whether to plot individual sims, even if combine() or reduce() has been used
            color_by_sim (bool) : if True, set colors based on the simulation type; otherwise, color by result type; True implies a scenario-style plotting, False implies sim-style plotting
            max_sims     (int)  : maximum number of sims to use with color-by-sim; can be overridden by other options
            colors       (list) : if supplied, override default colors for color_by_sim
            labels       (list) : if supplied, override default labels for color_by_sim
            alpha_range  (list) : a 2-element list/tuple/array providing the range of alpha values to use to distinguish the lines
            plot_args    (dict) : passed to sim.plot()
            show_args    (dict) : passed to sim.plot()
            kwargs       (dict) : passed to sim.plot()

        Returns:
            fig: Figure handle

        **Examples**::

            sim = cv.Sim()
            msim = cv.MultiSim(sim)
            msim.run()
            msim.plot() # Plots individual sims
            msim.reduce()
            msim.plot() # Plots the combined sim
        """

        # Plot a single curve, possibly with a range
        if not plot_sims and self.which in ['combined', 'reduced']:
            fig = self.base_sim.plot(to_plot=to_plot, colors=colors, plot_args=plot_args, show_args=show_args, **kwargs)

        # PLot individual sims on top of each other
        else:

            # Initialize
            fig          = kwargs.pop('fig', None)
            orig_show    = kwargs.get('do_show', None)
            orig_close   = cvo.close
            orig_setylim = kwargs.get('setylim', True)
            kwargs['legend_args'] = sc.mergedicts({'show_legend':True}, kwargs.get('legend_args')) # Only plot the legend the first time

            # Handle indices
            if inds is None:
                inds = np.arange(len(self.sims))
            n_sims = len(inds)

            # Handle what style of plotting to use:
            if color_by_sim is None:
                if n_sims <= max_sims:
                    color_by_sim = True
                else:
                    color_by_sim = False

            # Handle what to plot -- if None, the default plots for this kind are used by sim.plot()
            kind = 'scens'
            if to_plot is None:
                kind = 'scens' if color_by_sim else 'sim'

            # Handle colors
            if colors is None:
                if color_by_sim:
                    colors = sc.gridcolors(ncolors=n_sims)
                else:
                    colors = [None]*n_sims # So we can iterate over it
            else:
                colors = [colors]*n_sims # Again, for iteration

            # Handle alpha if not using colors
            if alpha_range is None:
                if color_by_sim:
                    alpha_range = [0.8, 0.8] # We're using color to distinguish sims, so don't need alpha
                else:
                    alpha_range = [0.8, 0.3] # We're using alpha to distinguish sims
            alphas = np.linspace(alpha_range[0], alpha_range[1], n_sims)

            # Plot
            for s,ind in enumerate(inds):

                sim = self.sims[ind]
                final_plot = (s == n_sims-1) # Check if this is the final plot

                # Handle the legend and labels
                if final_plot:
                    merged_show_args  = sc.mergedicts(dict(returnfig=cvo.returnfig), show_args)
                    kwargs['do_show'] = orig_show
                    kwargs['setylim'] = orig_setylim
                    cvo.set(close=orig_close) # Reset original closing settings
                else:
                    merged_show_args  = dict(annotations=False, returnfig=True) # Only show things like data the last time it's plotting
                    kwargs['do_show'] = False # On top of that, don't show the plot at all unless it's the last time
                    kwargs['setylim'] = False # Don't set the y limits until we have all the data
                    cvo.set(close=False) # Do not close figures if we're in the middle of plotting

                # Optionally set the label for the first max_sims sims
                if color_by_sim is True and s<max_sims:
                    if labels is None:
                        merged_labels = sim.label
                    else:
                        merged_labels = labels[s]
                elif final_plot and not color_by_sim:
                    merged_labels = labels
                else:
                    merged_labels = ''

                # Actually plot
                merged_plot_args = sc.mergedicts({'alpha':alphas[s]}, plot_args) # Need a new variable to avoid overwriting
                fig = sim.plot(fig=fig, to_plot=(kind, to_plot), colors=colors[s], labels=merged_labels, plot_args=merged_plot_args, show_args=merged_show_args, **kwargs)

        return cvplt.handle_show_return(fig=fig)

    def plot_result(self, key, colors=None, labels=None, *args, **kwargs):
        """ Convenience method for plotting -- arguments passed to sim.plot_result() """
        if self.which in ['combined', 'reduced']:
            fig = self.base_sim.plot_result(key, *args, **kwargs)
        else:
            fig = None
            if colors is None:
                colors = sc.gridcolors(len(self))
            if labels is None:
                labels = [sim.label for sim in self.sims]
            orig_setylim = kwargs.get('setylim', True)
            for s,sim in enumerate(self.sims):
                if s == len(self.sims)-1:
                    kwargs['setylim'] = orig_setylim
                else:
                    kwargs['setylim'] = False
                fig = sim.plot_result(key=key, fig=fig, color=colors[s], label=labels[s], *args, **kwargs)
        return cvplt.handle_show_return(fig=fig)

    def plot_compare(self, t=-1, sim_inds=None, log_scale=True, **kwargs):
        """
        Plot a comparison between sims, using bars to show different values for
        each result. For an explanation of other available arguments, see Sim.plot().

        Args:
            t         (int)  : index of results, passed to compare()
            sim_inds  (list) : which sims to include, passed to compare()
            log_scale (bool) : whether to plot with a logarithmic x-axis
            kwargs    (dict) : standard plotting arguments, see Sim.plot() for explanation

        Returns:
            fig: Figure handle
        """
        df = self.compare(t=t, sim_inds=sim_inds, output=True)
        return cvplt.plot_compare(df, log_scale=log_scale, **kwargs)

    def save(self, filename=None, keep_people=False, **kwargs):
        """
        Save to disk as a gzipped pickle. Load with cv.load(filename) or
        cv.MultiSim.load(filename).

        Args:
            filename    (str)  : the name or path of the file to save to; if None, uses default
            keep_people (bool) : whether or not to store the population in the Sim objects (NB, very large)
            kwargs      (dict) : passed to ``sc.makefilepath()``

        Returns:
            msimfile (str): the validated absolute path to the saved file

        **Example**::

            msim.save() # Saves to an .msim file
        """
        if filename is None:
            filename = 'covasim.msim'
        msimfile = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename # Store the actual saved filename

        # Store sims separately
        sims = self.sims
        self.sims = None # Remove for now
        try:
            obj = sc.dcp(self) # This should be quick once we've removed the sims
            if keep_people:
                obj.sims = sims # Just restore the object in full
                print('Note: saving people, which may produce a large file!')
            else:
                if obj.base_sim.initialized: # Nothing to shrink otherwise
                    obj.base_sim.shrink(in_place=True)
                obj.sims = [sim.shrink(in_place=False) for sim in sims]
            cvm.save(filename=msimfile, obj=obj) # Actually save
        finally:
            self.sims = sims # Restore, even if saving failed
        return msimfile

    @staticmethod
    def load(msimfile, *args, **kwargs):
        """
        Load from disk from a gzipped pickle.

        Args:
            msimfile (str): the name or path of the file to load from
            kwargs: passed to cv.load()

        Returns:
            msim (MultiSim): the loaded MultiSim object

        **Example**::

            msim = cv.MultiSim.load('my-multisim.msim')
        """
        msim = cvm.load(msimfile, *args, **kwargs)
        if not isinstance(msim, MultiSim):
            errormsg = f'Cannot load object of {type(msim)} as a MultiSim object'
            raise TypeError(errormsg)
        return msim


#%% Scenarios

class Scenarios(cvc.V3Scenarios):
    """
    Class for running multiple sets of multiple simulations -- e.g., scenarios.
    Note that most users are recommended to use MultiSim rather than Scenarios,
    as it gives more control over run options. Scenarios should be used primarily
    for quick investigations. See the examples folder for example usage.

    Each scenario is run as a cv.MultiSim, and the results are stored as
    ``scens.results[reskey][scenkey]``, with ``name``, ``best`` (the median),
    ``low``, and ``high`` entries. For the results by variant (e.g. ``cum_infections_by_variant``),
    these have shape (time, variant), as in the sim, rather than (variant, time) as in v3.

    Args:
        sim       (Sim)  : if supplied, use a pre-created simulation as the basis for the scenarios
        metapars  (dict) : meta-parameters for the run, e.g. number of runs; see make_metapars() for structure
        scenarios (dict) : a dictionary defining the scenarios; see examples folder for examples; see below for baseline
        basepars  (dict) : a dictionary of sim parameters to be used for the basis of the scenarios (not required if sim is provided)
        scenfile  (str)  : a filename for saving (defaults to the creation date)
        label     (str)  : the name of the scenarios

    **Example**::

        scenarios = {'base': {'name':'Base','pars': {}}, 'beta': {'name':'Beta', 'pars': {'beta': 0.020}}}
        scens = cv.Scenarios(scenarios=scenarios, basepars={'pop_size':5000})
        scens.run()
        scens.plot()

    Returns:
        scens: a Scenarios object
    """

    def __init__(self, sim=None, metapars=None, scenarios=None, basepars=None, scenfile=None, label=None):

        # For this object, metapars are the foundation
        self.pars = make_metapars() # Start with default pars
        self.created = sc.now()

        # Handle filename
        if scenfile is None:
            datestr = sc.getdate(obj=self.created, dateformat='%Y-%b-%d_%H.%M.%S')
            scenfile = f'covasim_scenarios_{datestr}.scens'
        self.scenfile = scenfile
        self.label = label

        # Handle scenarios -- by default, create the simplest possible baseline scenario
        if scenarios is None:
            scenarios = {'baseline':{'name':'Baseline', 'pars':{}}}
        self.scenarios = scenarios

        # Handle metapars
        self.metapars = sc.dcp(sc.mergedicts(metapars))
        self.update_pars(self.metapars)

        # Create the simulation and handle basepars
        if sim is None:
            sim = cvs.Sim()
        self.base_sim = sim.new() if sim.initialized else sc.dcp(sim) # An initialized sim can't have its parameters changed, so it's made again
        self.basepars = sc.dcp(sc.mergedicts(basepars))
        self.base_sim.update_pars(self.basepars)

        # Copy quantities from the base sim to the main object
        self.npts    = self.base_sim.npts
        self.tvec    = self.base_sim.tvec
        self.datevec = self.base_sim.datevec
        if 'verbose' not in self.metapars: # Use the sim's verbosity unless it's supplied as a metaparameter
            self['verbose'] = self.base_sim['verbose']

        # The sims and results are populated by run(); the results order is: results key, scenario, name/best/low/high
        self.sims = sc.objdict()
        self.results = sc.objdict()
        self._kept_people = False
        return

    def result_keys(self, which='all'):
        """
        The result keys of the scenarios; see sim.result_keys(). These are only
        available once the scenarios have been run.

        Args:
            which (str): 'main', 'variant', or 'all'
        """
        choices = ['main', 'variant', 'all']
        if which not in choices:
            errormsg = f'Choice "{which}" not available; choices are: {sc.strjoin(choices)}'
            raise ValueError(errormsg)
        keys = []
        for key in self.results.keys():
            is_variant = key.endswith('_by_variant')
            if which == 'all':
                keys.append(key)
            elif is_variant == (which == 'variant'):
                keys.append(key)
        return keys

    def run(self, debug=False, keep_people=False, verbose=None, **kwargs):
        """
        Run the specified scenarios.

        Args:
            debug       (bool) : if True, runs a single run instead of multiple, which makes debugging easier
            keep_people (bool) : whether to keep the people in each sim after it has run
            verbose     (int)  : level of detail to print, passed to sim.run()
            kwargs      (dict) : passed to MultiSim.run(), and thence to multi_run()

        Returns:
            scens (Scenarios): the Scenarios, modified in place
        """
        if verbose is None:
            verbose = self['verbose']

        def print_heading(string):
            """ Choose whether to print a heading, regular text, or nothing """
            if verbose >= 2:
                sc.heading(string)
            elif verbose == 1:
                print(string)
            return

        # Loop over scenarios
        self.sims = sc.objdict()
        self.results = sc.objdict()
        for scenkey,scen in self.scenarios.items():
            scenname = scen['name']
            scenpars = scen['pars']

            # This is necessary for plotting, and since self.npts is defined prior to run
            if 'n_days' in scenpars.keys():
                errormsg = 'Scenarios cannot be run with different numbers of days; set via basepars instead'
                raise ValueError(errormsg)

            # Create the sim for this scenario
            print_heading(f'Multirun for {scenkey}')
            scen_sim = sc.dcp(self.base_sim)
            scen_sim.scenkey = scenkey
            scen_sim.label = scenname
            scen_sim.scen = scen
            scen_sim.update_pars(sc.dcp(scenpars)) # Interventions and analyzers replace those in the base sim

            # Run the sims, and reduce them to the median and quantiles
            n_runs = 1 if debug else self['n_runs']
            if debug:
                print('Running in debug mode (not parallelized)')
                kwargs['parallel'] = False
            msim = MultiSim(scen_sim, n_runs=n_runs, noise=self['noise'], noisepar=self['noisepar'], keep_people=keep_people, verbose=verbose)
            msim.run(**kwargs)
            print_heading(f'Processing {scenkey}')
            msim.median(quantiles=self['quantiles'])

            # Store the results, including those by variant (with time as the first axis, as in the sim)
            for reskey in msim.base_sim.result_keys('all'):
                res = msim.base_sim.results['variant'][reskey] if reskey.endswith('_by_variant') else msim.base_sim.results[reskey]
                if reskey not in self.results:
                    self.results[reskey] = sc.objdict()
                self.results[reskey][scenkey] = sc.objdict(name=scenname, best=res.values, low=res.low, high=res.high)
            self.sims[scenkey] = msim.sims

        # Print statistics
        if verbose:
            self.compare()

        # Save details about the run
        self._kept_people = keep_people

        return self

    def compare(self, t=None, output=False):
        """
        Print out a comparison of each scenario.

        Args:
            t (int/str)   : the day (or date) to do the comparison; default, the end
            output (bool) : if true, return the dataframe instead of printing output

        **Example**::

            scenarios = {'base': {'name':'Base','pars': {}}, 'beta': {'name':'Beta', 'pars': {'beta': 0.020}}}
            scens = cv.Scenarios(scenarios=scenarios, label='Example scenarios')
            scens.run()
            scens.compare(t=30) # Prints comparison for day 30
        """
        # Handle time
        if t is None:
            t = -1
            daystr = 'the last day'
        else:
            daystr = f'day {t}'
        day = self.base_sim.day(t) # Unlike MultiSims, scenarios must have the same start day

        # Compute dataframe
        x = sc.ddict(dict)
        for scenkey in self.scenarios.keys():
            for reskey in self.results.keys():
                vals = sc.toarray(self.results[reskey][scenkey].best[day]) # One value, or one per variant for the results by variant
                for v,val in enumerate(vals):
                    key = reskey
                    if reskey.endswith('_by_variant'):
                        key = reskey + str(v) # As in v3, add the variant number to the key
                    is_whole = np.isfinite(val) and (val == np.round(val))
                    if is_whole: # Show whole numbers as integers
                        val = int(val)
                    x[scenkey][key] = val
        df = pd.DataFrame.from_dict(x).astype(object)

        if not output:
            print(f'Results for {daystr} in each scenario:')
            print(df)
        else:
            return df

    def plot(self, *args, **kwargs):
        """
        Plot the results of a scenario. For an explanation of available arguments,
        see Sim.plot().

        Returns:
            fig: Figure handle


        **Example**::

            scens = cv.Scenarios()
            scens.run()
            scens.plot()
        """
        return cvplt.plot_scens(scens=self, *args, **kwargs)

    def plot_result(self, key, *args, **kwargs):
        """ Convenience method for plotting a single result; see Sim.plot() for other arguments """
        return self.plot(to_plot=[key], *args, **kwargs)

    def to_json(self, filename=None, tostring=True, indent=2, verbose=False, *args, **kwargs):
        """
        Export results as JSON.

        Args:
            filename (str): if None, return string; else, write to file

        Returns:
            A unicode string containing a JSON representation of the results,
            or writes the JSON file to disk
        """
        d = {'t':         self.tvec,
             'results':   self.results,
             'basepars':  self.basepars,
             'metapars':  self.metapars,
             'simpars':   self.base_sim.export_pars(),
             'scenarios': self.scenarios
             }
        if filename is None:
            output = sc.jsonify(d, tostring=tostring, indent=indent, verbose=verbose, *args, **kwargs)
        else:
            output = sc.savejson(filename=filename, obj=d, indent=indent, *args, **kwargs)
        return output

    def to_excel(self, filename=None):
        """
        Export results as XLSX

        Args:
            filename (str): if None, return string; else, write to file

        Returns:
            An sc.Spreadsheet with an Excel file, or writes the file to disk
        """
        spreadsheet = sc.Spreadsheet()
        spreadsheet.freshbytes()
        with pd.ExcelWriter(spreadsheet.bytes, engine='xlsxwriter') as writer:
            for key in self.results.keys():
                columns = {}
                for colkey,val in sc.flattendict(self.results[key], sep='_').items():
                    if np.ndim(val) == 2: # Results by variant: one column per variant, numbered as in compare()
                        for v in range(val.shape[1]):
                            columns[f'{colkey}{v}'] = val[:,v]
                    else:
                        columns[colkey] = val
                result_df = pd.DataFrame.from_dict(columns)
                result_df.to_excel(writer, sheet_name=key[:31]) # Excel sheet names are limited to 31 characters
        spreadsheet.load()

        if filename is None:
            output = spreadsheet
        else:
            output = spreadsheet.save(filename)
        return output

    def save(self, scenfile=None, keep_sims=True, keep_people=False, **kwargs):
        """
        Save to disk as a gzipped pickle.

        Args:
            scenfile    (str)  : the name or path of the file to save to; if None, uses stored
            keep_sims   (bool) : whether or not to store the actual Sim objects in the Scenarios object
            keep_people (bool) : whether or not to store the population in the Sim objects (NB, very large)
            kwargs      (dict) : passed to makefilepath()

        Returns:
            scenfile (str): the validated absolute path to the saved file

        **Example**::

            scens.save() # Saves to a .scens file with the date and time of creation by default
        """
        if scenfile is None:
            scenfile = self.scenfile
        scenfile = sc.makefilepath(filename=scenfile, **kwargs)
        self.scenfile = scenfile # Store the actual saved filename

        # Store sims separately
        sims = self.sims
        self.sims = None # Remove for now
        try:
            obj = sc.dcp(self) # This should be quick once we've removed the sims
            if not keep_people and obj.base_sim.initialized: # Nothing to shrink otherwise
                obj.base_sim.shrink(in_place=True)

            if keep_sims or keep_people:
                if keep_people:
                    if not obj._kept_people:
                        warnmsg = 'Warning: there are no people because they were not saved during the run. If you want people, please rerun with keep_people=True.'
                        cvm.warn(warnmsg)
                    obj.sims = sims # Just restore the object in full
                    print('Note: saving people, which may produce a large file!')
                else:
                    obj.sims = sc.objdict()
                    for key in sims.keys():
                        obj.sims[key] = [sim.shrink(in_place=False) for sim in sims[key]]

            cvm.save(filename=scenfile, obj=obj) # Actually save
        finally:
            self.sims = sims # Restore, even if saving failed
        return scenfile

    @staticmethod
    def load(scenfile, *args, **kwargs):
        """
        Load from disk from a gzipped pickle.

        Args:
            scenfile (str): the name or path of the file to load from
            kwargs: passed to cv.load()

        Returns:
            scens (Scenarios): the loaded scenarios object

        **Example**::

            scens = cv.Scenarios.load('my-scenarios.scens')
        """
        scens = cvm.load(scenfile, *args, **kwargs)
        if not isinstance(scens, Scenarios):
            errormsg = f'Cannot load object of {type(scens)} as a Scenarios object'
            raise TypeError(errormsg)
        return scens

    def disp(self, output=False):
        """
        Display a verbose description of the scenarios. See also scenarios.summarize()
        (medium length output) and scenarios.brief() (short output).

        Args:
            output (bool): if true, return a string instead of printing output
        """
        string = sc.prepr(self)
        if not output:
            print(string)
        else:
            return string

    def summarize(self, output=False):
        """
        Print a moderate length summary of the scenarios. See also scenarios.disp()
        (detailed output) and scenarios.brief() (short output).

        Args:
            output (bool): if true, return a string instead of printing output
        """
        labelstr = f' "{self.label}"' if self.label else ''
        string  = f'Scenarios{labelstr} summary:\n'
        string += f'  Number of scenarios: {len(self.sims)}\n'
        string += f'  Base: {self.base_sim.brief(output=True)}\n'
        if self.sims:
            string +=  '  Scenarios:\n'
            for k,key,simlist in self.sims.enumitems():
                string += f'      {k}: "{key}"\n'
                for s,sim in enumerate(simlist):
                    string += '          ' + f'{s}: {sim.brief(output=True)}\n'
        if not output:
            print(string)
        else:
            return string

    def __repr__(self):
        """ A brief description of the scenarios; see also brief() """
        try:
            labelstr = f'"{self.label}"; ' if self.label else ''
            n_scenarios = 0 if not self.scenarios else len(self.scenarios)
            string = f'Scenarios({labelstr}n_scenarios: {n_scenarios}; base: {self.base_sim.brief(output=True)})'
        except Exception as E:
            string = sc.objectid(self)
            string += f'Warning, scenarios appear to be malformed:\n{str(E)}'
        return string

    def brief(self, output=False):
        """
        Print a compact representation of the scenarios. See also scenarios.disp()
        (detailed output) and scenarios.summarize() (medium length output).

        Args:
            output (bool): if true, return a string instead of printing output
        """
        string = repr(self)
        if not output:
            print(string)
        else:
            return string


#%% Running functions

def keep_people(kwargs):
    """ The options for running sims, with v3's keep_people as Starsim's shrink (its opposite) """
    kwargs = dict(kwargs)
    if 'keep_people' in kwargs:
        keep = kwargs.pop('keep_people')
        if keep is not None:
            kwargs['shrink'] = not keep
    return kwargs


def single_run(sim, ind=0, reseed=True, noise=0.0, noisepar=None, keep_people=False, run_args=None, sim_args=None, verbose=None, do_run=True, **kwargs):
    """
    Convenience function to perform a single simulation run. Mostly used for
    parallelization, but can also be used directly.

    Args:
        sim         (Sim)   : the sim instance to be run
        ind         (int)   : the index of this sim
        reseed      (bool)  : whether or not to generate a fresh seed for each run
        noise       (float) : the amount of noise to add to each run
        noisepar    (str)   : the name of the parameter to add noise to
        keep_people (bool)  : whether to keep the people after the sim run
        run_args    (dict)  : arguments passed to sim.run()
        sim_args    (dict)  : extra parameters to pass to the sim, e.g. 'beta'
        verbose     (int)   : detail to print
        do_run      (bool)  : whether to actually run the sim (if not, just initialize it)
        kwargs      (dict)  : also passed to the sim

    Returns:
        sim (Sim): a single sim object with results

    **Example**::

        import covasim as cv
        sim = cv.Sim() # Create a default simulation
        sim = cv.single_run(sim) # Run it, equivalent(ish) to sim.run()
        sim = cv.single_run(cv.Sim(), beta=0.02) # Change a parameter, then run
    """
    seed = sim['rand_seed'] + (ind if reseed else 0) # As in v3
    sim = MultiSim(sim).make_run(sim, sc.mergedicts(sim_args, kwargs), seed, noise=noise, noisepar=noisepar)
    if not sim.label:
        sim.label = f'Sim {ind}'
    if not do_run:
        return sim.init()
    return ss.run.single_run(sim, ind=ind, shrink=not keep_people, run_args=run_args, verbose=verbose)


def multi_run(sim, n_runs=4, reseed=None, noise=0.0, noisepar=None, iterpars=None, combine=False, keep_people=None,
              run_args=None, sim_args=None, par_args=None, do_run=True, parallel=True, n_cpus=None, verbose=None, retry=None, **kwargs):
    """
    For running multiple runs in parallel, via ``cv.MultiSim``. If the first argument is a list of sims,
    exactly these will be run and most other arguments will be ignored.

    Args:
        sim         (Sim)   : the sim instance to be run, or a list of sims.
        n_runs      (int)   : the number of parallel runs
        reseed      (bool)  : not used (each run has its own seed, unless it is one of a list of sims); accepted for compatibility
        noise       (float) : the amount of noise to add to each run
        noisepar    (str)   : the name of the parameter to add noise to
        iterpars    (dict)  : any other parameters to iterate over the runs, e.g. dict(beta=[0.01, 0.02]); one run for each value
        combine     (bool)  : whether or not to combine all results into one sim, rather than return multiple sim objects
        keep_people (bool)  : whether to keep the people after the sim run (default false)
        run_args    (dict)  : arguments passed to sim.run()
        sim_args    (dict)  : extra parameters to pass to the sim
        par_args    (dict)  : arguments passed to sc.parallelize()
        do_run      (bool)  : whether to actually run the sim (if not, just initialize it)
        parallel    (bool)  : whether to run in parallel using multiprocessing (else, just run in a loop)
        n_cpus      (int)   : the number of CPUs to run on (if blank, set automatically; otherwise, passed to par_args)
        verbose     (int)   : detail to print
        retry       (str)   : not used (in v3, how to handle failed runs); accepted for compatibility
        kwargs      (dict)  : also passed to the sim

    Returns:
        If combine is True, a single sim object with the combined results from each sim.
        Otherwise, a list of sim objects (default).

    **Example**::

        import covasim as cv
        sim = cv.Sim()
        sims = cv.multi_run(sim, n_runs=6, noise=0.2)
        sims = cv.multi_run(sim, iterpars={'beta':[0.01, 0.02, 0.03]}) # Run 3 sims with different betas
    """
    sim_args = sc.mergedicts(sim_args, kwargs)
    is_list = isinstance(sim, list)
    pars = None
    if iterpars: # One run for each value of iterpars, as in v3
        pars = [sc.mergedicts(sim_args, p) for p in ss.run.par_sets(iterpars)]
        n_runs = None
    elif sim_args:
        pars = [sim_args]
    opts = dict(noise=noise, noisepar=noisepar, keep_people=keep_people, run_args=run_args, par_args=par_args, parallel=parallel, n_cpus=n_cpus, verbose=verbose)
    msim = MultiSim(sim, n_runs=None if is_list else n_runs, pars=pars, **opts)
    if not do_run:
        msim.init_sims()
        return list(msim.sims)
    msim.run()
    if combine:
        return msim.combine(output=True)
    return list(msim.sims)


def parallel(*args, **kwargs):
    """
    A shortcut to ``cv.MultiSim()``, allowing the quick running of multiple simulations
    at once.

    Args:
        args (list): The simulations to run
        kwargs (dict): passed to MultiSim.run()

    Returns:
        A run MultiSim object.

    **Examples**::

        s1 = cv.Sim(beta=0.01, label='Low')
        s2 = cv.Sim(beta=0.02, label='High')
        cv.parallel(s1, s2).plot()
        msim = cv.parallel([s1, s2], keep_people=True)
    """
    sims = sc.mergelists(*args)
    return MultiSim(sims=sims).run(**kwargs)
