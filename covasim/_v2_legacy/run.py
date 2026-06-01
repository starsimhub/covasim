'''
Functions and classes for running multiple Covasim runs.
'''

#%% Imports
import numpy as np
import pandas as pd
import sciris as sc
import pickle as pkl
from . import defaults as cvd
from . import misc as cvm
from . import base as cvb
from . import sim as cvs
from . import plotting as cvpl
from .settings import options as cvo


# Specify all externally visible functions this file defines
__all__ = ['make_metapars', 'MultiSim', 'Scenarios', 'single_run', 'multi_run', 'parallel']



def make_metapars():
    ''' Create default metaparameters for a Scenarios run '''
    metapars = sc.objdict(
        n_runs    = 3, # Number of parallel runs; change to 3 for quick, 11 for real
        noise     = 0.0, # Use noise, optionally
        noisepar  = 'beta',
        rand_seed = 1,
        quantiles = {'low':0.1, 'high':0.9},
        verbose   = cvo.verbose,
    )
    return metapars


class MultiSim(cvb.FlexPretty):
    '''
    Class for running multiple copies of a simulation. The parameter n_runs
    controls how many copies of the simulation there will be, if a list of sims
    is not provided. This is the main class that's used to run multiple versions
    of a simulation (e.g., with different random seeds).

    Args:
        sims      (Sim/list) : a single sim or a list of sims
        base_sim  (Sim)      : the sim used for shared properties; if not supplied, the first of the sims provided
        label      (str)     : the name of the multisim
        initialize (bool)    : whether or not to initialize the sims (otherwise, initialize them during run)
        kwargs    (dict)     : stored in run_args and passed to run()

    Returns:
        msim: a MultiSim object

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
    '''

    def __init__(self, sims=None, base_sim=None, label=None, initialize=False, **kwargs):

        # Handle inputs
        if base_sim is None:
            if isinstance(sims, cvs.Sim):
                base_sim = sims
                sims = None
            elif isinstance(sims, list):
                base_sim = sims[0]
            else:
                errormsg = f'If base_sim is not supplied, sims must be either a single sim (treated as base_sim) or a list of sims, not {type(sims)}'
                raise TypeError(errormsg)

        # Set properties
        self.sims      = sims
        self.base_sim  = base_sim
        self.label     = base_sim.label if (label is None and base_sim is not None) else label
        self.run_args  = sc.mergedicts(kwargs)
        self.results   = None
        self.which     = None # Whether the multisim is to be reduced, combined, etc.
        cvb.set_metadata(self) # Set version, date, and git info

        # Optionally initialize
        if initialize:
            self.init_sims()

        return


    def __len__(self):
        try:
            return len(self.sims)
        except:
            return 0


    def result_keys(self):
        ''' Attempt to retrieve the results keys from the base sim '''
        try:
            keys = self.base_sim.result_keys()
        except Exception as E:
            errormsg = f'Could not retrieve result keys since base sim not accessible: {str(E)}'
            raise ValueError(errormsg)
        return keys


    def init_sims(self, **kwargs):
        '''
        Initialize the sims, but don't actually run them. Syntax is the same
        as MultiSim.run(). Note: in most cases you can just call run() directly,
        there is no need to call this separately.

        Args:
            kwargs  (dict): passed to multi_run()
        '''

        # Handle which sims to use
        if self.sims is None:
            sims = self.base_sim
        else:
            sims = self.sims

        # Initialize the sims but don't run them
        kwargs = sc.mergedicts(self.run_args, kwargs, {'do_run':False}) # Never run, that's the point!
        self.sims = multi_run(sims, **kwargs)

        return


    def run(self, reduce=False, combine=False, **kwargs):
        '''
        Run the actual sims

        Args:
            reduce  (bool): whether or not to reduce after running (see reduce())
            combine (bool): whether or not to combine after running (see combine(), not compatible with reduce)
            kwargs  (dict): passed to multi_run(); use run_args to pass arguments to sim.run()

        Returns:
            None (modifies MultiSim object in place)

        **Examples**::

            msim.run()
            msim.run(run_args=dict(until='2020-0601', restore_pars=False))
        '''
        # Handle which sims to use -- same as init_sims()
        if self.sims is None:
            sims = self.base_sim
        else:
            sims = self.sims

            # Handle missing labels
            for s,sim in enumerate(sims):
                if sim.label is None:
                    sim.label = f'Sim {s}'

        # Run
        kwargs = sc.mergedicts(self.run_args, kwargs)
        self.sims = multi_run(sims, **kwargs)

        # Reduce or combine
        if reduce:
            self.reduce()
        elif combine:
            self.combine()

        return self


    def _has_orig_sim(self):
        ''' Helper method for determining if an original base sim is present '''
        return hasattr(self, 'orig_base_sim')


    def _rm_orig_sim(self, reset=False):
        ''' Helper method for removing the original base sim, if present '''
        if self._has_orig_sim():
            if reset:
                self.base_sim = self.orig_base_sim
            delattr(self, 'orig_base_sim')
        return


    def shrink(self, **kwargs):
        '''
        Not to be confused with reduce(), this shrinks each sim in the msim;
        see sim.shrink() for more information.

        Args:
            kwargs (dict): passed to sim.shrink() for each sim
        '''
        self.base_sim.shrink(**kwargs)
        self._rm_orig_sim()
        for sim in self.sims:
            sim.shrink(**kwargs)
        return


    def reset(self):
        ''' Undo a combine() or reduce() by resetting the base sim, which, and results '''
        self._rm_orig_sim(reset=True)
        self.which = None
        self.results = None
        return


    def reduce(self, quantiles=None, use_mean=False, bounds=None, output=False):
        '''
        Combine multiple sims into a single sim statistically: by default, use
        the median value and the 10th and 90th percentiles for the lower and upper
        bounds. If use_mean=True, then use the mean and ±2 standard deviations
        for lower and upper bounds.

        Args:
            quantiles (dict): the quantiles to use, e.g. [0.1, 0.9] or {'low : '0.1, 'high' : 0.9}
            use_mean (bool): whether to use the mean instead of the median
            bounds (float): if use_mean=True, the multiplier on the standard deviation for upper and lower bounds (default 2)
            output (bool): whether to return the "reduced" sim (in any case, modify the multisim in-place)

        **Example**::

            msim = cv.MultiSim(cv.Sim())
            msim.run()
            msim.reduce()
            msim.summarize()
        '''

        if use_mean:
            if bounds is None:
                bounds = 2
        else:
            if quantiles is None:
                quantiles = make_metapars()['quantiles']
            if not isinstance(quantiles, dict):
                try:
                    quantiles = {'low':float(quantiles[0]), 'high':float(quantiles[1])}
                except Exception as E:
                    errormsg = f'Could not figure out how to convert {quantiles} into a quantiles object: must be a dict with keys low, high or a 2-element array ({str(E)})'
                    raise ValueError(errormsg)

        # Store information on the sims
        n_runs = len(self)
        reduced_sim = sc.dcp(self.sims[0])
        reduced_sim.metadata = dict(parallelized=True, combined=False, n_runs=n_runs, quantiles=quantiles, use_mean=use_mean, bounds=bounds) # Store how this was parallelized

        # Perform the statistics
        raw = {}
        mainkeys = reduced_sim.result_keys('main')
        variantkeys = reduced_sim.result_keys('variant')
        for reskey in mainkeys:
            raw[reskey] = np.zeros((reduced_sim.npts, len(self.sims)))
            for s,sim in enumerate(self.sims):
                vals = sim.results[reskey].values
                raw[reskey][:, s] = vals
        for reskey in variantkeys:
            raw[reskey] = np.zeros((reduced_sim['n_variants'], reduced_sim.npts, len(self.sims)))
            for s,sim in enumerate(self.sims):
                vals = sim.results['variant'][reskey].values
                raw[reskey][:, :, s] = vals

        for reskey in mainkeys + variantkeys:
            if reskey in mainkeys:
                axis = 1
                results = reduced_sim.results
            else:
                axis = 2
                results = reduced_sim.results['variant']
            if use_mean:
                r_mean = np.mean(raw[reskey], axis=axis)
                r_std = np.std(raw[reskey], axis=axis)
                results[reskey].values[:] = r_mean
                results[reskey].low = r_mean - bounds * r_std
                results[reskey].high = r_mean + bounds * r_std
            else:
                results[reskey].values[:] = np.quantile(raw[reskey], q=0.5, axis=axis)
                results[reskey].low = np.quantile(raw[reskey], q=quantiles['low'], axis=axis)
                results[reskey].high = np.quantile(raw[reskey], q=quantiles['high'], axis=axis)

        # Compute and store final results
        reduced_sim.compute_summary()
        self.orig_base_sim = self.base_sim
        self.base_sim = reduced_sim
        self.results = reduced_sim.results
        self.summary = reduced_sim.summary
        self.which = 'reduced'

        if output:
            return self.base_sim
        else:
            return


    def mean(self, bounds=None, **kwargs):
        '''
        Alias for reduce(use_mean=True). See reduce() for full description.

        Args:
            bounds (float): multiplier on the standard deviation for the upper and lower bounds (default, 2)
            kwargs (dict): passed to reduce()
        '''
        return self.reduce(use_mean=True, bounds=bounds, **kwargs)


    def median(self, quantiles=None, **kwargs):
        '''
        Alias for reduce(use_mean=False). See reduce() for full description.

        Args:
            quantiles (list or dict): upper and lower quantiles (default, 0.1 and 0.9)
            kwargs (dict): passed to reduce()
        '''
        return self.reduce(use_mean=False, quantiles=quantiles, **kwargs)


    def combine(self, output=False):
        '''
        Combine multiple sims into a single sim with scaled results.

        **Example**::

            msim = cv.MultiSim(cv.Sim())
            msim.run()
            msim.combine()
            msim.summarize()
        '''

        n_runs = len(self)
        combined_sim = sc.dcp(self.sims[0])
        combined_sim.parallelized = dict(parallelized=True, combined=True, n_runs=n_runs)  # Store how this was parallelized

        for s,sim in enumerate(self.sims[1:]): # Skip the first one
            if combined_sim.people: # If the people are there, add them and increment the population size accordingly
                combined_sim.people += sim.people
                combined_sim['pop_size'] = combined_sim.people.pars['pop_size']
            else: # If not, manually update population size
                combined_sim['pop_size'] += sim['pop_size']  # Record the number of people
            for key in sim.result_keys():
                vals = sim.results[key].values
                if len(vals) != combined_sim.npts:
                    errormsg = f'Cannot combine sims with inconsistent numbers of days: {combined_sim.npts} vs. {len(vals)}'
                    raise ValueError(errormsg)
                combined_sim.results[key].values += vals

        # For non-count results (scale=False), rescale them
        for key in combined_sim.result_keys():
            if not combined_sim.results[key].scale:
                combined_sim.results[key].values /= n_runs

        # Compute and store final results
        combined_sim.compute_summary()
        self.orig_base_sim = self.base_sim
        self.base_sim = combined_sim
        self.results = combined_sim.results
        self.summary = combined_sim.summary

        self.which = 'combined'

        if output:
            return self.base_sim
        else:
            return


    def compare(self, t=None, sim_inds=None, output=False, do_plot=False, **kwargs):
        '''
        Create a dataframe compare sims at a single point in time.

        Args:
            t        (int/str) : the day (or date) to do the comparison; default, the end
            sim_inds (list)    : list of integers of which sims to include (default: all)
            output   (bool)    : whether or not to return the comparison as a dataframe
            do_plot  (bool)    : whether or not to plot the comparison (see also plot_compare())
            kwargs   (dict)    : passed to plot_compare()

        Returns:
            df (dataframe): a dataframe comparison
        '''

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
                label = f'Sim {i}'
            if label in resdict: # Avoid duplicates
                label += f' ({i})'
            for reskey in sim.result_keys():
                res = sim.results[reskey]
                val = res.values[day]
                if res.scale: # Results that are scaled by population are ints
                    val = int(val)
                resdict[label][reskey] = val

        if do_plot:
            self.plot_compare(**kwargs)

        df = pd.DataFrame.from_dict(resdict).astype(object) # astype is necessary to prevent type coercion
        if not output:
            print(f'Results for {daystr} in each sim:')
            print(df)
        else:
            return df


    def plot(self, to_plot=None, inds=None, plot_sims=False, color_by_sim=None, max_sims=5, colors=None, labels=None, alpha_range=None, plot_args=None, show_args=None, **kwargs):
        '''
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
        '''

        # Plot a single curve, possibly with a range
        if not plot_sims and self.which in ['combined', 'reduced']:
            fig = self.base_sim.plot(to_plot=to_plot, colors=colors, **kwargs)

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

            # Handle what to plot
            if to_plot is None:
                kind = 'scens' if color_by_sim else 'sim'
                to_plot = cvd.get_default_plots(kind=kind)

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
                fig = sim.plot(fig=fig, to_plot=('scens', to_plot), colors=colors[s], labels=merged_labels, plot_args=merged_plot_args, show_args=merged_show_args, **kwargs)

        return cvpl.handle_show_return(fig=fig)


    def plot_result(self, key, colors=None, labels=None, *args, **kwargs):
        ''' Convenience method for plotting -- arguments passed to sim.plot_result() '''
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
        return cvpl.handle_show_return(fig=fig)


    def plot_compare(self, t=-1, sim_inds=None, log_scale=True, **kwargs):
        '''
        Plot a comparison between sims, using bars to show different values for
        each result. For an explanation of other available arguments, see Sim.plot().

        Args:
            t         (int)  : index of results, passed to compare()
            sim_inds  (list) : which sims to include, passed to compare()
            log_scale (bool) : whether to plot with a logarithmic x-axis
            kwargs    (dict) : standard plotting arguments, see Sim.plot() for explanation

        Returns:
            fig: Figure handle
        '''
        df = self.compare(t=t, sim_inds=sim_inds, output=True)
        return cvpl.plot_compare(df, log_scale=log_scale, **kwargs)


    def save(self, filename=None, keep_people=False, **kwargs):
        '''
        Save to disk as a gzipped pickle. Load with cv.load(filename) or
        cv.MultiSim.load(filename).

        Args:
            filename    (str)  : the name or path of the file to save to; if None, uses default
            keep_people (bool) : whether or not to store the population in the Sim objects (NB, very large)
            kwargs      (dict) : passed to ``sc.makefilepath()``

        Returns:
            scenfile (str): the validated absolute path to the saved file

        **Example**::

            msim.save() # Saves to an .msim file
        '''
        if filename is None:
            filename = 'covasim.msim'
        msimfile = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename # Store the actual saved filename

        # Store sims separately
        sims = self.sims
        self.sims = None # Remove for now

        obj = sc.dcp(self) # This should be quick once we've removed the sims
        if keep_people:
            obj.sims = sims # Just restore the object in full
            print('Note: saving people, which may produce a large file!')
        else:
            obj.base_sim.shrink(in_place=True)
            obj.sims = []
            for sim in sims:
                obj.sims.append(sim.shrink(in_place=False))

        cvm.save(filename=msimfile, obj=obj) # Actually save

        self.sims = sims # Restore
        return msimfile


    @staticmethod
    def load(msimfile, *args, **kwargs):
        '''
        Load from disk from a gzipped pickle.

        Args:
            msimfile (str): the name or path of the file to load from
            kwargs: passed to cv.load()

        Returns:
            msim (MultiSim): the loaded MultiSim object

        **Example**::

            msim = cv.MultiSim.load('my-multisim.msim')
        '''
        msim = cvm.load(msimfile, *args, **kwargs)
        if not isinstance(msim, MultiSim):
            errormsg = f'Cannot load object of {type(msim)} as a MultiSim object'
            raise TypeError(errormsg)
        return msim


    @staticmethod
    def merge(*args, base=False):
        '''
        Convenience method for merging two MultiSim objects.

        Args:
            args (MultiSim): the MultiSims to merge (either a list, or separate)
            base (bool): if True, make a new list of sims from the multisim's two base sims; otherwise, merge the multisim's lists of sims

        Returns:
            msim (MultiSim): a new MultiSim object

        **Examples**:

            mm1 = cv.MultiSim.merge(msim1, msim2, base=True)
            mm2 = cv.MultiSim.merge([m1, m2, m3, m4], base=False)
        '''

        # Handle arguments
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0] # A single list of MultiSims has been provided

        # Create the multisim from the base sim of the first argument
        msim = MultiSim(base_sim=sc.dcp(args[0].base_sim), label=args[0].label)
        msim.sims = []
        msim.chunks = [] # This is used to enable automatic splitting later

        # Handle different options for combining
        if base: # Only keep the base sims
            for i,ms in enumerate(args):
                sim = sc.dcp(ms.base_sim)
                sim.label = ms.label
                msim.sims.append(sim)
                msim.chunks.append([[i]])
        else: # Keep all the sims
            for ms in args:
                len_before = len(msim.sims)
                msim.sims += sc.dcp(ms.sims)
                len_after= len(msim.sims)
                msim.chunks.append(list(range(len_before, len_after)))

        return msim


    def split(self, inds=None, chunks=None):
        '''
        Convenience method for splitting one MultiSim into several. You can specify
        either individual indices of simulations to extract, via inds, or consecutive
        chunks of indices, via chunks. If this function is called on a merged MultiSim,
        the chunks can be retrieved automatically and no arguments are necessary.

        Args:
            inds (list): a list of lists of indices, with each list turned into a MultiSim
            chunks (int or list): if an int, split the MultiSim into that many chunks; if a list return chunks of that many sims

        Returns:
            A list of MultiSim objects

        **Examples**::

            m1 = cv.MultiSim(cv.Sim(label='sim1'), initialize=True)
            m2 = cv.MultiSim(cv.Sim(label='sim2'), initialize=True)
            m3 = cv.MultiSim.merge(m1, m2)
            m3.run()
            m1b, m2b = m3.split()

            msim = cv.MultiSim(cv.Sim(), n_runs=6)
            msim.run()
            m1, m2 = msim.split(inds=[[0,2,4], [1,3,5]])
            mlist1 = msim.split(chunks=[2,4]) # Equivalent to inds=[[0,1], [2,3,4,5]]
            mlist2 = msim.split(chunks=2) # Equivalent to inds=[[0,1,2], [3,4,5]]
        '''

        # Process indices and chunks
        if inds is None: # Indices not supplied
            if chunks is None: # Chunks not supplied
                if hasattr(self, 'chunks'): # Created from a merged MultiSim
                    inds = self.chunks
                else: # No indices or chunks and not created from a merge
                    errormsg = 'If a MultiSim has not been created via merge(), you must supply either inds or chunks to split it'
                    raise ValueError(errormsg)
            else: # Chunks supplied, but not inds
                inds = [] # Initialize
                sim_inds = np.arange(len(self)) # Indices for the simulations
                if sc.isiterable(chunks): # e.g. chunks = [2,4]
                    chunk_inds = np.cumsum(chunks)[:-1]
                    inds = np.split(sim_inds, chunk_inds)
                else: # e.g. chunks = 3
                    inds = np.split(sim_inds, chunks) # This will fail if the length is wrong

        # Do the conversion
        mlist = []
        for indlist in inds:
            sims = sc.dcp([self.sims[i] for i in indlist])
            msim = MultiSim(sims=sims)
            mlist.append(msim)

        return mlist


    def disp(self, output=False):
        '''
        Display a verbose description of a multisim. See also multisim.summarize()
        (medium length output) and multisim.brief() (short output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            msim = cv.MultiSim(cv.Sim(verbose=0), label='Example multisim')
            msim.run()
            msim.disp() # Displays detailed output
        '''
        string = self._disp()
        if not output:
            print(string)
        else:
            return string


    def summarize(self, output=False):
        '''
        Print a moderate length summary of the MultiSim. See also multisim.disp()
        (detailed output) and multisim.brief() (short output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            msim = cv.MultiSim(cv.Sim(verbose=0), label='Example multisim')
            msim.run()
            msim.summarize() # Prints moderate length output
        '''
        labelstr = f' "{self.label}"' if self.label else ''
        simlenstr = f'{len(self.sims)}' if self.sims else '0'
        string  = f'MultiSim{labelstr} summary:\n'
        string += f'  Number of sims: {simlenstr}\n'
        string += f'  Reduced/combined: {self.which}\n'
        string += f'  Base: {self.base_sim.brief(output=True)}\n'
        if self.sims:
            string += '  Sims:\n'
            for s,sim in enumerate(self.sims):
                string += f'    {s}: {sim.brief(output=True)}\n'
        if not output:
            print(string)
        else:
            return string


    def _brief(self):
        '''
        Return a brief description of a multisim -- used internally and by repr();
        see multisim.brief() for the user version.
        '''
        try:
            labelstr = f'"{self.label}"; ' if self.label else ''
            n_sims = 0 if not self.sims else len(self.sims)
            string   = f'MultiSim({labelstr}n_sims: {n_sims}; base: {self.base_sim.brief(output=True)})'
        except Exception as E:
            string = sc.objectid(self)
            string += f'Warning, multisim appears to be malformed:\n{str(E)}'
        return string


    def brief(self, output=False):
        '''
        Print a compact representation of the multisim. See also multisim.disp()
        (detailed output) and multisim.summarize() (medium length output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            msim = cv.MultiSim(cv.Sim(verbose=0), label='Example multisim')
            msim.run()
            msim.brief() # Prints one-line output
         '''
        string = self._brief()
        if not output:
            print(string)
        else:
            return string


    def to_json(self, *args, **kwargs):
        ''' Shortcut for base_sim.to_json() '''
        if not self.base_sim.results_ready: # pragma: no cover
            errormsg = 'JSON export only available for reduced sim; please run msim.mean() or msim.median() first'
            raise RuntimeError(errormsg)
        return self.base_sim.to_json(*args, **kwargs)


    def to_excel(self, *args, **kwargs):
        ''' Shortcut for base_sim.to_excel() '''
        if not self.base_sim.results_ready: # pragma: no cover
            errormsg = 'Excel export only available for reduced sim; please run msim.mean() or msim.median() first'
            raise RuntimeError(errormsg)
        return self.base_sim.to_excel(*args, **kwargs)


class Scenarios(cvb.ParsObj):
    '''
    Class for running multiple sets of multiple simulations -- e.g., scenarios.
    Note that most users are recommended to use MultiSim rather than Scenarios,
    as it gives more control over run options. Scenarios should be used primarily
    for quick investigations. See the examples folder for example usage.

    Args:
        sim       (Sim)  : if supplied, use a pre-created simulation as the basis for the scenarios
        metapars  (dict) : meta-parameters for the run, e.g. number of runs; see make_metapars() for structure
        scenarios (dict) : a dictionary defining the scenarios; see examples folder for examples; see below for baseline
        basepars  (dict) : a dictionary of sim parameters to be used for the basis of the scenarios (not required if sim is provided)
        scenfile  (str)  : a filename for saving (defaults to the creation date)
        label     (str)  : the name of the scenarios

    **Example**::

        scens = cv.Scenarios()

    Returns:
        scens: a Scenarios object
    '''

    def __init__(self, sim=None, metapars=None, scenarios=None, basepars=None, scenfile=None, label=None):

        # For this object, metapars are the foundation
        default_pars = make_metapars() # Start with default pars
        super().__init__(default_pars) # Initialize and set the parameters as attributes
        cvb.set_metadata(self) # Set version, date, and git info

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
        self.base_sim = sc.dcp(sim)
        self.basepars = sc.dcp(sc.mergedicts(basepars))
        self.base_sim.update_pars(self.basepars)
        self.base_sim.validate_pars()
        if not self.base_sim.initialized:
            self.base_sim.init_variants()
            self.base_sim.init_immunity()
            self.base_sim.init_results()

        # Copy quantities from the base sim to the main object
        self.npts       = self.base_sim.npts
        self.tvec       = self.base_sim.tvec
        self.datevec    = self.base_sim.datevec
        self['verbose'] = self.base_sim['verbose']

        # Create the results object; order is: results key, scenario, best/low/high
        self.sims = sc.objdict()
        self.results = sc.objdict()
        for reskey in self.result_keys():
            self.results[reskey] = sc.objdict()
            for scenkey in scenarios.keys():
                self.results[reskey][scenkey] = sc.objdict()
                for nblh in ['name', 'best', 'low', 'high']:
                    self.results[reskey][scenkey][nblh] = None # This will get populated below
        return


    def result_keys(self, which='all'):
        ''' Attempt to retrieve the results keys from the base sim '''
        try:
            keys = self.base_sim.result_keys(which=which)
        except Exception as E:
            errormsg = f'Could not retrieve result keys since base sim not accessible: {str(E)}'
            raise ValueError(errormsg)
        return keys


    def run(self, debug=False, keep_people=False, verbose=None, **kwargs):
        '''
        Run the specified scenarios.

        Args:
            debug   (bool) : if True, runs a single run instead of multiple, which makes debugging easier
            verbose (int)  : level of detail to print, passed to sim.run()
            kwargs  (dict) : passed to multi_run() and thence to sim.run()

        Returns:
            None (modifies Scenarios object in place)
        '''

        if verbose is None:
            verbose = self['verbose']

        def print_heading(string):
            ''' Choose whether to print a heading, regular text, or nothing '''
            if verbose >= 2:
                sc.heading(string)
            elif verbose == 1:
                print(string)
            return

        mainkeys   = self.result_keys('main')
        variantkeys = self.result_keys('variant')

        # Loop over scenarios
        for scenkey,scen in self.scenarios.items():
            scenname = scen['name']
            scenpars = scen['pars']

            # This is necessary for plotting, and since self.npts is defined prior to run
            if 'n_days' in scenpars.keys():
                errormsg = 'Scenarios cannot be run with different numbers of days; set via basepars instead'
                raise ValueError(errormsg)

            # Create and run the simulations
            print_heading(f'Multirun for {scenkey}')
            scen_sim = sc.dcp(self.base_sim)
            scen_sim.scenkey = scenkey
            scen_sim.label = scenname
            scen_sim.scen = scen

            # Update the parameters, if provided, and re-initialize aspects of the simulation
            scen_sim.update_pars(scenpars)
            scen_sim.initialized = False # Ensure it gets re-initialized

            run_args = dict(n_runs=self['n_runs'], noise=self['noise'], noisepar=self['noisepar'], keep_people=keep_people, verbose=verbose)
            if debug:
                print('Running in debug mode (not parallelized)')
                run_args.pop('n_runs', None) # Remove n_runs argument, not used for a single run
                scen_sims = [single_run(scen_sim, **run_args, **kwargs)]
            else:
                scen_sims = multi_run(scen_sim, **run_args, **kwargs) # This is where the sims actually get run

            # Process the simulations
            print_heading(f'Processing {scenkey}')
            ns = scen_sims[0]['n_variants'] # Get number of variants
            scenraw = {}
            for reskey in mainkeys:
                scenraw[reskey] = np.zeros((self.npts, len(scen_sims)))
                for s,sim in enumerate(scen_sims):
                    scenraw[reskey][:,s] = sim.results[reskey].values
            for reskey in variantkeys:
                scenraw[reskey] = np.zeros((ns, self.npts, len(scen_sims)))
                for s,sim in enumerate(scen_sims):
                    scenraw[reskey][:,:,s] = sim.results['variant'][reskey].values

            scenres = sc.objdict()
            scenres.best = {}
            scenres.low = {}
            scenres.high = {}
            for reskey in mainkeys + variantkeys:
                axis = 1 if reskey in mainkeys else 2
                scenres.best[reskey] = np.quantile(scenraw[reskey], q=0.5, axis=axis) # Changed from median to mean for smoother plots
                scenres.low[reskey]  = np.quantile(scenraw[reskey], q=self['quantiles']['low'], axis=axis)
                scenres.high[reskey] = np.quantile(scenraw[reskey], q=self['quantiles']['high'], axis=axis)

            for reskey in mainkeys + variantkeys:
                self.results[reskey][scenkey]['name'] = scenname
                for blh in ['best', 'low', 'high']:
                    self.results[reskey][scenkey][blh] = scenres[blh][reskey]

            self.sims[scenkey] = scen_sims

        #%% Print statistics
        if verbose:
            self.compare()

        # Save details about the run
        self._kept_people = keep_people

        return self


    def compare(self, t=None, output=False):
        '''
        Print out a comparison of each scenario.

        Args:
            t (int/str)   : the day (or date) to do the comparison; default, the end
            output (bool) : if true, return the dataframe instead of printing output

        **Example**::

            scenarios = {'base': {'name':'Base','pars': {}}, 'beta': {'name':'Beta', 'pars': {'beta': 0.020}}}
            scens = cv.Scenarios(scenarios=scenarios, label='Example scenarios')
            scens.run()
            scens.compare(t=30) # Prints comparison for day 30
        '''

        # Handle time
        if t is None:
            t = -1
            daystr = 'the last day'
        else:
            daystr = f'day {t}'
        day = self.base_sim.day(t) # Unlike MultiSims, scenarios must have the same start day

        # Compute dataframe
        x = sc.ddict(dict)
        variantkeys = self.result_keys('variant')
        for scenkey in self.scenarios.keys():
            for reskey in self.result_keys():
                if reskey in variantkeys:
                    for variant in range(self.base_sim['n_variants']):
                        val = self.results[reskey][scenkey].best[variant, day] # Only prints results for infections by first variant
                        variantkey = reskey + str(variant) # Add variant number to the summary output
                        x[scenkey][variantkey] = int(val)
                else:
                    val = self.results[reskey][scenkey].best[day]
                    if reskey not in ['r_eff', 'doubling_time']:
                        val = int(val)
                    x[scenkey][reskey] = val
        df = pd.DataFrame.from_dict(x).astype(object)

        if not output:
            print(f'Results for {daystr} in each scenario:')
            print(df)
        else:
            return df


    def plot(self, *args, **kwargs):
        '''
        Plot the results of a scenario. For an explanation of available arguments,
        see Sim.plot().

        Returns:
            fig: Figure handle


        **Example**::

            scens = cv.Scenarios()
            scens.run()
            scens.plot()
        '''
        return cvpl.plot_scens(scens=self, *args, **kwargs)


    def to_json(self, filename=None, tostring=True, indent=2, verbose=False, *args, **kwargs):
        '''
        Export results as JSON.

        Args:
            filename (str): if None, return string; else, write to file

        Returns:
            A unicode string containing a JSON representation of the results,
            or writes the JSON file to disk

        '''
        d = {'t':self.tvec,
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
        '''
        Export results as XLSX

        Args:
            filename (str): if None, return string; else, write to file

        Returns:
            An sc.Spreadsheet with an Excel file, or writes the file to disk

        '''
        spreadsheet = sc.Spreadsheet()
        spreadsheet.freshbytes()
        with pd.ExcelWriter(spreadsheet.bytes, engine='xlsxwriter') as writer:
            for key in self.result_keys('main'): # Multidimensional variant keys can't be exported
                result_df = pd.DataFrame.from_dict(sc.flattendict(self.results[key], sep='_'))
                result_df.to_excel(writer, sheet_name=key)
        spreadsheet.load()

        if filename is None:
            output = spreadsheet
        else:
            output = spreadsheet.save(filename)

        return output


    def save(self, scenfile=None, keep_sims=True, keep_people=False, **kwargs):
        '''
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

        '''
        if scenfile is None:
            scenfile = self.scenfile
        scenfile = sc.makefilepath(filename=scenfile, **kwargs)
        self.scenfile = scenfile # Store the actual saved filename

        # Store sims separately
        sims = self.sims
        self.sims = None # Remove for now

        obj = sc.dcp(self) # This should be quick once we've removed the sims
        if not keep_people:
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
                    obj.sims[key] = []
                    for sim in sims[key]:
                        obj.sims[key].append(sim.shrink(in_place=False))

        cvm.save(filename=scenfile, obj=obj) # Actually save

        self.sims = sims # Restore
        return scenfile


    @staticmethod
    def load(scenfile, *args, **kwargs):
        '''
        Load from disk from a gzipped pickle.

        Args:
            scenfile (str): the name or path of the file to load from
            kwargs: passed to cv.load()

        Returns:
            scens (Scenarios): the loaded scenarios object

        **Example**::

            scens = cv.Scenarios.load('my-scenarios.scens')
        '''
        scens = cvm.load(scenfile, *args, **kwargs)
        if not isinstance(scens, Scenarios):
            errormsg = f'Cannot load object of {type(scens)} as a Scenarios object'
            raise TypeError(errormsg)
        return scens


    def disp(self, output=False):
        '''
        Display a verbose description of the scenarios. See also scenarios.summarize()
        (medium length output) and scenarios.brief() (short output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            scens = cv.Scenarios(cv.Sim(), label='Example scenarios')
            scens.run(verbose=0) # Run silently
            scens.disp() # Displays detailed output
        '''
        string = self._disp()
        if not output:
            print(string)
        else:
            return string


    def summarize(self, output=False):
        '''
        Print a moderate length summary of the scenarios. See also scenarios.disp()
        (detailed output) and scenarios.brief() (short output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            scens = cv.Scenarios(cv.Sim(), label='Example scenarios')
            scens.run(verbose=0) # Run silently
            scens.summarize() # Prints moderate length output
        '''
        labelstr = f' "{self.label}"' if self.label else ''
        string  = f'Scenarios{labelstr} summary:\n'
        string += f'  Number of scenarios: {len(self.sims)}\n'
        string += f'  Base: {self.base_sim.brief(output=True)}\n'
        if self.sims:
            string +=  '  Scenarios:\n'
            for k,key,simlist in self.sims.enumitems():
                keystr = f'      {k}: "{key}"\n'
                string += keystr
                for s,sim in enumerate(simlist):
                    simstr = f'{sim.brief(output=True)}'
                    string += '          ' + f'{s}: {simstr}\n'
        if not output:
            print(string)
        else:
            return string


    def _brief(self):
        '''
        Return a brief description of the scenarios -- used internally and by repr();
        see scenarios.brief() for the user version.
        '''
        try:
            labelstr = f'"{self.label}"; ' if self.label else ''
            n_scenarios = 0 if not self.scenarios else len(self.scenarios)
            string   = f'Scenarios({labelstr}n_scenarios: {n_scenarios}; base: {self.base_sim.brief(output=True)})'
        except Exception as E:
            string = sc.objectid(self)
            string += f'Warning, scenarios appear to be malformed:\n{str(E)}'
        return string


    def brief(self, output=False):
        '''
        Print a compact representation of the scenarios. See also scenarios.disp()
        (detailed output) and scenarios.summarize() (medium length output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            scens = cv.Scenarios(label='Example scenarios')
            scens.run()
            scens.brief() # Prints one-line output
         '''
        string = self._brief()
        if not output:
            print(string)
        else:
            return string


def single_run(sim, ind=0, reseed=True, noise=0.0, noisepar=None, keep_people=False, run_args=None, sim_args=None, verbose=None, do_run=True, **kwargs):
    '''
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
        sim_args    (dict)  : extra parameters to pass to the sim, e.g. 'n_infected'
        verbose     (int)   : detail to print
        do_run      (bool)  : whether to actually run the sim (if not, just initialize it)
        kwargs      (dict)  : also passed to the sim

    Returns:
        sim (Sim): a single sim object with results

    **Example**::

        import covasim as cv
        sim = cv.Sim() # Create a default simulation
        sim = cv.single_run(sim) # Run it, equivalent(ish) to sim.run()
    '''

    # Set sim and run arguments
    sim_args = sc.mergedicts(sim_args, kwargs)
    run_args = sc.mergedicts({'verbose':verbose}, run_args)
    if verbose is None:
        verbose = sim['verbose']

    if not sim.label:
        sim.label = f'Sim {ind}'

    if reseed:
        sim['rand_seed'] += ind # Reset the seed, otherwise no point of parallel runs
        sim.set_seed()

    # If the noise parameter is not found, guess what it should be
    if noisepar is None:
        noisepar = 'beta'
        if noisepar not in sim.pars.keys():
            raise sc.KeyNotFoundError(f'Noise parameter {noisepar} was not found in sim parameters')

    # Handle noise -- normally distributed fractional error
    noiseval = noise*np.random.normal()
    if noiseval > 0:
        noisefactor = 1 + noiseval
    else:
        noisefactor = 1/(1-noiseval)
    sim[noisepar] *= noisefactor

    if verbose>=1:
        verb = 'Running' if do_run else 'Creating'
        print(f'{verb} a simulation using seed={sim["rand_seed"]} and noise={noiseval}')

    # Handle additional arguments
    for key,val in sim_args.items():
        print(f'Processing {key}:{val}')
        if key in sim.pars.keys():
            if verbose>=1:
                print(f'Setting key {key} from {sim[key]} to {val}')
                sim[key] = val
        else:
            raise sc.KeyNotFoundError(f'Could not set key {key}: not a valid parameter name')

    # Run
    if do_run:
        sim.run(**run_args)

    # Shrink the sim to save memory
    if not keep_people:
        sim.shrink()

    return sim


def multi_run(sim, n_runs=4, reseed=None, noise=0.0, noisepar=None, iterpars=None, 
              combine=False, keep_people=None, run_args=None, sim_args=None, par_args=None, 
              do_run=True, parallel=True, n_cpus=None, verbose=None, retry='warn', **kwargs):
    '''
    For running multiple runs in parallel. If the first argument is a list of sims,
    exactly these will be run and most other arguments will be ignored.

    Args:
        sim         (Sim)   : the sim instance to be run, or a list of sims.
        n_runs      (int)   : the number of parallel runs
        reseed      (bool)  : whether or not to generate a fresh seed for each run (default: true for single, false for list of sims)
        noise       (float) : the amount of noise to add to each run
        noisepar    (str)   : the name of the parameter to add noise to
        iterpars    (dict)  : any other parameters to iterate over the runs; see sc.parallelize() for syntax
        combine     (bool)  : whether or not to combine all results into one sim, rather than return multiple sim objects
        keep_people (bool)  : whether to keep the people after the sim run (default false)
        run_args    (dict)  : arguments passed to sim.run()
        sim_args    (dict)  : extra parameters to pass to the sim
        par_args    (dict)  : arguments passed to sc.parallelize()
        do_run      (bool)  : whether to actually run the sim (if not, just initialize it)
        parallel    (bool)  : whether to run in parallel using multiprocessing (else, just run in a loop)
        n_cpus      (int)   : the number of CPUs to run on (if blank, set automatically; otherwise, passed to par_args)
        verbose     (int)   : detail to print
        retry       (str)   : what to do if default parallelizer fails: choices are 'warn' (default), 'die' (raise exception), or 'silent' (keep going)
        kwargs      (dict)  : also passed to the sim

    Returns:
        If combine is True, a single sim object with the combined results from each sim.
        Otherwise, a list of sim objects (default).

    **Example**::

        import covasim as cv
        sim = cv.Sim()
        sims = cv.multi_run(sim, n_runs=6, noise=0.2)
    '''

    # Handle inputs
    sim_args = sc.mergedicts(sim_args, kwargs) # Handle blank
    par_args = sc.mergedicts({'ncpus':n_cpus, 'parallelizer':'robust'}, par_args) # Handle blank

    # Handle iterpars
    if iterpars is None:
        iterpars = {}
    else:
        n_runs = None # Reset and get from length of dict instead
        for key,val in iterpars.items():
            new_n = len(val)
            if n_runs is not None and new_n != n_runs:
                raise ValueError(f'Each entry in iterpars must have the same length, not {n_runs} and {len(val)}')
            else:
                n_runs = new_n

    # Run the sims
    if isinstance(sim, cvs.Sim): # One sim
        if reseed is None: reseed = True
        iterkwargs = dict(ind=np.arange(n_runs))
        iterkwargs.update(iterpars)
        kwargs = dict(sim=sim, reseed=reseed, noise=noise, noisepar=noisepar, verbose=verbose, keep_people=keep_people, sim_args=sim_args, run_args=run_args, do_run=do_run)
    elif isinstance(sim, list): # List of sims
        if reseed is None: reseed = False
        iterkwargs = dict(sim=sim, ind=np.arange(len(sim)))
        kwargs = dict(reseed=reseed, verbose=verbose, keep_people=keep_people, sim_args=sim_args, run_args=run_args, do_run=do_run)
    else:
        errormsg = f'Must be Sim object or list, not {type(sim)}'
        raise TypeError(errormsg)

    # Actually run!
    if parallel:
        kw = dict(iterkwargs=iterkwargs, kwargs=kwargs, **par_args)
        try:
            sims = sc.parallelize(single_run, **kw) # Run in parallel
        except RuntimeError as E: # Handle if run outside of __main__ on Windows
            if 'freeze_support' in E.args[0]: # For this error, add additional information
                errormsg = '''
 Uh oh! It appears you are trying to run with multiprocessing on Windows outside
 of the __main__ block; please see https://docs.python.org/3/library/multiprocessing.html
 for more information. The correct syntax to use is e.g.

     import covasim as cv
     sim = cv.Sim()
     msim = cv.MultiSim(sim)

     if __name__ == '__main__':
         msim.run()

Alternatively, to run without multiprocessing, set parallel=False.
 '''
                raise RuntimeError(errormsg) from E
            else: # For all other runtime errors, raise the original exception
                raise E
        except pkl.PicklingError as E:
            parallelizer = par_args.get('parallelizer')
            if retry in ['warn', 'silent'] and parallelizer not in ['multiprocess', 'robust']:
                if retry == 'warn':
                    warnmsg = f'multi_run() failed with parallelizer={parallelizer}, trying more robust "multiprocess"...'
                    cvm.warn(warnmsg)
                kw['parallelizer'] = 'multiprocess'
                sims = sc.parallelize(single_run, **kw) # Try again to run in parallel
            else:
                errormsg = 'Parallel run failed due to a pickling error; this is usually due to including a lambda function or other complex object'
                raise pkl.PicklingError(errormsg) from E
                
    else: # Run in serial, not in parallel
        sims = []
        n_sims = len(list(iterkwargs.values())[0]) # Must have length >=1 and all entries must be the same length
        for s in range(n_sims):
            this_iter = {k:v[s] for k,v in iterkwargs.items()} # Pull out items specific to this iteration
            this_iter.update(kwargs) # Merge with the kwargs
            this_iter['sim'] = this_iter['sim'].copy() # Ensure we have a fresh sim; this happens implicitly on pickling with multiprocessing
            sim = single_run(**this_iter) # Run in series
            sims.append(sim)

    return sims


def parallel(*args, **kwargs):
    '''
    A shortcut to ``cv.MultiSim()``, allowing the quick running of multiple simulations
    at once.

    Args:
        args (list): The simulations to run
        kwargs (dict): passed to multi_run()

    Returns:
        A run MultiSim object.

    **Examples**::

        s1 = cv.Sim(beta=0.01, label='Low')
        s2 = cv.Sim(beta=0.02, label='High')
        cv.parallel(s1, s2).plot()
        msim = cv.parallel([s1, s2], keep_people=True)

    New in version 3.1.1.
    '''
    sims = sc.mergelists(*args)
    return MultiSim(sims=sims).run(**kwargs)
