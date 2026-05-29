'''
Additional analysis functions that are not part of the core Covasim workflow,
but which are useful for particular investigations.
'''

import os
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
from . import utils as cvu
from . import misc as cvm
from . import interventions as cvi
from . import plotting as cvpl
from . import run as cvr
from .settings import options as cvo # For setting global options


__all__ = ['Analyzer', 'snapshot', 'age_histogram', 'daily_age_stats', 'daily_stats', 'nab_histogram',
           'Fit', 'Calibration', 'TransTree']


class Analyzer(sc.prettyobj):
    '''
    Base class for analyzers. Based on the Intervention class. Analyzers are used
    to provide more detailed information about a simulation than is available by
    default -- for example, pulling states out of sim.people on a particular timestep
    before it gets updated in the next timestep.

    To retrieve a particular analyzer from a sim, use sim.get_analyzer().

    Args:
        label (str): a label for the Analyzer (used for ease of identification)
    '''

    def __init__(self, label=None):
        if label is None:
            label = self.__class__.__name__ # Use the class name if no label is supplied
        self.label = label # e.g. "Record ages"
        self.initialized = False
        self.finalized = False
        return


    def __call__(self, *args, **kwargs):
        # Makes Analyzer(sim) equivalent to Analyzer.apply(sim)
        if not self.initialized:
            errormsg = f'Analyzer (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(*args, **kwargs)


    def initialize(self, sim=None):
        '''
        Initialize the analyzer, e.g. convert date strings to integers.
        '''
        self.initialized = True
        self.finalized = False
        return


    def finalize(self, sim=None):
        '''
        Finalize analyzer

        This method is run once as part of `sim.finalize()` enabling the analyzer to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized:
            raise RuntimeError('Analyzer already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        self.finalized = True
        return


    def apply(self, sim):
        '''
        Apply analyzer at each time point. The analyzer has full access to the
        sim object, and typically stores data/results in itself. This is the core
        method which each analyzer object needs to implement.

        Args:
            sim: the Sim instance
        '''
        raise NotImplementedError


    def shrink(self, in_place=False):
        '''
        Remove any excess stored data from the intervention; for use with sim.shrink().

        Args:
            in_place (bool): whether to shrink the intervention (else shrink a copy)
        '''
        if in_place:
            return self
        else:
            return sc.dcp(self)


    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. This method will attempt to JSONify each attribute of the
        intervention, skipping any that fail.

        Returns:
            JSON-serializable representation
        '''
        # Set the name
        json = {}
        json['analyzer_name'] = self.label if hasattr(self, 'label') else None
        json['analyzer_class'] = self.__class__.__name__

        # Loop over the attributes and try to process
        attrs = self.__dict__.keys()
        for attr in attrs:
            try:
                data = getattr(self, attr)
                try:
                    attjson = sc.jsonify(data)
                    json[attr] = attjson
                except Exception as E:
                    json[attr] = f'Could not jsonify "{attr}" ({type(data)}): "{str(E)}"'
            except Exception as E2:
                json[attr] = f'Could not jsonify "{attr}": "{str(E2)}"'
        return json


def validate_recorded_dates(sim, requested_dates, recorded_dates, die=True):
    '''
    Helper method to ensure that dates recorded by an analyzer match the ones
    requested.
    '''
    requested_dates = sorted(list(requested_dates))
    recorded_dates = sorted(list(recorded_dates))
    if recorded_dates != requested_dates: # pragma: no cover
        errormsg = f'The dates {requested_dates} were requested but only {recorded_dates} were recorded: please check the dates fall between {sim.date(sim["start_day"])} and {sim.date(sim["start_day"])} and the sim was actually run'
        if die:
            raise RuntimeError(errormsg)
        else:
            print(errormsg)
    return



class snapshot(Analyzer):
    '''
    Analyzer that takes a "snapshot" of the sim.people array at specified points
    in time, and saves them to itself. To retrieve them, you can either access
    the dictionary directly, or use the get() method.

    Args:
        days   (list): list of ints/strings/date objects, the days on which to take the snapshot
        args   (list): additional day(s)
        die    (bool): whether or not to raise an exception if a date is not found (default true)
        kwargs (dict): passed to Analyzer()


    **Example**::

        sim = cv.Sim(analyzers=cv.snapshot('2020-04-04', '2020-04-14'))
        sim.run()
        snapshot = sim['analyzers'][0]
        people = snapshot.snapshots[0]            # Option 1
        people = snapshot.snapshots['2020-04-04'] # Option 2
        people = snapshot.get('2020-04-14')       # Option 3
        people = snapshot.get(34)                 # Option 4
        people = snapshot.get()                   # Option 5
    '''

    def __init__(self, days, *args, die=True, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        days = sc.tolist(days) # Combine multiple days
        days.extend(args) # Include additional arguments, if present
        self.days      = days # Converted to integer representations
        self.die       = die  # Whether or not to raise an exception
        self.dates     = None # String representations
        self.start_day = None # Store the start date of the simulation
        self.snapshots = sc.odict() # Store the actual snapshots
        return


    def initialize(self, sim):
        self.start_day = sim['start_day'] # Store the simulation start day
        self.days, self.dates = cvi.process_days(sim, self.days, return_dates=True) # Ensure days are in the right format
        max_snapshot_day = self.days[-1]
        max_sim_day = sim.day(sim['end_day'])
        if max_snapshot_day > max_sim_day: # pragma: no cover
            errormsg = f'Cannot create snapshot for {self.dates[-1]} (day {max_snapshot_day}) because the simulation ends on {self.end_day} (day {max_sim_day})'
            raise ValueError(errormsg)
        self.initialized = True
        return


    def apply(self, sim):
        for ind in cvi.find_day(self.days, sim.t):
            date = self.dates[ind]
            self.snapshots[date] = sc.dcp(sim.people) # Take snapshot!


    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.snapshots.keys(), die=self.die)
        return


    def get(self, key=None):
        ''' Retrieve a snapshot from the given key (int, str, or date) '''
        if key is None:
            key = self.days[0]
        day  = sc.day(key, start_date=self.start_day)
        date = sc.date(day, start_date=self.start_day, as_date=False)
        if date in self.snapshots:
            snapshot = self.snapshots[date]
        else: # pragma: no cover
            dates = ', '.join(list(self.snapshots.keys()))
            errormsg = f'Could not find snapshot date {date} (day {day}): choices are {dates}'
            raise sc.KeyNotFoundError(errormsg)
        return snapshot



class age_histogram(Analyzer):
    '''
    Calculate statistics across age bins, including histogram plotting functionality.

    Args:
        days    (list): list of ints/strings/date objects, the days on which to calculate the histograms (default: last day)
        states  (list): which states of people to record (default: exposed, tested, diagnosed, dead)
        edges   (list): edges of age bins to use (default: 10 year bins from 0 to 100)
        datafile (str): the name of the data file to load in for comparison, or a dataframe of data (optional)
        sim      (Sim): only used if the analyzer is being used after a sim has already been run
        die     (bool): whether to raise an exception if dates are not found (default true)
        kwargs  (dict): passed to Analyzer()

    **Examples**::

        sim = cv.Sim(analyzers=cv.age_histogram())
        sim.run()

        agehist = sim.get_analyzer()
        agehist = cv.age_histogram(sim=sim) # Alternate method
        agehist.plot()
    '''

    def __init__(self, days=None, states=None, edges=None, datafile=None, sim=None, die=True, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        self.days      = days # To be converted to integer representations
        self.edges     = edges # Edges of age bins
        self.states    = states # States to save
        self.datafile  = datafile # Data file to load
        self.die       = die # Whether to raise an exception if dates are not found
        self.bins      = None # Age bins, calculated from edges
        self.dates     = None # String representations of dates
        self.start_day = None # Store the start date of the simulation
        self.data      = None # Store the loaded data
        self.hists = sc.odict() # Store the actual snapshots
        self.window_hists = None # Store the histograms for individual windows -- populated by compute_windows()
        if sim is not None: # Process a supplied simulation
            self.from_sim(sim)
        return


    def from_sim(self, sim):
        ''' Create an age histogram from an already run sim '''
        if self.days is not None: # pragma: no cover
            errormsg = 'If a simulation is being analyzed post-run, no day can be supplied: only the last day of the simulation is available'
            raise ValueError(errormsg)
        self.initialize(sim)
        self.apply(sim)
        return


    def initialize(self, sim):
        super().initialize()

        # Handle days
        self.start_day = sc.date(sim['start_day'], as_date=False) # Get the start day, as a string
        self.end_day   = sc.date(sim['end_day'],   as_date=False) # Get the start day, as a string
        if self.days is None:
            self.days = self.end_day # If no day is supplied, use the last day
        self.days, self.dates = cvi.process_days(sim, self.days, return_dates=True) # Ensure days are in the right format
        max_hist_day = self.days[-1]
        max_sim_day = sim.day(self.end_day)
        if max_hist_day > max_sim_day: # pragma: no cover
            errormsg = f'Cannot create histogram for {self.dates[-1]} (day {max_hist_day}) because the simulation ends on {self.end_day} (day {max_sim_day})'
            raise ValueError(errormsg)

        # Handle edges and age bins
        if self.edges is None: # Default age bins
            self.edges = np.linspace(0,100,11)
        self.bins = self.edges[:-1] # Don't include the last edge in the bins

        # Handle states
        if self.states is None:
            self.states = ['exposed', 'severe', 'dead', 'tested', 'diagnosed']
        self.states = sc.tolist(self.states)
        for s,state in enumerate(self.states):
            self.states[s] = state.replace('date_', '') # Allow keys starting with date_ as input, but strip it off here

        # Handle the data file
        if self.datafile is not None:
            if sc.isstring(self.datafile):
                self.data = cvm.load_data(self.datafile, check_date=False)
            else:
                self.data = self.datafile # Use it directly
                self.datafile = None

        return


    def apply(self, sim):
        for ind in cvi.find_day(self.days, sim.t):
            date = self.dates[ind] # Find the date for this index
            self.hists[date] = sc.objdict() # Initialize the dictionary
            scale  = sim.rescale_vec[sim.t] # Determine current scale factor
            age    = sim.people.age # Get the age distribution,since used heavily
            self.hists[date]['bins'] = self.bins # Copy here for convenience
            for state in self.states: # Loop over each state
                inds = sim.people.defined(f'date_{state}') # Pull out people for which this state is defined
                self.hists[date][state] = np.histogram(age[inds], bins=self.edges)[0]*scale # Actually count the people


    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.hists.keys(), die=self.die)
        return


    def get(self, key=None):
        ''' Retrieve a specific histogram from the given key (int, str, or date) '''
        if key is None:
            key = self.days[0]
        day  = sc.day(key, start_date=self.start_day)
        date = sc.date(day, start_date=self.start_day, as_date=False)
        if date in self.hists:
            hists = self.hists[date]
        else: # pragma: no cover
            dates = ', '.join(list(self.hists.keys()))
            errormsg = f'Could not find histogram date {date} (day {day}): choices are {dates}'
            raise sc.KeyNotFoundError(errormsg)
        return hists


    def compute_windows(self):
        ''' Convert cumulative histograms to windows '''
        if len(self.hists)<2:
            errormsg = 'You must have at least two dates specified to compute a window'
            raise ValueError(errormsg)

        self.window_hists = sc.objdict()
        for d,end_date,hists in self.hists.enumitems():
            if d==0: # Copy the first one
                start_date = self.start_day
                self.window_hists[f'{start_date} to {end_date}'] = self.hists[end_date]
            else:
                start_date = self.dates[d-1]
                datekey = f'{start_date} to {end_date}'
                self.window_hists[datekey] = sc.objdict() # Initialize the dictionary
                self.window_hists[datekey]['bins'] = self.hists[end_date]['bins']
                for state in self.states: # Loop over each state
                    self.window_hists[datekey][state] = self.hists[end_date][state] - self.hists[start_date][state]

        return


    def plot(self, windows=False, width=0.8, color='#F8A493', fig_args=None, axis_args=None, data_args=None, **kwargs):
        '''
        Simple method for plotting the histograms.

        Args:
            windows (bool): whether to plot windows instead of cumulative counts
            width (float): width of bars
            color (hex or rgb): the color of the bars
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
            kwargs (dict): passed to ``cv.options.with_style()``; see that function for choices
        '''

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))

        # Initialize
        n_plots = len(self.states)
        n_rows, n_cols = sc.get_rows_cols(n_plots)
        figs = []

        # Handle windows and what to plot
        if windows:
            if self.window_hists is None:
                self.compute_windows()
            histsdict = self.window_hists
        else:
            histsdict = self.hists
        if not len(histsdict): # pragma: no cover
            errormsg = f'Cannot plot since no histograms were recorded (schuled days: {self.days})'
            raise ValueError(errormsg)

        # Make the figure(s)
        with cvo.with_style(**kwargs):
            for date,hists in histsdict.items():
                figs += [pl.figure(**fig_args)]
                pl.subplots_adjust(**axis_args)
                bins = hists['bins']
                barwidth = width*(bins[1] - bins[0]) # Assume uniform width
                for s,state in enumerate(self.states):
                    ax = pl.subplot(n_rows, n_cols, s+1)
                    ax.bar(bins, hists[state], width=barwidth, facecolor=color, label=f'Number {state}')
                    if self.data and state in self.data:
                        data = self.data[state]
                        ax.bar(bins+d_args.offset, data, width=barwidth*d_args.width, facecolor=d_args.color, label='Data')
                    ax.set_xlabel('Age')
                    ax.set_ylabel('Count')
                    ax.set_xticks(ticks=bins)
                    ax.legend()
                    preposition = 'from' if windows else 'by'
                    ax.set_title(f'Number of people {state} {preposition} {date}')

        return cvpl.handle_show_return(figs=figs)


class daily_age_stats(Analyzer):
    '''
    Calculate daily counts by age, saving for each day of the simulation. Can
    plot either time series by age or a histogram over all time.

    Args:
        states  (list): which states of people to record (default: ['diagnoses', 'deaths', 'tests', 'severe'])
        edges   (list): edges of age bins to use (default: 10 year bins from 0 to 100)
        kwargs  (dict): passed to Analyzer()

    **Examples**::

        sim = cv.Sim(analyzers=cv.daily_age_stats())
        sim = cv.Sim(pars, analyzers=daily_age)
        sim.run()
        daily_age = sim.get_analyzer()
        daily_age.plot()
        daily_age.plot(total=True)

    '''

    def __init__(self, states=None, edges=None, **kwargs):
        super().__init__(**kwargs)
        self.edges = edges
        self.bins = None  # Age bins, calculated from edges
        self.states = states
        self.results = sc.odict()
        self.start_day = None
        self.df = None
        self.total_df = None
        return


    def initialize(self, sim):
        super().initialize()

        if self.states is None:
            self.states = ['exposed', 'severe', 'dead', 'tested', 'diagnosed']

        # Handle edges and age bins
        if self.edges is None:  # Default age bins
            self.edges = np.linspace(0, 100, 11)
        self.bins = self.edges[:-1]  # Don't include the last edge in the bins

        self.start_day = sim['start_day']

        return


    def apply(self, sim):
        df_entry = {}
        for state in self.states:
            inds = sc.findinds(sim.people[f'date_{state}'], sim.t)
            b, _ = np.histogram(sim.people.age[inds], self.edges)
            df_entry.update({state: b * sim.rescale_vec[sim.t]})
        df_entry.update({'day':sim.t, 'age': self.bins})
        self.results.update({sim.date(sim.t): df_entry})


    def to_df(self):
        '''Create dataframe totals for each day'''
        mapper = {f'{k}': f'new_{k}' for k in self.states}
        df = pd.DataFrame()
        for date, k in self.results.items():
            df_ = pd.DataFrame(k)
            df_['date'] = date
            df_.rename(mapper, inplace=True, axis=1)
            df = pd.concat((df, df_))
        cols = list(df.columns.values)
        cols = [cols[-1]] + [cols[-2]] + cols[:-2]
        self.df = df[cols]
        return self.df


    def to_total_df(self):
        ''' Create dataframe totals across days '''
        if self.df is None:
            self.to_df()
        cols = list(self.df.columns)
        cum_cols = [c for c in cols if c.split('_')[0] == 'new']
        mapper = {f'new_{c.split("_")[1]}': f'cum_{c.split("_")[1]}' for c in cum_cols}
        df_dict = {'age': []}
        df_dict.update({c: [] for c in mapper.values()})
        for age, group in self.df.groupby('age'):
            cum_vals = group.sum()
            df_dict['age'].append(age)
            for k, v in mapper.items():
                df_dict[v].append(cum_vals[k])
        df = pd.DataFrame(df_dict)
        if ('cum_diagnoses' in df.columns) and ('cum_tests' in df.columns):
            df['yield'] = df['cum_diagnoses'] / df['cum_tests']
        self.total_df = df
        return df


    def plot(self, total=False, do_show=None, fig_args=None, axis_args=None, plot_args=None,
             dateformat=None, width=0.8, color='#F8A493', **kwargs):
        '''
        Plot the results.

        Args:
            total     (bool): whether to plot the total histograms rather than time series
            do_show   (bool): whether to show the plot
            fig_args  (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            plot_args (dict): passed to pl.plot()
            dateformat (str): the format to use for the x-axes (only used for time series)
            width    (float): width of bars (only used for histograms)
            color  (hex/rgb): the color of the bars (only used for histograms)
            kwargs    (dict): passed to ``cv.options.with_style()``
        '''
        if self.df is None:
            self.to_df()
        if self.total_df is None:
            self.to_total_df()

        fig_args  = sc.mergedicts(dict(figsize=(18,11)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.25, hspace=0.4), axis_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.5, marker='o'), plot_args)

        with cvo.with_style(**kwargs):
            nplots = len(self.states)
            nrows, ncols = sc.get_rows_cols(nplots)
            fig, axs = pl.subplots(nrows=nrows, ncols=ncols, **fig_args)
            pl.subplots_adjust(**axis_args)

            for count,state in enumerate(self.states):
                row,col = np.unravel_index(count, (nrows,ncols))
                ax = axs[row,col]
                ax.set_title(state.title())
                ages = self.df.age.unique()

                # Plot time series
                if not total:
                    colors = sc.vectocolor(len(ages))
                    has_data = False
                    for a,age in enumerate(ages):
                        label = f'Age {age}'
                        df = self.df[self.df.age==age]
                        ax.plot(df.date, df[f'new_{state}'], c=colors[a], label=label)
                        has_data = has_data or len(df)
                    if has_data:
                        ax.legend()
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Count')
                        sc.dateformatter(dateformat=dateformat, ax=ax)

                # Plot total histograms
                else:
                    df = self.total_df
                    barwidth = width*(df.age[1] - df.age[0]) # Assume uniform width
                    ax.bar(df.age, df[f'cum_{state}'], width=barwidth, facecolor=color)
                    ax.set_xlabel('Age')
                    ax.set_ylabel('Count')
                    ax.set_xticks(ticks=df.age)

        return cvpl.handle_show_return(fig=fig, do_show=do_show)


class daily_stats(Analyzer):
    '''
    Print out daily statistics about the simulation. Note that this analyzer takes
    a considerable amount of time, so should be used primarily for debugging, not
    in production code. To keep the intervention but toggle it off, pass an empty
    list of days.

    To show the stats for a day after a run has finished, use e.g. ``daily_stats.report('2020-04-04')``.

    Args:
        days (list): days on which to print out statistics (if None, assume all)
        verbose (bool): whether to print on each timestep
        reporter (func): if supplied, a custom parser of the stats object into a report (see make_report() function for syntax)
        save_inds (bool): whether to save the indices of every infection at every timestep (also recoverable from the infection log)

    **Example**::

        sim = cv.Sim(analyzers=cv.daily_stats())
        sim.run()
        sim['analyzers'][0].plot()
    '''

    def __init__(self, days=None, verbose=True, reporter=None, save_inds=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        self.days      = days # Converted to integer representations
        self.verbose   = verbose # Print on each timestep
        self.reporter  = reporter # Custom way of reporting the stats
        self.save_inds = save_inds # Whether to save infection log indices
        self.stats     = sc.objdict() # Store the actual stats
        self.reports   = sc.objdict() # Textual representation of the statistics
        return


    def initialize(self, sim):
        super().initialize()
        if self.days is None:
            self.days = sc.dcp(sim.tvec)
        else:
            self.days = sim.day(self.days)

        self.keys =  ['exposed', 'infectious', 'symptomatic', 'severe', 'critical', 'known_contact', 'quarantined', 'diagnosed', 'recovered', 'dead']
        self.basekeys = ['stocks', 'trans', 'source', 'test', 'quar'] # Categories of things to plot
        self.extrakeys = ['layer_counts', 'extra']
        return


    def intersect(self, *args):
        '''
        Compute the intersection between arrays of indices, handling either keys
        to precomputed indices or lists of indices. With two array inputs, simply
        performs np.intersect1d(arr1, arr2).
        '''
        # Optionally pull precomputed indices
        args = list(args) # Convert from tuple to list
        for i,inds in enumerate(args):
            if isinstance(inds, str):
                args[i] = self.inds[inds]

        # Find the intersection
        output = args[0] # Start with the first set of indices
        for inds in args[1:]: # Loop over remaining sets
            output = np.intersect1d(output, inds, assume_unique=True)

        return output


    def apply(self, sim):
        for ind in cvi.find_day(self.days, sim.t):

            # Initialize
            ppl = sim.people
            all_inds = np.arange(len(ppl))
            stats = sc.objdict()
            stats.empty = sc.objdict()
            for basekey in self.basekeys:
                stats[basekey] = sc.objdict()
                stats.empty[basekey] = []

            # Get the indices for each of the states
            self.inds = {}
            for key in self.keys:
                self.inds[key] = ppl.true(key)

            # Basic stocks
            for key in self.keys:
                stats.stocks[key] = len(self.inds[key])

            # Transmission stats
            newinfs = cvu.true(ppl.date_exposed == sim.t)
            stats.trans.new_infections = len(newinfs)
            for key in ['known_contact', 'quarantined']:
                stats.trans[key] = len(self.intersect(newinfs, key))
                if not stats.trans[key]:
                    stats.empty.trans.append(key)

            # Source stats
            inflog = sim.people.infection_log
            infloginds = [i for i,e in enumerate(inflog) if (e['date']==sim.t and e['source'] is not None)] # Person was infected today and was not a seed infection
            sourceinds = list(set([inflog[i]['source'] for i in infloginds]))
            stats.source.new_sources = len(sourceinds)
            for key in self.keys:
                stats.source[key] = len(self.intersect(sourceinds, key))
                if not stats.source[key]:
                    stats.empty.source.append(key)

            # Testing stats
            newtests = cvu.true(ppl.date_tested == sim.t)
            stats.test.new_tests = len(newtests)
            for key in self.keys:
                stats.test[key] = len(self.intersect(newtests,key))
                if not stats.test[key]:
                    stats.empty.test.append(key)

            # Quarantine stats
            q_inds = np.union1d(self.inds['quarantined'], cvu.true(ppl.date_end_quarantine == sim.t)) # Append people who finished quarantine today
            eq_inds = cvu.true(ppl.date_quarantined == sim.t-1) # People entering quarantine the day before (their first full day of quarantine)
            fq_inds = cvu.true(ppl.date_end_quarantine == sim.t+1) # People finishing quarantine; +1 since on the date of quarantine end, they are released back and can get infected at normal rates
            stats.quar.in_quarantine = len(q_inds) # Similar to stats.quar.quarantined, but slightly more
            stats.quar.entered_quar  = len(eq_inds)
            stats.quar.finished_quar = len(fq_inds)
            for key in self.keys:
                stats.quar[key] = len(self.intersect('quarantined', key))
                if not stats.quar[key]:
                    stats.empty.quar.append(key)

            # Calculate extras for the source
            stats.extra = sc.objdict() # Additional quantities not stored in the main counts
            symp_inds = self.inds['symptomatic']
            asymp_inds = ppl.false('symptomatic')
            stats.extra.symp    = len(self.intersect(sourceinds, 'symptomatic')) # Redefine in case empty above
            stats.extra.presymp = len(self.intersect(sourceinds, asymp_inds, ppl.defined('date_symptomatic')))
            stats.extra.asymp   = len(self.intersect(sourceinds, asymp_inds,  ppl.undefined('date_symptomatic')))
            per_factor = 100/max(1, stats.source.new_sources) # Convert to a percentage and avoid division by zero
            stats.extra.per_symp    = stats.extra.symp*per_factor # Percentage symptomatic
            stats.extra.per_presymp = stats.extra.presymp*per_factor
            stats.extra.per_asymp   = stats.extra.asymp*per_factor
            stats.layer_counts = {k:0 for k in sim.layer_keys()}
            for i in infloginds:
                stats.layer_counts[inflog[i]['layer']] += 1

            # Calculate extras for quarantine testing
            t_inds = newtests # Everyone who tested this timestep
            d_inds = self.intersect(newtests, 'infectious') # Everyone infectious will test positive
            u_inds = self.intersect('infectious', ppl.false('diagnosed'))
            nq_inds = np.setdiff1d(all_inds, q_inds) # We can't use ppl.false('quarantined') since that will miss people who left quarantine because they were diagnosed
            for tk,ti in zip(['test', 'diag', 'undiag'], [t_inds, d_inds, u_inds]): # People tested vs diagnosed
                for sk,si in zip(['symp', 'asymp'], [symp_inds, asymp_inds]): # Symptomatic vs asymptomatic
                    for qk,qi in zip(['q', 'nq', 'eq', 'fq'], [q_inds, nq_inds, eq_inds, fq_inds]): # In quarantine, not in quarantine, entering quarantine, finishing quarantine
                        stats.extra[f'{tk}_{sk}_{qk}']  = len(self.intersect(ti, si,  qi)) # E.g. stats.extra.diag_asymp_nq = len(self.intersect(d_inds, asymp_inds, nq_inds))

            # Final calculations
            stats.extra.prev = stats.stocks.infectious/sim["pop_size"] # Overall prevalence
            stats.extra.dead = stats.stocks.dead/sim["pop_size"] # Fraction dead
            stats.extra.quar_prev     = len(self.intersect(q_inds, 'infectious'))/max(1,len(q_inds)) # Prevalence of people in quarantine
            stats.extra.e_quar_prev   = len(self.intersect(eq_inds, 'infectious'))/max(1,len(eq_inds)) # Prevalence of people entering quarantine
            stats.extra.f_quar_prev   = len(self.intersect(fq_inds, 'infectious'))/max(1,len(fq_inds)) # Prevalence of people finishing quarantine
            stats.extra.non_quar_prev = len(self.intersect(nq_inds, 'infectious'))/max(1,len(nq_inds)) # Prevalence of people outside quarantine

            # Indices aren't usually saved for memory reasons, but may be helpful for extra debugging
            if self.save_inds:
                stats.inds = sc.objdict()
                stats.inds.inflog  = infloginds
                stats.inds.targets = newinfs
                stats.inds.sources = sourceinds
                stats.inds.t_inds = t_inds
                stats.inds.d_inds = d_inds
                stats.inds.eq_inds = eq_inds
                stats.inds.fq_inds = fq_inds

            # Turn into report
            if self.reporter is not None:
                report = self.reporter(self, sim, stats)
            else:
                report = self.make_report(sim, stats)

            # Save
            today = sim.date(sim.t)
            self.stats[today] = stats
            self.reports[today] = report

            if self.verbose:
                self.report(today)

        return


    def report(self, day=None):
        ''' Print out one or all reports -- take a date string or an int '''
        if day is None:
            print(self.reports)
        else:
            print(self.reports[day])
        return


    def make_report(self, sim, stats, show_empty='count'):
        ''' Turn the statistics into a report '''

        def make_entry(basekey, show_empty=show_empty):
            ''' For each key, print the key and the count if the count is >0, and optionally any empty states '''
            string  = '\n'.join([f'  {k:13s} = {v}' for k,v in stats[basekey].items() if v>0])
            if show_empty is True:
                string += f'\n  Empty states: {stats.empty[basekey]}'
            elif show_empty == 'count':
                string += f'\n  Number of empty states: {len(stats.empty[basekey])}'
            string = '\n' + string + '\n'
            return string

        datestr = f'day {sim.t} ({sim.date(sim.t)})'
        report  = f'*** Statistics report for {datestr} ***\n\n'
        report += 'Overall stocks:'
        report += make_entry('stocks', show_empty=False)
        report += '  Derived statistics:\n'
        report += f'    Percentage infectious: {stats.extra.prev*100:6.3f}%\n'
        report += f'    Percentage dead:       {stats.extra.dead*100:6.3f}%\n'
        report += '\nTransmission target statistics:'
        report += make_entry('trans')
        report += '  Infections by layer:\n'
        report += '\n'.join([f'    {k} = {v}' for k,v in stats.layer_counts.items()])
        report += '\n\nTransmission source statistics:'
        report += make_entry('source')
        report += '  Derived statistics:\n'
        report += f'    Pre-symptomatic: {stats.extra.presymp} ({stats.extra.per_presymp:0.1f})%\n'
        report += f'    Asymptomatic:    {stats.extra.asymp} ({stats.extra.per_asymp:0.1f})%\n'
        report += f'    Symptomatic:     {stats.extra.symp} ({stats.extra.per_symp:0.1f})%\n'
        report += '\nTesting statistics:'
        report += make_entry('test')
        report += '  Derived statistics:\n'
        report += '    Tests:\n'
        report += f'      Symp/asymp not in quar: {stats.extra.test_symp_nq}/{stats.extra.test_asymp_nq}\n'
        report += f'      Symp/asymp in quar:     {stats.extra.test_symp_q}/{stats.extra.test_asymp_q}\n'
        report += f'      Symp/asymp enter quar:  {stats.extra.test_symp_eq}/{stats.extra.test_asymp_eq}\n'
        report += f'      Symp/asymp finish quar: {stats.extra.test_symp_fq}/{stats.extra.test_asymp_fq}\n'
        report += '    Diagnoses:\n'
        report += f'      Symp/asymp not in quar: {stats.extra.diag_symp_nq}/{stats.extra.diag_asymp_nq}\n'
        report += f'      Symp/asymp in quar:     {stats.extra.diag_symp_q}/{stats.extra.diag_asymp_q}\n'
        report += f'      Symp/asymp enter quar:  {stats.extra.diag_symp_eq}/{stats.extra.diag_asymp_eq}\n'
        report += f'      Symp/asymp finish quar: {stats.extra.diag_symp_fq}/{stats.extra.diag_asymp_fq}\n'
        report += '    Undiagnosed:\n'
        report += f'      Symp/asymp not in quar: {stats.extra.undiag_symp_nq}/{stats.extra.undiag_asymp_nq}\n'
        report += f'      Symp/asymp in quar:     {stats.extra.undiag_symp_q}/{stats.extra.undiag_asymp_q}\n'
        report += f'      Symp/asymp enter quar:  {stats.extra.undiag_symp_eq}/{stats.extra.undiag_asymp_eq}\n'
        report += f'      Symp/asymp finish quar: {stats.extra.undiag_symp_fq}/{stats.extra.undiag_asymp_fq}\n'
        report += '\nQuarantine statistics:'
        report += make_entry('quar')
        report += '  Derived statistics:\n'
        report += f'    Percentage infectious not in quarantine:    {stats.extra.non_quar_prev*100:6.3f}%\n'
        report += f'    Percentage infectious in quarantine:        {stats.extra.quar_prev*100:6.3f}%\n'
        report += f'    Percentage infectious entering quarantine:  {stats.extra.e_quar_prev*100:6.3f}%\n'
        report += f'    Percentage infectious finishing quarantine: {stats.extra.f_quar_prev*100:6.3f}%\n'
        report += f'\n*** End of report for day {datestr} ***\n'

        return report


    def transpose(self, keys=None):
        ''' Transpose the data from a list-of-dicts-of-dicts to a dict-of-dicts-of-lists '''
        if keys is None:
            keys = self.basekeys + self.extrakeys

        # Initialize
        data = {}
        for k1 in keys:
            data[k1] = {}
            for k2 in self.stats[0][k1].keys():
                data[k1][k2] = []

        # Populate
        for stats in self.stats.values():
            for k1 in keys:
                for k2 in stats[k1].keys():
                    data[k1][k2].append(stats[k1][k2])

        return data


    def plot(self, fig_args=None, axis_args=None, plot_args=None, do_show=None, **kwargs):
        '''
        Plot the daily statistics recorded. Some overlap with e.g. ``sim.plot(to_plot='overview')``.

        Args:
            fig_args  (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            plot_args (dict): passed to pl.plot()
            do_show   (bool): whether to show the plot
            kwargs    (dict): passed to ``cv.options.with_style()``
        '''

        fig_args  = sc.mergedicts(dict(figsize=(18,11)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.25, hspace=0.4), axis_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.5, marker='o'), plot_args)

        # Transform the data into time series
        data = self.transpose()

        # Do the plotting
        with cvo.with_style(**kwargs):
            nplots = sum([len(data[k].keys()) for k in data.keys()]) # Figure out how many plots there are
            nrows,ncols = sc.get_rows_cols(nplots)
            fig, axs = pl.subplots(nrows=nrows, ncols=ncols, **fig_args)
            pl.subplots_adjust(**axis_args)

            count = -1
            for k1 in data.keys():
                for k2 in data[k1].keys():
                    count += 1
                    row,col = np.unravel_index(count, (nrows,ncols))
                    ax = axs[row,col]
                    y = data[k1][k2]
                    ax.plot(y, **plot_args)
                    ax.set_title(f'{k1}: {k2}')

        return cvpl.handle_show_return(fig=fig, do_show=do_show)


class nab_histogram(Analyzer):
    '''
    Store histogram of log_{10}(NAb) distribution

    Args:
        days (list): days on which calculate the NAb histogram (if None, assume last day)
        edges (list): log10 bin edges for histogram

    **Example**::

        sim = cv.Sim(analyzers=cv.nab_histogram())
        sim.run()
        sim.get_analyzer().plot()

    New in version 3.1.0.
    '''
    def __init__(self, days=None, edges=None, **kwargs):
        super().__init__(**kwargs)  # Initialize the Analyzer object
        self.days = days  # To be converted to integer representations
        self.edges = edges  # Edges of age bins in log10
        self.hists = sc.odict()  # Store the actual snapshots


    def initialize(self, sim):

        # Check that the simulation parameters are correct
        if not sim['use_waning']:
            errormsg = 'The cv.nab_histogram() analyzer requires use_waning=True. Please enable waning.'
            raise RuntimeError(errormsg)

        super().initialize()

        # Handle days
        self.start_day = sc.date(sim['start_day'], as_date=False)  # Get the start day, as a string
        self.end_day = sc.date(sim['end_day'], as_date=False)  # Get the start day, as a string
        if self.days is None:
            self.days = self.end_day  # If no day is supplied, use the last day
        self.days, self.dates = cvi.process_days(sim, self.days,
                                                 return_dates=True)  # Ensure days are in the right format

        # Handle edges and nab bins
        if self.edges is None:  # Default  bins
            self.edges = np.arange(-4, 3)
        self.bins = self.edges[:-1]  # Don't include the last edge in the bins

        return


    def apply(self, sim):
        nonzero = sim.people.nab > 0
        log_nabs = np.log10(sim.people.nab[nonzero])
        for ind in cvi.find_day(self.days, sim.t):
            date = self.dates[ind]  # Find the date for this index
            self.hists[date] = sc.objdict()  # Initialize the dictionary
            scale = sim.rescale_vec[sim.t]  # Determine current scale factor
            self.hists[date]['bins'] = self.bins  # Copy here for convenience
            self.hists[date]['n'] = np.histogram(log_nabs, bins=self.edges)[0] * scale  # Actually count the people
            self.hists[date]['s'] = np.std(log_nabs)    # keep the std
            self.hists[date]['m'] = np.mean(log_nabs)   # keep the mean


    def plot(self, fig_args=None, axis_args=None, plot_args=None, do_show=None, **kwargs):
        '''
        Plot the results

        Args:
            fig_args  (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            plot_args (dict): passed to pl.plot()
            do_show   (bool): whether to show the plot
            kwargs    (dict): passed to ``cv.options.with_style()``
        '''

        fig_args  = sc.mergedicts(dict(figsize=(9,5)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.10, right=0.95, bottom=0.10, top=0.95, wspace=0.25, hspace=0.4), axis_args)
        plot_args = sc.mergedicts(dict(lw=2), plot_args)

        with cvo.with_style(**kwargs):
            fig, axs = pl.subplots(nrows=1, ncols=1, **fig_args)
            pl.subplots_adjust(**axis_args)
            for date, hist in self.hists.items():
                axs.stairs(hist['n'], edges=self.edges, label=date, **plot_args)
            axs.set_xlabel('Log10(NAb)')
            axs.set_ylabel('Count')
            axs.legend()

        return cvpl.handle_show_return(fig=fig, do_show=do_show)


class Fit(Analyzer):
    '''
    A class for calculating the fit between the model and the data. Note the
    following terminology is used here:

        - fit: nonspecific term for how well the model matches the data
        - difference: the absolute numerical differences between the model and the data (one time series per result)
        - goodness-of-fit: the result of passing the difference through a statistical function, such as mean squared error
        - loss: the goodness-of-fit for each result multiplied by user-specified weights (one time series per result)
        - mismatches: the sum of all the losses (a single scalar value per time series)
        - mismatch: the sum of the mismatches -- this is the value to be minimized during calibration

    Args:
        sim (Sim): the sim object
        weights (dict): the relative weight to place on each result (by default: 10 for deaths, 5 for diagnoses, 1 for everything else)
        keys (list): the keys to use in the calculation
        custom (dict): a custom dictionary of additional data to fit; format is e.g. {'my_output':{'data':[1,2,3], 'sim':[1,2,4], 'weights':2.0}}
        compute (bool): whether to compute the mismatch immediately
        verbose (bool): detail to print
        die (bool): whether to raise an exception if no data are supplied
        label (str): the label for the analyzer
        kwargs (dict): passed to cv.compute_gof() -- see this function for more detail on goodness-of-fit calculation options

    **Example**::

        sim = cv.Sim(datafile='my-data-file.csv')
        sim.run()
        fit = sim.compute_fit()
        fit.plot()
    '''

    def __init__(self, sim, weights=None, keys=None, custom=None, compute=True, verbose=False, die=True, label=None, **kwargs):
        super().__init__(label=label) # Initialize the Analyzer object

        # Handle inputs
        self.weights    = weights
        self.custom     = sc.mergedicts(custom)
        self.verbose    = verbose
        self.weights    = sc.mergedicts({'cum_deaths':10, 'cum_diagnoses':5}, weights)
        self.keys       = keys
        self.gof_kwargs = kwargs
        self.die        = die

        # Copy data
        if sim.data is None: # pragma: no cover
            errormsg = 'Model fit cannot be calculated until data are loaded'
            if self.die:
                raise RuntimeError(errormsg)
            else:
                cvm.warn(errormsg)
                sim.data = pd.DataFrame() # Use an empty dataframe
        self.data = sim.data

        # Copy sim results
        if not sim.results_ready: # pragma: no cover
            errormsg = 'Model fit cannot be calculated until results are run'
            if self.die: raise RuntimeError(errormsg)
            else:        cvm.warn(errormsg)
        self.sim_results = sc.objdict()
        for key in sim.result_keys() + ['t', 'date']:
            self.sim_results[key] = sim.results[key]
        self.sim_npts = sim.npts # Number of time points in the sim

        # Copy other things
        self.sim_dates = sim.datevec.tolist()

        # These are populated during initialization
        self.inds         = sc.objdict() # To store matching indices between the data and the simulation
        self.inds.sim     = sc.objdict() # For storing matching indices in the sim
        self.inds.data    = sc.objdict() # For storing matching indices in the data
        self.date_matches = sc.objdict() # For storing matching dates, largely for plotting
        self.pair         = sc.objdict() # For storing perfectly paired points between the data and the sim
        self.diffs        = sc.objdict() # Differences between pairs
        self.gofs         = sc.objdict() # Goodness-of-fit for differences
        self.losses       = sc.objdict() # Weighted goodness-of-fit
        self.mismatches   = sc.objdict() # Final mismatch values
        self.mismatch     = None # The final value

        if compute:
            self.compute()

        return


    def compute(self):
        ''' Perform all required computations '''
        self.reconcile_inputs() # Find matching values
        self.compute_diffs() # Perform calculations
        self.compute_gofs()
        self.compute_losses()
        self.compute_mismatch()
        return self.mismatch


    def reconcile_inputs(self):
        ''' Find matching keys and indices between the model and the data '''

        data_cols = self.data.columns
        if self.keys is None:
            sim_keys = [k for k in self.sim_results.keys() if k.startswith('cum_')] # Default sim keys, only keep cumulative keys if no keys are supplied
            intersection = list(set(sim_keys).intersection(data_cols)) # Find keys in both the sim and data
            self.keys = [key for key in sim_keys if key in intersection] # Maintain key order
            if not len(self.keys): # pragma: no cover
                errormsg = f'No matches found between simulation result keys:\n{sc.strjoin(sim_keys)}\n\nand data columns:\n{sc.strjoin(data_cols)}'
                if self.die: raise sc.KeyNotFoundError(errormsg)
                else:        cvm.warn(errormsg)
        mismatches = [key for key in self.keys if key not in data_cols]
        if len(mismatches): # pragma: no cover
            mismatchstr = ', '.join(mismatches)
            errormsg = f'The following requested key(s) were not found in the data: {mismatchstr}'
            if self.die: raise sc.KeyNotFoundError(errormsg)
            else:        cvm.warn(errormsg)

        for key in self.keys: # For keys present in both the results and in the data
            self.inds.sim[key]  = []
            self.inds.data[key] = []
            self.date_matches[key] = []
            count = -1
            for d, datum in self.data[key].items():
                count += 1
                if np.isfinite(datum):
                    if d in self.sim_dates:
                        self.date_matches[key].append(d)
                        self.inds.sim[key].append(self.sim_dates.index(d))
                        self.inds.data[key].append(count)
            self.inds.sim[key]  = np.array(self.inds.sim[key])
            self.inds.data[key] = np.array(self.inds.data[key])

        # Convert into paired points
        matches = 0 # Count how many data points match
        for key in self.keys:
            self.pair[key] = sc.objdict()
            sim_inds = self.inds.sim[key]
            data_inds = self.inds.data[key]
            n_inds = len(sim_inds)
            self.pair[key].sim  = np.zeros(n_inds)
            self.pair[key].data = np.zeros(n_inds)
            for i in range(n_inds):
                matches += 1
                self.pair[key].sim[i]  = self.sim_results[key].values[sim_inds[i]]
                self.pair[key].data[i] = self.data[key].values[data_inds[i]]

        # Process custom inputs
        self.custom_keys = list(self.custom.keys())
        for key in self.custom.keys():
            matches += 1 # If any of these exist, count it as  amatch

            # Initialize and do error checking
            custom = self.custom[key]
            c_keys = list(custom.keys())
            if 'sim' not in c_keys or 'data' not in c_keys:
                errormsg = f'Custom input must have "sim" and "data" keys, not {c_keys}'
                raise sc.KeyNotFoundError(errormsg)
            c_data = custom['data']
            c_sim  = custom['sim']
            try:
                assert len(c_data) == len(c_sim)
            except: # pragma: no cover
                errormsg = f'Custom data and sim must be arrays, and be of the same length: data = {c_data}, sim = {c_sim} could not be processed'
                raise ValueError(errormsg)
            if key in self.pair: # pragma: no cover
                errormsg = f'You cannot use a custom key "{key}" that matches one of the existing keys: {self.pair.keys()}'
                raise ValueError(errormsg)

            # If all tests pass, simply copy the data
            self.pair[key] = sc.objdict()
            self.pair[key].sim  = c_sim
            self.pair[key].data = c_data

            # Process weight, if available
            wt = custom.get('weight', 1.0) # Attempt to retrieve key 'weight', or use the default if not provided
            wt = custom.get('weights', wt) # ...but also try "weights"
            self.weights[key] = wt # Set the weight

        if matches == 0:
            errormsg = 'No paired data points were found between the supplied data and the simulation; please check the dates for each'
            if self.die: raise ValueError(errormsg)
            else:        cvm.warn(errormsg)

        return


    def compute_diffs(self, absolute=False):
        ''' Find the differences between the sim and the data '''
        for key in self.pair.keys():
            self.diffs[key] = self.pair[key].sim - self.pair[key].data
            if absolute:
                self.diffs[key] = np.abs(self.diffs[key])
        return


    def compute_gofs(self, **kwargs):
        ''' Compute the goodness-of-fit '''
        kwargs = sc.mergedicts(self.gof_kwargs, kwargs)
        for key in self.pair.keys():
            actual    = sc.dcp(self.pair[key].data)
            predicted = sc.dcp(self.pair[key].sim)
            self.gofs[key] = cvm.compute_gof(actual, predicted, **kwargs)
        return


    def compute_losses(self):
        ''' Compute the weighted goodness-of-fit '''
        for key in self.gofs.keys():
            if key in self.weights:
                weight = self.weights[key]
                if sc.isiterable(weight): # It's an array
                    len_wt = len(weight)
                    len_sim = self.sim_npts
                    len_match = len(self.gofs[key])
                    if len_wt == len_match: # If the weight already is the right length, do nothing
                        pass
                    elif len_wt == len_sim: # Most typical case: it's the length of the simulation, must trim
                        weight = weight[self.inds.sim[key]] # Trim to matching indices
                    else: # pragma: no cover
                        errormsg = f'Could not map weight array of length {len_wt} onto simulation of length {len_sim} or data-model matches of length {len_match}'
                        raise ValueError(errormsg)
            else:
                weight = 1.0
            self.losses[key] = self.gofs[key]*weight
        return


    def compute_mismatch(self, use_median=False):
        ''' Compute the final mismatch '''
        for key in self.losses.keys():
            if use_median:
                self.mismatches[key] = np.median(self.losses[key])
            else:
                self.mismatches[key] = np.sum(self.losses[key])
        self.mismatch = self.mismatches[:].sum()
        return self.mismatch


    def summarize(self):
        ''' Print out results from the fit '''
        if self.mismatch is not None:
            print('Mismatch values for:')
            print(self.mismatches)
            print('\nTotal mismatch value:')
            print(self.mismatch)
        else:
            print('Mismatch values not yet calculated; please run sim.compute_fit().')
        return


    def plot(self, keys=None, width=0.8, fig_args=None, axis_args=None, plot_args=None,
             date_args=None, do_show=None, fig=None, **kwargs):
        '''
        Plot the fit of the model to the data. For each result, plot the data
        and the model; the difference; and the loss (weighted difference). Also
        plots the loss as a function of time.

        Args:
            keys      (list):  which keys to plot (default, all)
            width     (float): bar width
            fig_args  (dict):  passed to ``pl.figure()``
            axis_args (dict):  passed to ``pl.subplots_adjust()``
            plot_args (dict):  passed to ``pl.plot()``
            date_args (dict):  passed to ``cv.plotting.reset_ticks()`` (handle date format, rotation, etc.)
            do_show   (bool):  whether to show the plot
            fig       (fig):   if supplied, use this figure to plot in
            kwargs    (dict):  passed to ``cv.options.with_style()``

        Returns:
            Figure object
        '''

        fig_args  = sc.mergedicts(dict(figsize=(18,11)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3), axis_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.5, marker='o'), plot_args)
        date_args = sc.mergedicts(sc.objdict(as_dates=True, dateformat=None, rotation=None, start=None, end=None), date_args)

        if keys is None:
            keys = self.keys + self.custom_keys
        n_keys = len(keys)

        loss_ax = None
        colors = sc.gridcolors(n_keys)
        n_rows = 4

        # Plot
        with cvo.with_style(**kwargs):
            if fig is None:
                fig = pl.figure(**fig_args)
            pl.subplots_adjust(**axis_args)
            main_ax1 = pl.subplot(n_rows, 2, 1)
            main_ax2 = pl.subplot(n_rows, 2, 2)
            bottom = sc.objdict() # Keep track of the bottoms for plotting cumulative
            bottom.daily = np.zeros(self.sim_npts)
            bottom.cumul = np.zeros(self.sim_npts)
            for k,key in enumerate(keys):
                if key in self.keys: # It's a time series, plot with days and dates
                    days      = self.inds.sim[key] # The "days" axis (or not, for custom keys)
                    daylabel  = 'Date'
                else: #It's custom, we don't know what it is
                    days      = np.arange(len(self.losses[key])) # Just use indices
                    daylabel  = 'Index'

                # Cumulative totals can't mix daily and non-daily inputs, so skip custom keys
                if key in self.keys:
                    for i,ax in enumerate([main_ax1, main_ax2]):

                        if i == 0:
                            data = self.losses[key]
                            ylabel = 'Daily mismatch'
                            title = 'Daily total mismatch'
                        else:
                            data = np.cumsum(self.losses[key])
                            ylabel = 'Cumulative mismatch'
                            title = f'Cumulative mismatch: {self.mismatch:0.3f}'

                        dates = self.sim_results['date'][days] # Show these with dates, rather than days, as a reference point
                        ax.bar(dates, data, width=width, bottom=bottom[i][self.inds.sim[key]], color=colors[k], label=f'{key}')

                        if i == 0:
                            bottom.daily[self.inds.sim[key]] += self.losses[key]
                        else:
                            bottom.cumul = np.cumsum(bottom.daily)

                        if k == len(self.keys)-1:
                            ax.set_xlabel('Date')
                            ax.set_ylabel(ylabel)
                            ax.set_title(title)
                            cvpl.reset_ticks(ax=ax, date_args=date_args, start_day=self.sim_results['date'][0])
                            ax.legend()

                ts_ax = pl.subplot(n_rows, n_keys, k+1*n_keys+1)
                ts_ax.plot(days, self.pair[key].data, c='k', label='Data', **plot_args)
                ts_ax.plot(days, self.pair[key].sim, c=colors[k], label='Simulation', **plot_args)
                ts_ax.set_title(key)
                if k == 0:
                    ts_ax.set_ylabel('Time series (counts)')
                    ts_ax.legend()

                diff_ax = pl.subplot(n_rows, n_keys, k+2*n_keys+1)
                diff_ax.bar(days, self.diffs[key], width=width, color=colors[k], label='Difference')
                diff_ax.axhline(0, c='k')
                if k == 0:
                    diff_ax.set_ylabel('Differences (counts)')
                    diff_ax.legend()

                loss_ax = pl.subplot(n_rows, n_keys, k+3*n_keys+1, sharey=loss_ax)
                loss_ax.bar(days, self.losses[key], width=width, color=colors[k], label='Losses')
                loss_ax.set_xlabel(daylabel)
                loss_ax.set_title(f'Total loss: {self.losses[key].sum():0.3f}')
                if k == 0:
                    loss_ax.set_ylabel('Losses')
                    loss_ax.legend()

                if daylabel == 'Date':
                    for ax in [ts_ax, diff_ax, loss_ax]:
                        cvpl.reset_ticks(ax=ax, date_args=date_args, start_day=self.sim_results['date'][0])

        return cvpl.handle_show_return(fig=fig, do_show=do_show)


def import_optuna():
    ''' A helper function to import Optuna, which is an optional dependency '''
    try:
        import optuna as op # Import here since it's slow
    except ModuleNotFoundError as E: # pragma: no cover
        errormsg = f'Optuna import failed ({str(E)}), please install first (pip install optuna)'
        raise ModuleNotFoundError(errormsg)
    return op


class Calibration(Analyzer):
    '''
    A class to handle calibration of Covasim simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org), which must be installed separately (via
    pip install optuna).

    Note: running a calibration does not guarantee a good fit! You must ensure that
    you run for a sufficient number of iterations, have enough free parameters, and
    that the parameters have wide enough bounds. Please see the tutorial on calibration
    for more information.

    Args:
        sim          (Sim)  : the simulation to calibrate
        calib_pars   (dict) : a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
        fit_args     (dict) : a dictionary of options that are passed to sim.compute_fit() to calculate the goodness-of-fit
        par_samplers (dict) : an optional mapping from parameters to the Optuna sampler to use for choosing new points for each; by default, suggest_float
        custom_fn    (func) : a custom function for modifying the simulation; receives the sim and calib_pars as inputs, should return the modified sim
        n_trials     (int)  : the number of trials per worker
        n_workers    (int)  : the number of parallel workers (default: maximum
        total_trials (int)  : if n_trials is not supplied, calculate by dividing this number by n_workers)
        name         (str)  : the name of the database (default: 'covasim_calibration')
        db_name      (str)  : the name of the database file (default: 'covasim_calibration.db')
        keep_db      (bool) : whether to keep the database after calibration (default: false)
        storage      (str)  : the location of the database (default: sqlite)
        label        (str)  : a label for this calibration object
        die          (bool) : whether to stop if an exception is encountered (default: false)
        verbose      (bool) : whether to print details of the calibration
        kwargs       (dict) : passed to cv.Calibration()

    Returns:
        A Calibration object

    **Example**::

        sim = cv.Sim(datafile='data.csv')
        calib_pars = dict(beta=[0.015, 0.010, 0.020])
        calib = cv.Calibration(sim, calib_pars, total_trials=100)
        calib.calibrate()
        calib.plot()

    New in version 3.0.3.
    '''

    def __init__(self, sim, calib_pars=None, fit_args=None, custom_fn=None, par_samplers=None,
                 n_trials=None, n_workers=None, total_trials=None, name=None, db_name=None,
                 keep_db=None, storage=None, label=None, die=False, verbose=True):
        super().__init__(label=label) # Initialize the Analyzer object

        import multiprocessing as mp # Import here since it's also slow

        # Handle run arguments
        if n_trials  is None: n_trials  = 20
        if n_workers is None: n_workers = mp.cpu_count()
        if name      is None: name      = 'covasim_calibration'
        if db_name   is None: db_name   = f'{name}.db'
        if keep_db   is None: keep_db   = False
        if storage   is None: storage   = f'sqlite:///{db_name}'
        if total_trials is not None: n_trials = np.ceil(total_trials/n_workers)
        self.run_args   = sc.objdict(n_trials=int(n_trials), n_workers=int(n_workers), name=name, db_name=db_name, keep_db=keep_db, storage=storage)

        # Handle other inputs
        self.sim          = sim
        self.calib_pars   = calib_pars
        self.fit_args     = sc.mergedicts(fit_args)
        self.par_samplers = sc.mergedicts(par_samplers)
        self.custom_fn    = custom_fn
        self.die          = die
        self.verbose      = verbose
        self.calibrated   = False

        # Handle if the sim has already been run
        if self.sim.complete:
            warnmsg = 'Sim has already been run; re-initializing, but in future, use a sim that has not been run'
            cvm.warn(warnmsg)
            self.sim = self.sim.copy()
            self.sim.initialize()

        return


    def run_sim(self, calib_pars, label=None, return_sim=False):
        ''' Create and run a simulation '''
        sim = self.sim.copy()
        if label: sim.label = label
        valid_pars = {k:v for k,v in calib_pars.items() if k in sim.pars}
        sim.update_pars(valid_pars)
        if self.custom_fn:
            sim = self.custom_fn(sim, calib_pars)
        else:
            if len(valid_pars) != len(calib_pars):
                extra = set(calib_pars.keys()) - set(valid_pars.keys())
                errormsg = f'The following parameters are not part of the sim, nor is a custom function specified to use them: {sc.strjoin(extra)}'
                raise ValueError(errormsg)
        try:
            sim.run()
            sim.compute_fit(**self.fit_args)
            if return_sim:
                return sim
            else:
                return sim.fit.mismatch
        except Exception as E:
            if self.die:
                raise E
            else:
                warnmsg = f'Encountered error running sim!\nParameters:\n{valid_pars}\nTraceback:\n{sc.traceback()}'
                cvm.warn(warnmsg)
                output = None if return_sim else np.inf
                return output


    def run_trial(self, trial):
        ''' Define the objective for Optuna '''
        pars = {}
        for key, (best,low,high) in self.calib_pars.items():
            if key in self.par_samplers: # If a custom sampler is used, get it now
                try:
                    sampler_fn = getattr(trial, self.par_samplers[key])
                except Exception as E:
                    errormsg = 'The requested sampler function is not found: ensure it is a valid attribute of an Optuna Trial object'
                    raise AttributeError(errormsg) from E
            else:
                sampler_fn = trial.suggest_float
            pars[key] = sampler_fn(key, low, high) # Sample from values within this range
        mismatch = self.run_sim(pars)
        return mismatch


    def worker(self):
        ''' Run a single worker '''
        op = import_optuna()
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name)
        output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials)
        return output


    def run_workers(self):
        ''' Run multiple workers in parallel '''
        if self.run_args.n_workers > 1: # Normal use case: run in parallel
            output = sc.parallelize(self.worker, iterarg=self.run_args.n_workers)
        else: # Special case: just run one
            output = [self.worker()]
        return output
    
    
    def remove_db(self):
        '''
        Remove the database file if keep_db is false and the path exists.

        New in version 3.1.0.
        '''
        try:
            op = import_optuna()
            op.delete_study(study_name=self.run_args.name, storage=self.run_args.storage)
            if self.verbose:
                print(f'Deleted study {self.run_args.name} in {self.run_args.storage}')
        except Exception as E:
            print('Could not delete study, skipping...')
            print(str(E))
        if os.path.exists(self.run_args.db_name):
            os.remove(self.run_args.db_name)
            if self.verbose:
                print(f'Removed existing calibration {self.run_args.db_name}')
        return
    


    def make_study(self):
        ''' Make a study, deleting one if it already exists '''
        op = import_optuna()
        if not self.run_args.keep_db:
            self.remove_db()
        output = op.create_study(storage=self.run_args.storage, study_name=self.run_args.name)
        return output


    def calibrate(self, calib_pars=None, verbose=True, **kwargs):
        '''
        Actually perform calibration.

        Args:
            calib_pars (dict): if supplied, overwrite stored calib_pars
            verbose (bool): whether to print output from each trial
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        '''
        op = import_optuna()

        # Load and validate calibration parameters
        if calib_pars is not None:
            self.calib_pars = calib_pars
        if self.calib_pars is None:
            errormsg = 'You must supply calibration parameters either when creating the calibration object or when calling calibrate().'
            raise ValueError(errormsg)
        self.run_args.update(kwargs) # Update optuna settings

        # Run the optimization
        t0 = sc.tic()
        self.make_study()
        self.run_workers()
        self.study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name)
        self.best_pars = sc.objdict(self.study.best_params)
        self.elapsed = sc.toc(t0, output=True)

        # Compare the results
        self.initial_pars = sc.objdict({k:v[0] for k,v in self.calib_pars.items()})
        self.par_bounds   = sc.objdict({k:np.array([v[1], v[2]]) for k,v in self.calib_pars.items()})
        self.before = self.run_sim(calib_pars=self.initial_pars, label='Before calibration', return_sim=True)
        self.after  = self.run_sim(calib_pars=self.best_pars,    label='After calibration',  return_sim=True)
        self.parse_study()

        # Tidy up
        self.calibrated = True
        if not self.run_args.keep_db:
            self.remove_db()
        if verbose:
            self.summarize()

        return self


    def summarize(self):
        ''' Print out results from the calibration '''
        if self.calibrated:
            print(f'Calibration for {self.run_args.n_workers*self.run_args.n_trials} total trials completed in {self.elapsed:0.1f} s.')
            before = self.before.fit.mismatch
            after = self.after.fit.mismatch
            print('\nInitial parameter values:')
            print(self.initial_pars)
            print('\nBest parameter values:')
            print(self.best_pars)
            print(f'\nMismatch before calibration: {before:n}')
            print(f'Mismatch after calibration:  {after:n}')
            print(f'Percent improvement:         {((before-after)/before)*100:0.1f}%')
            return before, after
        else:
            print('Calibration not yet run; please run calib.calibrate()')
            return


    def parse_study(self):
        '''Parse the study into a data frame -- called automatically '''
        best = self.best_pars

        print('Making results structure...')
        results = []
        n_trials = len(self.study.trials)
        failed_trials = []
        for trial in self.study.trials:
            data = {'index':trial.number, 'mismatch': trial.value}
            for key,val in trial.params.items():
                data[key] = val
            if data['mismatch'] is None:
                failed_trials.append(data['index'])
            else:
                results.append(data)
        print(f'Processed {n_trials} trials; {len(failed_trials)} failed')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i,r in enumerate(results):
            for key in keys:
                if key not in r:
                    warnmsg = f'Key {key} is missing from trial {i}, replacing with default'
                    cvm.warn(warnmsg)
                    r[key] = best[key]
                data[key].append(r[key])
        self.data = data
        self.df = pd.DataFrame.from_dict(data)

        return


    def to_json(self, filename=None):
        '''
        Convert the data to JSON.

        New in version 3.1.1.
        '''
        order = np.argsort(self.df['mismatch'])
        json = []
        for o in order:
            row = self.df.iloc[o,:].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
            for key,val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        if filename:
            sc.savejson(filename, json, indent=2)
        else:
            return json


    def plot_sims(self, **kwargs):
        '''
        Plot sims, before and after calibration.

        New in version 3.1.1: renamed from plot() to plot_sims().
        '''
        msim = cvr.MultiSim([self.before, self.after])
        fig = msim.plot(**kwargs)
        return cvpl.handle_show_return(fig=fig)


    def plot_trend(self, best_thresh=2):
        '''
        Plot the trend in best mismatch over time.

        New in version 3.1.1.
        '''
        mismatch = sc.dcp(self.df['mismatch'].values)
        best_mismatch = np.zeros(len(mismatch))
        for i in range(len(mismatch)):
            best_mismatch[i] = mismatch[:i+1].min()
        smoothed_mismatch = sc.smooth(mismatch)
        fig = pl.figure(figsize=(16,12), dpi=120)

        ax1 = pl.subplot(2,1,1)
        pl.plot(mismatch, alpha=0.2, label='Original')
        pl.plot(smoothed_mismatch, lw=3, label='Smoothed')
        pl.plot(best_mismatch, lw=3, label='Best')

        ax2 = pl.subplot(2,1,2)
        max_mismatch = mismatch.min()*best_thresh
        inds = sc.findinds(mismatch<=max_mismatch)
        pl.plot(best_mismatch, lw=3, label='Best')
        pl.scatter(inds, mismatch[inds], c=mismatch[inds], label='Usable indices')
        for ax in [ax1, ax2]:
            pl.sca(ax)
            pl.grid(True)
            pl.legend()
            sc.setylim()
            sc.setxlim()
            pl.xlabel('Trial number')
            pl.ylabel('Mismatch')
        return cvpl.handle_show_return(fig=fig)


    def plot_all(self): # pragma: no cover
        '''
        Plot every point in the calibration. Warning, very slow for more than a few hundred trials.

        New in version 3.1.1.
        '''
        g = pairplotpars(self.data, color_column='mismatch', bounds=self.par_bounds)
        return g


    def plot_best(self, best_thresh=2): # pragma: no cover
        ''' Plot only the points with lowest mismatch. New in version 3.1.1. '''
        max_mismatch = self.df['mismatch'].min()*best_thresh
        inds = sc.findinds(self.df['mismatch'].values <= max_mismatch)
        g = pairplotpars(self.data, inds=inds, color_column='mismatch', bounds=self.par_bounds)
        return g


    def plot_stride(self, npts=200): # pragma: no cover
        '''
        Plot a fixed number of points in order across the results.

        New in version 3.1.1.
        '''
        npts = min(len(self.df), npts)
        inds = np.linspace(0, len(self.df)-1, npts).round()
        g = pairplotpars(self.data, inds=inds, color_column='mismatch', bounds=self.par_bounds)
        return g


def pairplotpars(data, inds=None, color_column=None, bounds=None, cmap='parula', bins=None, edgecolor='w', facecolor='#F8A493', figsize=(20,16)): # pragma: no cover
    ''' Plot scatterplots, histograms, and kernel densities for calibration results '''
    try:
        import seaborn as sns # Optional import
    except ModuleNotFoundError as E:
        errormsg = 'Calibration plotting requires Seaborn; please install with "pip install seaborn"'
        raise ModuleNotFoundError(errormsg) from E

    data = sc.odict(sc.dcp(data))

    # Create the dataframe
    df = pd.DataFrame.from_dict(data)
    if inds is not None:
        df = df.iloc[inds,:].copy()

    # Choose the colors
    if color_column:
        colors = sc.vectocolor(df[color_column].values, cmap=cmap)
    else:
        colors = [facecolor for i in range(len(df))]
    df['color_column'] = [sc.rgb2hex(rgba[:-1]) for rgba in colors]

    # Make the plot
    grid = sns.PairGrid(df)
    grid = grid.map_lower(pl.scatter, **{'facecolors':df['color_column']})
    grid = grid.map_diag(pl.hist, bins=bins, edgecolor=edgecolor, facecolor=facecolor)
    grid = grid.map_upper(sns.kdeplot)
    grid.fig.set_size_inches(figsize)
    grid.fig.tight_layout()

    # Set bounds
    if bounds:
        for ax in grid.axes.flatten():
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if xlabel in bounds:
                ax.set_xlim(bounds[xlabel])
            if ylabel in bounds:
                ax.set_ylim(bounds[ylabel])

    return grid



class TransTree(Analyzer):
    '''
    A class for holding a transmission tree. There are several different representations
    of the transmission tree: "infection_log" is copied from the people object and is the
    simplest representation. "detailed h" includes additional attributes about the source
    and target. If NetworkX is installed (required for most methods), "graph" includes an
    NX representation of the transmission tree.

    Args:
        sim (Sim): the sim object
        to_networkx (bool): whether to convert the graph to a NetworkX object

    **Example**::

        sim = cv.Sim().run()
        sim.run()
        tt = sim.make_transtree()
        tt.plot()
        tt.plot_histograms()

    New in version 2.1.0: ``tt.detailed`` is a dataframe rather than a list of dictionaries;
    for the latter, use ``tt.detailed.to_dict('records')``.
    '''

    def __init__(self, sim, to_networkx=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object

        # Pull out each of the attributes relevant to transmission
        attrs = {'age', 'date_exposed', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_quarantined', 'date_severe', 'date_critical', 'date_known_contact', 'date_recovered'}

        # Pull out the people and some of the sim results
        people = sim.people
        self.sim_start = sim['start_day'] # Used for filtering later
        self.sim_results = {}
        self.sim_results['t'] = sim.results['t']
        self.sim_results['cum_infections'] = sim.results['cum_infections'].values
        self.n_days = people.t  # people.t should be set to the last simulation timestep in the output (since the Transtree is constructed after the people have been stepped forward in time)
        self.pop_size = len(people)

        # Check that rescaling is not on
        if sim['rescale'] and sim['pop_scale']>1:
            warningmsg = 'Warning: transmission tree results are unreliable when' \
                         'dynamic rescaling is on, since agents are reused! Please '\
                         'rerun with rescale=False and pop_scale=1 for reliable results.'
            cvm.warn(warningmsg)

        # Include the basic line list -- copying directly is slow, so we'll make a copy later
        self.infection_log = people.infection_log

        # Parse into sources and targets
        self.sources = [None for i in range(self.pop_size)]
        self.targets = [[]   for i in range(self.pop_size)]
        self.source_dates = [None for i in range(self.pop_size)]
        self.target_dates = [[]   for i in range(self.pop_size)]

        for entry in self.infection_log:
            source = entry['source']
            target = entry['target']
            date   = entry['date']
            if source:
                self.sources[target] = source # Each target has at most one source
                self.targets[source].append(target) # Each source can have multiple targets
                self.source_dates[target] = date # Each target has at most one source
                self.target_dates[source].append(date) # Each source can have multiple targets

        # Count the number of targets each person has, and the list of transmissions
        self.count_targets()
        self.count_transmissions()

        # Include the detailed transmission tree as well, as a list and as a dataframe
        self.make_detailed(people)

        # Optionally convert to NetworkX -- must be done on import since the people object is not kept
        if to_networkx:

            # Initialization
            import networkx as nx
            self.graph = nx.DiGraph()

            # Add the nodes
            for i in range(len(people)):
                d = {}
                for attr in attrs:
                    d[attr] = people[attr][i]
                self.graph.add_node(i, **d)

            # Next, add edges from linelist
            for edge in people.infection_log:
                if edge['source'] is not None: # Skip seed infections
                    self.graph.add_edge(edge['source'],edge['target'],date=edge['date'],layer=edge['layer'])

        return


    def __len__(self):
        '''
        The length of the transmission tree is the length of the line list,
        which should equal the number of infections.
        '''
        try:
            return len(self.infection_log)
        except: # pragma: no cover
            return 0


    def day(self, day=None, which=None):
        ''' Convenience function for converting an input to an integer day '''
        if day is not None:
            day = sc.day(day, start_date=self.sim_start)
        elif which == 'start':
            day = 0
        elif which == 'end':
            day = self.n_days
        return day


    def count_targets(self, start_day=None, end_day=None):
        '''
        Count the number of targets each infected person has. If start and/or end
        days are given, it will only count the targets of people who got infected
        between those dates (it does not, however, filter on the date the target
        got infected).

        Args:
            start_day (int/str): the day on which to start counting people who got infected
            end_day (int/str): the day on which to stop counting people who got infected
        '''

        # Handle start and end days
        start_day = self.day(start_day, which='start')
        end_day   = self.day(end_day,   which='end')

        n_targets = np.nan+np.zeros(self.pop_size)
        for i in range(self.pop_size):
            if self.sources[i] is not None:
                if self.source_dates[i] >= start_day and self.source_dates[i] <= end_day:
                    n_targets[i] = len(self.targets[i])
        n_target_inds = sc.findinds(np.isfinite(n_targets))
        n_targets = n_targets[n_target_inds]
        self.n_targets = n_targets
        return n_targets


    def count_transmissions(self):
        """
        Iterable over edges corresponding to transmission events

        This excludes edges corresponding to seeded infections without a source
        """
        source_inds = []
        target_inds = []
        transmissions = []
        for d in self.infection_log:
            if d['source'] is not None:
                src = d['source']
                trg = d['target']
                source_inds.append(src)
                target_inds.append(trg)
                transmissions.append([src, trg])
        self.transmissions = transmissions
        self.source_inds = source_inds
        self.target_inds = target_inds
        return transmissions


    def make_detailed(self, people, reset=False):
        ''' Construct a detailed transmission tree, with additional information for each person '''

        def df_to_arrdict(df):
            ''' Convert a dataframe to a dictionary of arrays '''
            arrdict = {}
            for col in df.columns:
                arrdict[col] = df[col].values
            return arrdict

        # Convert infection log to a dataframe and from there to a dict of arrays
        inflog = df_to_arrdict(sc.dcp(pd.DataFrame(self.infection_log)))

        # Initialization
        n_people = len(people)
        src = 'src_'
        trg = 'trg_'
        attrs = ['age', 'date_exposed', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_severe', 'date_critical', 'date_known_contact']
        quar_attrs = ['date_quarantined', 'date_end_quarantine']
        date_attrs = [attr for attr in attrs if attr.startswith('date_')]
        is_attrs = [attr.replace('date_', 'is_') for attr in date_attrs]
        dd_arr = lambda: np.nan*np.zeros(n_people) # Create an empty array of the right size
        dd = sc.odict(defaultdict=dd_arr) # Data dictionary, to be converted to a dataframe later

        # Handle indices
        src_arr  = dd_arr()
        trg_arr  = dd_arr()
        date_arr = dd_arr()

        # Map onto arrays
        ti = np.array(inflog['target'], dtype=np.int64) # "Target indices", short since used so much
        src_arr[ti]  = inflog['source']
        trg_arr[ti]  = ti
        date_arr[ti] = inflog['date']

        # Further index wrangling
        vts_inds  = sc.findinds(np.isfinite(trg_arr) * np.isfinite(src_arr)) # Valid target-source indices
        vs_inds   = np.array(src_arr[vts_inds], dtype=np.int64) # Valid source indices
        vi        = np.array(trg_arr[vts_inds], dtype=np.int64) # Valid target indices, short since used so much
        vinfdates = date_arr[vi] # Valid target-source pair infection dates
        tinfdates = date_arr[ti] # All target infection dates

        # Populate main columns
        dd['source'][vi] = vs_inds
        dd['target'][ti] = ti
        dd['date'][ti]   = tinfdates
        dd['layer']      = np.array(dd['layer'], dtype=object)
        dd['layer'][ti]  = inflog['layer']

        # Populate from people
        for attr in attrs+quar_attrs:
            dd[trg+attr] = people[attr][:]
            dd[src+attr][vi] = people[attr][vs_inds]

        # Pull out valid indices for source and target
        lnot = np.logical_not # Shorten since used heavily
        dd[src+'is_quarantined'][vi] = (dd[src+'date_quarantined'][vi] <= vinfdates) & lnot(dd[src+'date_quarantined'][vi] <= vinfdates)
        for is_attr,date_attr in zip(is_attrs, date_attrs):
            dd[src+is_attr][vi] = np.array(dd[src+date_attr][vi] <= vinfdates, dtype=bool)

        # Populate remaining properties
        dd[src+'is_asymp'][vi] = np.isnan(dd[src+'date_symptomatic'][vi])
        dd[src+'is_presymp'][vi] = lnot(dd[src+'is_asymp'][vi]) & lnot(dd[src+'is_symptomatic'][vi])
        dd[trg+'is_quarantined'][ti] = (dd[trg+'date_quarantined'][ti] <= tinfdates) & lnot(dd[trg+'date_end_quarantine'][ti] <= tinfdates)

        # Also re-parse the log and convert to a simpler dataframe
        targets = np.array(self.target_inds)
        infdates = dd['date'][targets]
        dtr = {}
        dtr['date']      = infdates
        dtr['layer']     = dd['layer'][targets]
        dtr['s_asymp']   = np.isnan(dd['src_date_symptomatic'][targets])
        dtr['s_presymp'] = ~(dtr['s_asymp'][:]) & (infdates < dd['src_date_symptomatic'][targets])
        dtr['s_sev']     = dd['src_date_severe'][targets]       < infdates
        dtr['s_crit']    = dd['src_date_critical'][targets]     < infdates
        dtr['s_diag']    = dd['src_date_diagnosed'][targets]    < infdates
        dtr['s_quar']    = (dd['src_date_quarantined'][targets] < infdates) & lnot(dd['src_date_end_quarantine'][targets] <= infdates)
        dtr['t_quar']    = (dd['trg_date_quarantined'][targets] < infdates) & lnot(dd['trg_date_end_quarantine'][targets] <= infdates)

        df = pd.DataFrame(dtr)
        df = df.rename(columns={'date': 'Day'}) # For use in plotting
        df = df.loc[df['layer'] != 'seed_infection']

        df['Stage'] = 'Symptomatic'
        df.loc[df['s_asymp'], 'Stage'] = 'Asymptomatic'
        df.loc[df['s_presymp'], 'Stage'] = 'Presymptomatic'

        df['Severity'] = 'Mild'
        df.loc[df['s_sev'], 'Severity'] = 'Severe'
        df.loc[df['s_crit'], 'Severity'] = 'Critical'

        # Store
        self.detailed = pd.DataFrame(dd)
        self.df = df

        return


    def r0(self, recovered_only=False):
        """
        Return average number of transmissions per person

        This doesn't include seed transmissions. By default, it also doesn't adjust
        for length of infection (e.g. people infected towards the end of the simulation
        will have fewer transmissions because their infection may extend past the end
        of the simulation, these people are not included). If 'recovered_only=True'
        then the downstream transmissions will only be included for people that recover
        before the end of the simulation, thus ensuring they all had the same amount of
        time to transmit.
        """
        n_infected = []
        try:
            for i, node in self.graph.nodes.items():
                if i is None or np.isnan(node['date_exposed']) or (recovered_only and node['date_recovered']>self.n_days):
                    continue
                n_infected.append(self.graph.out_degree(i))
        except Exception as E: # pragma: no cover
            errormsg = f'Unable to compute r0 ({str(E)}): you may need to reinitialize the transmission tree with to_networkx=True'
            raise RuntimeError(errormsg)
        return np.mean(n_infected)


    def plot(self, fig_args=None, plot_args=None, do_show=None, fig=None):
        '''
        Plot the transmission tree.

        Args:
            fig_args  (dict):  passed to pl.figure()
            plot_args (dict):  passed to pl.plot()
            do_show   (bool):  whether to show the plot
            fig       (fig):   if supplied, use this figure
        '''

        fig_args = sc.mergedicts(dict(figsize=(8, 5)), fig_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.5, marker='o'), plot_args)

        if fig is None:
            fig = pl.figure(**fig_args)
        pl.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.95, wspace=0.4, hspace=0.4)
        n_rows = 2
        n_cols = 3

        def plot_quantity(key, title, i):
            dat = self.df.groupby(['Day', key]).size().unstack(key)
            ax = pl.subplot(n_rows, n_cols, i);
            dat.plot(ax=ax, legend=None, **plot_args)
            pl.legend(title=None)
            ax.set_title(title)
            sc.datenumformatter(start_date=self.sim_start, ax=ax)
            ax.set_ylabel('Count')

        to_plot = dict(
            layer    = 'Layer',
            Stage    = 'Source stage',
            s_diag   = 'Source diagnosed',
            s_quar   = 'Source quarantined',
            t_quar   = 'Target quarantined',
            Severity = 'Symptomatic source severity',
        )
        for i, (key, title) in enumerate(to_plot.items()):
            plot_quantity(key, title, i + 1)

        return cvpl.handle_show_return(fig=fig, do_show=do_show)


    def animate(self, *args, **kwargs):
        '''
        Animate the transmission tree.

        Args:
            animate    (bool):  whether to animate the plot (otherwise, show when finished)
            verbose    (bool):  print out progress of each frame
            markersize (int):   size of the markers
            sus_color  (list):  color for susceptibles
            fig_args   (dict):  arguments passed to pl.figure()
            axis_args  (dict):  arguments passed to pl.subplots_adjust()
            plot_args  (dict):  arguments passed to pl.plot()
            delay      (float): delay between frames in seconds
            colors     (list):  color of each person
            cmap       (str):   colormap for each person (if colors is not supplied)
            fig        (fig):   if supplied, use this figure

        Returns:
            fig: the figure object
        '''

        # Settings
        animate   = kwargs.get('animate', True)
        verbose   = kwargs.get('verbose', False)
        msize     = kwargs.get('markersize', 5)
        sus_color = kwargs.get('sus_color', [0.5, 0.5, 0.5])
        fig_args  = kwargs.get('fig_args', dict(figsize=(12, 8)))
        axis_args = kwargs.get('axis_args', dict(left=0.10, bottom=0.05, right=0.85, top=0.97, wspace=0.25, hspace=0.25))
        plot_args = kwargs.get('plot_args', dict(lw=1, alpha=0.5))
        delay     = kwargs.get('delay', 0.2)
        colors    = kwargs.get('colors', None)
        cmap      = kwargs.get('cmap', 'parula')
        fig       = kwargs.get('fig', None)
        if colors is None:
            colors = sc.vectocolor(self.pop_size, cmap=cmap)

        # Initialization
        n = self.n_days + 1
        frames = [list() for i in range(n)]
        tests = [list() for i in range(n)]
        diags = [list() for i in range(n)]
        quars = [list() for i in range(n)]

        # Construct each frame of the animation
        detailed = self.detailed.to_dict('records') # Convert to the old style
        for ddict in detailed:  # Loop over every person
            if np.isnan(ddict['source']):
                continue # Skip the 'None' node corresponding to seeded infections

            frame = {}
            tdq = {}  # Short for "tested, diagnosed, or quarantined"
            target_ind = ddict['target']

            if np.isfinite(ddict['date']): # If this person was infected

                source_ind = ddict['source'] # Index of the person who infected the target

                target_date = ddict['date']
                if np.isfinite(source_ind):  # Seed infections and importations won't have a source
                    source_ind = int(source_ind)
                    source_date = detailed[source_ind]['date']
                else:
                    source_ind = 0
                    source_date = 0

                # Construct this frame
                frame['x'] = [source_date, target_date]
                frame['y'] = [source_ind, target_ind]
                frame['c'] = colors[source_ind]
                frame['i'] = True  # If this person is infected
                frames[int(target_date)].append(frame)

                # Handle testing, diagnosis, and quarantine
                tdq['t'] = target_ind
                tdq['d'] = target_date
                tdq['c'] = colors[int(target_ind)]
                date_t = ddict['trg_date_tested']
                date_d = ddict['trg_date_diagnosed']
                date_q = ddict['trg_date_known_contact']
                if np.isfinite(date_t) and date_t < n:
                    tests[int(date_t)].append(tdq)
                if np.isfinite(date_d) and date_d < n:
                    diags[int(date_d)].append(tdq)
                if np.isfinite(date_q) and date_q < n:
                    quars[int(date_q)].append(tdq)

            else:
                frame['x'] = [0]
                frame['y'] = [target_ind]
                frame['c'] = sus_color
                frame['i'] = False
                frames[0].append(frame)

        # Configure plotting
        if fig is None:
            fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)
        ax = fig.add_subplot(1, 1, 1)

        # Create the legend
        ax2 = pl.axes([0.85, 0.05, 0.14, 0.9])
        ax2.axis('off')
        lcol = colors[0]
        na = np.nan  # Shorten
        pl.plot(na, na, '-', c=lcol, **plot_args, label='Transmission')
        pl.plot(na, na, 'o', c=lcol, markersize=msize, **plot_args, label='Source')
        pl.plot(na, na, '*', c=lcol, markersize=msize, **plot_args, label='Target')
        pl.plot(na, na, 'o', c=lcol, markersize=msize * 2, fillstyle='none', **plot_args, label='Tested')
        pl.plot(na, na, 's', c=lcol, markersize=msize * 1.2, **plot_args, label='Diagnosed')
        pl.plot(na, na, 'x', c=lcol, markersize=msize * 2.0, label='Known contact')
        pl.legend()

        # Plot the animation
        pl.sca(ax)
        for day in range(n):
            pl.title(f'Day: {day}')
            pl.xlim([0, n])
            pl.ylim([0, self.pop_size])
            pl.xlabel('Day')
            pl.ylabel('Person')
            flist = frames[day]
            tlist = tests[day]
            dlist = diags[day]
            qlist = quars[day]
            t_d = tdq['d']
            t_t = tdq['t']
            t_c = tdq['c']
            for f in flist:
                if verbose: print(f)
                x = f['x']
                y = f['y']
                c = f['c']
                pl.plot(x[0], y[0], 'o', c=c, markersize=msize, **plot_args)  # Plot sources
                pl.plot(x, y, '-', c=c, **plot_args)  # Plot transmission lines
                if f['i']:  # If this person is infected
                    pl.plot(x[1], y[1], '*', c=c, markersize=msize, **plot_args)  # Plot targets
            for tdq in tlist: pl.plot(t_d, t_t, 'o', c=t_c, markersize=msize * 2, fillstyle='none')  # Tested; No alpha for this
            for tdq in dlist: pl.plot(t_d, t_t, 's', c=t_c, markersize=msize * 1.2, **plot_args)  # Diagnosed
            for tdq in qlist: pl.plot(t_d, t_t, 'x', c=t_c, markersize=msize * 2.0)  # Quarantine; no alpha for this
            pl.plot([0, day], [0.5, 0.5], c='k', lw=3)  # Plot the endless march of time
            if animate:  # Whether to animate
                pl.pause(delay)

        return fig


    def plot_histograms(self, start_day=None, end_day=None, bins=None, width=0.8, fig_args=None, fig=None):
        '''
        Plots a histogram of the number of transmissions.

        Args:
            start_day (int/str): the day on which to start counting people who got infected
            end_day (int/str): the day on which to stop counting people who got infected
            bins (list): bin edges to use for the histogram
            width (float): width of bars
            fig_args (dict): passed to pl.figure()
            fig (fig): if supplied, use this figure
        '''

        # Process targets
        n_targets = self.count_targets(start_day, end_day)

        # Handle bins
        if bins is None:
            max_infections = n_targets.max()
            bins = np.arange(0, max_infections+2)

        # Analysis
        counts = np.histogram(n_targets, bins)[0]

        bins = bins[:-1] # Remove last bin since it's an edge
        total_counts = counts*bins
        n_bins = len(bins)
        index = np.linspace(0, 100, len(n_targets))
        sorted_arr = np.sort(n_targets)
        sorted_sum = np.cumsum(sorted_arr)
        sorted_sum = sorted_sum/sorted_sum.max()*100
        change_inds = sc.findinds(np.diff(sorted_arr) != 0)
        max_labels = 15 # Maximum number of ticks and legend entries to plot

        # Plotting
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        if fig is None:
            fig = pl.figure(**fig_args)
        pl.set_cmap('Spectral')
        pl.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)
        colors = sc.vectocolor(n_bins)

        pl.subplot(1,2,1)
        w05 = width*0.5
        w025 = w05*0.5
        pl.bar(bins-w025, counts, width=w05, facecolor='k', label='Number of events')
        for i in range(n_bins):
            label = 'Number of transmissions (events × transmissions per event)' if i==0 else None
            pl.bar(bins[i]+w025, total_counts[i], width=w05, facecolor=colors[i], label=label)
        pl.xlabel('Number of transmissions per person')
        pl.ylabel('Count')
        if n_bins<max_labels:
            pl.xticks(ticks=bins)
        pl.legend()
        pl.title('Numbers of events and transmissions')

        pl.subplot(2,2,2)
        total = 0
        for i in range(n_bins):
            pl.bar(bins[i:], total_counts[i], width=width, bottom=total, facecolor=colors[i])
            total += total_counts[i]
        if n_bins<max_labels:
            pl.xticks(ticks=bins)
        pl.xlabel('Number of transmissions per person')
        pl.ylabel('Number of infections caused')
        pl.title('Number of transmissions, by transmissions per person')

        pl.subplot(2,2,4)
        pl.plot(index, sorted_sum, lw=1.5, c='k', alpha=0.5)
        n_change_inds = len(change_inds)
        label_inds = np.linspace(0, n_change_inds, max_labels).round() # Don't allow more than this many labels
        for i in range(n_change_inds):
            if i in label_inds: # Don't plot more than this many labels
                label = f'Transmitted to {bins[i+1]:n} people'
            else:
                label = None
            pl.scatter([index[change_inds[i]]], [sorted_sum[change_inds[i]]], s=150, zorder=10, c=[colors[i]], label=label)
        pl.xlabel('Proportion of population, ordered by the number of people they infected (%)')
        pl.ylabel('Proportion of infections caused (%)')
        pl.legend()
        pl.ylim([0, 100])
        pl.grid(True)
        pl.title('Proportion of transmissions, by proportion of population')

        pl.axes([0.30, 0.65, 0.15, 0.2])
        berry      = [0.8, 0.1, 0.2]
        dirty_snow = [0.9, 0.9, 0.9]
        start_day  = self.day(start_day, which='start')
        end_day    = self.day(end_day, which='end')
        pl.axvspan(start_day, end_day, facecolor=dirty_snow)
        pl.plot(self.sim_results['t'], self.sim_results['cum_infections'], lw=1, c=berry)
        pl.xlabel('Day')
        pl.ylabel('Cumulative infections')

        return cvpl.handle_show_return(fig=fig)

