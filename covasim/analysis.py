"""
Additional analysis functions that are not part of the core Covasim workflow,
but which are useful for particular investigations: the analyzers (which run
during a sim, e.g. ``cv.snapshot``), ``cv.Fit`` and ``cv.Calibration`` (which
compare a sim with data), and ``cv.TransTree`` (the transmission tree).
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

from . import misc as cvm
from . import parameters as cvpar
from . import compat as cvc
from . import plotting as cvplt
from . import settings as cvset
from . import interventions as cvi
from . import run as cvr

# Lazy import (do not import unless actually used, since it is slow to load)
sns = sc.importbyname('seaborn', lazy=True)

__all__ = ['Analyzer', 'PeopleSnapshot', 'snapshot', 'age_histogram', 'daily_age_stats', 'daily_stats', 'nab_histogram',
           'Fit', 'Calibration', 'custom_pars', 'TransTree']


def make_infection_log(sim, start=None):
    """
    Return the sim's transmission log as a list of dicts with keys ``source`` (None for a seed
    infection or importation), ``target``, ``date`` (the day index), ``layer``, and ``variant``
    (the variant label). This is built from Starsim's infection log (``sim.diseases.covid.infection_log``,
    an ``ss.diseases.InfectionLog``; use its ``to_df()`` method for a dataframe).

    Args:
        sim (Sim): a sim that has been run (or is running)
        start (int): if supplied, only include the entries from this position in the log onwards, i.e. ``len(infection_log)`` at an earlier point (much faster than converting the whole log)
    """
    covid = sim.diseases.covid
    log = covid.infection_log

    # Optionally make a log of only the later entries; chunks are added whole, so any earlier length of the log is a chunk boundary
    if start:
        sublog = ss.diseases.InfectionLog(disease=log.disease, networks=log.networks)
        count = 0
        for chunk in log.chunks:
            if count >= start:
                sublog.chunks.append(chunk)
                sublog.n += len(chunk['target'])
            count += len(chunk['target'])
        for n, uids, kwargs in log.updates: # Each update applies to the entries before position n
            if n > start:
                sublog.updates.append((n - start, uids, kwargs))
        log = sublog

    # Convert to a list of dicts
    infection_log = []
    if len(log) == 0:
        return infection_log
    df = log.to_df()
    for source, target, day, layer, variant in zip(df.source.tolist(), df.target.tolist(), df.day.tolist(), df.network.tolist(), df.variant.tolist()):
        entry = dict(
            source  = None if (pd.isna(source) or source < 0) else source, # Seed infections and importations have no source
            target  = target,
            date    = day,
            layer   = layer,
            variant = covid.variant_map[variant],
        )
        infection_log.append(entry)
    return infection_log


def validate_recorded_dates(sim, requested_dates, recorded_dates, die=True):
    """
    Helper method to ensure that dates recorded by an analyzer match the ones requested.
    """
    requested_dates = sorted(list(requested_dates))
    recorded_dates = sorted(list(recorded_dates))
    if recorded_dates != requested_dates: # pragma: no cover
        errormsg = f'The dates {requested_dates} were requested but only {recorded_dates} were recorded: please check the dates fall between {sim.date(0)} and {sim.date(sim.npts-1)} and the sim was actually run'
        if die:
            raise RuntimeError(errormsg)
        else:
            print(errormsg)
    return


#%% Analyzers

class Analyzer(cvc.V3Module, ss.Analyzer):
    """
    Base class for Covasim analyzers, built on ``ss.Analyzer``.

    As in v3, an analyzer can define ``initialize(self, sim)``, ``apply(self, sim)`` (called on each
    timestep), and ``finalize(self, sim)``; or the Starsim equivalents ``init_post()``, ``step()``, and
    ``finalize()``.

    Note: a custom analyzer must not store data under the names Starsim reserves on a module
    (``t``, ``pars``, ``sim``, ``dists``, ``results``) -- use e.g. ``self.tvec`` instead of ``self.t``.
    """

    @property
    def today(self):
        """ The day the analyzer sees the people on (day 0 is the start day, before the first step), which days given to analyzers (e.g. ``days=10``) refer to; as for the results, the people as they are at the start of that day """
        return self.ti


# The per-agent states that cv.COVID derives (as properties) rather than stores
_DERIVED_STATES = ['exposed', 'infectious']


class PeopleSnapshot(sc.prettyobj):
    """
    A copy of the per-agent states of ``sim.people`` at one point in time (as stored by ``cv.snapshot``).

    Every array is full length and indexed by agent (UID): agents who have died keep their
    last values (with ``dead=True`` and ``alive=False``), so snapshots from different days can be compared
    elementwise. States are available as attributes or keys, e.g. ``people.exposed`` or ``people['age']``,
    including the v3 names ``date_exposed``, ``date_symptomatic``, etc. (the ``ti_*`` time indices, since
    the timestep is one day).

    Args:
        sim (Sim): the sim to take the snapshot from
        copy (bool): whether to copy the arrays (default); if False, the stored states are not copied, so only use the snapshot on the current timestep
    """

    def __init__(self, sim, copy=True):
        people = sim.people
        covid = sim.diseases.covid
        n = len(people.uid.raw) # Number of agents ever created (UIDs are 0..n-1, since Covasim has no births)
        self.t = sim.ti
        self._keys = []

        def add(key, arr):
            arr = np.asarray(arr)[:n]
            setattr(self, key, arr.copy() if copy else arr)
            if key not in self._keys:
                self._keys.append(key)
            return

        # People states (age, sex, alive, ...), then the disease states from the COVID module
        add('uid', people.uid.raw)
        for key, state in people.states.items():
            if '.' not in key: # Module states appear on People as e.g. "covid.exposed"; these are added below
                add(key, state.raw)
        for state in covid.state_list:
            add(state.name, state.raw)

        # States that are derived rather than stored (e.g. "infectious"); agents who have died get the default (False)
        auids = people.auids
        add('sex', (~people.female.raw).astype(int)) # v3: 0 for female, 1 for male
        for key in _DERIVED_STATES:
            values = getattr(covid, key).values
            full = np.zeros(n, dtype=values.dtype)
            full[auids] = values
            add(key, full)

        # The v3 date names, e.g. date_exposed for ti_exposed
        for key in list(self._keys):
            if key.startswith('ti_'):
                datekey = 'date_' + key[3:] # e.g. ti_exposed -> date_exposed
                if datekey not in self._keys:
                    setattr(self, datekey, getattr(self, key))
                    self._keys.append(datekey)
        return

    def __getitem__(self, key):
        """ Allow people['age'] as well as people.age """
        return getattr(self, key)

    def __contains__(self, key):
        """ Allow 'age' in people """
        return key in self._keys

    def __len__(self):
        return len(self.uid)

    def keys(self):
        """ The names of the stored states """
        return list(self._keys)

    def true(self, key):
        """ The indices of people for whom this state is true """
        return sc.findinds(self[key])

    def false(self, key):
        """ The indices of people for whom this state is false """
        return sc.findinds(~self[key].astype(bool))

    def defined(self, key):
        """ The indices of people for whom this state is not NaN """
        return sc.findinds(~np.isnan(self[key]))

    def undefined(self, key):
        """ The indices of people for whom this state is NaN """
        return sc.findinds(np.isnan(self[key]))

    def count(self, key):
        """ The number of people for whom this state is true """
        return np.count_nonzero(self[key])


class snapshot(Analyzer):
    """
    Analyzer that takes a "snapshot" of the sim.people array at specified points
    in time, and saves them to itself. To retrieve them, you can either access
    the dictionary directly, or use the get() method.

    Each snapshot is a ``cv.PeopleSnapshot``: full-length per-agent arrays, indexed by agent, with the
    same names as in v3 (e.g. ``people.exposed``, ``people.age``, ``people.date_symptomatic``).

    Args:
        days   (list): list of ints/strings/date objects, the days on which to take the snapshot
        args   (list): additional day(s)
        die    (bool): whether or not to raise an exception if a date is not found (default true)
        kwargs (dict): passed to Analyzer()

    **Example**::

        sim = cv.Sim(analyzers=cv.snapshot('2020-04-04', '2020-04-14'))
        sim.run()
        snapshot = sim.get_analyzer()
        people = snapshot.snapshots[0]            # Option 1
        people = snapshot.snapshots['2020-04-04'] # Option 2
        people = snapshot.get('2020-04-14')       # Option 3
        people = snapshot.get(34)                 # Option 4
        people = snapshot.get()                   # Option 5
    """

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
        self.start_day = sim.date(0) # Store the simulation start day
        self.days, self.dates = cvi.process_days(sim, self.days, return_dates=True) # Ensure days are in the right format
        max_snapshot_day = self.days[-1]
        max_sim_day = sim.npts - 1
        if max_snapshot_day > max_sim_day: # pragma: no cover
            errormsg = f'Cannot create snapshot for {self.dates[-1]} (day {max_snapshot_day}) because the simulation ends on {sim.date(max_sim_day)} (day {max_sim_day})'
            raise ValueError(errormsg)
        return

    def apply(self, sim):
        for ind in cvi.find_day(self.days, self.today):
            date = self.dates[ind]
            self.snapshots[date] = PeopleSnapshot(sim) # Take snapshot!
        return

    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.snapshots.keys(), die=self.die)
        return

    def get(self, key=None):
        """ Retrieve a snapshot from the given key (int, str, or date) """
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
    """
    Calculate statistics across age bins, including histogram plotting functionality.

    For each state, the histogram counts the people for whom that state's date is defined (e.g.
    ``date_exposed``), i.e. the cumulative number of people who have been (or are scheduled to be) in
    that state.

    Args:
        days    (list): list of ints/strings/date objects, the days on which to calculate the histograms (default: last day)
        states  (list): which states of people to record (default: exposed, severe, dead, tested, diagnosed)
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
    """

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
        """ Create an age histogram from an already run sim """
        if self.days is not None: # pragma: no cover
            errormsg = 'If a simulation is being analyzed post-run, no day can be supplied: only the last day of the simulation is available'
            raise ValueError(errormsg)
        self.initialize(sim)
        self.apply(sim)
        return

    def initialize(self, sim):

        # Handle days
        self.start_day = sim.date(0) # Get the start day, as a string
        self.end_day   = sim.date(sim.npts-1) # Get the end day, as a string
        if self.days is None:
            self.days = self.end_day # If no day is supplied, use the last day
        self.days, self.dates = cvi.process_days(sim, self.days, return_dates=True) # Ensure days are in the right format
        max_hist_day = self.days[-1]
        max_sim_day = sim.npts - 1
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
        for ind in cvi.find_day(self.days, self.today):
            date = self.dates[ind] # Find the date for this index
            self.hists[date] = sc.objdict() # Initialize the dictionary
            scale  = sim.current_scale # Determine current scale factor
            people = PeopleSnapshot(sim, copy=False) # All agents, including those who have died
            age    = people.age # Get the age distribution, since used heavily
            self.hists[date]['bins'] = self.bins # Copy here for convenience
            for state in self.states: # Loop over each state
                inds = people.defined(f'date_{state}') # Pull out people for which this state is defined
                self.hists[date][state] = np.histogram(age[inds], bins=self.edges)[0]*scale # Actually count the people
        return

    def finalize(self, sim):
        super().finalize()
        validate_recorded_dates(sim, requested_dates=self.dates, recorded_dates=self.hists.keys(), die=self.die)
        return

    def get(self, key=None):
        """ Retrieve a specific histogram from the given key (int, str, or date) """
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
        """ Convert cumulative histograms to windows """
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
        """
        Simple method for plotting the histograms.

        Args:
            windows (bool): whether to plot windows instead of cumulative counts
            width (float): width of bars
            color (hex or rgb): the color of the bars
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
            kwargs (dict): passed to ``cv.options.with_style()``; see that function for choices

        Returns:
            A list of figures, one per day
        """

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))

        # Initialize
        n_plots = len(self.states)
        n_rows, n_cols = sc.getrowscols(n_plots)
        figs = []

        # Handle windows and what to plot
        if windows:
            if self.window_hists is None:
                self.compute_windows()
            histsdict = self.window_hists
        else:
            histsdict = self.hists
        if not len(histsdict): # pragma: no cover
            errormsg = f'Cannot plot since no histograms were recorded (scheduled days: {self.days})'
            raise ValueError(errormsg)

        # Make the figure(s)
        with cvset.options.with_style(**kwargs):
            for date,hists in histsdict.items():
                figs += [plt.figure(**fig_args)]
                plt.subplots_adjust(**axis_args)
                bins = hists['bins']
                barwidth = width*(bins[1] - bins[0]) # Assume uniform width
                for s,state in enumerate(self.states):
                    ax = plt.subplot(n_rows, n_cols, s+1)
                    ax.bar(bins, hists[state], width=barwidth, facecolor=color, label=f'Number {state}')
                    if self.data is not None and state in self.data:
                        data = self.data[state]
                        ax.bar(bins+d_args.offset, data, width=barwidth*d_args.width, facecolor=d_args.color, label='Data')
                    ax.set_xlabel('Age')
                    ax.set_ylabel('Count')
                    ax.set_xticks(ticks=bins)
                    ax.legend()
                    preposition = 'from' if windows else 'by'
                    ax.set_title(f'Number of people {state} {preposition} {date}')

        return cvplt.handle_show_return(figs=figs)


class daily_age_stats(Analyzer):
    """
    Calculate daily counts by age, saving for each day of the simulation. Can
    plot either time series by age or a histogram over all time.

    The counts are stored in ``age_results`` (v3: ``results``, which is reserved by Starsim for
    time-series results): ``age_results[date][state]`` is the number of people in each age bin whose
    ``date_{state}`` is that day.

    Args:
        states  (list): which states of people to record (default: exposed, severe, dead, tested, diagnosed)
        edges   (list): edges of age bins to use (default: 10 year bins from 0 to 100)
        kwargs  (dict): passed to Analyzer()

    **Examples**::

        sim = cv.Sim(analyzers=cv.daily_age_stats())
        sim.run()
        daily_age = sim.get_analyzer()
        daily_age.plot()
        daily_age.plot(total=True)
    """

    def __init__(self, states=None, edges=None, **kwargs):
        super().__init__(**kwargs)
        self.edges = edges
        self.bins = None  # Age bins, calculated from edges
        self.states = states
        self.age_results = sc.odict()
        self.start_day = None
        self.df = None
        self.total_df = None
        return

    def initialize(self, sim):
        if self.states is None:
            self.states = ['exposed', 'severe', 'dead', 'tested', 'diagnosed']

        # Handle edges and age bins
        if self.edges is None:  # Default age bins
            self.edges = np.linspace(0, 100, 11)
        self.bins = self.edges[:-1]  # Don't include the last edge in the bins

        self.start_day = sim.date(0)
        return

    def apply(self, sim):
        people = PeopleSnapshot(sim, copy=False) # All agents, including those who have died
        df_entry = {}
        for state in self.states:
            inds = sc.findinds(people[f'date_{state}'], sim.ti)
            b, _ = np.histogram(people.age[inds], self.edges)
            df_entry.update({state: b * sim.current_scale})
        df_entry.update({'day':self.today, 'age': self.bins})
        self.age_results.update({sim.date(self.today): df_entry})
        return

    def to_df(self):
        """ Create dataframe totals for each day """
        mapper = {f'{k}': f'new_{k}' for k in self.states}
        df = pd.DataFrame()
        for date, k in self.age_results.items():
            df_ = pd.DataFrame(k)
            df_['date'] = date
            df_.rename(mapper, inplace=True, axis=1)
            df = pd.concat((df, df_))
        cols = list(df.columns.values)
        cols = [cols[-1]] + [cols[-2]] + cols[:-2]
        self.df = df[cols]
        return self.df

    def to_total_df(self):
        """ Create dataframe totals across days """
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
        if ('cum_diagnosed' in df.columns) and ('cum_tested' in df.columns):
            df['yield'] = df['cum_diagnosed'] / df['cum_tested'] # Test yield by age
        self.total_df = df
        return df

    def plot(self, total=False, do_show=None, fig_args=None, axis_args=None, plot_args=None,
             dateformat=None, width=0.8, color='#F8A493', **kwargs):
        """
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
        """
        if self.df is None:
            self.to_df()
        if self.total_df is None:
            self.to_total_df()

        fig_args  = sc.mergedicts(dict(figsize=(18,11)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.25, hspace=0.4), axis_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.5, marker='o'), plot_args)

        with cvset.options.with_style(**kwargs):
            nplots = len(self.states)
            nrows, ncols = sc.getrowscols(nplots)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, **fig_args)
            plt.subplots_adjust(**axis_args)

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
                        ax.plot(pd.to_datetime(df.date), df[f'new_{state}'], c=colors[a], label=label, **plot_args)
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

        return cvplt.handle_show_return(fig=fig, do_show=do_show)


class daily_stats(Analyzer):
    """
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
        sim.get_analyzer().plot()
    """

    def __init__(self, days=None, verbose=True, reporter=None, save_inds=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        self.days      = days # Converted to integer representations
        self.verbose   = verbose # Print on each timestep
        self.reporter  = reporter # Custom way of reporting the stats
        self.save_inds = save_inds # Whether to save infection log indices
        self.stats     = sc.objdict() # Store the actual stats
        self.reports   = sc.objdict() # Textual representation of the statistics
        self.log_start = 0 # The length of the transmission log at the end of the previous timestep
        return

    def initialize(self, sim):
        if self.days is None:
            self.days = np.arange(sim.npts)
        else:
            self.days = cvi.process_days(sim, self.days)

        self.keys =  ['exposed', 'infectious', 'symptomatic', 'severe', 'critical', 'known_contact', 'quarantined', 'diagnosed', 'recovered', 'dead']
        self.basekeys = ['stocks', 'trans', 'source', 'test', 'quar'] # Categories of things to plot
        self.extrakeys = ['layer_counts', 'extra']
        return

    def intersect(self, *args):
        """
        Compute the intersection between arrays of indices, handling either keys
        to precomputed indices or lists of indices. With two array inputs, simply
        performs np.intersect1d(arr1, arr2).
        """
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
        for ind in cvi.find_day(self.days, self.today):

            # Initialize
            t = sim.ti
            ppl = PeopleSnapshot(sim, copy=False) # All agents, including those who have died
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
            newinfs = sc.findinds(ppl.date_exposed == t)
            stats.trans.new_infections = len(newinfs)
            for key in ['known_contact', 'quarantined']:
                stats.trans[key] = len(self.intersect(newinfs, key))
                if not stats.trans[key]:
                    stats.empty.trans.append(key)

            # Source stats
            inflog = make_infection_log(sim, start=self.log_start) # Only the entries added since the last timestep
            infloginds = [i for i,e in enumerate(inflog) if (e['date']==t and e['source'] is not None)] # Person was infected today and was not a seed infection
            sourceinds = list(set([inflog[i]['source'] for i in infloginds]))
            stats.source.new_sources = len(sourceinds)
            for key in self.keys:
                stats.source[key] = len(self.intersect(sourceinds, key))
                if not stats.source[key]:
                    stats.empty.source.append(key)

            # Testing stats
            newtests = sc.findinds(ppl.date_tested == t)
            stats.test.new_tests = len(newtests)
            for key in self.keys:
                stats.test[key] = len(self.intersect(newtests,key))
                if not stats.test[key]:
                    stats.empty.test.append(key)

            # Quarantine stats
            q_inds = np.union1d(self.inds['quarantined'], sc.findinds(ppl.date_end_quarantine == t)) # Append people who finished quarantine today
            eq_inds = sc.findinds(ppl.date_quarantined == t-1) # People entering quarantine the day before (their first full day of quarantine)
            fq_inds = sc.findinds(ppl.date_end_quarantine == t+1) # People finishing quarantine; +1 since on the date of quarantine end, they are released back and can get infected at normal rates
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
            today = sim.date(t)
            self.stats[today] = stats
            self.reports[today] = report

            if self.verbose:
                self.report(today)

        self.log_start = len(sim.diseases.covid.infection_log) # So the next timestep only converts the new entries of the transmission log
        return

    def report(self, day=None):
        """ Print out one or all reports -- take a date string, or an int for the position in the list of reports (not a day of the sim) """
        if day is None:
            print(self.reports)
        else:
            print(self.reports[day])
        return

    def make_report(self, sim, stats, show_empty='count'):
        """ Turn the statistics into a report """

        def make_entry(basekey, show_empty=show_empty):
            """ For each key, print the key and the count if the count is >0, and optionally any empty states """
            string  = '\n'.join([f'  {k:13s} = {v}' for k,v in stats[basekey].items() if v>0])
            if show_empty is True:
                string += f'\n  Empty states: {stats.empty[basekey]}'
            elif show_empty == 'count':
                string += f'\n  Number of empty states: {len(stats.empty[basekey])}'
            string = '\n' + string + '\n'
            return string

        datestr = f'day {self.today} ({sim.date(self.today)})'
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
        """ Transpose the data from a list-of-dicts-of-dicts to a dict-of-dicts-of-lists """
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
        """
        Plot the daily statistics recorded. Some overlap with e.g. ``sim.plot(to_plot='overview')``.

        Args:
            fig_args  (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            plot_args (dict): passed to pl.plot()
            do_show   (bool): whether to show the plot
            kwargs    (dict): passed to ``cv.options.with_style()``
        """

        fig_args  = sc.mergedicts(dict(figsize=(18,11)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.25, hspace=0.4), axis_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.5, marker='o'), plot_args)

        # Transform the data into time series
        data = self.transpose()

        # Do the plotting
        with cvset.options.with_style(**kwargs):
            nplots = sum([len(data[k].keys()) for k in data.keys()]) # Figure out how many plots there are
            nrows,ncols = sc.getrowscols(nplots)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, **fig_args)
            plt.subplots_adjust(**axis_args)

            count = -1
            for k1 in data.keys():
                for k2 in data[k1].keys():
                    count += 1
                    row,col = np.unravel_index(count, (nrows,ncols))
                    ax = axs[row,col]
                    y = data[k1][k2]
                    ax.plot(y, **plot_args)
                    ax.set_title(f'{k1}: {k2}')

        return cvplt.handle_show_return(fig=fig, do_show=do_show)


class nab_histogram(Analyzer):
    """
    Store histogram of log_{10}(NAb) distribution

    Args:
        days (list): days on which calculate the NAb histogram (if None, assume last day)
        edges (list): log10 bin edges for histogram

    **Example**::

        sim = cv.Sim(analyzers=cv.nab_histogram())
        sim.run()
        sim.get_analyzer().plot()

    New in version 3.1.0.
    """
    def __init__(self, days=None, edges=None, **kwargs):
        super().__init__(**kwargs)  # Initialize the Analyzer object
        self.days = days  # To be converted to integer representations
        self.edges = edges  # Edges of age bins in log10
        self.hists = sc.odict()  # Store the actual snapshots
        return

    def initialize(self, sim):

        # Check that the simulation parameters are correct
        if not sim.diseases.covid.pars.use_waning:
            errormsg = 'The cv.nab_histogram() analyzer requires use_waning=True. Please enable waning.'
            raise RuntimeError(errormsg)

        # Handle days
        self.start_day = sim.date(0) # Get the start day, as a string
        self.end_day   = sim.date(sim.npts-1) # Get the end day, as a string
        if self.days is None:
            self.days = self.end_day  # If no day is supplied, use the last day
        self.days, self.dates = cvi.process_days(sim, self.days, return_dates=True) # Ensure days are in the right format

        # Handle edges and nab bins
        if self.edges is None:  # Default  bins
            self.edges = np.arange(-4, 3)
        self.bins = self.edges[:-1]  # Don't include the last edge in the bins
        return

    def apply(self, sim):
        for ind in cvi.find_day(self.days, self.today):
            nab = PeopleSnapshot(sim, copy=False).nab # All agents, including those who have died
            log_nabs = np.log10(nab[nab > 0])
            date = self.dates[ind]  # Find the date for this index
            self.hists[date] = sc.objdict()  # Initialize the dictionary
            scale = sim.current_scale  # Determine current scale factor
            self.hists[date]['bins'] = self.bins  # Copy here for convenience
            self.hists[date]['n'] = np.histogram(log_nabs, bins=self.edges)[0] * scale  # Actually count the people
            self.hists[date]['s'] = np.std(log_nabs)    # keep the std
            self.hists[date]['m'] = np.mean(log_nabs)   # keep the mean
        return

    def plot(self, fig_args=None, axis_args=None, plot_args=None, do_show=None, **kwargs):
        """
        Plot the results

        Args:
            fig_args  (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            plot_args (dict): passed to pl.plot()
            do_show   (bool): whether to show the plot
            kwargs    (dict): passed to ``cv.options.with_style()``
        """

        fig_args  = sc.mergedicts(dict(figsize=(9,5)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.10, right=0.95, bottom=0.10, top=0.95, wspace=0.25, hspace=0.4), axis_args)
        plot_args = sc.mergedicts(dict(lw=2), plot_args)

        with cvset.options.with_style(**kwargs):
            fig, axs = plt.subplots(nrows=1, ncols=1, **fig_args)
            plt.subplots_adjust(**axis_args)
            for date, hist in self.hists.items():
                axs.stairs(hist['n'], edges=self.edges, label=date, **plot_args)
            axs.set_xlabel('Log10(NAb)')
            axs.set_ylabel('Count')
            axs.legend()

        return cvplt.handle_show_return(fig=fig, do_show=do_show)


#%% Fitting and calibration

class Fit(sc.prettyobj):
    """
    A class for calculating the fit between the model and the data. Note the
    following terminology is used here:

        - fit: nonspecific term for how well the model matches the data
        - difference: the absolute numerical differences between the model and the data (one time series per result)
        - goodness-of-fit: the result of passing the difference through a statistical function, such as mean squared error
        - loss: the goodness-of-fit for each result multiplied by user-specified weights (one time series per result)
        - mismatches: the sum of all the losses (a single scalar value per time series)
        - mismatch: the sum of the mismatches -- this is the value to be minimized during calibration

    Args:
        sim (Sim): the sim object (after it has been run)
        weights (dict): the relative weight to place on each result (by default: 10 for deaths, 5 for diagnoses, 1 for everything else)
        keys (list): the keys to use in the calculation
        custom (dict): a custom dictionary of additional data to fit; format is e.g. {'my_output':{'data':[1,2,3], 'sim':[1,2,4], 'weights':2.0}}
        compute (bool): whether to compute the mismatch immediately
        verbose (bool): detail to print
        die (bool): whether to raise an exception if no data are supplied
        label (str): the label for the fit
        data (DataFrame): the data to fit, indexed by date or by integer day, with columns named like the sim results (e.g. ``cum_deaths``); by default, the sim's data (``sim.data``)
        kwargs (dict): passed to cv.compute_gof() -- see this function for more detail on goodness-of-fit calculation options

    **Example**::

        sim = cv.Sim(datafile='my-data-file.csv')
        sim.run()
        fit = sim.compute_fit()
        fit.plot()
    """

    def __init__(self, sim, weights=None, keys=None, custom=None, compute=True, verbose=False, die=True, label=None, data=None, **kwargs):

        # Handle inputs
        self.weights    = sc.mergedicts({'cum_deaths':10, 'cum_diagnoses':5}, weights)
        self.user_keys  = keys
        self.custom     = sc.mergedicts(custom)
        self.verbose    = verbose
        self.die        = die
        self.label      = label
        self.gof_kwargs = kwargs

        # Copy data
        if data is None:
            data = getattr(sim, 'data', None)
        if (data is None) or (len(data) == 0):
            if self.die and not self.custom:
                errormsg = 'Model fit cannot be calculated until data are loaded (or custom data are supplied)'
                raise RuntimeError(errormsg)
            data = pd.DataFrame() # Use an empty dataframe
        self.data = data

        # Copy sim results: the 1D results, plus the time points and dates
        self.sim_results = sc.objdict()
        for key in sim.results.keys():
            arr = getattr(sim.results[key], 'values', None)
            if (arr is not None) and (np.ndim(arr) == 1): # Skip nested results, e.g. by variant
                self.sim_results[key] = arr
        self.sim_dates = list(sim.datevec)
        self.sim_npts = len(self.sim_dates) # Number of time points in the sim
        self.sim_results['t'] = np.arange(self.sim_npts)
        self.sim_results['date'] = np.array(self.sim_dates)

        # These are populated during the computation
        self.keys         = None # The result keys being fit
        self.custom_keys  = list(self.custom.keys())
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
        """ Perform all required computations """
        self.reconcile_inputs() # Find matching values
        self.compute_diffs() # Perform calculations
        self.compute_gofs()
        self.compute_losses()
        self.compute_mismatch()
        return self.mismatch

    def reconcile_inputs(self):
        """ Find matching keys and indices between the model and the data """
        data_cols = list(self.data.columns) if len(self.data) else []
        if self.user_keys is None:
            sim_keys = [k for k in self.sim_results.keys() if k.startswith('cum_')] # Default sim keys, only keep cumulative keys if no keys are supplied
            self.keys = [k for k in sim_keys if k in data_cols] # Keys present in both the sim and the data
        else:
            self.keys = sc.tolist(self.user_keys)
            missing = [k for k in self.keys if k not in data_cols]
            if missing:
                errormsg = f'The following requested key(s) were not found in the data: {sc.strjoin(missing)}'
                if self.die:
                    raise sc.KeyNotFoundError(errormsg)
                cvm.warn(errormsg)
                self.keys = [k for k in self.keys if k not in missing] # Skip the missing keys

        # Map each data row (indexed by date or by integer day) to a sim time index
        start_day = self.sim_dates[0]
        matches = 0 # Count how many data points match
        for key in self.keys:
            sim_inds = []
            data_inds = []
            self.date_matches[key] = []
            series = self.data[key]
            for pos, (d, datum) in enumerate(series.items()):
                if np.isfinite(datum):
                    day = sc.day(d, start_date=start_day) # Raises an error if the date can't be interpreted
                    if (day >= 0) and (day < self.sim_npts):
                        self.date_matches[key].append(d)
                        sim_inds.append(day)
                        data_inds.append(pos)
            self.inds.sim[key] = np.array(sim_inds, dtype=int)
            self.inds.data[key] = np.array(data_inds, dtype=int)
            self.pair[key] = sc.objdict(
                sim  = self.sim_results[key][self.inds.sim[key]].astype(float),
                data = series.values[self.inds.data[key]].astype(float),
            )
            matches += len(sim_inds)

        # Process custom inputs: paired directly, with no date matching
        for key, custom in self.custom.items():
            if 'sim' not in custom or 'data' not in custom:
                raise sc.KeyNotFoundError(f'Custom input {key!r} must have "sim" and "data" keys.')
            if key in self.pair:
                errormsg = f'You cannot use a custom key "{key}" that matches one of the existing keys: {self.pair.keys()}'
                raise ValueError(errormsg)
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
        """ Find the differences between the sim and the data """
        for key in self.pair.keys():
            d = self.pair[key].sim - self.pair[key].data
            self.diffs[key] = np.abs(d) if absolute else d
        return

    def compute_gofs(self, **kwargs):
        """ Compute the goodness-of-fit """
        kwargs = sc.mergedicts(self.gof_kwargs, kwargs)
        for key in self.pair.keys():
            self.gofs[key] = cvm.compute_gof(self.pair[key].data, self.pair[key].sim, **kwargs)
        return

    def compute_losses(self):
        """ Compute the weighted goodness-of-fit """
        for key in self.gofs.keys():
            weight = self.weights.get(key, 1.0)
            if sc.isiterable(weight): # It's an array
                weight = np.asarray(weight)
                len_wt = len(weight)
                len_sim = self.sim_npts
                len_match = len(self.pair[key].data)
                if len_wt == len_match: # If the weight already is the right length, do nothing
                    pass
                elif (len_wt == len_sim) and (key in self.inds.sim): # Most typical case: it's the length of the simulation, must trim
                    weight = weight[self.inds.sim[key]] # Trim to matching indices
                else:
                    errormsg = f'Could not map weight array of length {len_wt} onto simulation of length {len_sim} or data-model matches of length {len_match}'
                    raise ValueError(errormsg)
            self.losses[key] = self.gofs[key] * weight
        return

    def compute_mismatch(self, use_median=False):
        """ Compute the final mismatch """
        for key in self.losses.keys():
            self.mismatches[key] = np.median(self.losses[key]) if use_median else np.sum(self.losses[key])
        self.mismatch = float(np.sum([v for v in self.mismatches.values()]))
        return self.mismatch

    def summarize(self):
        """ Print out results from the fit """
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
        """
        Plot the fit of the model to the data. For each result, plot the data
        and the model; the difference; and the loss (weighted difference). Also
        plots the loss as a function of time.

        Args:
            keys      (list):  which keys to plot (default, all)
            width     (float): bar width
            fig_args  (dict):  passed to ``plt.figure()``
            axis_args (dict):  passed to ``plt.subplots_adjust()``
            plot_args (dict):  passed to ``plt.plot()``
            date_args (dict):  passed to ``cv.plotting.reset_ticks()`` (handle date format, rotation, etc.)
            do_show   (bool):  whether to show the plot
            fig       (fig):   if supplied, use this figure to plot in
            kwargs    (dict):  passed to ``cv.options.with_style()``

        Returns:
            Figure object
        """

        fig_args  = sc.mergedicts(dict(figsize=(18,11)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3), axis_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.5, marker='o'), plot_args)
        date_args = sc.mergedicts(dict(as_dates=True, dateformat=None, rotation=None, start=None, end=None), date_args)

        if keys is None:
            keys = self.keys + self.custom_keys
        keys = sc.tolist(keys)
        n_keys = len(keys)

        loss_ax = None
        colors = sc.gridcolors(n_keys)
        n_rows = 4
        dates = self.sim_results['date']
        start_day = self.sim_dates[0]

        # Plot
        with cvset.options.with_style(**kwargs):
            if fig is None:
                fig = plt.figure(**fig_args)
            fig.subplots_adjust(**axis_args)
            main_ax1 = fig.add_subplot(n_rows, 2, 1)
            main_ax2 = fig.add_subplot(n_rows, 2, 2)
            bottom = np.zeros(self.sim_npts) # Keep track of the bottoms of the stacked daily losses
            for k,key in enumerate(keys):
                if key in self.keys: # It's a time series, plot with dates
                    x = dates[self.inds.sim[key]]
                    daylabel = 'Date'
                else: # It's custom, we don't know what it is
                    x = np.arange(len(self.losses[key])) # Just use indices
                    daylabel = 'Index'

                # Stacked daily and cumulative losses; skip custom keys, since they can't be combined with daily inputs
                if key in self.keys:
                    inds = self.inds.sim[key]
                    cum_bottom = np.cumsum(bottom)
                    main_ax1.bar(x, self.losses[key], width=width, bottom=bottom[inds], color=colors[k], label=f'{key}')
                    bottom[inds] += self.losses[key]
                    main_ax2.bar(x, np.cumsum(self.losses[key]), width=width, bottom=cum_bottom[inds], color=colors[k], label=f'{key}')

                ts_ax = fig.add_subplot(n_rows, n_keys, k+1*n_keys+1)
                ts_ax.plot(x, self.pair[key].data, c='k', label='Data', **plot_args)
                ts_ax.plot(x, self.pair[key].sim, c=colors[k], label='Simulation', **plot_args)
                ts_ax.set_title(key)
                if k == 0:
                    ts_ax.set_ylabel('Time series (counts)')
                    ts_ax.legend()

                diff_ax = fig.add_subplot(n_rows, n_keys, k+2*n_keys+1)
                diff_ax.bar(x, self.diffs[key], width=width, color=colors[k], label='Difference')
                diff_ax.axhline(0, c='k')
                if k == 0:
                    diff_ax.set_ylabel('Differences (counts)')
                    diff_ax.legend()

                loss_ax = fig.add_subplot(n_rows, n_keys, k+3*n_keys+1, sharey=loss_ax)
                loss_ax.bar(x, self.losses[key], width=width, color=colors[k], label='Losses')
                loss_ax.set_xlabel(daylabel)
                loss_ax.set_title(f'Total loss: {np.sum(self.losses[key]):0.3f}')
                if k == 0:
                    loss_ax.set_ylabel('Losses')
                    loss_ax.legend()

                if daylabel == 'Date':
                    for ax in [ts_ax, diff_ax, loss_ax]:
                        cvplt.reset_ticks(ax=ax, date_args=date_args, start_day=start_day)

            # Label the overall loss axes
            for ax, ylabel, title in zip([main_ax1, main_ax2], ['Daily mismatch', 'Cumulative mismatch'], ['Daily total mismatch', f'Cumulative mismatch: {self.mismatch:0.3f}']):
                ax.set_xlabel('Date')
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                cvplt.reset_ticks(ax=ax, date_args=date_args, start_day=start_day)
                ax.legend()

        return cvplt.handle_show_return(fig=fig, do_show=do_show)


class Calibration(sc.prettyobj):
    """
    A class to handle calibration of Covasim simulations, with Starsim's ``sim.fit()`` (using the Optuna
    hyperparameter optimization library, optuna.org).

    Each parameter to calibrate is given its bounds (as ``ss.par``) in a copy of the sim, which ``sim.fit()``
    then varies, minimizing the mismatch with the data (``sim.compute_fit()``). A parameter that is not one of
    the sim's (e.g. part of an intervention) is set by ``custom_fn`` once each trial's sim is initialized.

    Note: running a calibration does not guarantee a good fit! You must ensure that
    you run for a sufficient number of iterations, have enough free parameters, and
    that the parameters have wide enough bounds. Please see the tutorial on calibration
    for more information.

    Args:
        sim          (Sim)  : the simulation to calibrate (ideally not yet initialized, so that parameters used during initialization, such as pop_infected, can be calibrated)
        calib_pars   (dict) : a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
        fit_args     (dict) : a dictionary of options that are passed to sim.compute_fit() to calculate the goodness-of-fit (e.g. data, weights, keys)
        custom_fn    (func) : a custom function for modifying the simulation; receives the sim and calib_pars as inputs, should return the modified sim. The sim has been initialized, so e.g. sim.get_intervention() can be used.
        n_trials     (int)  : the number of trials per worker (default 20)
        n_workers    (int)  : the number of parallel workers (default: the number of CPUs)
        total_trials (int)  : if supplied, the total number of trials, divided between the workers (overrides n_trials)
        storage      (str)  : a file to keep the trials in, so that calibrating again resumes the calibration (see ``sim.fit()``)
        label        (str)  : a label for this calibration object
        verbose      (bool) : whether to print details of the calibration
        data         (df)   : shortcut for fit_args['data']: the data to fit to (default: the sim's data)
        weights      (dict) : shortcut for fit_args['weights']: the weight of each result key
        fit_kw       (dict) : alias for fit_args

    **Example**::

        sim = cv.Sim(datafile='data.csv')
        calib_pars = dict(beta=[0.015, 0.010, 0.020])
        calib = cv.Calibration(sim, calib_pars, total_trials=100)
        calib.calibrate()
        calib.plot_sims()
    """

    def __init__(self, sim, calib_pars=None, fit_args=None, custom_fn=None, n_trials=None, n_workers=None,
                 total_trials=None, storage=None, label=None, verbose=True, data=None, weights=None, fit_kw=None):
        self.sim = sim.new() if sim.initialized else sim # Calibrated by making it again, with other values, so it can't have been initialized
        self.calib_pars = calib_pars
        self.custom_fn = custom_fn
        self.n_workers = sc.ifelse(n_workers, sc.cpu_count())
        self.total_trials = sc.ifelse(total_trials, sc.ifelse(n_trials, 20)*self.n_workers) # v3 gave the number of trials per worker
        self.storage = storage
        self.label = label
        self.verbose = verbose
        self.fit_args = sc.mergedicts(fit_kw, fit_args)
        if data    is not None: self.fit_args['data']    = data
        if weights is not None: self.fit_args['weights'] = weights
        self.calibrated = False
        return

    def make_sim(self, calib_pars, label=None):
        """
        The sim made again (see ``sim.new()``) with these values, or bounds, of the parameters to calibrate:
        dict(key=value), or dict(key=ss.par(best, bounds=[low, high])). Those that are not the sim's are given
        to ``custom_fn`` once the sim is initialized, by a ``cv.custom_pars`` connector (which holds them as its
        parameters, so that ``sim.fit()`` can vary them).
        """
        sim_keys = set(cvpar.make_pars()) | set(self.sim._v3_covid().pars)
        sim_pars = {key:val for key,val in calib_pars.items() if key in sim_keys}
        custom = {key:val for key,val in calib_pars.items() if key not in sim_keys}
        sim = self.sim.new(**sim_pars)
        if custom:
            if self.custom_fn is None:
                errormsg = f'The following parameters are not part of the sim, nor is a custom function specified to use them: {sc.strjoin(custom)}'
                raise ValueError(errormsg)
            sim.pars.connectors = sc.tolist(sim.pars.connectors) + [custom_pars(self.custom_fn, **custom)] # Kept by sim.new(), and so in each trial of sim.fit()
        if label:
            sim.label = label
        return sim

    def mismatch(self, sim):
        """ The goodness of fit of a sim that has run: the mismatch of ``sim.compute_fit()`` """
        return sim.compute_fit(**self.fit_args).mismatch

    def run_sim(self, calib_pars, label=None):
        """ Run a sim with these values of the parameters, and compute its fit; returns the sim and its fit """
        sim = self.make_sim(calib_pars, label=label)
        sim.run()
        return sim, sim.compute_fit(**self.fit_args)

    def calibrate(self, calib_pars=None, verbose=None, **kwargs):
        """
        Actually perform calibration.

        Args:
            calib_pars (dict): if supplied, overwrite stored calib_pars
            verbose (bool): whether to print a summary of the results
            kwargs (dict): passed to ``sim.fit()``, e.g. n_trials=200
        """
        if calib_pars is not None:
            self.calib_pars = calib_pars
        if self.calib_pars is None:
            errormsg = 'You must supply calibration parameters either when creating the calibration object or when calling calibrate().'
            raise ValueError(errormsg)
        verbose = sc.ifelse(verbose, self.verbose)

        # Run the optimization
        bounded = {key:ss.par(best, bounds=[low, high]) for key,(best,low,high) in self.calib_pars.items()}
        sim = self.make_sim(bounded)
        kwargs = sc.mergedicts(dict(n_trials=self.total_trials, n_workers=self.n_workers, storage=self.storage), kwargs)
        self.fit = sim.fit(gof=self.mismatch, method='optuna', verbose=verbose, **kwargs)
        self.best_pars = sc.objdict({key:val for name,val in self.fit.pars.items() for key in self.calib_pars if name.split('.')[-1] == key}) # Named e.g. "custom_pars.test_prob" if the name is also another module's parameter
        self.df = self.fit.trials # In trial order
        self.calibrated = True

        # Compare the results
        self.initial_pars = sc.objdict({k:v[0] for k,v in self.calib_pars.items()})
        self.par_bounds   = sc.objdict({k:np.array([v[1], v[2]]) for k,v in self.calib_pars.items()})
        self.before, self.before_fit = self.run_sim(self.initial_pars, label='Before calibration')
        self.after,  self.after_fit  = self.run_sim(self.best_pars,    label='After calibration')
        if verbose:
            self.summarize()
        return self

    def summarize(self):
        """ Print out results from the calibration """
        if self.calibrated:
            print(f'Calibration for {len(self.df)} total trials completed.')
            before = self.before_fit.mismatch
            after = self.after_fit.mismatch
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

    def trials(self):
        """ The parameter values and mismatch of each trial, as a dataframe with the parameters named as in calib_pars """
        names = {name:key for name in self.df.columns for key in self.calib_pars if name.split('.')[-1] == key}
        return self.df.rename(columns=dict(names, objective='mismatch'))

    def check_fit(self, do_plot=True):
        """
        Check whether calibration improved the fit, by comparing the mismatch of the sims run before and
        after calibration (see also summarize()).

        Args:
            do_plot (bool): whether to plot the sims before and after calibration

        Returns:
            True if the fit improved
        """
        if not self.calibrated:
            errormsg = 'Calibration not yet run; please run calib.calibrate()'
            raise RuntimeError(errormsg)
        before = self.before_fit.mismatch
        after = self.after_fit.mismatch
        print(f'Mismatch before calibration: {before:n}')
        print(f'Mismatch after calibration:  {after:n}')
        if do_plot:
            self.plot_sims()
        return after <= before

    def plot_sims(self, **kwargs):
        """
        Plot sims, before and after calibration.

        Args:
            kwargs (dict): passed to MultiSim.plot(), e.g. to_plot
        """
        msim = cvr.MultiSim([self.before, self.after])
        fig = msim.plot(**kwargs)
        return cvplt.handle_show_return(fig=fig)

    def plot(self, **kwargs):
        """ Alias to plot_sims() """
        return self.plot_sims(**kwargs)

    def plot_final(self, **kwargs):
        """ Alias to plot_sims() """
        return self.plot_sims(**kwargs)

    def plot_trend(self, best_thresh=2):
        """
        Plot the trend in best mismatch over time.

        Args:
            best_thresh (float): in the lower panel, show the trials with a mismatch within this factor of the best mismatch
        """
        mismatch = self.df['objective'].values.copy() # The mismatch of each trial
        best_mismatch = np.zeros(len(mismatch))
        for i in range(len(mismatch)):
            best_mismatch[i] = mismatch[:i+1].min()
        smoothed_mismatch = sc.smooth(mismatch)
        fig = plt.figure(figsize=(16,12), dpi=120)

        ax1 = plt.subplot(2,1,1)
        plt.plot(mismatch, alpha=0.2, label='Original')
        plt.plot(smoothed_mismatch, lw=3, label='Smoothed')
        plt.plot(best_mismatch, lw=3, label='Best')

        ax2 = plt.subplot(2,1,2)
        max_mismatch = mismatch.min()*best_thresh
        inds = sc.findinds(mismatch<=max_mismatch)
        plt.plot(best_mismatch, lw=3, label='Best')
        plt.scatter(inds, mismatch[inds], c=mismatch[inds], label='Usable indices')
        for ax in [ax1, ax2]:
            plt.sca(ax)
            plt.grid(True)
            plt.legend()
            sc.setylim()
            sc.setxlim()
            plt.xlabel('Trial number')
            plt.ylabel('Mismatch')
        return cvplt.handle_show_return(fig=fig)

    def plot_all(self): # pragma: no cover
        """ Plot every point in the calibration. Warning, very slow for more than a few hundred trials. """
        g = pairplotpars(self.trials(), color_column='mismatch', bounds=self.par_bounds)
        return g

    def plot_best(self, best_thresh=2): # pragma: no cover
        """
        Plot only the points with lowest mismatch.

        Args:
            best_thresh (float): plot the trials with a mismatch within this factor of the best mismatch
        """
        max_mismatch = self.df['objective'].min()*best_thresh
        inds = sc.findinds(self.df['objective'].values <= max_mismatch)
        g = pairplotpars(self.trials(), inds=inds, color_column='mismatch', bounds=self.par_bounds)
        return g

    def plot_stride(self, npts=200): # pragma: no cover
        """
        Plot a fixed number of points in order across the results.

        Args:
            npts (int): the number of points to plot
        """
        npts = min(len(self.df), npts)
        inds = np.linspace(0, len(self.df)-1, npts).round()
        g = pairplotpars(self.trials(), inds=inds, color_column='mismatch', bounds=self.par_bounds)
        return g


class custom_pars(ss.Connector):
    """
    Parameters to calibrate that are not the sim's (e.g. part of an intervention), held as the parameters of this
    module, so that ``sim.fit()`` can vary them, and set by a function of the sim once it is initialized (the
    ``custom_fn`` of ``cv.Calibration``, called as ``fn(sim, pars)``)
    """
    def __init__(self, fn, **pars):
        super().__init__()
        self.define_pars(**pars)
        self.fn = fn
        return

    def init_post(self):
        super().init_post()
        self.fn(self.sim, dict(self.pars))
        return


def pairplotpars(data, inds=None, color_column=None, bounds=None, cmap='parula', bins=None, edgecolor='w', facecolor='#F8A493', figsize=(20,16)): # pragma: no cover
    """ Plot scatterplots, histograms, and kernel densities for calibration results (requires Seaborn) """
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
    grid = grid.map_lower(plt.scatter, **{'facecolors':df['color_column']})
    grid = grid.map_diag(plt.hist, bins=bins, edgecolor=edgecolor, facecolor=facecolor)
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


#%% Transmission tree

class TransTree(Analyzer):
    """
    A class for holding a transmission tree. There are several different representations
    of the transmission tree: "infection_log" is copied from the sim and is the
    simplest representation. "detailed" includes additional attributes about the source
    and target. If NetworkX is installed (required for most methods), "graph" includes an
    NX representation of the transmission tree.

    The transmission log is always recorded by the COVID module (``sim.diseases.covid.infection_log``),
    so the tree can be made from any sim after it has run. It can also be added as an analyzer
    (``analyzers=cv.TransTree()``), in which case it is made when the sim finishes.

    Args:
        sim (Sim): the sim object (if None, make the tree at the end of the run the analyzer is part of)
        to_networkx (bool): whether to convert the graph to a NetworkX object

    **Example**::

        sim = cv.Sim()
        sim.run()
        tt = sim.make_transtree()
        tt.plot()
        tt.plot_histograms()

    New in version 2.1.0: ``tt.detailed`` is a dataframe rather than a list of dictionaries;
    for the latter, use ``tt.detailed.to_dict('records')``.
    """

    def __init__(self, sim=None, to_networkx=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        self.to_networkx = to_networkx
        self.infection_log = None
        if sim is not None:
            self.make_tree(sim)
        return

    def finalize(self):
        super().finalize()
        self.make_tree(self.sim)
        return

    def make_tree(self, sim):
        """ Make the transmission tree from a sim that has been run """

        # Pull out each of the attributes relevant to transmission
        attrs = {'age', 'date_exposed', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_quarantined', 'date_severe', 'date_critical', 'date_known_contact', 'date_recovered'}

        # Pull out the people and some of the sim results
        people = PeopleSnapshot(sim, copy=False) # All agents, including those who have died
        self.sim_start = sim.date(0) # Used for filtering later
        self.sim_results = {}
        self.sim_results['t'] = np.arange(sim.npts)
        self.sim_results['cum_infections'] = sim.diseases.covid.results['cum_infections'].values # Not sim.results, since this may be called before these are made at the end of the run
        self.n_days = sim.ti  # The last simulation timestep (since the TransTree is constructed after the sim has been run)
        self.pop_size = len(people)

        # Check that rescaling is not on
        if sim['rescale'] and (sim['pop_scale'] > 1):
            warnmsg = 'Transmission tree results are unreliable when dynamic rescaling is on, since agents are reused! ' \
                      'Please rerun with rescale=False and pop_scale=1 for reliable results.'
            cvm.warn(warnmsg)

        # Include the basic line list
        self.infection_log = make_infection_log(sim)

        # Parse into sources and targets
        self.sources = [None for i in range(self.pop_size)]
        self.targets = [[]   for i in range(self.pop_size)]
        self.source_dates = [None for i in range(self.pop_size)]
        self.target_dates = [[]   for i in range(self.pop_size)]

        for entry in self.infection_log:
            source = entry['source']
            target = entry['target']
            date   = entry['date']
            if source is not None:
                self.sources[target] = source # Each target has at most one source
                self.targets[source].append(target) # Each source can have multiple targets
                self.source_dates[target] = date # Each target has at most one source
                self.target_dates[source].append(date) # Each source can have multiple targets

        # Count the number of targets each person has, and the list of transmissions
        self.count_targets()
        self.count_transmissions()

        # Include the detailed transmission tree as well, as a list and as a dataframe
        self.make_detailed(people)

        # Optionally convert to NetworkX -- must be done on creation since the people object is not kept
        if self.to_networkx:

            # Initialization
            import networkx as nx
            self.graph = nx.DiGraph()

            # Add the nodes
            arrs = {attr:self.get_people_attr(people, attr) for attr in attrs}
            for i in range(len(people)):
                d = {}
                for attr in attrs:
                    d[attr] = arrs[attr][i]
                self.graph.add_node(i, **d)

            # Next, add edges from linelist
            for edge in self.infection_log:
                if edge['source'] is not None: # Skip seed infections
                    self.graph.add_edge(edge['source'],edge['target'],date=edge['date'],layer=edge['layer'])

        return

    @staticmethod
    def get_people_attr(people, attr):
        """ Get a per-agent array from the people; the date someone became a known contact is not recorded, so that is all NaN """
        if attr in people.keys():
            return people[attr]
        else:
            return np.full(len(people), np.nan)

    def __len__(self):
        """
        The length of the transmission tree is the length of the line list,
        which should equal the number of infections.
        """
        try:
            return len(self.infection_log)
        except: # pragma: no cover
            return 0

    def day(self, day=None, which=None):
        """ Convenience function for converting an input to an integer day """
        if day is not None:
            day = sc.day(day, start_date=self.sim_start)
        elif which == 'start':
            day = 0
        elif which == 'end':
            day = self.n_days
        return day

    def count_targets(self, start_day=None, end_day=None):
        """
        Count the number of targets each infected person has. If start and/or end
        days are given, it will only count the targets of people who got infected
        between those dates (it does not, however, filter on the date the target
        got infected).

        Args:
            start_day (int/str): the day on which to start counting people who got infected
            end_day (int/str): the day on which to stop counting people who got infected
        """

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

    def make_detailed(self, people):
        """ Construct a detailed transmission tree, with additional information for each person (people is a cv.PeopleSnapshot) """

        def df_to_arrdict(df):
            """ Convert a dataframe to a dictionary of arrays """
            arrdict = {}
            for col in df.columns:
                arrdict[col] = df[col].values
            return arrdict

        # Convert infection log to a dataframe and from there to a dict of arrays
        inflog = df_to_arrdict(pd.DataFrame(self.infection_log, columns=['source', 'target', 'date', 'layer', 'variant']))

        # Initialization
        n_people = len(people)
        src = 'src_'
        trg = 'trg_'
        attrs = ['age', 'date_exposed', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_severe', 'date_critical', 'date_known_contact']
        quar_attrs = ['date_quarantined', 'date_end_quarantine']
        date_attrs = [attr for attr in attrs if attr.startswith('date_')]
        is_attrs = [attr.replace('date_', 'is_') for attr in date_attrs]
        def dd_arr(): # Create an empty array of the right size
            return np.nan*np.zeros(n_people)
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
            values = self.get_people_attr(people, attr)
            dd[trg+attr] = values[:]
            dd[src+attr][vi] = values[vs_inds]

        # Pull out valid indices for source and target
        lnot = np.logical_not # Shorten since used heavily
        dd[src+'is_quarantined'][vi] = (dd[src+'date_quarantined'][vi] <= vinfdates) & lnot(dd[src+'date_end_quarantine'][vi] <= vinfdates)
        for is_attr,date_attr in zip(is_attrs, date_attrs):
            dd[src+is_attr][vi] = np.array(dd[src+date_attr][vi] <= vinfdates, dtype=bool)

        # Populate remaining properties
        dd[src+'is_asymp'][vi] = np.isnan(dd[src+'date_symptomatic'][vi])
        dd[src+'is_presymp'][vi] = lnot(dd[src+'is_asymp'][vi]) & lnot(dd[src+'is_symptomatic'][vi])
        dd[trg+'is_quarantined'][ti] = (dd[trg+'date_quarantined'][ti] <= tinfdates) & lnot(dd[trg+'date_end_quarantine'][ti] <= tinfdates)

        # Also re-parse the log and convert to a simpler dataframe
        targets = np.array(self.target_inds, dtype=int)
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
        """
        Plot the transmission tree.

        Args:
            fig_args  (dict):  passed to plt.figure()
            plot_args (dict):  passed to plt.plot()
            do_show   (bool):  whether to show the plot
            fig       (fig):   if supplied, use this figure
        """

        fig_args = sc.mergedicts(dict(figsize=(8, 5)), fig_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.5, marker='o'), plot_args)

        if fig is None:
            fig = plt.figure(**fig_args)
        plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.95, wspace=0.4, hspace=0.4)
        n_rows = 2
        n_cols = 3

        def plot_quantity(key, title, i):
            dat = self.df.groupby(['Day', key]).size().unstack(key)
            ax = plt.subplot(n_rows, n_cols, i);
            dat.plot(ax=ax, legend=None, **plot_args)
            plt.legend(title=None)
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

        return cvplt.handle_show_return(fig=fig, do_show=do_show)

    def animate(self, *args, **kwargs):
        """
        Animate the transmission tree.

        Args:
            animate    (bool):  whether to animate the plot (otherwise, show when finished)
            verbose    (bool):  print out progress of each frame
            markersize (int):   size of the markers
            sus_color  (list):  color for susceptibles
            fig_args   (dict):  arguments passed to plt.figure()
            axis_args  (dict):  arguments passed to plt.subplots_adjust()
            plot_args  (dict):  arguments passed to plt.plot()
            delay      (float): delay between frames in seconds
            colors     (list):  color of each person
            cmap       (str):   colormap for each person (if colors is not supplied)
            fig        (fig):   if supplied, use this figure

        Returns:
            fig: the figure object
        """

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
            fig = plt.figure(**fig_args)
        plt.subplots_adjust(**axis_args)
        ax = fig.add_subplot(1, 1, 1)

        # Create the legend
        ax2 = plt.axes([0.85, 0.05, 0.14, 0.9])
        ax2.axis('off')
        lcol = colors[0]
        na = np.nan  # Shorten
        plt.plot(na, na, '-', c=lcol, **plot_args, label='Transmission')
        plt.plot(na, na, 'o', c=lcol, markersize=msize, **plot_args, label='Source')
        plt.plot(na, na, '*', c=lcol, markersize=msize, **plot_args, label='Target')
        plt.plot(na, na, 'o', c=lcol, markersize=msize * 2, fillstyle='none', **plot_args, label='Tested')
        plt.plot(na, na, 's', c=lcol, markersize=msize * 1.2, **plot_args, label='Diagnosed')
        plt.plot(na, na, 'x', c=lcol, markersize=msize * 2.0, label='Known contact')
        plt.legend()

        # Plot the animation
        plt.sca(ax)
        for day in range(n):
            plt.title(f'Day: {day}')
            plt.xlim([0, n])
            plt.ylim([0, self.pop_size])
            plt.xlabel('Day')
            plt.ylabel('Person')
            flist = frames[day]
            tlist = tests[day]
            dlist = diags[day]
            qlist = quars[day]
            for f in flist:
                if verbose: print(f)
                x = f['x']
                y = f['y']
                c = f['c']
                plt.plot(x[0], y[0], 'o', c=c, markersize=msize, **plot_args)  # Plot sources
                plt.plot(x, y, '-', c=c, **plot_args)  # Plot transmission lines
                if f['i']:  # If this person is infected
                    plt.plot(x[1], y[1], '*', c=c, markersize=msize, **plot_args)  # Plot targets
            for tdq in tlist: plt.plot(tdq['d'], tdq['t'], 'o', c=tdq['c'], markersize=msize * 2, fillstyle='none')  # Tested; No alpha for this
            for tdq in dlist: plt.plot(tdq['d'], tdq['t'], 's', c=tdq['c'], markersize=msize * 1.2, **plot_args)  # Diagnosed
            for tdq in qlist: plt.plot(tdq['d'], tdq['t'], 'x', c=tdq['c'], markersize=msize * 2.0)  # Quarantine; no alpha for this
            plt.plot([0, day], [0.5, 0.5], c='k', lw=3)  # Plot the endless march of time
            if animate:  # Whether to animate
                plt.pause(delay)

        return fig

    def plot_histograms(self, start_day=None, end_day=None, bins=None, width=0.8, fig_args=None, fig=None):
        """
        Plots a histogram of the number of transmissions.

        Args:
            start_day (int/str): the day on which to start counting people who got infected
            end_day (int/str): the day on which to stop counting people who got infected
            bins (list): bin edges to use for the histogram
            width (float): width of bars
            fig_args (dict): passed to plt.figure()
            fig (fig): if supplied, use this figure
        """

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
            fig = plt.figure(**fig_args)
        plt.set_cmap('Spectral')
        plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)
        colors = sc.vectocolor(n_bins)

        plt.subplot(1,2,1)
        w05 = width*0.5
        w025 = w05*0.5
        plt.bar(bins-w025, counts, width=w05, facecolor='k', label='Number of events')
        for i in range(n_bins):
            label = 'Number of transmissions (events × transmissions per event)' if i==0 else None
            plt.bar(bins[i]+w025, total_counts[i], width=w05, facecolor=colors[i], label=label)
        plt.xlabel('Number of transmissions per person')
        plt.ylabel('Count')
        if n_bins<max_labels:
            plt.xticks(ticks=bins)
        plt.legend()
        plt.title('Numbers of events and transmissions')

        plt.subplot(2,2,2)
        total = 0
        for i in range(n_bins):
            plt.bar(bins[i:], total_counts[i], width=width, bottom=total, facecolor=colors[i])
            total += total_counts[i]
        if n_bins<max_labels:
            plt.xticks(ticks=bins)
        plt.xlabel('Number of transmissions per person')
        plt.ylabel('Number of infections caused')
        plt.title('Number of transmissions, by transmissions per person')

        plt.subplot(2,2,4)
        plt.plot(index, sorted_sum, lw=1.5, c='k', alpha=0.5)
        n_change_inds = len(change_inds)
        label_inds = np.linspace(0, n_change_inds, max_labels).round() # Don't allow more than this many labels
        for i in range(n_change_inds):
            if i in label_inds: # Don't plot more than this many labels
                label = f'Transmitted to {bins[i+1]:n} people'
            else:
                label = None
            plt.scatter([index[change_inds[i]]], [sorted_sum[change_inds[i]]], s=150, zorder=10, c=[colors[i]], label=label)
        plt.xlabel('Proportion of population, ordered by the number of people they infected (%)')
        plt.ylabel('Proportion of infections caused (%)')
        plt.legend()
        plt.ylim([0, 100])
        plt.grid(True)
        plt.title('Proportion of transmissions, by proportion of population')

        plt.axes([0.30, 0.65, 0.15, 0.2])
        berry      = [0.8, 0.1, 0.2]
        dirty_snow = [0.9, 0.9, 0.9]
        start_day  = self.day(start_day, which='start')
        end_day    = self.day(end_day, which='end')
        plt.axvspan(start_day, end_day, facecolor=dirty_snow)
        plt.plot(self.sim_results['t'], self.sim_results['cum_infections'], lw=1, c=berry)
        plt.xlabel('Day')
        plt.ylabel('Cumulative infections')

        return cvplt.handle_show_return(fig=fig)
