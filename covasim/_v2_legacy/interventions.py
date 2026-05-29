'''
Specify the core interventions available in Covasim. Other interventions can be
defined by the user by inheriting from these classes.
'''

import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import inspect
import datetime as dt
from . import misc as cvm
from . import utils as cvu
from . import base as cvb
from . import defaults as cvd
from . import parameters as cvpar
from . import immunity as cvi


#%% Helper functions

def find_day(arr, t=None, interv=None, sim=None, which='first'):
    '''
    Helper function to find if the current simulation time matches any day in the
    intervention. Although usually never more than one index is returned, it is
    returned as a list for the sake of easy iteration.

    Args:
        arr (list/function): list of days in the intervention, or a boolean array; or a function that returns these
        t (int): current simulation time (can be None if a boolean array is used)
        which (str): what to return: 'first', 'last', or 'all' indices
        interv (intervention): the intervention object (usually self); only used if arr is callable
        sim (sim): the simulation object; only used if arr is callable

    Returns:
        inds (list): list of matching days; length zero or one unless which is 'all'

    New in version 2.1.2: arr can be a function with arguments interv and sim.
    '''
    if callable(arr):
        arr = arr(interv, sim)
        arr = sc.toarray(arr)
    all_inds = sc.findinds(arr=arr, val=t)
    if len(all_inds) == 0 or which == 'all':
        inds = all_inds
    elif which == 'first':
        inds = [all_inds[0]]
    elif which == 'last':
        inds = [all_inds[-1]]
    else: # pragma: no cover
        errormsg = f'Argument "which" must be "first", "last", or "all", not "{which}"'
        raise ValueError(errormsg)
    return inds


def preprocess_day(day, sim):
    '''
    Preprocess a day: leave it as-is if it's a function, or try to convert it to
    an integer if it's anything else.
    '''
    if callable(day):  # pragma: no cover
        return day # If it's callable, leave it as-is
    else:
        day = sim.day(day) # Otherwise, convert it to an int
    return day


def get_day(day, interv=None, sim=None):
    '''
    Return the day if it's an integer, or call it if it's a function.
    '''
    if callable(day): # pragma: no cover
        return day(interv, sim) # If it's callable, call it
    else:
        return day # Otherwise, leave it as-is


def process_days(sim, days, return_dates=False):
    '''
    Ensure lists of days are in consistent format. Used by change_beta, clip_edges,
    and some analyzers. If day is 'end' or -1, use the final day of the simulation.
    Optionally return dates as well as days. If days is callable, leave unchanged.
    '''
    if callable(days):
        return days
    if sc.isstring(days) or not sc.isiterable(days):
        days = sc.tolist(days)
    for d,day in enumerate(days):
        if day in ['end', -1]: # pragma: no cover
            day = sim['end_day']
        days[d] = preprocess_day(day, sim) # Ensure it's an integer and not a string or something
    days = np.sort(sc.toarray(days)) # Ensure they're an array and in order
    if return_dates:
        dates = [sim.date(day) for day in days] # Store as date strings
        return days, dates
    else:
        return days


def process_changes(sim, changes, days):
    '''
    Ensure lists of changes are in consistent format. Used by change_beta and clip_edges.
    '''
    changes = sc.toarray(changes)
    if sc.isiterable(days) and len(days) != len(changes): # pragma: no cover
        errormsg = f'Number of days supplied ({len(days)}) does not match number of changes ({len(changes)})'
        raise ValueError(errormsg)
    return changes


def process_daily_data(daily_data, sim, start_day, as_int=False):
    '''
    This function performs one of three things: if the daily test data are supplied as
    a number, then it converts it to an array of the right length. If the daily
    data are supplied as a Pandas series or dataframe with a date index, then it
    reindexes it to match the start date of the simulation. If the daily data are
    supplied as a string, then it will convert it to a column and try to read from
    that. Otherwise, it does nothing.

    Args:
        daily_data (str, number, dataframe, or series): the data to convert to standardized format
        sim (Sim): the simulation object
        start_day (date): the start day of the simulation, in already-converted datetime.date format
        as_int (bool): whether to convert to an integer
    '''
    # Handle string arguments
    if sc.isstring(daily_data):
        if daily_data == 'data':
            daily_data = sim.data['new_tests'] # Use default name
        else: # pragma: no cover
            try:
                daily_data = sim.data[daily_data]
            except Exception as E:
                errormsg = f'Tried to load testing data from sim.data["{daily_data}"], but that failed: {str(E)}.\nPlease ensure data are loaded into the sim and the column exists.'
                raise ValueError(errormsg) from E

    # Handle other arguments
    if sc.isnumber(daily_data):  # If a number, convert to an array
        if as_int: daily_data = int(daily_data) # Make it an integer
        daily_data = np.array([daily_data] * sim.npts)
    elif isinstance(daily_data, (pd.Series, pd.DataFrame)):
        start_date = sim['start_day'] + dt.timedelta(days=start_day)
        end_date = daily_data.index[-1]
        dateindex = pd.date_range(start_date, end_date)
        daily_data = daily_data.reindex(dateindex, fill_value=0).to_numpy()

    return daily_data


def get_subtargets(subtarget, sim):
    '''
    A small helper function to see if subtargeting is a list of indices to use,
    or a function that needs to be called. If a function, it must take a single
    argument, a sim object, and return a list of indices. Also validates the values.
    Currently designed for use with testing interventions, but could be generalized
    to other interventions. Not typically called directly by the user.

    Args:
        subtarget (dict): dict with keys 'inds' and 'vals'; see test_num() for examples of a valid subtarget dictionary
        sim (Sim): the simulation object
    '''

    # Validation
    if callable(subtarget):
        subtarget = subtarget(sim)

    if 'inds' not in subtarget: # pragma: no cover
        errormsg = f'The subtarget dict must have keys "inds" and "vals", but you supplied {subtarget}'
        raise ValueError(errormsg)

    # Handle the two options of type
    if callable(subtarget['inds']): # A function has been provided
        subtarget_inds = subtarget['inds'](sim) # Call the function to get the indices
    else:
        subtarget_inds = subtarget['inds'] # The indices are supplied directly

    # Validate the values
    if callable(subtarget['vals']): # A function has been provided
        subtarget_vals = subtarget['vals'](sim) # Call the function to get the indices
    else:
        subtarget_vals = subtarget['vals'] # The indices are supplied directly
    if sc.isiterable(subtarget_vals):
        if len(subtarget_vals) != len(subtarget_inds): # pragma: no cover
            errormsg = f'Length of subtargeting indices ({len(subtarget_inds)}) does not match length of values ({len(subtarget_vals)})'
            raise ValueError(errormsg)

    return subtarget_inds, subtarget_vals

def InterventionDict(which, pars):
    '''
    Generate an intervention from a dictionary. Although a function, it acts
    like a class, since it returns a class instance.

    **Example**::

        interv = cv.InterventionDict(which='change_beta', pars={'days': 30, 'changes': 0.5, 'layers': None})
    '''
    mapping = dict(
        dynamic_pars    = dynamic_pars,
        sequence        = sequence,
        change_beta     = change_beta,
        clip_edges      = clip_edges,
        test_num        = test_num,
        test_prob       = test_prob,
        contact_tracing = contact_tracing,
        )
    try:
        IntervClass = mapping[which]
    except:
        available = ', '.join(mapping.keys())
        errormsg = f'Only interventions "{available}" are available in dictionary representation, not "{which}"'
        raise sc.KeyNotFoundError(errormsg)
    intervention = IntervClass(**pars)
    return intervention



#%% Generic intervention classes

__all__ = ['InterventionDict', 'Intervention', 'dynamic_pars', 'sequence']


class Intervention:
    '''
    Base class for interventions. By default, interventions are printed using a
    dict format, which they can be recreated from. To display all the attributes
    of the intervention, use disp() instead.

    To retrieve a particular intervention from a sim, use sim.get_intervention().

    Args:
        label       (str): a label for the intervention (used for plotting, and for ease of identification)
        show_label (bool): whether or not to include the label in the legend
        do_plot    (bool): whether or not to plot the intervention
        line_args  (dict): arguments passed to pl.axvline() when plotting
    '''
    def __init__(self, label=None, show_label=False, do_plot=None, line_args=None):
        self._store_args() # Store the input arguments so the intervention can be recreated
        if label is None: label = self.__class__.__name__ # Use the class name if no label is supplied
        self.label = label # e.g. "Close schools"
        self.show_label = show_label # Do not show the label by default
        self.do_plot = do_plot if do_plot is not None else True # Plot the intervention, including if None
        self.line_args = sc.mergedicts(dict(linestyle='--', c='#aaa', lw=1.0), line_args) # Do not set alpha by default due to the issue of overlapping interventions
        self.days = [] # The start and end days of the intervention
        self.initialized = False # Whether or not it has been initialized
        self.finalized = False # Whether or not it has been initialized
        return


    def __repr__(self, jsonify=False):
        ''' Return a JSON-friendly output if possible, else revert to short repr '''

        if self.__class__.__name__ in __all__ or jsonify:
            try:
                json = self.to_json()
                which = json['which']
                pars = json['pars']
                parstr = ', '.join([f'{k}={v}' for k,v in pars.items()])
                output = f"cv.{which}({parstr})"
            except Exception as E:
                output = f'{type(self)} (error: {str(E)})' # If that fails, print why
            return output
        else:
            return f'{self.__module__}.{self.__class__.__name__}()'


    def __call__(self, *args, **kwargs):
        # Makes Intervention(sim) equivalent to Intervention.apply(sim)
        if not self.initialized:  # pragma: no cover
            errormsg = f'Intervention (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(*args, **kwargs)


    def disp(self):
        ''' Print a detailed representation of the intervention '''
        return sc.pr(self)


    def _store_args(self):
        ''' Store the user-supplied arguments for later use in to_json '''
        self.input_args = {} # Create a place to store the input arguments
        f0 = inspect.currentframe() # This "frame", i.e. Intervention.__init__()
        f1 = inspect.getouterframes(f0) # The list of outer frames
        parent = f1[2].frame # The parent frame, e.g. change_beta.__init__()
        arginfo = inspect.getargvalues(parent) # Get the values of the arguments
        for arg in arginfo.args:
            if arg != 'self': # Don't store this
                self.input_args[arg] = arginfo.locals[arg] # Store normal arguments, including defined keyword arguments
        if arginfo.keywords is not None:
            for key,value in arginfo.locals['kwargs'].items(): # Store additional arguments captured by **kwargs
                self.input_args[key] = value
        return


    def initialize(self, sim=None):
        '''
        Initialize intervention -- this is used to make modifications to the intervention
        that can't be done until after the sim is created.
        '''
        self.initialized = True
        self.finalized = False
        return


    def finalize(self, sim=None):
        '''
        Finalize intervention

        This method is run once as part of `sim.finalize()` enabling the intervention to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized: # pragma: no cover
            raise RuntimeError('Intervention already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        self.finalized = True
        return


    def apply(self, sim):
        '''
        Apply the intervention. This is the core method which each derived intervention
        class must implement. This method gets called at each timestep and can make
        arbitrary changes to the Sim object, as well as storing or modifying the
        state of the intervention.

        Args:
            sim: the Sim instance

        Returns:
            None
        '''
        raise NotImplementedError


    def shrink(self, in_place=False):
        '''
        Remove any excess stored data from the intervention; for use with sim.shrink().

        Args:
            in_place (bool): whether to shrink the intervention (else shrink a copy)
        '''
        if in_place: # pragma: no cover
            return self
        else:
            return sc.dcp(self)


    def plot_intervention(self, sim, ax=None, **kwargs):
        '''
        Plot the intervention

        This can be used to do things like add vertical lines on days when
        interventions take place. Can be disabled by setting self.do_plot=False.

        Note 1: you can modify the plotting style via the ``line_args`` argument when
        creating the intervention.

        Note 2: By default, the intervention is plotted at the days stored in self.days.
        However, if there is a self.plot_days attribute, this will be used instead.

        Args:
            sim: the Sim instance
            ax: the axis instance
            kwargs: passed to ax.axvline()

        Returns:
            None
        '''
        line_args = sc.mergedicts(self.line_args, kwargs)
        if self.do_plot or self.do_plot is None:
            if ax is None:
                ax = pl.gca()
            if hasattr(self, 'plot_days'):
                days = self.plot_days
            else:
                days = self.days
            if sc.isiterable(days):
                label_shown = False # Don't show the label more than once
                for day in days:
                    if sc.isnumber(day):
                        if self.show_label and not label_shown: # Choose whether to include the label in the legend
                            label = self.label
                            label_shown = True
                        else:
                            label = None
                        date = sc.date(sim.date(day))
                        ax.axvline(date, label=label, **line_args)
        return


    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. In the first instance, the object dict will be returned.
        However, if an intervention itself contains non-standard variables as
        attributes, then its `to_json` method will need to handle those.

        Note that simply printing an intervention will usually return a representation
        that can be used to recreate it.

        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        which = self.__class__.__name__
        pars = sc.jsonify(self.input_args)
        output = dict(which=which, pars=pars)
        return output


class dynamic_pars(Intervention):
    '''
    A generic intervention that modifies a set of parameters at specified points
    in time.

    The intervention takes a single argument, pars, which is a dictionary of which
    parameters to change, with following structure: keys are the parameters to change,
    then subkeys 'days' and 'vals' are either a scalar or list of when the change(s)
    should take effect and what the new value should be, respectively.

    You can also pass parameters to change directly as keyword arguments.

    Args:
        pars (dict): described above
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = cv.dynamic_pars(n_imports=dict(days=10, vals=100))
        interv = cv.dynamic_pars({'beta':{'days':[14, 28], 'vals':[0.005, 0.015]}, 'rel_death_prob':{'days':30, 'vals':2.0}}) # Change beta, and make diagnosed people stop transmitting

    '''

    def __init__(self, pars=None, **kwargs):

        # Find valid sim parameters and move matching keyword arguments to the pars dict
        pars = sc.mergedicts(pars) # Ensure it's a dictionary
        sim_par_keys = list(cvpar.make_pars().keys()) # Get valid sim parameters
        kwarg_keys = [k for k in kwargs.keys() if k in sim_par_keys]
        for kkey in kwarg_keys:
            pars[kkey] = kwargs.pop(kkey)

        # Do standard initialization
        super().__init__(**kwargs) # Initialize the Intervention object

        # Handle the rest of the initialization
        subkeys = ['days', 'vals']
        for parkey in pars.keys():
            for subkey in subkeys:
                if subkey not in pars[parkey].keys(): # pragma: no cover
                    errormsg = f'Parameter {parkey} is missing subkey {subkey}'
                    raise sc.KeyNotFoundError(errormsg)
                if sc.isnumber(pars[parkey][subkey]): # Allow scalar values or dicts, but leave everything else unchanged
                    pars[parkey][subkey] = sc.toarray(pars[parkey][subkey])
            days = pars[parkey]['days']
            vals = pars[parkey]['vals']
            if sc.isiterable(days):
                len_days = len(days)
                len_vals = len(vals)
                if len_days != len_vals: # pragma: no cover
                    raise ValueError(f'Length of days ({len_days}) does not match length of values ({len_vals}) for parameter {parkey}')
        self.pars = pars
        return


    def apply(self, sim):
        ''' Loop over the parameters, and then loop over the days, applying them if any are found '''
        t = sim.t
        for parkey,parval in self.pars.items():
            for ind in find_day(parval['days'], t, interv=self, sim=sim):
                self.days.append(t)
                val = parval['vals'][ind]
                if isinstance(val, dict):
                    sim[parkey].update(val) # Set the parameter if a nested dict
                else:
                    sim[parkey] = val # Set the parameter if not a dict
        return


class sequence(Intervention):
    '''
    This is an example of a meta-intervention which switches between a sequence of interventions.

    Args:
        days (list): the days on which to start applying each intervention
        interventions (list): the interventions to apply on those days
        kwargs (dict): passed to Intervention()

    **Example**::

        interv = cv.sequence(days=[10, 51], interventions=[
                    cv.test_num(n_tests=[100]*npts),
                    cv.test_prob(symptomatic_prob=0.2, asymptomatic_prob=0.002),
                ])
    '''

    def __init__(self, days, interventions, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        if sc.isiterable(days):
            assert len(days) == len(interventions)
        self.days = days
        self.interventions = interventions
        return


    def initialize(self, sim):
        ''' Fix the dates '''
        super().initialize()
        self.days = [sim.day(day) for day in self.days]
        self.days_arr = np.array(self.days + [sim.npts])
        for intervention in self.interventions:
            intervention.initialize(sim)
        return


    def apply(self, sim):
        ''' Find the matching day, and see which intervention to activate '''
        inds = find_day(self.days_arr <= sim.t, which='last')
        if len(inds):
            return self.interventions[inds[0]].apply(sim)


#%% Beta interventions

__all__+= ['change_beta', 'clip_edges']


class change_beta(Intervention):
    '''
    The most basic intervention -- change beta (transmission) by a certain amount
    on a given day or days. This can be used to represent physical distancing (although
    clip_edges() is more appropriate for overall changes in mobility, e.g. school
    or workplace closures), as well as hand-washing, masks, and other behavioral
    changes that affect transmission rates.

    Args:
        days    (int/arr):   the day or array of days to apply the interventions
        changes (float/arr): the changes in beta (1 = no change, 0 = no transmission)
        layers  (str/list):  the layers in which to change beta (default: all)
        kwargs  (dict):      passed to Intervention()

    **Examples**::

        interv = cv.change_beta(25, 0.3) # On day 25, reduce overall beta by 70% to 0.3
        interv = cv.change_beta([14, 28], [0.7, 1], layers='s') # On day 14, reduce beta by 30%, and on day 28, return to 1 for schools
    '''

    def __init__(self, days, changes, layers=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.days       = sc.dcp(days)
        self.changes    = sc.dcp(changes)
        self.layers     = sc.dcp(layers)
        self.orig_betas = None
        return


    def initialize(self, sim):
        ''' Fix days and store beta '''
        super().initialize()
        self.days    = process_days(sim, self.days)
        self.changes = process_changes(sim, self.changes, self.days)
        self.layers  = sc.tolist(self.layers, keepnone=True)
        self.orig_betas = {}
        for lkey in self.layers:
            if lkey is None:
                self.orig_betas['overall'] = sim['beta']
            else:
                self.orig_betas[lkey] = sim['beta_layer'][lkey]

        return


    def apply(self, sim):

        # If this day is found in the list, apply the intervention
        for ind in find_day(self.days, sim.t, interv=self, sim=sim):
            for lkey,new_beta in self.orig_betas.items():
                new_beta = new_beta * self.changes[ind]
                if lkey == 'overall':
                    sim['beta'] = new_beta
                else:
                    sim['beta_layer'][lkey] = new_beta

        return


class clip_edges(Intervention):
    '''
    Isolate contacts by removing them from the simulation. Contacts are treated as
    "edges", and this intervention works by removing them from sim.people.contacts
    and storing them internally. When the intervention is over, they are moved back.
    This intervention has quite similar effects as change_beta(), but is more appropriate
    for modeling the effects of mobility reductions such as school and workplace
    closures. The main difference is that since clip_edges() actually removes contacts,
    it affects the number of people who would be traced and placed in quarantine
    if an individual tests positive. It also alters the structure of the network
    -- i.e., compared to a baseline case of 20 contacts and a 2% chance of infecting
    each, there are slightly different statistics for a beta reduction (i.e., 20 contacts
    and a 1% chance of infecting each) versus an edge clipping (i.e., 10 contacts
    and a 2% chance of infecting each).

    Args:
        days (int or array): the day or array of days to isolate contacts
        changes (float or array): the changes in the number of contacts (1 = no change, 0 = no contacts)
        layers (str or list): the layers in which to isolate contacts (if None, then all layers)
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = cv.clip_edges(25, 0.3) # On day 25, reduce overall contacts by 70% to 0.3
        interv = cv.clip_edges([14, 28], [0.7, 1], layers='s') # On day 14, remove 30% of school contacts, and on day 28, restore them
    '''

    def __init__(self, days, changes, layers=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.days     = sc.dcp(days)
        self.changes  = sc.dcp(changes)
        self.layers   = sc.dcp(layers)
        self.contacts = None
        return


    def initialize(self, sim):
        super().initialize()
        self.days    = process_days(sim, self.days)
        self.changes = process_changes(sim, self.changes, self.days)
        if self.layers is None:
            self.layers = sim.layer_keys()
        else:
            self.layers = sc.tolist(self.layers)
        self.contacts = cvb.Contacts(layer_keys=self.layers)
        return


    def apply(self, sim):

        # If this day is found in the list, apply the intervention
        for ind in find_day(self.days, sim.t, interv=self, sim=sim):

            # Do the contact moving
            for lkey in self.layers:
                s_layer = sim.people.contacts[lkey] # Contact layer in the sim
                i_layer = self.contacts[lkey] # Contact layer in the intervention
                n_sim = len(s_layer) # Number of contacts in the simulation layer
                n_int = len(i_layer) # Number of contacts in the intervention layer
                n_contacts = n_sim + n_int # Total number of contacts
                if n_contacts:
                    current_prop = n_sim/n_contacts # Current proportion of contacts in the sim, e.g. 1.0 initially
                    desired_prop = self.changes[ind] # Desired proportion, e.g. 0.5
                    prop_to_move = current_prop - desired_prop # Calculate the proportion of contacts to move
                    n_to_move = int(prop_to_move*n_contacts) # Number of contacts to move
                    from_sim = (n_to_move>0) # Check if we're moving contacts from the sim
                    if from_sim: # We're moving from the sim to the intervention
                        inds = cvu.choose(max_n=n_sim, n=n_to_move)
                        to_move = s_layer.pop_inds(inds)
                        i_layer.append(to_move)
                    else: # We're moving from the intervention back to the sim
                        inds = cvu.choose(max_n=n_int, n=abs(n_to_move))
                        to_move = i_layer.pop_inds(inds)
                        s_layer.append(to_move)
                else: # pragma: no cover
                    warnmsg = f'Warning: clip_edges() was applied to layer "{lkey}", but no edges were found; please check sim.people.contacts["{lkey}"]'
                    cvm.warn(warnmsg)
        return


    def finalize(self, sim):
        ''' Ensure the edges get deleted at the end '''
        super().finalize()
        self.contacts = None # Reset to save memory
        return



#%% Testing interventions

__all__+= ['test_num', 'test_prob', 'contact_tracing']


def get_quar_inds(quar_policy, sim):
    '''
    Helper function to return the appropriate indices for people in quarantine
    based on the current quarantine testing "policy". Used by test_num and test_prob.
    Not for use by the user.

    If quar_policy is a number or a list of numbers, then it is interpreted as
    the number of days after the start of quarantine when a test is performed.
    It can also be a function that returns the list of indices.

    Args:
        quar_policy (str, int, list, func): 'start', people entering quarantine; 'end', people leaving; 'both', entering and leaving; 'daily', every day in quarantine
        sim (Sim): the simulation object
    '''
    t = sim.t
    if   quar_policy is None:    quar_test_inds = np.array([])
    elif quar_policy == 'start': quar_test_inds = cvu.true(sim.people.date_quarantined==t-1) # Actually do the day after since testing usually happens before contact tracing
    elif quar_policy == 'end':   quar_test_inds = cvu.true(sim.people.date_end_quarantine==t+1) # +1 since they are released on date_end_quarantine, so do the day before
    elif quar_policy == 'both':  quar_test_inds = np.concatenate([cvu.true(sim.people.date_quarantined==t-1), cvu.true(sim.people.date_end_quarantine==t+1)])
    elif quar_policy == 'daily': quar_test_inds = cvu.true(sim.people.quarantined)
    elif sc.isnumber(quar_policy) or (sc.isiterable(quar_policy) and not sc.isstring(quar_policy)):
        quar_policy = sc.toarray(quar_policy)
        quar_test_inds = np.unique(np.concatenate([cvu.true(sim.people.date_quarantined==t-1-q) for q in quar_policy]))
    elif callable(quar_policy):
        quar_test_inds = quar_policy(sim)
    else: # pragma: no cover
        errormsg = f'Quarantine policy "{quar_policy}" not recognized: must be a string (start, end, both, daily), int, list, array, set, tuple, or function'
        raise ValueError(errormsg)
    return quar_test_inds


class test_num(Intervention):
    '''
    Test the specified number of people per day. Useful for including historical
    testing data. The probability of a given person getting a test is dependent
    on the total number of tests, population size, and odds ratios. Compare this
    intervention with cv.test_prob().

    Args:
        daily_tests (arr)   : number of tests per day, can be int, array, or dataframe/series; if integer, use that number every day; if 'data' or another string, use loaded data
        symp_test   (float) : odds ratio of a symptomatic person testing (default: 100x more likely)
        quar_test   (float) : probability of a person in quarantine testing (default: no more likely)
        quar_policy (str)   : policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget   (dict)  : subtarget intervention to people with particular indices (format: {'ind': array of indices, or function to return indices from the sim, 'vals': value(s) to apply}
        ili_prev    (arr)   : prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        sensitivity (float) : test sensitivity (default 100%, i.e. no false negatives)
        loss_prob   (float) : probability of the person being lost-to-follow-up (default 0%, i.e. no one lost to follow-up)
        test_delay  (int)   : days for test result to be known (default 0, i.e. results available instantly)
        start_day   (int)   : day the intervention starts (default: 0, i.e. first day of the simulation)
        end_day     (int)   : day the intervention ends
        swab_delay  (dict)  : distribution for the delay from onset to swab; if this is present, it is used instead of test_delay
        kwargs      (dict)  : passed to Intervention()

    **Examples**::

        interv = cv.test_num(daily_tests=[0.10*n_people]*npts)
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': cv.true(sim.people.age>50), 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': lambda sim: cv.true(sim.people.age>50), 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests='data') # Take number of tests from loaded data using default column name (new_tests)
        interv = cv.test_num(daily_tests='swabs_per_day') # Take number of tests from loaded data using a custom column name
    '''

    def __init__(self, daily_tests, symp_test=100.0, quar_test=1.0, quar_policy=None, subtarget=None,
                 ili_prev=None, sensitivity=1.0, loss_prob=0, test_delay=0,
                 start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.daily_tests = daily_tests # Should be a list of length matching time
        self.symp_test   = symp_test   # Set probability of testing symptomatics
        self.quar_test   = quar_test # Probability of testing people in quarantine
        self.quar_policy = quar_policy if quar_policy else 'start'
        self.subtarget   = subtarget  # Set any other testing criteria
        self.ili_prev    = ili_prev     # Should be a list of length matching time or a float or a dataframe
        self.sensitivity = sensitivity
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        self.pdf         = cvu.get_pdf(**sc.mergedicts(swab_delay)) # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        return


    def initialize(self, sim):
        ''' Fix the dates and number of tests '''

        # Handle days
        super().initialize()

        self.start_day   = preprocess_day(self.start_day, sim)
        self.end_day     = preprocess_day(self.end_day,   sim)
        self.days        = [self.start_day, self.end_day]

        # Process daily data
        self.daily_tests = process_daily_data(self.daily_tests, sim, self.start_day)
        self.ili_prev    = process_daily_data(self.ili_prev,    sim, self.start_day)

        return


    def finalize(self, sim):
        ''' Ensure variables with large memory footprints get erased '''
        super().finalize()
        self.subtarget = None # Reset to save memory
        return


    def apply(self, sim):

        t = sim.t
        start_day = get_day(self.start_day, self, sim)
        end_day   = get_day(self.end_day,   self, sim)
        if t < start_day:
            return
        elif end_day is not None and t > end_day:
            return

        # Check that there are still tests
        rel_t = t - start_day
        if rel_t < len(self.daily_tests):
            n_tests = sc.randround(self.daily_tests[rel_t]/sim.rescale_vec[t]) # Correct for scaling that may be applied by rounding to the nearest number of tests
            if not (n_tests and pl.isfinite(n_tests)): # If there are no tests today, abort early
                return
            else:
                sim.results['new_tests'][t] += n_tests
        else:
            return

        test_probs = np.ones(sim.n) # Begin by assigning equal testing weight (converted to a probability) to everyone

        # Calculate test probabilities for people with symptoms
        symp_inds = cvu.true(sim.people.symptomatic)
        symp_test = self.symp_test
        if self.pdf: # Handle the onset to swab delay
            symp_time = cvd.default_int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape) # Initialize the count
            count[inv_count != 0] = 1/inv_count[inv_count != 0] # Update the counts where defined
            symp_test *= self.pdf.pdf(symp_time) * count[symp_time] # Put it all together

        test_probs[symp_inds] *= symp_test # Update the test probabilities

        # Handle symptomatic testing, taking into account prevalence of ILI symptoms
        if self.ili_prev is not None:
            if rel_t < len(self.ili_prev):
                n_ili = int(self.ili_prev[rel_t] * sim['pop_size'])  # Number with ILI symptoms on this day
                ili_inds = cvu.choose(sim['pop_size'], n_ili) # Give some people some symptoms. Assuming that this is independent of COVID symptomaticity...
                ili_inds = np.setdiff1d(ili_inds, symp_inds)
                test_probs[ili_inds] *= self.symp_test

        # Handle quarantine testing
        quar_test_inds = get_quar_inds(self.quar_policy, sim)
        test_probs[quar_test_inds] *= self.quar_test

        # Handle any other user-specified testing criteria
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = test_probs[subtarget_inds]*subtarget_vals

        # Don't re-diagnose people
        diag_inds  = cvu.true(sim.people.diagnosed)
        test_probs[diag_inds] = 0.0

        # With dynamic rescaling, we have to correct for uninfected people outside of the population who would test
        if sim.rescale_vec[t]/sim['pop_scale'] < 1: # We still have rescaling to do
            in_pop_tot_prob = test_probs.sum()*sim.rescale_vec[t] # Total "testing weight" of people in the subsampled population
            out_pop_tot_prob = sim.scaled_pop_size - sim.rescale_vec[t]*sim['pop_size'] # Find out how many people are missing and assign them each weight 1
            in_frac = in_pop_tot_prob/(in_pop_tot_prob + out_pop_tot_prob) # Fraction of tests which should fall in the sample population
            n_tests = sc.randround(n_tests*in_frac) # Recompute the number of tests

        # Now choose who gets tested and test them
        n_tests = min(n_tests, (test_probs!=0).sum()) # Don't try to test more people than have nonzero testing probability
        test_inds = cvu.choose_w(probs=test_probs, n=n_tests, unique=True) # Choose who actually tests
        sim.people.test(test_inds, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)

        return test_inds


class test_prob(Intervention):
    '''
    Assign each person a probability of being tested for COVID based on their
    symptom state, quarantine state, and other states. Unlike test_num, the
    total number of tests not specified, but rather is an output.

    Args:
        symp_prob        (float)     : probability of testing a symptomatic (unquarantined) person
        asymp_prob       (float)     : probability of testing an asymptomatic (unquarantined) person (default: 0)
        symp_quar_prob   (float)     : probability of testing a symptomatic quarantined person (default: same as symp_prob)
        asymp_quar_prob  (float)     : probability of testing an asymptomatic quarantined person (default: same as asymp_prob)
        quar_policy      (str)       : policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget        (dict)      : subtarget intervention to people with particular indices  (see test_num() for details)
        ili_prev         (float/arr) : prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        sensitivity      (float)     : test sensitivity (default 100%, i.e. no false negatives)
        loss_prob        (float)     : probability of the person being lost-to-follow-up (default 0%, i.e. no one lost to follow-up)
        test_delay       (int)       : days for test result to be known (default 0, i.e. results available instantly)
        start_day        (int)       : day the intervention starts (default: 0, i.e. first day of the simulation)
        end_day          (int)       : day the intervention ends (default: no end)
        swab_delay       (dict)      : distribution for the delay from onset to swab; if this is present, it is used instead of test_delay
        kwargs           (dict)      : passed to Intervention()

    **Examples**::

        interv = cv.test_prob(symp_prob=0.1, asymp_prob=0.01) # Test 10% of symptomatics and 1% of asymptomatics
        interv = cv.test_prob(symp_quar_prob=0.4) # Test 40% of those in quarantine with symptoms
    '''
    def __init__(self, symp_prob, asymp_prob=0.0, symp_quar_prob=None, asymp_quar_prob=None, quar_policy=None, subtarget=None, ili_prev=None,
                 sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.symp_prob        = symp_prob
        self.asymp_prob       = asymp_prob
        self.symp_quar_prob   = symp_quar_prob  if  symp_quar_prob is not None else  symp_prob
        self.asymp_quar_prob  = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self.quar_policy      = quar_policy if quar_policy else 'start'
        self.subtarget        = subtarget
        self.ili_prev         = ili_prev
        self.sensitivity      = sensitivity
        self.loss_prob        = loss_prob
        self.test_delay       = test_delay
        self.start_day        = start_day
        self.end_day          = end_day
        self.pdf              = cvu.get_pdf(**sc.mergedicts(swab_delay)) # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        return


    def initialize(self, sim):
        ''' Fix the dates '''
        super().initialize()
        self.start_day = preprocess_day(self.start_day, sim)
        self.end_day   = preprocess_day(self.end_day,   sim)
        self.days      = [self.start_day, self.end_day]
        self.ili_prev  = process_daily_data(self.ili_prev, sim, self.start_day)
        return


    def finalize(self, sim):
        ''' Ensure variables with large memory footprints get erased '''
        super().finalize()
        self.subtarget = None # Reset to save memory
        return


    def apply(self, sim):
        ''' Perform testing '''

        t = sim.t
        start_day = get_day(self.start_day, self, sim)
        end_day   = get_day(self.end_day,   self, sim)
        if t < start_day:
            return
        elif end_day is not None and t > end_day:
            return

        # Find probablity for symptomatics to be tested
        symp_inds  = cvu.true(sim.people.symptomatic)
        symp_prob = self.symp_prob
        if self.pdf:
            symp_time = cvd.default_int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape)
            count[inv_count != 0] = 1/inv_count[inv_count != 0]
            symp_prob = np.ones(len(symp_time))
            inds = 1 > (symp_time*self.symp_prob)
            symp_prob[inds] = self.symp_prob/(1-symp_time[inds]*self.symp_prob)
            symp_prob = self.pdf.pdf(symp_time) * symp_prob * count[symp_time]

        # Define symptomatics, accounting for ILI prevalence
        pop_size = sim['pop_size']
        ili_inds = []
        if self.ili_prev is not None:
            rel_t = t - start_day
            if rel_t < len(self.ili_prev):
                n_ili = int(self.ili_prev[rel_t] * pop_size)  # Number with ILI symptoms on this day
                ili_inds = cvu.choose(pop_size, n_ili) # Give some people some symptoms, assuming that this is independent of COVID symptomaticity...
                ili_inds = np.setdiff1d(ili_inds, symp_inds)

        # Define asymptomatics: those who neither have COVID symptoms nor ILI symptoms
        asymp_inds = np.setdiff1d(np.setdiff1d(np.arange(pop_size), symp_inds), ili_inds)

        # Handle quarantine and other testing criteria
        quar_test_inds = get_quar_inds(self.quar_policy, sim)
        symp_quar_inds  = np.intersect1d(quar_test_inds, symp_inds)
        asymp_quar_inds = np.intersect1d(quar_test_inds, asymp_inds)
        diag_inds       = cvu.true(sim.people.diagnosed)

        # Construct the testing probabilities piece by piece -- complicated, since need to do it in the right order
        test_probs = np.zeros(sim['pop_size']) # Begin by assigning equal testing probability to everyone
        test_probs[symp_inds]       = symp_prob            # People with symptoms (true positive)
        test_probs[ili_inds]        = self.symp_prob       # People with symptoms (false positive) -- can't use swab delay since no date symptomatic
        test_probs[asymp_inds]      = self.asymp_prob      # People without symptoms
        test_probs[symp_quar_inds]  = self.symp_quar_prob  # People with symptoms in quarantine
        test_probs[asymp_quar_inds] = self.asymp_quar_prob # People without symptoms in quarantine
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted
        test_probs[diag_inds] = 0.0 # People who are diagnosed don't test
        test_inds = cvu.true(cvu.binomial_arr(test_probs)) # Finally, calculate who actually tests

        # Actually test people
        sim.people.test(test_inds, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay) # Actually test people
        sim.results['new_tests'][t] += len(test_inds)*sim['pop_scale']/sim.rescale_vec[t] # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec

        return test_inds


class contact_tracing(Intervention):
    '''
    Contact tracing of people who are diagnosed. When a person is diagnosed positive
    (by either test_num() or test_prob(); this intervention has no effect if there
    is not also a testing intervention active), a certain proportion of the index
    case's contacts (defined by trace_prob) are contacted after a certain number
    of days (defined by trace_time). After they are contacted, they are placed
    into quarantine (with effectiveness quar_factor, a simulation parameter) for
    a certain period (defined by quar_period, another simulation parameter). They
    may also change their testing probability, if test_prob() is defined.

    Args:
        trace_probs (float/dict): probability of tracing, per layer (default: 100%, i.e. everyone is traced)
        trace_time  (float/dict): days required to trace, per layer (default: 0, i.e. no delay)
        start_day   (int):        intervention start day (default: 0, i.e. the start of the simulation)
        end_day     (int):        intervention end day (default: no end)
        presumptive (bool):       whether or not to begin isolation and contact tracing on the presumption of a positive diagnosis (default: no)
        capacity    (int):        optionally specify a maximum number of newly diagnosed people to trace each day
        quar_period (int):        number of days to quarantine when notified as a known contact. Default value is ``pars['quar_period']``
        kwargs      (dict):       passed to Intervention()

    **Example**::

        tp = cv.test_prob(symp_prob=0.1, asymp_prob=0.01)
        ct = cv.contact_tracing(trace_probs=0.5, trace_time=2)
        sim = cv.Sim(interventions=[tp, ct]) # Note that without testing, contact tracing has no effect
    '''
    def __init__(self, trace_probs=None, trace_time=None, start_day=0, end_day=None, presumptive=False, quar_period=None, capacity=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.trace_probs = trace_probs
        self.trace_time  = trace_time
        self.start_day   = start_day
        self.end_day     = end_day
        self.presumptive = presumptive
        self.capacity = capacity
        self.quar_period = quar_period # If quar_period is None, it will be drawn from sim.pars at initialization
        return


    def initialize(self, sim):
        ''' Process the dates and dictionaries '''
        super().initialize()
        self.start_day = preprocess_day(self.start_day, sim)
        self.end_day   = preprocess_day(self.end_day,   sim)
        self.days      = [self.start_day, self.end_day]
        if self.trace_probs is None:
            self.trace_probs = 1.0
        if self.trace_time is None:
            self.trace_time = 0.0
        if self.quar_period is None:
            self.quar_period = sim['quar_period']
        if sc.isnumber(self.trace_probs):
            val = self.trace_probs
            self.trace_probs = {k:val for k in sim.people.layer_keys()}
        if sc.isnumber(self.trace_time):
            val = self.trace_time
            self.trace_time = {k:val for k in sim.people.layer_keys()}
        return


    def apply(self, sim):
        '''
        Trace and notify contacts

        Tracing involves three steps that can independently be overloaded or extended
        by derived classes

        - Select which confirmed cases get interviewed by contact tracers
        - Identify the contacts of the confirmed case
        - Notify those contacts that they have been exposed and need to take some action
        '''
        t = sim.t
        start_day = get_day(self.start_day, self, sim)
        end_day   = get_day(self.end_day,   self, sim)
        if t < start_day:
            return
        elif end_day is not None and t > end_day:
            return

        trace_inds = self.select_cases(sim)
        contacts = self.identify_contacts(sim, trace_inds)
        self.notify_contacts(sim, contacts)
        return contacts


    def select_cases(self, sim):
        '''
        Return people to be traced at this time step
        '''
        if not self.presumptive:
            inds = cvu.true(sim.people.date_diagnosed == sim.t) # Diagnosed this time step, time to trace
        else:
            just_tested = cvu.true(sim.people.date_tested == sim.t) # Tested this time step, time to trace
            inds = cvu.itruei(sim.people.exposed, just_tested) # This is necessary to avoid infinite chains of asymptomatic testing

        # If there is a tracing capacity constraint, limit the number of agents that can be traced
        if self.capacity is not None:
            capacity = int(self.capacity / sim.rescale_vec[sim.t])  # Convert capacity into a number of agents
            if len(inds) > capacity:
                inds = np.random.choice(inds, capacity, replace=False)

        return inds


    def identify_contacts(self, sim, trace_inds):
        '''
        Return contacts to notify by trace time

        In the base class, the trace time is the same per-layer, but derived classes might
        provide different functionality e.g. sampling the trace time from a distribution. The
        return value of this method is a dict keyed by trace time so that the `Person` object
        can be easily updated in `contact_tracing.notify_contacts`

        Args:
            sim: Simulation object
            trace_inds: Indices of people to trace

        Returns: {trace_time: np.array(inds)} dictionary storing which people to notify
        '''

        if not len(trace_inds):
            return {}

        contacts = sc.ddict(list)

        for lkey, this_trace_prob in self.trace_probs.items():

            if this_trace_prob == 0:
                continue

            traceable_inds = sim.people.contacts[lkey].find_contacts(trace_inds)
            if len(traceable_inds):
                contacts[self.trace_time[lkey]].extend(cvu.binomial_filter(this_trace_prob, traceable_inds)) # Filter the indices according to the probability of being able to trace this layer

        array_contacts = {}
        for trace_time, inds in contacts.items():
            array_contacts[trace_time] = np.fromiter(inds, dtype=cvd.default_int)

        return array_contacts


    def notify_contacts(self, sim, contacts):
        '''
        Notify contacts

        This method represents notifying people that they have had contact with a confirmed case.
        In this base class, that involves

        - Setting the 'known_contact' flag and recording the 'date_known_contact'
        - Scheduling quarantine

        Args:
            sim: Simulation object
            contacts: {trace_time: np.array(inds)} dictionary storing which people to notify
        '''
        is_dead = cvu.true(sim.people.dead) # Find people who are not alive
        for trace_time, contact_inds in contacts.items():
            contact_inds = np.setdiff1d(contact_inds, is_dead) # Do not notify contacts who are dead
            sim.people.known_contact[contact_inds] = True
            sim.people.date_known_contact[contact_inds] = np.fmin(sim.people.date_known_contact[contact_inds], sim.t + trace_time)
            sim.people.schedule_quarantine(contact_inds, start_date=sim.t + trace_time, period=self.quar_period - trace_time)  # Schedule quarantine for the notified people to start on the date they will be notified
        return



#%% Treatment and prevention interventions

__all__+= ['simple_vaccine', 'BaseVaccination', 'vaccinate', 'vaccinate_prob', 'vaccinate_num']


class simple_vaccine(Intervention):
    '''
    Apply a simple vaccine to a subset of the population. In addition to changing the
    relative susceptibility and the probability of developing symptoms if still
    infected, this intervention stores several types of data:

        - ``doses``:      the number of vaccine doses per person
        - ``vaccination_dates``: list of dates per person
        - ``orig_rel_sus``:      relative susceptibility per person at the beginning of the simulation
        - ``orig_symp_prob``:    probability of developing symptoms per person at the beginning of the simulation
        - ``mod_rel_sus``:       modifier on default susceptibility due to the vaccine
        - ``mod_symp_prob``:     modifier on default symptom probability due to the vaccine

    Args:
        days (int or array): the day or array of days to apply the interventions
        prob      (float): probability of being vaccinated (i.e., fraction of the population)
        rel_sus   (float): relative change in susceptibility; 0 = perfect, 1 = no effect
        rel_symp  (float): relative change in symptom probability for people who still get infected; 0 = perfect, 1 = no effect
        subtarget  (dict): subtarget intervention to people with particular indices (see test_num() for details)
        cumulative (bool): whether cumulative doses have cumulative effects (default false); can also be an array for efficacy per dose, with the last entry used for multiple doses; thus True = [1] and False = [1,0]
        kwargs     (dict): passed to Intervention()

    Note: this intervention is still under development and should be used with caution.
    It is intended for use with use_waning=False.

    **Examples**::

        interv = cv.simple_vaccine(days=50, prob=0.3, rel_sus=0.5, rel_symp=0.1)
        interv = cv.simple_vaccine(days=[10,20,30,40], prob=0.8, rel_sus=0.5, cumulative=[1, 0.3, 0.1, 0]) # A vaccine with efficacy up to the 3rd dose
    '''
    def __init__(self, days, prob=1.0, rel_sus=0.0, rel_symp=0.0, subtarget=None, cumulative=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.days      = sc.dcp(days)
        self.prob      = prob
        self.rel_sus   = rel_sus
        self.rel_symp  = rel_symp
        self.subtarget = subtarget
        if cumulative in [0, False]:
            cumulative = [1,0] # First dose has full efficacy, second has none
        elif cumulative in [1, True]:
            cumulative = [1] # All doses have full efficacy
        self.cumulative = np.array(cumulative, dtype=cvd.default_float) # Ensure it's an array
        return


    def initialize(self, sim):
        ''' Fix the dates and store the doses '''
        super().initialize()
        self.days = process_days(sim, self.days)
        self.doses             = np.zeros(sim.n, dtype=cvd.default_int) # Number of doses given per person of this vaccine
        self.vaccination_dates = [[]] * sim.n # Store the dates when people are vaccinated
        self.orig_rel_sus      = sc.dcp(sim.people.rel_sus) # Keep a copy of pre-vaccination susceptibility
        self.orig_symp_prob    = sc.dcp(sim.people.symp_prob) # ...and symptom probability
        self.mod_rel_sus       = np.ones(sim.n, dtype=cvd.default_float) # Store the final modifiers
        self.mod_symp_prob     = np.ones(sim.n, dtype=cvd.default_float) # Store the final modifiers
        self.vacc_inds         = None
        return


    def apply(self, sim):
        ''' Perform vaccination '''

        # If this day is found in the list, apply the intervention
        for ind in find_day(self.days, sim.t, interv=self, sim=sim):

            # Construct the testing probabilities piece by piece -- complicated, since need to do it in the right order
            vacc_probs = np.full(sim.n, self.prob) # Begin by assigning equal vaccination probability to everyone
            if self.subtarget is not None:
                subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
                vacc_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted
            vacc_inds = cvu.true(cvu.binomial_arr(vacc_probs)) # Calculate who actually gets vaccinated

            # Calculate the effect per person
            vacc_doses = self.doses[vacc_inds] # Calculate current doses
            eff_doses = np.minimum(vacc_doses, len(self.cumulative)-1) # Convert to a valid index
            vacc_eff = self.cumulative[eff_doses] # Pull out corresponding effect sizes
            rel_sus_eff  = (1.0 - vacc_eff) + vacc_eff*self.rel_sus
            rel_symp_eff = (1.0 - vacc_eff) + vacc_eff*self.rel_symp

            # Apply the vaccine to people
            sim.people.rel_sus[vacc_inds]   *= rel_sus_eff
            sim.people.symp_prob[vacc_inds] *= rel_symp_eff

            # Update counters
            self.mod_rel_sus[vacc_inds]   *= rel_sus_eff
            self.mod_symp_prob[vacc_inds] *= rel_symp_eff
            self.doses[vacc_inds] += 1
            for v_ind in vacc_inds:
                self.vaccination_dates[v_ind].append(sim.t)

            # Update vaccine attributes in sim
            sim.people.vaccinated[vacc_inds] = True
            sim.people.doses[vacc_inds] += 1

        return


class BaseVaccination(Intervention):
    '''
    Apply a vaccine to a subset of the population.

    This base class implements the mechanism of vaccinating people to modify their immunity.
    It does not implement allocation of the vaccines, which is implemented by derived classes
    such as `cv.vaccinate`. The idea is that vaccination involves a series of standard operations
    to modify `cv.People` and applications will likely need to modify the vaccine parameters and
    test potentially complex allocation strategies. These should be accounted for by:

        - Custom vaccine parameters being passed in as a dictionary to the vaccine intervention
        - Custom vaccine allocations being implemented by a derived class overloading
          `BaseVaccination.select_people`. Any additional attributes required to manage the allocation
          can be defined in the derived class. Refer to `cv.vaccinate` or `cv.vaccinate_sequential` for
          an example of how to implement this.

    Some quantities are tracked during execution for reporting after running the simulation.
    These are:

        - ``doses``:             the number of vaccine doses per person
        - ``vaccination_dates``: integer; dates of all doses for this vaccine

    Args:
        vaccine (dict/str) : which vaccine to use; see below for dict parameters
        label   (str)      : if vaccine is supplied as a dict, the name of the vaccine
        booster (boolean)  : whether the vaccine is a booster, i.e. whether vaccinated people are eligible
        kwargs  (dict)     : passed to Intervention()

    If ``vaccine`` is supplied as a dictionary, it must have the following parameters:

        EITHER
        - ``nab_init``:  the initial antibody level (higher = more protection)
        - ``nab_boost``: how much of a boost being vaccinated on top of a previous dose or natural infection provides
        OR
        - ``target_eff``: the target efficacy from which to calculate initial antibody and boosting.
        must be supplied as a list, where length of list is equal to number of doses
        - ``nab_eff``:   the waning efficacy of neutralizing antibodies at preventing infection
        - ``doses``:     the number of doses required to be fully vaccinated with this vaccine
        - ``interval``:  the interval between doses (integer)
        - entries for efficacy against each of the variants (e.g. ``b117``)

    See ``parameters.py`` for additional examples of these parameters.


    '''
    def __init__(self, vaccine, label=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.index = None # Index of the vaccine in the sim; set later
        self.label = label # Vaccine label (used as a dict key)
        self.p     = None # Vaccine parameters
        self.doses = None # Record the number of doses given per person *by this intervention*
        self.vaccination_dates = None # Store the dates that each person was last vaccinated *by this intervention*

        self._parse_vaccine_pars(vaccine=vaccine) # Populate
        return


    def _parse_vaccine_pars(self, vaccine=None):
        ''' Unpack vaccine information, which may be given as a string or dict '''

        # Option 1: vaccines can be chosen from a list of pre-defined vaccines
        if isinstance(vaccine, str):

            choices, mapping = cvpar.get_vaccine_choices()
            variant_pars = cvpar.get_vaccine_variant_pars()
            dose_pars = cvpar.get_vaccine_dose_pars()

            label = vaccine.lower()
            for txt in ['.', ' ', '&', '-', 'vaccine']:
                label = label.replace(txt, '')

            if label in mapping:
                label = mapping[label]
                vaccine_pars = sc.mergedicts(variant_pars[label], dose_pars[label])
            else: # pragma: no cover
                errormsg = f'The selected vaccine "{vaccine}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)

            if self.label is None:
                self.label = label

        # Option 2: variants can be specified as a dict of pars
        elif isinstance(vaccine, dict):

            # Parse label
            vaccine_pars = vaccine
            label = vaccine_pars.pop('label', None) # Allow including the label in the parameters
            if self.label is None: # pragma: no cover
                if label is None:
                    self.label = 'custom'
                else:
                    self.label = label

        else: # pragma: no cover
            errormsg = f'Could not understand {type(vaccine)}, please specify as a string indexing a predefined vaccine or a dict.'
            raise ValueError(errormsg)

        # Set label and parameters
        self.p = sc.objdict(vaccine_pars)

        return


    def initialize(self, sim):
        super().initialize()

        # Check that the simulation parameters are correct
        if not sim['use_waning']: # pragma: no cover
            errormsg = f'The cv.{self.__class__.__name__} intervention requires use_waning=True. Please enable waning, or else use cv.simple_vaccine().'
            raise RuntimeError(errormsg)

        # Populate any missing keys -- must be here, after variants are initialized
        default_variant_pars = cvpar.get_vaccine_variant_pars(default=True)
        default_dose_pars    = cvpar.get_vaccine_dose_pars(default=True)
        variant_labels       = list(sim['variant_pars'].keys())
        dose_keys            = list(default_dose_pars.keys())

        # Handle dose keys
        for key in dose_keys:
            if key not in self.p:
                self.p[key] = default_dose_pars[key]

        # Handle variants
        for key in variant_labels:
            if key not in self.p:
                if key in default_variant_pars:
                    val = default_variant_pars[key]
                else: # pragma: no cover
                    val = 1.0
                    if sim['verbose']: print(f'Note: No cross-immunity specified for vaccine {self.label} and variant {key}, setting to 1.0')
                self.p[key] = val

        # If an efficacy target was specified, calculate what NAb level it maps onto
        if 'target_eff' in self.p.keys():
            # check to make sure length is equal to number of doses
            if self.p['doses'] == len(self.p['target_eff']):
                # determine efficacy of first dose (assume efficacy supplied is against symptomatic disease)
                nabs = np.arange(-8, 4, 0.1) # Pick a range of trial NAbs to use
                VE_symp = cvi.calc_VE_symp(2**nabs, sim.pars['nab_eff'])
                peak_nab = nabs[np.argmax(VE_symp>self.p['target_eff'][0])]
                self.p['nab_init'] = dict(dist='normal', par1=peak_nab, par2=2)
                if self.p['doses'] == 2:
                    boosted_nab = nabs[np.argmax(VE_symp>self.p['target_eff'][1])]
                    boost = (2**boosted_nab)/(2**peak_nab)
                    self.p['nab_boost'] = boost
            else: # pragma: no cover
                errormsg = 'Provided mismatching efficacies and doses.'
                raise ValueError(errormsg)

        self.doses = np.zeros(sim['pop_size'], dtype=cvd.default_int) # Number of doses given per person
        self.vaccination_dates = [[] for _ in range(sim.n)] # Store the dates when people are vaccinated

        sim['vaccine_pars'][self.label] = self.p # Store the parameters
        self.index = list(sim['vaccine_pars'].keys()).index(self.label) # Find where we are in the list
        sim['vaccine_map'][self.index]  = self.label # Use that to populate the reverse mapping

        return


    def finalize(self, sim):
        ''' Ensure variables with large memory footprints get erased '''
        super().finalize()
        self.subtarget = None # Reset to save memory
        return


    def select_people(self, sim):
        """
        Return an array of indices of people to vaccinate
        Derived classes must implement this function to determine who to vaccinate at each timestep
        Args:
            sim: A cv.Sim instance
        Returns: Array of person indices
        """
        raise NotImplementedError


    def vaccinate(self, sim, vacc_inds, t=None):
        '''
        Vaccinate people

        This method applies the vaccine to the requested people indices. The indices of people vaccinated
        is returned. These may be different to the requested indices, because anyone that is dead will be
        skipped, as well as anyone already fully vaccinated (if booster=False). This could
        occur if a derived class does not filter out such people in its `select_people` method.

        Args:
            sim: A cv.Sim instance
            vacc_inds: An array of person indices to vaccinate
            t: Optionally override the day on which vaccinations are recorded for historical vaccination

        Returns: An array of person indices of people vaccinated
        '''

        if t is None:
            t = sim.t
        else: # pragma: no cover
            assert t <= sim.t, 'Overriding the vaccination day should only be used for historical vaccination' # High potential for errors to creep in if future vaccines could be scheduled here

        # Perform checks
        vacc_inds = vacc_inds[~sim.people.dead[vacc_inds]] # Skip anyone that is dead
        # Skip anyone that has already had all the doses of *this* vaccine (not counting boosters).
        # Otherwise, they will receive the 2nd dose boost cumulatively for every subsequent dose.
        # Note, this does not preclude someone from getting additional doses of another vaccine (e.g. a booster)
        vacc_inds = vacc_inds[self.doses[vacc_inds] < self.p['doses']]

        # Extract indices of already-vaccinated people and get indices of newly-vaccinated
        prior_vacc = cvu.true(sim.people.vaccinated)
        new_vacc   = np.setdiff1d(vacc_inds, prior_vacc)

        if len(vacc_inds):
            self.doses[vacc_inds] += 1
            for v_ind in vacc_inds:
                self.vaccination_dates[v_ind].append(t)

            sim.people.vaccinated[vacc_inds] = True
            sim.people.vaccine_source[vacc_inds] = self.index
            sim.people.doses[vacc_inds] += 1
            sim.people.date_vaccinated[vacc_inds] = t

            # Update the NAbs, resetting the time for historical vaccination if needed
            orig_t = sim.people.t
            sim.people.t = t
            cvi.update_peak_nab(sim.people, vacc_inds, self.p)
            sim.people.t = orig_t

            if t >= 0: # Only update the flows if it's *not* a historical dose
                factor = sim['pop_scale']/sim.rescale_vec[t] # Scale up by pop_scale, but then down by the current rescale_vec, which gets applied again when results are finalized
                sim.people.flows['new_doses']      += len(vacc_inds)*factor # Count number of doses given
                sim.people.flows['new_vaccinated'] += len(new_vacc)*factor # Count number of people not already vaccinated given doses

        return vacc_inds


    def apply(self, sim):
        ''' Perform vaccination each timestep '''

        inds = self.select_people(sim)
        if len(inds):
            inds = self.vaccinate(sim, inds)
        return inds


    def shrink(self, in_place=True):
        ''' Shrink vaccination intervention '''
        obj = super().shrink(in_place=in_place)
        obj.vaccinated = None
        obj.doses = None
        obj.vaccination_dates = None
        if hasattr(obj, 'second_dose_days'):
            obj.second_dose_days = None
        return obj


def check_doses(doses, interval):
    ''' Check that doses and intervals are supplied in correct formats '''

    # First check that they're both numbers
    if not sc.checktype(doses, int):
        raise ValueError(f'Doses must be an integer, not {doses}.')
    if interval is not None and not sc.isnumber(interval):
        errormsg = f"Can't understand the dosing interval given by '{interval}'. Dosing interval should be a number."
        raise ValueError(errormsg)

    # Now check that they're compatible
    if doses == 1 and interval is not None:
        raise ValueError("Can't use dosing intervals for vaccines with only one dose.")
    elif doses == 2 and interval is None:
        raise ValueError('Must specify a dosing interval if using a vaccine with more than one dose.')
    elif doses > 2:
        raise NotImplementedError('Scheduling three or more doses not yet supported; use a booster vaccine instead')

    return


def process_doses(num_doses, sim):
    ''' Handle different types of dose data'''
    if sc.isnumber(num_doses):
        num_people = num_doses
    elif callable(num_doses):
        num_people = num_doses(sim)
    elif sim.t in num_doses:
        num_people = num_doses[sim.t]
    else:
        num_people = 0
    return num_people


def process_sequence(sequence, sim):
    ''' Handle different types of prioritization sequence for vaccination '''
    if callable(sequence):
        sequence = sequence(sim.people)
    elif sequence == 'age':
        sequence = np.argsort(-sim.people.age)
    elif sequence is None:
        sequence = np.random.permutation(sim.n)
    elif sc.checktype(sequence, 'arraylike'):
        sequence = sc.toarray(sequence)
    else:
        errormsg = f'Unable to interpret sequence {type(sequence)}: must be None, "age", callable, or an array'
        raise TypeError(errormsg)
    return sequence


def vaccinate(*args, **kwargs):
    '''
    Wrapper function for ``vaccinate_prob()`` and ``vaccinate_num()``. If the ``num_doses``
    argument is used, will call ``vaccinate_num()``; else, calls ``vaccinate_prob()``.

    **Examples**::

        vx1 = cv.vaccinate(vaccine='pfizer', days=30, prob=0.7)
        vx2 = cv.vaccinate(vaccine='pfizer', num_doses=100)
    '''
    if 'num_doses' in kwargs:
        return vaccinate_num(*args, **kwargs)
    else:
        return vaccinate_prob(*args, **kwargs)


class vaccinate_prob(BaseVaccination):
    '''
    Probability-based vaccination

    This vaccine intervention allocates vaccines parametrized by the daily probability
    of being vaccinated.

    Args:
        vaccine (dict/str): which vaccine to use; see below for dict parameters
        label        (str): if vaccine is supplied as a dict, the name of the vaccine
        days     (int/arr): the day or array of days to apply the interventions
        prob       (float): probability of being vaccinated (i.e., fraction of the population)
        booster     (bool): whether it's a booster (i.e. targeted to vaccinated people) or not
        subtarget   (dict): subtarget intervention to people with particular indices (see test_num() for details)
        kwargs      (dict): passed to Intervention()

    If ``vaccine`` is supplied as a dictionary, it must have the following parameters:

        - ``nab_eff``:   the waning efficacy of neutralizing antibodies at preventing infection
        - ``nab_init``:  the initial antibody level (higher = more protection)
        - ``nab_boost``: how much of a boost being vaccinated on top of a previous dose or natural infection provides
        - ``doses``:     the number of doses required to be fully vaccinated
        - ``interval``:  the interval between doses (integer)
        - entries for efficacy against each of the strains (e.g. ``b117``)

    See ``parameters.py`` for additional examples of these parameters.

    **Example**::

        pfizer = cv.vaccinate_prob(vaccine='pfizer', days=30, prob=0.7)
        cv.Sim(interventions=pfizer, use_waning=True).run().plot()
    '''
    def __init__(self, vaccine, days, label=None, prob=None, subtarget=None, booster=False, **kwargs):
        super().__init__(vaccine,label=label,**kwargs) # Initialize the Intervention object
        self.days      = sc.dcp(days)
        if prob is None: # Populate default value of probability: 1 if no subtargeting, 0 if subtargeting
            prob = 1.0 if subtarget is None else 0.0
        self.prob      = prob
        self.booster   = booster
        self.subtarget = subtarget
        self.booster   = booster
        self.second_dose_days = None  # Track scheduled second doses
        return


    def initialize(self, sim):
        super().initialize(sim)
        self.days = process_days(sim, self.days) # days that group becomes eligible
        self.second_dose_days     = [None]*sim.npts # People who get second dose (if relevant)
        check_doses(self.p['doses'], self.p['interval'])
        return


    def select_people(self, sim):

        vacc_inds = np.array([], dtype=int)  # Initialize in case no one gets their first dose

        if sim.t >= np.min(self.days):

            # Vaccinate people with their first dose
            for _ in find_day(self.days, sim.t, interv=self, sim=sim):

                vacc_probs = np.zeros(sim['pop_size'])

                # Find eligible people
                vacc_probs[cvu.true(sim.people.dead)] *= 0.0  # Do not vaccinate dead people
                # Eligibility depends on whether it's a booster or not
                # If this is a booster, exclude unvaccinated people; otherwise, exclude vaccinated people
                if self.booster:    eligible_inds = sc.findinds(sim.people.vaccinated)
                else:               eligible_inds = sc.findinds(~sim.people.vaccinated)
                vacc_probs[eligible_inds] = self.prob  # Assign equal vaccination probability to everyone

                # Apply any subtargeting
                if self.subtarget is not None:
                    subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
                    vacc_probs[subtarget_inds] = subtarget_vals  # People being explicitly subtargeted

                vacc_inds = cvu.true(cvu.binomial_arr(vacc_probs))  # Calculate who actually gets vaccinated

                if len(vacc_inds):
                    if self.p.interval is not None:
                        # Schedule the doses
                        next_dose_days = sim.t + self.p.interval
                        if next_dose_days < sim['n_days']:
                            self.second_dose_days[next_dose_days] = vacc_inds

            # Also, if appropriate, vaccinate people with their second dose
            vacc_inds_dose2 = self.second_dose_days[sim.t]
            if vacc_inds_dose2 is not None:
                vacc_inds = np.concatenate((vacc_inds, vacc_inds_dose2), axis=None)

        return vacc_inds


class vaccinate_num(BaseVaccination):
    '''
    This vaccine intervention allocates vaccines in a pre-computed order of
    distribution, at a specified rate of doses per day. Second doses are prioritized
    each day.

    Args:
        vaccine (dict/str): which vaccine to use; see below for dict parameters
        label        (str): if vaccine is supplied as a dict, the name of the vaccine
        booster    (bool): whether it's a booster (i.e. targeted to vaccinated people) or not
        subtarget  (dict): subtarget intervention to people with particular indices (see test_num() for details)
        sequence: Specify the order in which people should get vaccinated. This can be

            - An array of person indices in order of vaccination priority
            - A callable that takes in `cv.People` and returns an ordered sequence. For example, to
              vaccinate people in descending age order, ``def age_sequence(people): return np.argsort(-people.age)``
              would be suitable.
            - The shortcut 'age', which does prioritization by age (see below for implementation)
              If not specified, people will be randomly ordered.
        num_doses: Specify the number of doses per day. This can take three forms

            - A scalar number of doses per day
            - A dict keyed by day/date with the number of doses e.g. ``{2:10000, '2021-05-01':20000}``.
              Any dates are converted to simulation days in `initialize()` which will also copy the
              dictionary passed in.
            - A callable that takes in a ``cv.Sim`` and returns a scalar number of doses. For example,
              ``def doses(sim): return 100 if sim.t > 10 else 0`` would be suitable
        **kwargs: Additional arguments passed to ``cv.BaseVaccination``

    **Example**::
        pfizer = cv.vaccinate_num(vaccine='pfizer', sequence='age', num_doses=100)
        cv.Sim(interventions=pfizer, use_waning=True).run().plot()
    '''

    def __init__(self, vaccine, num_doses, booster=False, subtarget=None, sequence=None, **kwargs):
        super().__init__(vaccine,**kwargs) # Initialize the Intervention object
        self.sequence   = sequence
        self.num_doses  = num_doses
        self.booster    = booster
        self.subtarget  = subtarget
        self._scheduled_doses = sc.ddict(set)  # Track scheduled second doses, where applicable
        return


    def initialize(self, sim):

        super().initialize(sim)

        # Perform checks and process inputs
        if isinstance(self.num_doses, dict): # Convert any dates to simulation days
            self.num_doses = {sim.day(k):v for k, v in self.num_doses.items()}
        self.sequence = process_sequence(self.sequence, sim)
        check_doses(self.p['doses'], self.p['interval'])

        return


    def select_people(self, sim):

        # Work out how many people to vaccinate today
        num_people = process_doses(self.num_doses, sim)
        if num_people == 0:
            self._scheduled_doses[sim.t + 1].update(self._scheduled_doses[sim.t])  # Defer any extras
            return np.array([])
        num_agents = sc.randround(num_people / sim['pop_scale'])

        # First, see how many scheduled second doses we are going to deliver
        if self._scheduled_doses[sim.t]:
            scheduled = np.fromiter(self._scheduled_doses[sim.t], dtype=cvd.default_int) # Everyone scheduled today
            scheduled = scheduled[(self.doses[scheduled] < self.p['doses']) & ~sim.people.dead[scheduled]] # Remove anyone who's already had all doses of this vaccine, also dead people

            # If there are more people due for a second dose than there are doses, vaccinate as many second doses
            # as possible, and add the remainder to tomorrow's doses. At the moment, they don't get priority
            # because the order of the scheduling doesn't matter (so there is a chance someone could go for several days
            # before being allocated their second dose) but then there is some flexibility in the dosing schedules anyway
            # e.g. Pfizer being 3-6 weeks in some jurisdictions
            if len(scheduled) > num_agents:
                np.random.shuffle(scheduled) # Randomly pick who to defer
                self._scheduled_doses[sim.t+1].update(scheduled[num_agents:]) # Defer any extras
                return scheduled[:num_agents]
        else:
            scheduled = np.array([], dtype=cvd.default_int)

        # Next, work out who is eligible for their first dose
        vacc_probs = np.ones(sim.n)  # Begin by assigning equal weight (converted to a probability) to everyone
        vacc_probs[cvu.true(sim.people.dead)] = 0.0  # Dead people are not eligible

        # Apply any subtargeting for this vaccination
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            vacc_probs[subtarget_inds] = vacc_probs[subtarget_inds]*subtarget_vals

        # If this is a booster, exclude unvaccinated people; otherwise, exclude vaccinated people
        if self.booster: vacc_probs[cvu.false(sim.people.vaccinated)] = 0.0
        else:            vacc_probs[cvu.true(sim.people.vaccinated)]  = 0.0 # Anyone who's received at least one dose is counted as vaccinated

        # All remaining people can be vaccinated, although anyone who has received half of a multi-dose
        # vaccine would have had subsequent doses scheduled and therefore should not be selected here
        first_dose_eligible = self.sequence[cvu.binomial_arr(vacc_probs[self.sequence])]

        if len(first_dose_eligible) == 0:
            return scheduled  # Just return anyone that is scheduled

        elif len(first_dose_eligible) > num_agents:
            # Truncate it to the number of agents for performance when checking whether anyone scheduled overlaps with first doses to allocate
            first_dose_eligible = first_dose_eligible[:num_agents] # This is the maximum number of people we could vaccinate this timestep, if there are no second doses allocated

        # It's *possible* that someone has been *scheduled* for a first dose by some other mechanism externally
        # Therefore, we need to check and remove them from the first dose list, otherwise they could be vaccinated
        # twice here (which would amount to wasting a dose)
        first_dose_eligible = first_dose_eligible[~np.isin(first_dose_eligible, scheduled)]

        if (len(first_dose_eligible)+len(scheduled)) > num_agents:
            first_dose_inds = first_dose_eligible[:(num_agents - len(scheduled))]
        else:
            first_dose_inds = first_dose_eligible

        # Schedule subsequent doses
        # For vaccines with >2 doses, scheduled doses will also need to be checked
        if self.p['doses'] > 1:
            self._scheduled_doses[sim.t+self.p.interval].update(first_dose_inds)

        vacc_inds = np.concatenate([scheduled, first_dose_inds])

        return vacc_inds



#%% Prior/historical immunity interventions

__all__ += ['prior_immunity', 'historical_vaccinate_prob', 'historical_wave']


def prior_immunity(*args, **kwargs):
    '''
    Wrapper function for ``historical_wave`` and ``historical_vaccinate_prob``. If the ``vaccine`` keyword is
    present then ``historical_vaccinate_prob`` will be used. Otherwise ``historical_wave`` is used.

    **Examples**::

        pim1 = cv.prior_immunity(vaccine='pfizer', days=[-30], prob=0.7)
        pim2 = cv.prior_immunity(120, 0.05)

    New in version 3.1.0.
    '''

    if 'vaccine' in kwargs:
        return historical_vaccinate_prob(*args, **kwargs)
    else:
        return historical_wave(*args, **kwargs)


class historical_vaccinate_prob(BaseVaccination):
    '''
    Probability-based historical vaccination

    This vaccine intervention allocates vaccines parametrized by the daily probability
    of being vaccinated.  Unlike cv.vaccinate_prob this function allows vaccination
    prior to t=0 (and continuing into the simulation).

    If any people are infected at the t=0 timestep (e.g. seed infections), this
    finds those people and will re-infect  them at the end of the historical vaccination.
    Thus you may have breakthrough infections and this might affect other interventions
    to initialize a population.

    Args:
        vaccine    (dict/str)  : which vaccine to use; see below for dict parameters
        label      (str)       : if vaccine is supplied as a dict, the name of the vaccine
        days       (int/arr)   : the day or array of days to apply the interventions
        prob       (float)     : probability of being vaccinated (i.e., fraction of the population)
        subtarget  (dict)      : subtarget intervention to people with particular indices (see test_num() for details)
        compliance (float/arr) : compliance of the person to take each dose (if scalar then applied per dose)
        kwargs     (dict)      : passed to Intervention()

    If ``vaccine`` is supplied as a dictionary, it must have the following parameters:

        - ``nab_eff``:   the waning efficacy of neutralizing antibodies at preventing infection
        - ``nab_init``:  the initial antibody level (higher = more protection)
        - ``nab_boost``: how much of a boost being vaccinated on top of a previous dose or natural infection provides
        - ``doses``:     the number of doses required to be fully vaccinated
        - ``interval``:  the interval between doses
        - entries for efficacy against each of the strains (e.g. ``b117``)

    See ``parameters.py`` for additional examples of these parameters.

    **Example**::

        pfizer = cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-30,0), prob=0.007) # 30-day vaccination campaign
        cv.Sim(interventions=pfizer).run().plot()

    New in version 3.1.0.
    '''
    def __init__(self,  vaccine, days, label=None, prob=1.0, subtarget=None, compliance=1.0, **kwargs):
        super().__init__(vaccine, label=label, **kwargs)
        self.days      = sc.dcp(days)
        self.prob      = prob
        self.subtarget = subtarget
        self.compliance = sc.dcp(compliance)
        return


    def initialize(self, sim):
        super().initialize(sim)

        if isinstance(self.compliance, (int, float)):
            self.compliance = 2*[self.compliance]
        else:
            if len(self.compliance) != 2:
                raise ValueError('compliance must either be a scalar or 2 element vector')

        # extend nab profiles
        self.extra_days = np.abs(np.min(self.days).astype(cvd.default_int))
        new_nab_length = sim.npts + self.extra_days
        if new_nab_length > len(sim.pars['nab_kin']):
            sim.pars['nab_kin'] = cvi.precompute_waning(length=new_nab_length, pars=sim['nab_decay'])
            if sim.people:
                sim.people.pars['nab_kin'] = sim['nab_kin']

        # handle days
        self.days             = self.process_days(sim, self.days) # days that group becomes eligible
        self.second_dose_days = [None]*new_nab_length # People who get second dose (if relevant)
        self.vaccinated       = [None]*new_nab_length # Keep track of inds of people vaccinated on each day

        # find the seed infections (set during sim.init_people()) and blank them out
        seed_inds = cvu.true(sim.people.date_exposed == 0)
        sim.people.make_naive(seed_inds)

        # administer vaccines before t=0
        times = np.arange(np.min(self.days), 0)
        for t in times: # step through time, init flows, including zeroing out the seed infections.
            sim.people.init_flows()

            # run daily vaccination
            inds = self.select_people(sim, t)
            if len(inds):
                inds = self.vaccinate(sim, inds, t=t)
                sim.results['new_doses'][0] += len(inds)
                sim.results['new_vaccinated'][0] += np.count_nonzero(sim.people.doses[inds] == 1)

            # we need to update the NAbs as it is a cumulative effect
            # this will mess up those who are the seed infections if not reset to naive (see above)
            sim.people.t = t
            to_update = cvu.true(self.doses > 0)  # Update nabs for anyone vaccinated using this intervention
            if len(to_update):
                cvi.update_nab(sim.people, inds=to_update)

        # Re-compute immunity so that seed infection prognoses will reflect the NAb level
        sim.people.t = 0
        cvi.check_immunity(sim.people)

        # Re-infect the seed cases so they get updated prognoses
        sim.people.infect(seed_inds, layer='seed_infection')

        # Restore the time index so that it matches sim.t (noting that these would both usually be 0)
        sim.people.t = sim.t

        return


    def select_people(self, sim, t=None):

        vacc_inds = np.array([], dtype=int)  # Initialize in case no one gets their first dose

        if t is None:
            t = sim.t

        # our vaccination arrays are prepended with extra days
        rel_t = t + self.extra_days

        if t >= np.min(self.days):

            # Vaccinate people with their first dose
            for _ in find_day(self.days, t, interv=self, sim=sim):
                vacc_probs = np.zeros(sim['pop_size'])
                unvacc_inds = sc.findinds(~sim.people.vaccinated)
                if self.subtarget is not None:
                    subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
                    vacc_probs[subtarget_inds] = subtarget_vals  # People being explicitly subtargeted
                else:
                    vacc_probs[unvacc_inds] = self.prob  # Assign equal vaccination probability to everyone
                vacc_probs[cvu.true(sim.people.dead)] *= 0.0  # Do not vaccinate dead people
                vacc_inds = cvu.true(cvu.binomial_arr(vacc_probs))  # Calculate who actually gets vaccinated

                # first dose compliance
                vacc_inds = cvu.binomial_filter(self.compliance[0], vacc_inds)

                if len(vacc_inds):
                    if self.p.interval is not None:
                        next_dose_day = rel_t + self.p.interval
                        if next_dose_day < (sim['n_days'] + self.extra_days):
                            # second dose compliance
                            second_dose_vacc_inds = cvu.binomial_filter(self.compliance[1], vacc_inds)
                            self.second_dose_days[next_dose_day] = second_dose_vacc_inds

            # Also, if appropriate, vaccinate people with their second dose
            vacc_inds_dose2 = self.second_dose_days[rel_t]
            if vacc_inds_dose2 is not None:
                vacc_inds = np.concatenate((vacc_inds, vacc_inds_dose2), axis=None)

        return vacc_inds


    @staticmethod
    def process_days(sim, days, return_dates=False):
        '''
        Ensure lists of days are in consistent format. Used by change_beta, clip_edges,
        and some analyzers.
        Optionally return dates as well as days. If days is callable, leave unchanged.
        '''
        if callable(days):
            return days
        if sc.isstring(days) or not sc.isiterable(days):
            days = sc.tolist(days)
        for d,day in enumerate(days):
            days[d] = preprocess_day(day, sim) # Ensure it's an integer and not a string or something
        days = np.sort(sc.toarray(days)) # Ensure they're an array and in order
        if return_dates:
            dates = [sim.date(day) for day in days] # Store as date strings
            return days, dates
        else:
            return days


    @staticmethod
    def estimate_prob(duration, coverage):
        '''
        Estimate the per-day probability to achieve desired population coverage for a campaign of fixed duration and
        fixed per-day probability of a person being vaccinated

        Args:
            duration: length of campign in days
            coverage: target coverage of campaign

        **Example**::

            prob = historical_vaccinate.estimate_prob(duration=180, coverage=0.70)
        '''
        from scipy import optimize, special # Not used elsewhere, and can't import scipy as sp
        
        def NB_cdf(k, p, r=1):
            '''note that the NB distribution shows the fraction '''
            return 1 - special.betainc(k + 1, r, p)

        # Note that NB distribution is defined as k number of successes *before* r=1 failures (vaccination) occur.
        # Mapping onto the vaccination campaign this means we need k+1 days of a campaign (k days to not be
        # vaccinated prior) before 1 day of getting the vaccine. p is the probability of *not* being vaccinated
        k = duration - 1
        # since the probability of not being vaccinated is ~ 1 and newton method is defined without bounds, we'll use the
        # inverse logit function to map onto [0,1]
        def invlogit(y):
            return np.exp(y)/(np.exp(y)+1)
        # this method can be finicky
        p = optimize.newton(lambda y: NB_cdf(k, invlogit(y)) - coverage, 0, x1=5)
        # p is the probability of *not* being vaccinated per day so we return 1-p
        return 1 - invlogit(p)
    


class historical_wave(Intervention):
    '''
    Imprint a historical (pre t=0) wave of infections in the population NAbs

    Args:
        days_prior (int/str/list) : offset relative to t=0 for the wave (median/par1 value) or median date if a string like "2021-11-15"
        prob       (float/list)   : probability of infection during the wave
        dist       (dict/list)    : passed to covasim.utils.sample to set wave shape (default gaussian with FWHM of 5 weeks)
        subtarget  (dict/list)    : subtarget intervention to people with particular indices  (see test_num() for details)
        variants   (str/list)     : name of variant associated with the wave
        kwargs     (dict)         : passed to Intervention()

    **Example**::
        cv.Sim(interventions=cv.historical_wave(120, 0.30)).run().plot()

    New in version 3.1.0.
    '''

    def __init__(self, days_prior, prob, dist=None, subtarget=None, variant=None, **kwargs):
        super().__init__(**kwargs)
        self.days_prior = sc.dcp(days_prior)
        self.dist = {'dist': 'normal', 'par1': 0, 'par2': 5*7/2.355} if dist is None else sc.dcp(dist) # default is FWHM 5 weeks
        self.prob = sc.dcp(prob)
        self.subtarget = subtarget
        self.variants = 'wild' if variant is None else variant


    def apply(self, sim):
        if sim.t != 0:
            return

        # Check that the simulation parameters are correct
        if not sim['use_waning']: # pragma: no cover
            errormsg = 'cv.historical_wave() requires use_waning=True. Please enable waning.'
            raise RuntimeError(errormsg)
        if sim['rescale'] and sim['pop_scale'] > 1:
            errormsg = 'cv.historical_wave() requires rescale=False, since rescaling assumes non-included agents are naive. Please disable dynamic rescaling.'
            raise RuntimeError(errormsg)

        # deal with values for multiple waves
        if isinstance(self.days_prior, (float, int, str)):
            self.days_prior = sc.tolist(self.days_prior)
        n_waves = len(self.days_prior)
        if not isinstance(self.subtarget, list):
            self.subtarget = n_waves*[self.subtarget]
        if not isinstance(self.prob, list):
            self.prob = n_waves*[self.prob]
        if isinstance(self.dist, dict):
            self.dist = n_waves*[self.dist]
        if isinstance(self.variants, str):
            self.variants = n_waves*[self.variants]

        # we use the people object often
        people = sim.people

        # find the seed infections (set during sim.init_people()) and blank them out
        seed_inds = cvu.true(sim.people.date_exposed == 0)
        people.make_naive(seed_inds)

        # pick variant mapping index (integer value)
        variants = []
        mapping = {v: k for k, v in sim['variant_map'].items()}  # Swap
        for variant in self.variants:
            if variant in mapping:
                variants += [mapping[variant]]
            else:
                errormsg = f'cv.historical_wave() cannot add the new variant "{variant}", must be added to sim via cv.variant(). Current variants are: {sc.strjoin(mapping.keys())}'
                raise ValueError(errormsg)

        # pick individuls for each wave
        inf_offset_days = []
        wave_inds = []
        wave_id = []
        for wave in range(n_waves):
            # per-individual probability to be part of the wave
            wave_probs = np.ones(sim['pop_size']) * self.prob[wave]
            if self.subtarget[wave] is not None:
                subtarget_inds, subtarget_vals = get_subtargets(self.subtarget[wave], sim)
                wave_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted

            # select members of the population to be infected
            this_wave_inds = cvu.true(cvu.binomial_arr(wave_probs)) # Finally, calculate who actually was infected

            days_prior = self.days_prior[wave]
            if isinstance(days_prior, str):
                # Interpret as sim day
                days_prior = sc.daydiff(days_prior, sim['start_day'])

            # select day for those to be infected / exposed
            this_inf_offset_days = cvu.sample(**self.dist[wave], size=len(this_wave_inds)) - days_prior

            # require that all offsets are before the start of the sim
            filtered_wave_inds = cvu.true(this_inf_offset_days <= 0)
            if len(filtered_wave_inds) == 0: # pragma: no cover
                warnmsg = f'Wave with days_prior of {days_prior} and prob of {self.prob} did not result in any historical infections - skipping this wave'
                cvm.warn(warnmsg)
                continue

            wave_inds = wave_inds + this_wave_inds[filtered_wave_inds].tolist()
            inf_offset_days = inf_offset_days + np.round(this_inf_offset_days[filtered_wave_inds]).astype(cvd.default_int).tolist()
            wave_id += len(filtered_wave_inds)*[wave]

        if len(wave_id) == 0: # pragma: no cover
            warnmsg = 'No waves resulted in any infections prior to the start of the simulation'
            cvm.warn(warnmsg)
            return

        wave_id = np.array(wave_id)
        wave_inds = np.array(wave_inds)
        inf_offset_days = np.array(inf_offset_days)

        if len(wave_id) != len(inf_offset_days): # pragma: no cover
            raise  RuntimeError(f'arrays mismatch: {len(wave_id)} != {len(inf_offset_days)}')

        # we will need to extend the nab profiles
        new_nab_length = sim.npts - np.floor(np.min(inf_offset_days)).astype(cvd.default_int)
        if new_nab_length > len(sim.pars['nab_kin']):
            sim.pars['nab_kin'] = cvi.precompute_waning(length=new_nab_length, pars=sim['nab_decay'])
            people.pars['nab_kin'] = sim['nab_kin']

        # update nab, states, and count flows
        flow_keys_to_save = ['new_infections', 'new_reinfections']
        flow_variant_keys_to_save = ['new_infections_by_variant', 'new_symptomatic_by_variant', 'new_severe_by_variant']
        nv = sim['n_variants']
        for t in np.arange(np.min(inf_offset_days), 0):

            flows = {fkey:0 for fkey in flow_keys_to_save}
            flows_variant = {fkey:[0 for v in range(nv)] for fkey in flow_variant_keys_to_save}
            for wave in range(n_waves):
                inds = cvu.true(np.logical_and(inf_offset_days == t, wave_id == wave))

                # set infection
                people.t = t
                people.infect(wave_inds[inds], layer='historical', variant=variants[wave])

            for fkey in flow_keys_to_save:
                flows[fkey] += people.flows[fkey]
            for v in range(nv):
                for fkey in flow_variant_keys_to_save:
                    flows_variant[fkey][v] += people.flows_variant[fkey][v]

            # this is potentially an issue with multiple waves close together as someone who is technically still
            # exposed from the first wave would be re-exposed during the second (assuming they are recovered by t=0)
            people.update_states_pre(t=t)

            # Update counts for t=0 step: flows
            # Does this count the seed infections twice?
            for key,count in people.flows.items():
                sim.results[key][0] += count

            for key,count in people.flows_variant.items():
                for variant in range(nv):
                    sim.results['variant'][key][variant][0] += count[variant]

            for key,count in flows.items():
                sim.results[key][0] += count

            for key,count in flows_variant.items():
                for v in range(nv):
                    sim.results['variant'][key][v][0] += count[v]


            # we need to update the NAbs as it is a cumulative effect
            # this will mess up those who are the seed infections if not reset to naive (see above)
            sim.people.t = t
            has_nabs = cvu.true(sim.people.peak_nab)
            if len(has_nabs):
                cvi.update_nab(sim.people, inds=has_nabs)

        # update states for t=0
        people.update_states_pre(t=0)

        # reset the seed infections
        sim.people.infect(seed_inds, layer='seed_infection')

        return
