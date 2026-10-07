"""
Specify the core interventions available in Covasim: testing, contact tracing, vaccination,
prior immunity, and changes to transmission and parameters. Other interventions can be defined
by the user by inheriting from these classes.
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

from . import compat as cvc
from . import utils as cvu
from . import parameters as cvpar
from . import immunity as cvimm
from . import misc as cvm
from . import covid as cvcovid

__all__ = ['InterventionDict', 'Intervention', 'test_num', 'test_prob', 'contact_tracing']


def InterventionDict(which, pars):
    """
    Generate an intervention from a dictionary. Although a function, it acts
    like a class, since it returns a class instance.

    **Example**::

        interv = cv.InterventionDict(which='change_beta', pars={'days': 30, 'changes': 0.5, 'layers': None})
    """
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
    except KeyError as E:
        available = ', '.join(mapping.keys())
        errormsg = f'Only interventions "{available}" are available in dictionary representation, not "{which}"'
        raise sc.KeyNotFoundError(errormsg) from E
    intervention = IntervClass(**pars)
    return intervention


class Intervention(cvc.V3Module, ss.Intervention):
    """
    Base class for interventions. A custom intervention defines ``step(self)`` (or, as in v3,
    ``apply(self, sim)``), which is called on each timestep, and optionally ``init_post(self)``
    (or ``initialize(self, sim)``).

    Args:
        label      (str):  a label for the intervention (used for plotting, and for ease of identification)
        show_label (bool): whether to include the label in the legend
        do_plot    (bool): whether to plot the intervention
        line_args  (dict): arguments passed to ax.axvline() when plotting

    **Example**::

        class halve_beta(cv.Intervention):
            def step(self):
                if self.ti == 30:
                    self.sim['beta'] = self.sim['beta']/2

        sim = cv.Sim(interventions=halve_beta()).run()
    """
    start_day = None # The first day on which the intervention is active (if used)
    end_day   = None # The last day on which the intervention is active (if used)

    def init_post(self):
        super().init_post()
        # v3 accepted date strings (or datetimes) for start_day/end_day; convert them to day indices.
        for attr in ('start_day', 'end_day'):
            val = getattr(self, attr, None)
            if val is not None and not isinstance(val, (int, np.integer, float, np.floating)):
                setattr(self, attr, self.sim.day(val))
        return

    def is_active(self, ti):
        """Whether the intervention is active on day index ``ti`` (within [start_day, end_day])."""
        if self.start_day is not None and ti < self.start_day:
            return False
        if self.end_day is not None and ti > self.end_day:
            return False
        return True


def get_quar_inds(quar_policy, sim):
    """
    Return the UIDs of people in quarantine who should be tested, based on the quarantine
    testing policy (the v3 ``get_quar_inds``). Used by test_num and test_prob.

    Args:
        quar_policy (str, int, list, func): 'start', people entering quarantine; 'end', people leaving; 'both', entering and leaving; 'daily', every day in quarantine; a number or list of numbers, the days after the start of quarantine to test; or a function ``quar_policy(sim)`` that returns the UIDs
        sim (Sim): the simulation object
    """
    covid = sim.diseases.covid
    t = sim.ti
    if   quar_policy is None:    quar_test_inds = np.array([], dtype=int)
    elif quar_policy == 'start': quar_test_inds = (covid.date_quarantined == t-1).uids # Actually do the day after since testing usually happens before contact tracing
    elif quar_policy == 'end':   quar_test_inds = (covid.date_end_quarantine == t+1).uids # +1 since they are released on date_end_quarantine, so do the day before
    elif quar_policy == 'both':  quar_test_inds = np.concatenate([(covid.date_quarantined == t-1).uids, (covid.date_end_quarantine == t+1).uids])
    elif quar_policy == 'daily': quar_test_inds = covid.quarantined.uids
    elif sc.isnumber(quar_policy) or (sc.isiterable(quar_policy) and not sc.isstring(quar_policy)):
        quar_test_inds = np.unique(np.concatenate([(covid.date_quarantined == t-1-q).uids for q in sc.toarray(quar_policy)]))
    elif callable(quar_policy):
        quar_test_inds = quar_policy(sim)
    else:
        errormsg = f'Quarantine policy "{quar_policy}" not recognized: must be a string (start, end, both, daily), int, list, array, set, tuple, or function'
        raise ValueError(errormsg)
    return np.asarray(quar_test_inds, dtype=int)


def process_daily_data(daily_data, sim, start_day, as_int=False):
    """
    Convert daily data (e.g. the number of tests) to an array indexed by the days since the
    intervention started (the v3 ``process_daily_data``).

    Args:
        daily_data (str, number, array, dataframe, or series): if a number, convert to an array of the right length; if a Pandas series or dataframe with a date index, reindex to match the start day; if a string, use that column of the sim's data ('data' means 'new_tests')
        sim (Sim): the simulation object
        start_day (int): the first day of the intervention
        as_int (bool): whether to convert a number to an integer
    """
    if sc.isstring(daily_data):
        key = 'new_tests' if daily_data == 'data' else daily_data
        try:
            daily_data = sim.data[key]
        except Exception as E:
            errormsg = f'Tried to load testing data from sim.data["{key}"], but that failed: {str(E)}.\nPlease ensure data are loaded into the sim and the column exists.'
            raise ValueError(errormsg) from E

    if sc.isnumber(daily_data):
        if as_int: daily_data = int(daily_data)
        daily_data = np.array([daily_data] * sim.npts)
    elif isinstance(daily_data, (pd.Series, pd.DataFrame)):
        start_date = sc.datedelta(sim['start_day'], days=start_day)
        end_date = daily_data.index[-1]
        dateindex = pd.date_range(start_date, end_date)
        daily_data = daily_data.reindex(dateindex, fill_value=0).to_numpy()
    return daily_data


def randround(x, dist):
    """
    Round a number up or down to an integer at random, in proportion to its fractional part
    (e.g. 2.3 becomes 3 with probability 0.3), using ``dist`` (an ``ss.bernoulli``) rather
    than NumPy's global random number generator. A number that isn't finite gives 0.
    """
    if not np.isfinite(x):
        return 0
    n = int(np.floor(x))
    dist.set(p=x - n)
    n += int(dist.rvs(1)[0])
    return n


class BaseTest(Intervention):
    """
    Base class for test_num and test_prob, with the arguments they share (see test_num()
    for details), the quarantine testing policy, influenza-like illness, and the swab delay.
    """

    def __init__(self, quar_policy=None, subtarget=None, ili_prev=None, sensitivity=1.0, loss_prob=0.0,
                 test_delay=0, start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs)
        self.quar_policy = quar_policy if quar_policy else 'start'
        self.subtarget   = subtarget
        self.ili_prev    = ili_prev
        self.sensitivity = sensitivity
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        self.pdf         = cvu.get_pdf(**sc.mergedicts(swab_delay)) # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        self._choose_ili = ss.choose_n() # Who has influenza-like illness on each day
        self._round      = ss.bernoulli(p=0.0) # For rounding the number of tests up or down at random
        return

    def init_post(self):
        super().init_post()
        self.ili_prev = process_daily_data(self.ili_prev, self.sim, self.start_day)
        return

    def get_symp_time(self, symp_inds):
        """Days since symptom onset, for the swab delay"""
        return (self.ti - self.sim.diseases.covid.ti_symptomatic[symp_inds]).astype(int)

    def get_swab_factor(self, symp_inds):
        """
        For the swab delay: the days since symptom onset of each symptomatic person, and the factor
        by which to multiply their chance of testing, so that the delays from onset to testing follow
        the swab delay distribution
        """
        symp_time = self.get_symp_time(symp_inds)
        inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
        count = np.nan * np.ones(inv_count.shape) # Initialize the count
        count[inv_count != 0] = 1/inv_count[inv_count != 0] # Update the counts where defined
        factor = self.pdf.pdf(symp_time) * count[symp_time] # Put it all together
        return symp_time, factor

    def get_ili_inds(self, symp_inds):
        """Choose people with influenza-like illness (independent of COVID symptoms) on this day"""
        if self.ili_prev is None:
            return np.array([], dtype=int)
        rel_t = self.ti - self.start_day
        if rel_t >= len(self.ili_prev):
            return np.array([], dtype=int)
        self._choose_ili.set(n=int(self.ili_prev[rel_t] * len(self.sim.people))) # Number with ILI symptoms on this day
        ili_inds = self._choose_ili.filter()
        return np.setdiff1d(ili_inds, symp_inds)


class test_prob(BaseTest):
    """
    Assign each person a probability of being tested for COVID based on their
    symptom state, quarantine state, and other states. Unlike test_num, the
    total number of tests is not specified, but rather is an output.

    Args:
        symp_prob        (float)     : probability of testing a symptomatic (unquarantined) person
        asymp_prob       (float)     : probability of testing an asymptomatic (unquarantined) person (default: 0)
        symp_quar_prob   (float)     : probability of testing a symptomatic quarantined person (default: same as symp_prob)
        asymp_quar_prob  (float)     : probability of testing an asymptomatic quarantined person (default: same as asymp_prob)
        quar_policy      (str)       : policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget        (dict)      : subtarget intervention to people with particular indices (see test_num() for details)
        ili_prev         (float/arr) : prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        sensitivity      (float)     : test sensitivity (default 100%, i.e. no false negatives)
        loss_prob        (float)     : probability of the person being lost-to-follow-up (default 0%, i.e. no one lost to follow-up)
        test_delay       (int)       : days for test result to be known (default 0, i.e. results available instantly)
        start_day        (int/str)   : day the intervention starts (default: 0, i.e. first day of the simulation)
        end_day          (int/str)   : day the intervention ends (default: no end)
        swab_delay       (dict)      : distribution for the delay from onset to swab, e.g. dict(dist='lognormal', par1=10, par2=170); if this is present, it is used instead of test_delay. For 'lognormal', par2 is the variance, not the standard deviation (see cv.utils.get_pdf())
        kwargs           (dict)      : passed to Intervention()

    **Examples**::

        interv = cv.test_prob(symp_prob=0.1, asymp_prob=0.01) # Test 10% of symptomatics and 1% of asymptomatics
        interv = cv.test_prob(symp_prob=0.1, symp_quar_prob=0.4) # Test 40% of those in quarantine with symptoms
    """

    def __init__(self, symp_prob, asymp_prob=0.0, symp_quar_prob=None, asymp_quar_prob=None, quar_policy=None, subtarget=None, ili_prev=None,
                 sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(quar_policy=quar_policy, subtarget=subtarget, ili_prev=ili_prev, sensitivity=sensitivity, loss_prob=loss_prob,
                         test_delay=test_delay, start_day=start_day, end_day=end_day, swab_delay=swab_delay, **kwargs)
        self.symp_prob       = symp_prob
        self.asymp_prob      = asymp_prob
        self.symp_quar_prob  = symp_quar_prob  if symp_quar_prob  is not None else symp_prob
        self.asymp_quar_prob = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self._select = ss.bernoulli(p=0.0) # Who is tested
        return

    def step(self):
        if not self.is_active(self.ti):
            return
        covid = self.sim.diseases.covid
        alive = self.sim.people.auids

        # Find probability of testing for symptomatic people, optionally with a swab delay
        symp_inds = covid.symptomatic.uids
        symp_prob = self.symp_prob
        if self.pdf:
            symp_time, swab_factor = self.get_swab_factor(symp_inds)
            symp_prob = np.ones(len(symp_time))
            inds = 1 > (symp_time*self.symp_prob)
            symp_prob[inds] = self.symp_prob/(1-symp_time[inds]*self.symp_prob)
            symp_prob = symp_prob * swab_factor

        # Define the groups of people, as boolean arrays by UID (much faster than set operations on the UIDs)
        n = len(covid.symptomatic.raw)
        ili_inds = self.get_ili_inds(symp_inds)
        is_symp = np.zeros(n, dtype=bool)
        is_symp[symp_inds] = True
        is_asymp = np.zeros(n, dtype=bool)
        is_asymp[alive] = True
        is_asymp[symp_inds] = False
        is_asymp[ili_inds] = False
        is_quar_test = np.zeros(n, dtype=bool)
        is_quar_test[get_quar_inds(self.quar_policy, self.sim)] = True

        # Assign testing probabilities by UID
        test_probs = np.zeros(n)
        test_probs[symp_inds] = symp_prob                                 # People with symptoms (true positive)
        test_probs[ili_inds]  = self.symp_prob                            # People with symptoms (false positive) -- can't use swab delay since no date symptomatic
        test_probs[is_asymp]  = self.asymp_prob                           # People without symptoms
        test_probs[is_quar_test & is_symp]  = self.symp_quar_prob         # People with symptoms in quarantine
        test_probs[is_quar_test & is_asymp] = self.asymp_quar_prob        # People without symptoms in quarantine
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, self.sim)
            test_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted
        test_probs[covid.diagnosed.uids] = 0.0 # People who are diagnosed don't test

        # Test
        self._select.set(p=test_probs[alive])
        test_uids = alive[self._select.rvs(alive)]
        if len(test_uids):
            covid.test(test_uids, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob,
                       test_delay=self.test_delay)
        return


class test_num(BaseTest):
    """
    Test the specified number of people per day. Useful for including historical
    testing data. The probability of a given person getting a test is dependent
    on the total number of tests, population size, and odds ratios. Compare this
    intervention with cv.test_prob().

    Args:
        daily_tests (arr)   : number of tests per day, can be int, array, or dataframe/series; if integer, use that number every day; if 'data' or another string, use loaded data
        symp_test   (float) : odds ratio of a symptomatic person testing (default: 100x more likely)
        quar_test   (float) : probability of a person in quarantine testing (default: no more likely)
        quar_policy (str)   : policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget   (dict)  : subtarget intervention to people with particular indices (format: {'inds': array of indices, or function to return indices from the sim, 'vals': value(s) to apply})
        ili_prev    (arr)   : prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        sensitivity (float) : test sensitivity (default 100%, i.e. no false negatives)
        loss_prob   (float) : probability of the person being lost-to-follow-up (default 0%, i.e. no one lost to follow-up)
        test_delay  (int)   : days for test result to be known (default 0, i.e. results available instantly)
        start_day   (int/str) : day the intervention starts (default: 0, i.e. first day of the simulation)
        end_day     (int/str) : day the intervention ends (default: no end)
        swab_delay  (dict)  : distribution for the delay from onset to swab, e.g. dict(dist='lognormal', par1=10, par2=170); if this is present, it is used instead of test_delay. For 'lognormal', par2 is the variance, not the standard deviation (see cv.utils.get_pdf())
        kwargs      (dict)  : passed to Intervention()

    **Examples**::

        interv = cv.test_num(daily_tests=[0.10*n_people]*npts)
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': cv.true(sim.people.age>50), 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': lambda sim: cv.true(sim.people.age>50), 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests='data') # Take number of tests from loaded data using default column name (new_tests)
        interv = cv.test_num(daily_tests='swabs_per_day') # Take number of tests from loaded data using a custom column name
    """

    def __init__(self, daily_tests, symp_test=100.0, quar_test=1.0, quar_policy=None, subtarget=None, ili_prev=None,
                 sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(quar_policy=quar_policy, subtarget=subtarget, ili_prev=ili_prev, sensitivity=sensitivity, loss_prob=loss_prob,
                         test_delay=test_delay, start_day=start_day, end_day=end_day, swab_delay=swab_delay, **kwargs)
        self.daily_tests = daily_tests
        self.symp_test   = symp_test
        self.quar_test   = quar_test
        self._choose_tests = ss.choose_n() # Who is tested
        return

    def init_post(self):
        super().init_post()
        self.daily_tests = process_daily_data(self.daily_tests, self.sim, self.start_day)
        return

    def step(self):
        ti = self.ti
        if not self.is_active(ti):
            return

        # Check that there are tests today, correcting for the population scale factor
        rel_t = ti - self.start_day
        if rel_t >= len(self.daily_tests):
            return
        n_tests = randround(self.daily_tests[rel_t]/self.sim.current_scale, self._round)
        if not n_tests:
            return

        # Assign testing weights by UID, starting with equal weight for everyone alive
        covid = self.sim.diseases.covid
        alive = self.sim.people.auids
        test_probs = np.zeros(len(covid.symptomatic.raw))
        test_probs[alive] = 1.0

        # Handle symptomatic testing, optionally with a swab delay
        symp_inds = covid.symptomatic.uids
        symp_test = self.symp_test
        if self.pdf:
            symp_time, swab_factor = self.get_swab_factor(symp_inds)
            symp_test = symp_test * swab_factor
        test_probs[symp_inds] *= symp_test

        # Handle the other groups
        test_probs[self.get_ili_inds(symp_inds)] *= self.symp_test
        test_probs[get_quar_inds(self.quar_policy, self.sim)] *= self.quar_test
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, self.sim)
            test_probs[subtarget_inds] = test_probs[subtarget_inds]*subtarget_vals
        test_probs[covid.diagnosed.uids] = 0.0

        # With dynamic rescaling, correct for the uninfected people outside of the population who would test
        n_requested = n_tests # As in v3, record the number of tests requested, which includes those outside the population
        scale = self.sim.current_scale
        pop_scale = self.sim.pars.pop_scale
        if scale < pop_scale: # We still have rescaling to do
            in_pop_tot_prob = test_probs.sum()*scale # Total "testing weight" of people in the subsampled population
            out_pop_tot_prob = (pop_scale - scale)*len(self.sim.people) # Find out how many people are missing and assign them each weight 1
            in_frac = in_pop_tot_prob/(in_pop_tot_prob + out_pop_tot_prob) # Fraction of tests which should fall in the sample population
            n_tests = randround(n_tests*in_frac, self._round) # Recompute the number of tests

        # Choose who tests, without replacement, weighted by the testing probabilities
        eligible = ss.uids(test_probs.nonzero()[0])
        self._choose_tests.set(n=n_tests, weights=test_probs[eligible])
        chosen = self._choose_tests.filter(eligible)
        covid.test(chosen, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay, n_tests=n_requested)
        return


class contact_tracing(Intervention):
    """
    Contact tracing of people who are diagnosed. When a person is diagnosed positive
    (by either test_num() or test_prob(); this intervention has no effect if there
    is not also a testing intervention active), a certain proportion of the index
    case's contacts (defined by trace_probs) are contacted after a certain number
    of days (defined by trace_time). After they are contacted, they are placed
    into quarantine (with effectiveness quar_factor, a simulation parameter) for
    a certain period (defined by quar_period). They may also change their testing
    probability, if test_prob() is defined.

    Tracing involves three steps that can independently be overridden or extended by
    derived classes: select_cases(), identify_contacts(), and notify_contacts().

    Args:
        trace_probs (float/dict): probability of tracing, per layer (default: 100%, i.e. everyone is traced); a layer not in the dict is not traced
        trace_time  (float/dict): days required to trace, per layer (default: 0, i.e. no delay); rounded to a whole number of days
        start_day   (int/str):    intervention start day (default: 0, i.e. the start of the simulation)
        end_day     (int/str):    intervention end day (default: no end)
        presumptive (bool):       whether or not to begin isolation and contact tracing on the presumption of a positive diagnosis (default: no)
        quar_period (int):        number of days to quarantine when notified as a known contact (default: the sim's ``quar_period``)
        capacity    (int):        optionally specify a maximum number of newly diagnosed people to trace each day
        kwargs      (dict):       passed to Intervention()

    **Example**::

        tp = cv.test_prob(symp_prob=0.1, asymp_prob=0.01)
        ct = cv.contact_tracing(trace_probs=0.5, trace_time=2)
        sim = cv.Sim(interventions=[tp, ct]) # Note that without testing, contact tracing has no effect
    """

    def __init__(self, trace_probs=None, trace_time=None, start_day=0, end_day=None,
                 presumptive=False, quar_period=None, capacity=None, **kwargs):
        super().__init__(**kwargs)
        self.trace_probs = trace_probs
        self.trace_time  = trace_time
        self.start_day   = start_day
        self.end_day     = end_day
        self.presumptive = presumptive
        self.quar_period = quar_period
        self.capacity    = capacity
        self._trace = ss.bernoulli(p=1.0) # Which contacts are traced
        self._choose_capacity = ss.choose_n() # Who is traced, if there are more index cases than the capacity
        return

    def init_post(self):
        """ Fill in the defaults, and expand trace_probs and trace_time into dicts with a value for each layer """
        super().init_post()
        layer_keys = list(self.sim.networks.keys())
        trace_probs = 1.0 if self.trace_probs is None else self.trace_probs
        trace_time  = 0.0 if self.trace_time  is None else self.trace_time
        if not isinstance(trace_probs, dict):
            trace_probs = {lkey:trace_probs for lkey in layer_keys}
        if not isinstance(trace_time, dict):
            trace_time = {lkey:trace_time for lkey in layer_keys}
        self.trace_probs = {lkey:float(trace_probs.get(lkey, 0.0)) for lkey in layer_keys} # A layer that isn't listed isn't traced
        self.trace_time  = {lkey:int(round(trace_time.get(lkey, 0.0))) for lkey in layer_keys}
        if self.quar_period is None:
            self.quar_period = self.sim.diseases.covid.pars.quar_period
        return

    def step(self):
        """ Trace and notify contacts """
        if not self.is_active(self.ti):
            return
        trace_inds = self.select_cases(self.sim)
        contacts = self.identify_contacts(self.sim, trace_inds)
        self.notify_contacts(self.sim, contacts)
        return contacts

    def select_cases(self, sim):
        """
        Return the people to be traced on this timestep: those diagnosed today or, if
        presumptive, those tested today who are infected
        """
        covid = sim.diseases.covid
        if not self.presumptive:
            inds = (covid.date_diagnosed == sim.ti).uids # Diagnosed this time step, time to trace
        else:
            just_tested = (covid.date_tested == sim.ti).uids # Tested this time step, time to trace
            inds = just_tested[covid.exposed[just_tested]] # This is necessary to avoid infinite chains of asymptomatic testing

        # If there is a tracing capacity constraint, limit the number of agents that can be traced
        if (self.capacity is not None) and len(inds):
            self._choose_capacity.set(n=int(self.capacity/sim.current_scale)) # Convert capacity into a number of agents
            inds = self._choose_capacity.filter(inds)
        return inds

    def identify_contacts(self, sim, trace_inds):
        """
        Return the contacts to notify, by trace time.

        In the base class, the trace time is the same for everyone in a layer, but derived classes
        might provide different functionality, e.g. sampling the trace time from a distribution.

        Args:
            sim: the simulation object
            trace_inds: the UIDs of the people to trace

        Returns:
            a dict ``{trace_time: uids}`` of the people to notify
        """
        if not len(trace_inds):
            return {}
        contacts = {}
        for lkey, trace_prob in self.trace_probs.items():
            if trace_prob == 0:
                continue
            traceable = ss.uids(sim.networks[lkey].find_contacts(trace_inds))
            if len(traceable):
                self._trace.set(p=trace_prob)
                traced = traceable[self._trace.rvs(traceable)] # Filter the contacts according to the probability of being able to trace this layer
                contacts.setdefault(self.trace_time[lkey], []).append(traced)
        return {trace_time:ss.uids(np.unique(np.concatenate(uids))) for trace_time, uids in contacts.items()}

    def notify_contacts(self, sim, contacts):
        """
        Notify people that they have had contact with a confirmed case: set their
        ``known_contact`` flag, and schedule their quarantine to start on the day they
        are notified.

        Args:
            sim: the simulation object
            contacts: a dict ``{trace_time: uids}`` of the people to notify
        """
        covid = sim.diseases.covid
        for trace_time, contact_inds in contacts.items():
            contact_inds = contact_inds[~covid.dead[contact_inds]] # Do not notify contacts who are dead
            if len(contact_inds):
                covid.known_contact[contact_inds] = True
                covid.schedule_quarantine(contact_inds, start_date=sim.ti + trace_time, period=max(1, self.quar_period - trace_time))
        return


# %% Vaccination interventions -----------------------------------------------------------------

__all__ += ['BaseVaccination', 'vaccinate', 'vaccinate_prob', 'vaccinate_num', 'simple_vaccine']


def check_doses(doses, interval):
    """ Check that doses and intervals are supplied in correct formats """
    if not isinstance(doses, (int, np.integer)):
        raise ValueError(f'Doses must be an integer, not {doses}.')
    if interval is not None and not np.isscalar(interval):
        raise ValueError(f"Dosing interval should be a number, not {interval!r}.")
    if doses == 1 and interval is not None:
        raise ValueError("Can't use a dosing interval for a single-dose vaccine.")
    if doses == 2 and interval is None:
        raise ValueError('Must specify a dosing interval for a 2-dose vaccine.')
    if doses > 2:
        raise NotImplementedError('Three or more scheduled doses are not supported; use a booster.')
    return


def get_subtargets(subtarget, sim):
    """
    Return the indices and values of a subtarget, used by the testing and vaccination interventions.

    ``subtarget`` is a dict ``{'inds': ..., 'vals': ...}`` (or a function ``f(sim)`` returning one).
    ``inds`` (the UIDs of the people subtargeted) and ``vals`` (a value for each, e.g. a probability)
    may each be a function ``f(sim)``; a single number for ``vals`` applies to everyone in ``inds``.
    Returns ``(inds, vals)`` as arrays (``vals`` may be None).
    """
    if callable(subtarget):
        subtarget = subtarget(sim)
    if 'inds' not in subtarget:
        raise ValueError(f"A subtarget must have keys 'inds' and 'vals'; got {subtarget}.")
    inds = subtarget['inds']
    if callable(inds):
        inds = inds(sim)
    inds = np.asarray(inds)
    vals = subtarget.get('vals', None)
    if callable(vals):
        vals = vals(sim)
    if vals is not None:
        vals = np.asarray(vals, dtype=float)
        if vals.ndim == 0:
            vals = np.full(len(inds), float(vals))
    return inds, vals


def apply_subtargets(probs, pool, subtarget, sim):
    """ Replace the probabilities ``probs`` (one for each UID in ``pool``) with the subtarget values for the people subtargeted """
    if subtarget is None:
        return probs
    inds, vals = get_subtargets(subtarget, sim)
    if vals is None or not len(inds):
        return probs
    pool = np.asarray(pool)
    sel = np.isin(inds, pool)                       # subtarget UIDs that are in the pool
    if sel.any():
        sorter = np.argsort(pool)
        pos = sorter[np.searchsorted(pool, inds[sel], sorter=sorter)]
        probs[pos] = vals[sel]
    return probs


class BaseVaccination(Intervention):
    """
    Apply a vaccine to a subset of the population.

    This base class implements the mechanism of vaccinating people to modify their immunity
    (through neutralizing antibodies, so it requires ``use_waning=True``; otherwise, use
    ``cv.simple_vaccine``). It does not implement allocation of the vaccines, which is implemented
    by derived classes such as ``cv.vaccinate_prob`` and ``cv.vaccinate_num``. Custom vaccine
    allocations can be implemented by a derived class overriding ``select_people(sim)``, which
    returns the UIDs of the people to vaccinate on the current timestep.

    The number of doses of this vaccine given to each person (indexed by UID) is stored in ``doses``.

    Args:
        vaccine   (dict/str) : which vaccine to use; see below for dict parameters
        label     (str)      : if vaccine is supplied as a dict, the name of the vaccine
        booster   (bool)     : whether the vaccine is a booster, i.e. whether vaccinated people are eligible
        subtarget (dict)     : subtarget intervention to people with particular indices (see test_num() for details)
        kwargs    (dict)     : passed to Intervention()

    If ``vaccine`` is supplied as a dictionary, it must have the following parameters:

        EITHER
        - ``nab_init``:  the initial antibody level (higher = more protection)
        - ``nab_boost``: how much of a boost being vaccinated on top of a previous dose or natural infection provides
        OR
        - ``target_eff``: the target efficacy from which to calculate initial antibody and boosting; must be supplied as a list, where the length of the list is equal to the number of doses
        AND
        - ``doses``:     the number of doses required to be fully vaccinated with this vaccine
        - ``interval``:  the interval between doses (integer)
        - entries for efficacy against each of the variants (e.g. ``b117``)

    See ``parameters.py`` for additional examples of these parameters.

    **Example**::

        class vaccinate_over_70(cv.BaseVaccination):
            def select_people(self, sim):
                if sim.ti == 5:
                    return cv.true(sim.people.age > 70)
                return np.array([], dtype=int)

        sim = cv.Sim(interventions=vaccinate_over_70('pfizer'), use_waning=True).run()
    """

    def __init__(self, vaccine, label=None, booster=False, subtarget=None, **kwargs):
        super().__init__(**kwargs)
        self.index = None # Index of the vaccine in the sim; set later
        self.label = label # Vaccine label (used as a dict key)
        self.p = None # Vaccine parameters
        self.booster = booster
        self.subtarget = subtarget
        self.doses = None # The number of doses given to each person by this intervention
        self._parse_vaccine_pars(vaccine)
        return

    def _parse_vaccine_pars(self, vaccine):
        """ Unpack vaccine information, which may be given as a string or dict """
        if isinstance(vaccine, str):
            choices, mapping = cvpar.get_vaccine_choices()
            variant_pars = cvpar.get_vaccine_variant_pars()
            dose_pars = cvpar.get_vaccine_dose_pars()
            label = vaccine.lower()
            for txt in ['.', ' ', '&', '-', 'vaccine']:
                label = label.replace(txt, '')
            if label not in mapping:
                raise NotImplementedError(f'Vaccine "{vaccine}" not known; choices: {sorted(choices)}')
            label = mapping[label]
            vaccine_pars = sc.mergedicts(variant_pars[label], dose_pars[label])
            if self.label is None:
                self.label = label
        elif isinstance(vaccine, dict):
            vaccine_pars = dict(vaccine)
            label = vaccine_pars.pop('label', None)
            if self.label is None:
                self.label = label if label is not None else 'custom'
        else:
            raise ValueError(f'Could not understand vaccine {type(vaccine)}; use a product name or a dict.')
        self.p = sc.objdict(vaccine_pars)
        return

    def register_vaccine(self, covid):
        """Populate missing dose/variant pars, back-calculate target_eff, and register in the module."""
        default_dose = cvpar.get_vaccine_dose_pars(default=True)
        default_var  = cvpar.get_vaccine_variant_pars(default=True)
        for key in default_dose:                       # fill missing dose pars (nab_init/nab_boost/doses/interval)
            if key not in self.p:
                self.p[key] = default_dose[key]
        for key in covid.variant_map.values():         # per-variant efficacy for every variant in the sim
            if key not in self.p:
                self.p[key] = default_var.get(key, 1.0)
        # target_eff -> back-calculate nab_init/nab_boost (v3 BaseVaccination.initialize)
        if 'target_eff' in self.p:
            if self.p['doses'] != len(self.p['target_eff']):
                raise ValueError('target_eff length must equal the number of doses.')
            nabs = np.arange(-8, 4, 0.1)
            nab_eff = self.p.get('nab_eff', covid.pars.nab_eff)  # Use the vaccine's own nab_eff if supplied
            VE_symp = cvimm.calc_VE_symp(2 ** nabs, nab_eff)
            peak_nab = nabs[np.argmax(VE_symp > self.p['target_eff'][0])]
            self.p['nab_init'] = dict(dist='normal', par1=float(peak_nab), par2=2)
            if self.p['doses'] == 2:
                boosted = nabs[np.argmax(VE_symp > self.p['target_eff'][1])]
                self.p['nab_boost'] = float((2 ** boosted) / (2 ** peak_nab))
        check_doses(self.p['doses'], self.p['interval'])
        covid.vaccine_pars[self.label] = self.p
        self.index = list(covid.vaccine_pars.keys()).index(self.label)
        covid.vaccine_map[self.index] = self.label
        self.doses = np.zeros(len(covid.rel_sus.raw), dtype=int)
        return

    def init_post(self):
        super().init_post()
        covid = self.sim.diseases.covid
        if not covid.pars.use_waning:
            raise RuntimeError(f'cv.{type(self).__name__} requires use_waning=True; else use cv.simple_vaccine().')
        self.register_vaccine(covid)
        return

    def select_people(self, sim):
        """
        Return the UIDs of the people to vaccinate on this timestep. Derived classes must
        implement this method to determine who to vaccinate.

        Args:
            sim: the simulation object
        """
        raise NotImplementedError

    def vaccinate(self, sim, vacc_inds, t=None):
        """
        Vaccinate people.

        This applies the vaccine to the requested people, and returns the UIDs of the people vaccinated.
        These may be fewer than were requested, since anyone who is dead, or who has already had all the
        doses of this vaccine, is skipped.

        Args:
            sim: the simulation object
            vacc_inds: the UIDs of the people to vaccinate
            t: the day of vaccination, if before the start of the sim (for historical vaccination)
        """
        covid = sim.diseases.covid
        uids = ss.uids(np.unique(vacc_inds))
        uids = uids[~covid.dead[uids]] # Skip anyone who is dead
        uids = uids[self.doses[uids] < self.p['doses']] # Skip anyone who has already had all the doses of this vaccine
        if len(uids):
            self.doses[uids] += 1
            covid.vaccinate_agents(uids, self.label, self.index)
            if t is not None: # Back-date the dose
                covid.date_vaccinated[uids] = t
                covid.t_nab_event[uids] = t
        return uids

    def step(self):
        uids = self.select_people(self.sim)
        if (uids is not None) and len(uids):
            uids = self.vaccinate(self.sim, uids)
        return uids


class vaccinate_prob(BaseVaccination):
    """
    Probability-based vaccination

    This vaccine intervention allocates vaccines parametrized by the daily probability
    of being vaccinated. People given a first dose of a two-dose vaccine are given
    their second dose after the vaccine's dosing interval.

    Args:
        vaccine (dict/str): which vaccine to use; see BaseVaccination for dict parameters
        days     (int/arr): the day or array of days to apply the interventions
        label        (str): if vaccine is supplied as a dict, the name of the vaccine
        prob       (float): probability of being vaccinated (i.e., fraction of the population); default 1 with no subtarget, or 0 with one (so only the people subtargeted are vaccinated)
        subtarget   (dict): subtarget intervention to people with particular indices (see test_num() for details)
        booster     (bool): whether it's a booster (i.e. targeted to vaccinated people) or not
        kwargs      (dict): passed to Intervention()

    **Example**::

        pfizer = cv.vaccinate_prob(vaccine='pfizer', days=30, prob=0.7)
        cv.Sim(interventions=pfizer, use_waning=True).run().plot()
    """

    def __init__(self, vaccine, days, label=None, prob=None, subtarget=None, booster=False, **kwargs):
        super().__init__(vaccine, label=label, booster=booster, subtarget=subtarget, **kwargs)
        self.days = days
        if prob is None: # Populate default value of probability: 1 if no subtargeting, 0 if subtargeting
            prob = 1.0 if subtarget is None else 0.0
        self.prob = prob
        self._second_dose = None # People who are due a second dose: {day: [UIDs]}
        self._select = ss.bernoulli(p=0.0) # Who is given a first dose
        return

    def init_post(self):
        super().init_post()
        self.days = process_days(self.sim, self.days) # Days that the group becomes eligible
        self._second_dose = {}
        return

    def select_first_doses(self, sim, t):
        """ Choose the people to give a first dose to on day t """
        covid = sim.diseases.covid
        alive = sim.people.auids
        vaccinated = covid.vaccinated[alive]
        if self.booster:
            eligible = alive[vaccinated] # If this is a booster, only vaccinated people are eligible
        else:
            eligible = alive[~vaccinated]
        probs = np.full(len(eligible), float(self.prob)) # Assign equal vaccination probability to everyone
        probs = apply_subtargets(probs, eligible, self.subtarget, sim) # Apply any subtargeting
        self._select.set(p=probs)
        return eligible[self._select.rvs(eligible)]

    def schedule_second_doses(self, t, uids):
        """ Schedule the second dose for people given a first dose on day t """
        interval = self.p['interval']
        if (interval is not None) and len(uids):
            self._second_dose.setdefault(t + int(interval), []).append(uids)
        return

    def select_people(self, sim, t=None):
        """ Return the people to give a first dose to on day t (by default, today), and those due a second dose """
        if t is None:
            t = sim.ti
        vacc_inds = [ss.uids()]
        if len(find_day(self.days, t, interv=self, sim=sim)):
            first = self.select_first_doses(sim, t)
            self.schedule_second_doses(t, first)
            vacc_inds.append(first)
        vacc_inds += self._second_dose.pop(t, []) # Also, if appropriate, vaccinate people with their second dose
        return ss.uids(np.unique(np.concatenate(vacc_inds)))


class vaccinate_num(BaseVaccination):
    """
    This vaccine intervention allocates vaccines in a pre-computed order of
    distribution, at a specified rate of doses per day. Second doses are prioritized
    each day.

    Args:
        vaccine (dict/str): which vaccine to use; see BaseVaccination for dict parameters
        num_doses: Specify the number of doses per day. This can take three forms

            - A scalar number of doses per day
            - A dict keyed by day/date with the number of doses e.g. ``{2:10000, '2021-05-01':20000}``.
              Any dates are converted to simulation days when the sim is initialized.
            - A callable that takes in a ``cv.Sim`` and returns a scalar number of doses. For example,
              ``def doses(sim): return 100 if sim.ti > 10 else 0`` would be suitable
        booster    (bool): whether it's a booster (i.e. targeted to vaccinated people) or not
        subtarget  (dict): subtarget intervention to people with particular indices (see test_num() for details); the values multiply each person's chance of being offered a first dose
        sequence: Specify the order in which people should get vaccinated. This can be

            - An array of person indices in order of vaccination priority
            - A callable that takes in ``cv.People`` and returns an ordered sequence. For example, to
              vaccinate people in descending age order, ``def age_sequence(people): return np.argsort(-people.age)``
              would be suitable.
            - The shortcut 'age', which does prioritization by age (see below for implementation)
              If not specified, people will be randomly ordered.
        label       (str): if vaccine is supplied as a dict, the name of the vaccine
        kwargs     (dict): passed to Intervention()

    **Example**::

        pfizer = cv.vaccinate_num(vaccine='pfizer', sequence='age', num_doses=100)
        cv.Sim(interventions=pfizer, use_waning=True).run().plot()
    """

    def __init__(self, vaccine, num_doses, booster=False, subtarget=None, sequence=None, label=None, **kwargs):
        super().__init__(vaccine, label=label, booster=booster, subtarget=subtarget, **kwargs)
        self.num_doses = num_doses
        self.sequence = sequence
        self._sequence = None
        self._scheduled = {} # People who are due a second dose: {day: set of UIDs}
        self._random_order = ss.random() # The order in which to vaccinate people, if no sequence is given
        self._eligible = ss.bernoulli(p=0.0) # Who is eligible for a first dose on each day, if subtargeted
        self._round = ss.bernoulli(p=0.0) # For rounding the number of doses up or down at random
        return

    def init_post(self):
        super().init_post()
        if isinstance(self.num_doses, dict): # Convert any dates to simulation days
            self.num_doses = {self.sim.day(k):v for k,v in self.num_doses.items()}
        self._sequence = self.process_sequence(self.sequence)
        return

    def process_sequence(self, sequence):
        """ Handle the different types of prioritization sequence for vaccination """
        alive = np.asarray(self.sim.people.auids)
        if sequence is None:
            return alive[np.argsort(self._random_order.rvs(ss.uids(alive)))]
        if sc.isstring(sequence) and sequence == 'age':
            ages = self.sim.people.age[ss.uids(alive)]
            return alive[np.argsort(-ages)]
        if callable(sequence):
            return np.asarray(sequence(self.sim.people))
        return np.asarray(sequence)

    def get_num_doses(self, ti):
        nd = self.num_doses
        if callable(nd):
            return int(nd(self.sim))
        if isinstance(nd, dict):
            return int(nd.get(ti, 0))
        return int(nd)

    def select_people(self, sim):
        """ Return the people due a second dose, then those next in the sequence for a first dose, up to the number of doses today """
        ti = sim.ti
        covid = sim.diseases.covid
        n_doses = self.get_num_doses(ti)
        scheduled_today = self._scheduled.pop(ti, set())
        if n_doses <= 0:
            if scheduled_today:
                self._scheduled.setdefault(ti + 1, set()).update(scheduled_today) # Defer any extras
            return ss.uids()
        n_agents = randround(n_doses / sim.pars.pop_scale, self._round)

        # First, see how many scheduled second doses we are going to deliver
        scheduled = np.fromiter(scheduled_today, dtype=int) # Everyone scheduled today
        scheduled = scheduled[~covid.dead[ss.uids(scheduled)] & (self.doses[scheduled] < self.p['doses'])] # Remove anyone who's already had all doses of this vaccine, also dead people
        if len(scheduled) > n_agents: # If there are more people due for a second dose than there are doses, add the remainder to tomorrow's doses
            self._scheduled.setdefault(ti + 1, set()).update(scheduled[n_agents:].tolist())
            return ss.uids(np.sort(scheduled[:n_agents]))

        # Next, work out who is eligible for their first dose: everyone alive, with any subtargeting
        alive = sim.people.auids
        vacc_probs = np.zeros(len(covid.vaccinated.raw))
        vacc_probs[alive] = 1.0
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            if subtarget_vals is not None:
                vacc_probs[subtarget_inds] = vacc_probs[subtarget_inds]*subtarget_vals
        if self.booster: # If this is a booster, exclude unvaccinated people; otherwise, exclude vaccinated people
            vacc_probs[~covid.vaccinated.raw] = 0.0
        else:
            vacc_probs[covid.vaccinated.raw] = 0.0
        self._eligible.set(p=np.clip(vacc_probs[alive], 0, 1))
        is_eligible = np.zeros(len(vacc_probs), dtype=bool)
        is_eligible[alive[self._eligible.rvs(alive)]] = True

        # Take people in the order of the sequence, skipping anyone already scheduled
        first_pool = self._sequence[is_eligible[self._sequence]]
        first_pool = first_pool[~np.isin(first_pool, scheduled)]
        n_first = max(0, n_agents - len(scheduled))
        first = first_pool[:n_first]
        if (self.p['doses'] > 1) and len(first): # Schedule the second doses
            self._scheduled.setdefault(ti + int(self.p['interval']), set()).update(first.tolist())
        vacc_inds = np.concatenate([scheduled, first])
        return ss.uids(np.sort(np.unique(vacc_inds)))


def vaccinate(*args, **kwargs):
    """
    Wrapper function for ``vaccinate_prob()`` and ``vaccinate_num()``. If the ``num_doses``
    argument is used, will call ``vaccinate_num()``; else, calls ``vaccinate_prob()``.

    **Examples**::

        vx1 = cv.vaccinate(vaccine='pfizer', days=30, prob=0.7)
        vx2 = cv.vaccinate(vaccine='pfizer', num_doses=100)
    """
    if 'num_doses' in kwargs:
        return vaccinate_num(*args, **kwargs)
    return vaccinate_prob(*args, **kwargs)


class simple_vaccine(Intervention):
    """
    Apply a simple vaccine to a subset of the population. Rather than conferring neutralizing
    antibodies, this directly changes the relative susceptibility and the probability of
    developing symptoms if still infected. The number of doses given to each person (indexed
    by UID) is stored in ``doses``.

    Args:
        days       (int/arr/str): the day or array of days to apply the interventions
        prob       (float): probability of being vaccinated (i.e., fraction of the population)
        rel_sus    (float): relative change in susceptibility; 0 = perfect, 1 = no effect
        rel_symp   (float): relative change in symptom probability for people who still get infected; 0 = perfect, 1 = no effect
        subtarget  (dict): subtarget intervention to people with particular indices (see test_num() for details)
        cumulative (bool): whether cumulative doses have cumulative effects (default false); can also be an array for efficacy per dose, with the last entry used for multiple doses; thus True = [1] and False = [1,0]
        kwargs     (dict): passed to Intervention()

    Note: this intervention is intended for use with use_waning=False.

    **Examples**::

        interv = cv.simple_vaccine(days=50, prob=0.3, rel_sus=0.5, rel_symp=0.1)
        interv = cv.simple_vaccine(days=[10,20,30,40], prob=0.8, rel_sus=0.5, cumulative=[1, 0.3, 0.1, 0]) # A vaccine with efficacy up to the 3rd dose
    """

    def __init__(self, days, prob=1.0, rel_sus=0.0, rel_symp=0.0, subtarget=None, cumulative=False, **kwargs):
        super().__init__(**kwargs)
        self.days = days
        self.prob = prob
        self.rel_sus = rel_sus
        self.rel_symp = rel_symp
        self.subtarget = subtarget
        if cumulative in [0, False]:
            cumulative = [1, 0] # First dose has full efficacy, second has none
        elif cumulative in [1, True]:
            cumulative = [1] # All doses have full efficacy
        self.cumulative = np.array(cumulative, dtype=float)
        self.doses = None # The number of doses given to each person by this intervention
        self._select = ss.bernoulli(p=0.0) # Who is vaccinated
        return

    def init_post(self):
        super().init_post()
        self.days = process_days(self.sim, self.days)
        self.doses = np.zeros(len(self.sim.diseases.covid.rel_sus.raw), dtype=int)
        return

    def step(self):
        if not len(find_day(self.days, self.ti, interv=self, sim=self.sim)):
            return
        covid = self.sim.diseases.covid
        alive = self.sim.people.auids
        probs = np.full(len(alive), float(self.prob)) # Begin by assigning equal vaccination probability to everyone
        probs = apply_subtargets(probs, alive, self.subtarget, self.sim) # People being explicitly subtargeted
        self._select.set(p=probs)
        vacc = alive[self._select.rvs(alive)] # Calculate who actually gets vaccinated
        if not len(vacc):
            return

        # Calculate the effect per person
        eff_doses = np.minimum(self.doses[vacc], len(self.cumulative) - 1) # Convert the current doses to a valid index
        vacc_eff = self.cumulative[eff_doses] # Pull out the corresponding effect sizes
        rel_sus_eff  = (1.0 - vacc_eff) + vacc_eff * self.rel_sus
        rel_symp_eff = (1.0 - vacc_eff) + vacc_eff * self.rel_symp

        # Apply the vaccine to people
        covid.rel_sus[vacc]   = covid.rel_sus[vacc] * rel_sus_eff
        covid.symp_prob[vacc] = covid.symp_prob[vacc] * rel_symp_eff

        # Update the counters
        prior = covid.vaccinated[vacc]
        self.doses[vacc] += 1
        covid.vaccinated[vacc] = True
        covid.doses[vacc] = covid.doses[vacc] + 1
        covid.count_doses(vacc, prior)
        return


# %% Prior (historical) immunity interventions -----------------------------------------------------

__all__ += ['historical_vaccinate_prob', 'historical_wave', 'prior_immunity']


class historical_vaccinate_prob(vaccinate_prob):
    """
    Probability-based historical vaccination

    This vaccine intervention allocates vaccines parametrized by the daily probability
    of being vaccinated. Unlike cv.vaccinate_prob, this allows vaccination before the
    start of the sim (negative days), continuing into the simulation. The days before the
    start are stepped through when the sim is initialized, giving first doses and any
    second doses that are due, so people start the sim with waned NAbs; second doses due
    on or after the start are given during the sim. The doses given before the start are
    counted in the results on day 0. Requires ``use_waning=True``.

    Args:
        vaccine    (dict/str)  : which vaccine to use; see BaseVaccination for dict parameters
        days       (int/arr)   : the day or array of days to apply the interventions; negative days are before the start of the sim (so -1 is the day before the start)
        label      (str)       : if vaccine is supplied as a dict, the name of the vaccine
        prob       (float)     : probability of being vaccinated (i.e., fraction of the population); default 1 with no subtarget, or 0 with one
        subtarget  (dict)      : subtarget intervention to people with particular indices (see test_num() for details)
        compliance (float/arr) : probability of a person taking each dose; a single number applies to both doses, or give two numbers for the first and second
        kwargs     (dict)      : passed to Intervention()

    **Example**::

        pfizer = cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-30,0), prob=0.007) # 30-day vaccination campaign
        cv.Sim(interventions=pfizer, use_waning=True).run().plot()
    """

    def __init__(self, vaccine, days, label=None, prob=None, subtarget=None, compliance=1.0, **kwargs):
        super().__init__(vaccine, days, label=label, prob=prob, subtarget=subtarget, **kwargs)
        compliance = sc.toarray(compliance).astype(float)
        if len(compliance) == 1:
            compliance = np.array([compliance[0], compliance[0]]) # The same for both doses
        if len(compliance) != 2:
            errormsg = f'compliance must be a single number or two numbers (for the first and second doses), not {compliance}'
            raise ValueError(errormsg)
        self.compliance = compliance
        self._comply = ss.bernoulli(p=1.0) # Who takes each dose
        return

    def init_post(self):
        BaseVaccination.init_post(self) # Not vaccinate_prob.init_post(), since -1 is the day before the start here, rather than the last day of the sim
        self.days = np.sort(sc.toarray(self.sim.day(self.days)))
        self._second_dose = {}

        # Extend the NAb waning kernel to cover the days before the start, as in v3
        covid = self.sim.diseases.covid
        first_day = min(0, int(self.days.min()))
        n_kin = covid.t.npts - first_day
        if len(covid.nab_kin) < n_kin:
            covid.nab_kin = cvimm.precompute_waning(n_kin, covid.pars.nab_decay)

        # Step through the days before the start of the sim, vaccinating people and updating their NAbs
        for t in range(first_day, 0):
            uids = self.select_people(self.sim, t)
            if len(uids):
                self.vaccinate(self.sim, uids, t=t)
            self.update_historical_nab(covid, t)
        return

    def update_historical_nab(self, covid, t):
        """ Step the NAbs of the people vaccinated by this intervention forward a day, on day t before the start of the sim """
        uids = ss.uids(self.doses.nonzero()[0])
        if len(uids):
            t_since = (t - covid.t_nab_event[uids]).astype(int)
            peak = covid.peak_nab[uids]
            nab = covid.nab[uids] + covid.nab_kin[t_since]*peak
            covid.nab[uids] = np.clip(nab, 0, peak) # NAbs can't go below 0 or above the peak
        return

    def filter_compliance(self, uids, dose):
        """ Keep the people who comply with this dose (0 for the first, 1 for the second) """
        self._comply.set(p=self.compliance[dose])
        return uids[self._comply.rvs(uids)]

    def select_first_doses(self, sim, t):
        """ Choose the people to give a first dose to on day t, allowing for compliance """
        uids = super().select_first_doses(sim, t)
        return self.filter_compliance(uids, 0)

    def schedule_second_doses(self, t, uids):
        """ Schedule the second dose for people given a first dose on day t, allowing for compliance """
        if self.p['interval'] is not None:
            uids = self.filter_compliance(uids, 1)
        return super().schedule_second_doses(t, uids)


class historical_wave(Intervention):
    """
    Imprint a historical (pre t=0) wave of infections in the population.

    At the start of the first timestep, agents are infected as if on a day before the start of the sim,
    with the usual prognoses, so by day 0 most have recovered (with waned NAbs), some have died, and any
    infected shortly before the start are still infectious. As in v3, the infections, recoveries, deaths
    etc. before the start are counted in the results on day 0. Requires ``use_waning=True``.

    Args:
        days_prior (int/str/list): offset relative to t=0 for the wave (the median of the default distribution), or the median date if a string like "2021-11-15"
        prob       (float/list):   probability of infection during the wave
        dist       (dict/list):    the shape of the wave, as a v3 distribution, e.g. dict(dist='normal', par1=0, par2=15) (default: normal with a full width at half maximum of 5 weeks)
        subtarget  (dict/list):    subtarget the wave to people with particular indices (see test_num() for details)
        variant    (str/list):     name of the variant associated with the wave (default wild)
        kwargs     (dict):         passed to Intervention()

    For multiple waves, supply lists; a single value applies to every wave.

    **Examples**::

        cv.Sim(interventions=cv.historical_wave(120, 0.30)).run().plot()
        cv.Sim(interventions=cv.historical_wave(days_prior=[300, 120], prob=[0.1, 0.3])).run().plot()
    """

    def __init__(self, days_prior, prob, dist=None, subtarget=None, variant=None, **kwargs):
        super().__init__(**kwargs)
        default_dist = dict(dist='normal', par1=0, par2=5*7/2.355) # Default is a full width at half maximum of 5 weeks
        self.days_prior = sc.tolist(days_prior)
        n_waves = len(self.days_prior)
        self.prob      = self.per_wave(prob, n_waves)
        self.dist      = self.per_wave(default_dist if dist is None else dist, n_waves)
        self.subtarget = self.per_wave(subtarget, n_waves)
        self.variants  = self.per_wave('wild' if variant is None else variant, n_waves)
        self._select = [ss.bernoulli(p=0.0) for wave in range(n_waves)] # Who is infected in each wave
        self._timing = [cvcovid.v3_durs(dict(wave=wave_dist))['dur_wave'] for wave_dist in self.dist] # When they are infected, in whole days
        self.day0_flows = {} # Counts of infections etc. before the start of the sim, which are added to the results on day 0
        self.day0_flows_variant = {}
        return

    @staticmethod
    def per_wave(val, n_waves):
        """A list is one value per wave; otherwise, use the same value for every wave"""
        return list(val) if isinstance(val, list) else [val]*n_waves

    def init_post(self):
        super().init_post()
        sim = self.sim
        covid = self.sim.diseases.covid
        if not covid.pars.use_waning:
            raise RuntimeError('cv.historical_wave() requires use_waning=True.')
        if sim.pars.rescale and (sim['pop_scale'] > 1):
            errormsg = 'cv.historical_wave() requires rescale=False, since rescaling assumes non-included agents are naive. Please disable dynamic rescaling.'
            raise RuntimeError(errormsg)
        mapping = {label:ind for ind,label in covid.variant_map.items()}
        for variant in self.variants:
            if variant not in mapping:
                errormsg = f'cv.historical_wave() cannot add the new variant "{variant}", must be added to sim via cv.variant(). Current variants are: {sc.strjoin(mapping.keys())}'
                raise ValueError(errormsg)
        return

    def start_step(self):
        """Infect the people in each wave, before the COVID module updates the states on the first timestep (which applies the recoveries, deaths, etc.)"""
        super().start_step()
        if self.ti != 0:
            return
        sim = self.sim
        covid = self.sim.diseases.covid
        mapping = {label:ind for ind,label in covid.variant_map.items()}

        flows_variant_before = {key:val.copy() for key,val in covid.flows_by_variant.items()}
        flows = dict(reinfections=-covid.flows['reinfections'], symptomatic=0, severe=0, critical=0, recoveries=0)
        flows_variant = dict(new_symptomatic=np.zeros(covid.nv), new_severe=np.zeros(covid.nv))

        # Infect the people in each wave
        dates = [covid.ti_infected, covid.ti_exposed, covid.ti_infectious, covid.ti_symptomatic, covid.ti_severe, covid.ti_critical, covid.ti_recovered, covid.ti_dead, covid.ti_vl_switch]
        alive = sim.people.auids
        for wave, days_prior in enumerate(self.days_prior):
            if isinstance(days_prior, str): # Interpret as a date
                days_prior = sc.daydiff(days_prior, sim['start_day'])

            # Choose who is infected, and when (days relative to the start of the sim, so negative)
            probs = apply_subtargets(np.full(len(alive), self.prob[wave]), alive, self.subtarget[wave], sim)
            self._select[wave].set(p=probs)
            uids = alive[self._select[wave].rvs(alive)]
            days = self._timing[wave].rvs(uids) - days_prior
            before_start = days <= 0 # As in v3, skip infections that would be after the start of the sim
            susceptible = covid.susceptible[uids] | (covid.ti_recovered[uids] <= days) # Not infected in an earlier wave, or have since recovered; this also excludes the seed infections (v3 included them, then infected them again, which counted them twice)
            keep = before_start & susceptible
            uids, days = uids[keep], days[keep]
            reinfected = uids[~covid.susceptible[uids]] # Infected in an earlier wave: count those outcomes now, since they are replaced by the new infection
            flows['symptomatic'] += np.count_nonzero(covid.ti_symptomatic.notnan[reinfected])
            flows['severe']      += np.count_nonzero(covid.ti_severe.notnan[reinfected])
            flows['critical']    += np.count_nonzero(covid.ti_critical.notnan[reinfected])
            flows['recoveries']  += len(reinfected)
            flows_variant['new_symptomatic'] += np.bincount(covid.exposed_variant[reinfected[covid.ti_symptomatic.notnan[reinfected]]].astype(int), minlength=covid.nv)
            flows_variant['new_severe']      += np.bincount(covid.exposed_variant[reinfected[covid.ti_severe.notnan[reinfected]]].astype(int), minlength=covid.nv)
            if not len(uids):
                warnmsg = f'Wave with days_prior of {days_prior} and prob of {self.prob[wave]} did not result in any historical infections - skipping this wave'
                cvm.warn(warnmsg)
                continue

            # Infect them as if on day 0, then move the dates of their infection and outcomes back
            covid.set_prognoses(uids, variant=mapping[self.variants[wave]])
            for date in dates:
                date[uids] = date[uids] + days
            for day in np.unique(days): # Wane their NAbs from the day they were infected
                covid.imprint_historical_nab(uids[days == day], day)
            covid.infection_log.add_data(uids, day=days, network='historical')

        # Store the counts to add to the results in step(), since the COVID module resets them when it updates the states
        flows['reinfections'] += covid.flows['reinfections']
        self.day0_flows = flows
        self.day0_flows_variant = {key:val - flows_variant_before[key] + flows_variant.get(key, 0) for key,val in covid.flows_by_variant.items()}
        return

    def step(self):
        """As in v3, count the infections before the start of the sim in the results on day 0"""
        if self.ti == 0:
            covid = self.sim.diseases.covid
            for key,val in self.day0_flows.items():
                covid.flows[key] += val
            for key,val in self.day0_flows_variant.items():
                covid.flows_by_variant[key] += val
        return


def prior_immunity(*args, **kwargs):
    """
    Wrapper function for ``historical_wave`` and ``historical_vaccinate_prob``. If the ``vaccine`` keyword is
    present then ``historical_vaccinate_prob`` will be used. Otherwise ``historical_wave`` is used.

    **Examples**::

        pim1 = cv.prior_immunity(vaccine='pfizer', days=[-30], prob=0.7)
        pim2 = cv.prior_immunity(120, 0.05)
    """
    if 'vaccine' in kwargs:
        return historical_vaccinate_prob(*args, **kwargs)
    return historical_wave(*args, **kwargs)


# %% Beta / parameter / meta interventions ----------------------------------------------------------

__all__ += ['change_beta', 'clip_edges', 'dynamic_pars', 'sequence']


def find_day(arr, t=None, interv=None, sim=None, which='first'):
    """
    Find which days of an intervention match the current timestep.

    Args:
        arr (list/function): list of days in the intervention, or a boolean array; or a function ``arr(interv, sim)`` that returns these
        t (int): current simulation timestep
        which (str): what to return: 'first', 'last', or 'all' indices
        interv (intervention): the intervention object (usually self); only used if arr is callable
        sim (sim): the simulation object; only used if arr is callable

    Returns:
        inds (list): list of matching days; length zero or one unless which is 'all'
    """
    if callable(arr):
        arr = sc.toarray(arr(interv, sim))
    all_inds = sc.findinds(arr=arr, val=t)
    if len(all_inds) == 0 or which == 'all':
        inds = all_inds
    elif which == 'first':
        inds = [all_inds[0]]
    elif which == 'last':
        inds = [all_inds[-1]]
    else:
        errormsg = f'Argument "which" must be "first", "last", or "all", not "{which}"'
        raise ValueError(errormsg)
    return inds


def process_days(sim, days, return_dates=False):
    """
    Ensure lists of days are in a consistent format: a sorted array of day indices. Used by
    change_beta, clip_edges, the vaccination interventions, and some analyzers. If a day is 'end'
    or -1, use the final day of the simulation. Optionally return dates as well as days. If days
    is callable, leave it unchanged.

    Args:
        sim (Sim): the simulation object
        days (int/str/date/list/function): the day(s), as day indices, date strings, or dates
        return_dates (bool): whether to also return the days as date strings
    """
    if callable(days):
        return days
    days = sc.tolist(days)
    for d, day in enumerate(days):
        if sc.isstring(day) and day == 'end':
            day = sim.npts - 1
        elif sc.isnumber(day) and day == -1:
            day = sim.npts - 1
        days[d] = sim.day(day) # Ensure it's an integer and not a string or something
    days = np.sort(np.array(days, dtype=int)) # Ensure they're an array and in order
    if return_dates:
        dates = [sim.date(day) for day in days] # Store as date strings
        return days, dates
    return days


def process_changes(changes, days):
    """ Ensure lists of changes are in a consistent format, matching the days. Used by change_beta and clip_edges. """
    changes = sc.toarray(changes).astype(float)
    if not callable(days):
        if len(changes) == 1:
            changes = np.full(len(days), changes[0])
        elif len(changes) != len(days):
            errormsg = f'Number of days supplied ({len(days)}) does not match number of changes ({len(changes)})'
            raise ValueError(errormsg)
    return changes


def process_days_changes(sim, days, changes):
    """ Process the days and changes of change_beta and clip_edges together, so that each change stays with its day when the days are sorted """
    if callable(days):
        return days, process_changes(changes, days)
    days = [process_days(sim, day)[0] for day in sc.tolist(days)] # Convert each day to an integer, keeping the order
    changes = process_changes(changes, days)
    order = np.argsort(days, kind='stable')
    return np.array(days, dtype=int)[order], changes[order]


class change_beta(Intervention):
    """
    The most basic intervention -- change beta (transmission) by a certain amount
    on a given day or days. This can be used to represent physical distancing (although
    clip_edges() is more appropriate for overall changes in mobility, e.g. school
    or workplace closures), as well as hand-washing, masks, and other behavioral
    changes that affect transmission rates. Each change is relative to the original
    beta, not cumulative.

    Args:
        days    (int/arr/str/function): the day or array of days to apply the interventions, or a function ``days(interv, sim)`` returning them
        changes (float/arr): the changes in beta (1 = no change, 0 = no transmission)
        layers  (str/list):  the layers in which to change beta (default: all, by changing the overall beta)
        kwargs  (dict):      passed to Intervention()

    **Examples**::

        interv = cv.change_beta(25, 0.3) # On day 25, reduce overall beta by 70% to 0.3
        interv = cv.change_beta([14, 28], [0.7, 1], layers='s') # On day 14, reduce beta by 30%, and on day 28, return to 1 for schools
    """

    def __init__(self, days, changes, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.days = days
        self.changes = changes
        self.layers = layers
        self.orig_betas = None # {layer: original beta_layer value}, or {'overall': original beta value}
        return

    def init_post(self):
        super().init_post()
        self.days, self.changes = process_days_changes(self.sim, self.days, self.changes)
        covid = self.sim.diseases.covid
        if self.layers is None:
            self.orig_betas = {'overall': ss.probperday(covid.pars.beta).value} # Per-day value, whether stored as a float or a rate
        else:
            self.orig_betas = {lk: covid.get_layer_par('beta_layer', lk) for lk in sc.tolist(self.layers)}
            if not isinstance(covid.pars.beta_layer, dict): # A scalar beta_layer: expand it so layers can be changed individually
                covid.pars.beta_layer = {lk: covid.pars.beta_layer for lk in self.sim.networks.keys()}
        return

    def step(self):
        covid = self.sim.diseases.covid
        for ind in find_day(self.days, self.ti, interv=self, sim=self.sim):
            for lk, orig in self.orig_betas.items():
                if lk == 'overall':
                    covid.pars.beta = ss.probperday(orig * self.changes[ind])
                else:
                    beta_layer = sc.cp(covid.pars.beta_layer) # Replace the dict rather than modifying it, so the original can be restored at the end of the run
                    beta_layer[lk] = orig * self.changes[ind]
                    covid.pars.beta_layer = beta_layer
        return


class clip_edges(Intervention):
    """
    Isolate contacts by removing them from the simulation. Contacts are treated as
    "edges", and this intervention works by removing them from the networks.
    When the intervention is over, they are restored.
    This intervention has quite similar effects as change_beta(), but is more appropriate
    for modeling the effects of mobility reductions such as school and workplace
    closures. The main difference is that since clip_edges() actually removes contacts,
    it affects the number of people who would be traced and placed in quarantine
    if an individual tests positive. It also alters the structure of the network
    -- i.e., compared to a baseline case of 20 contacts and a 2% chance of infecting
    each, there are slightly different statistics for a beta reduction (i.e., 20 contacts
    and a 1% chance of infecting each) versus an edge clipping (i.e., 10 contacts
    and a 2% chance of infecting each).

    Each change is the fraction of the original edges to keep, not cumulative. For a dynamic
    layer (see ``dynam_layer``), which is recreated on each timestep, that fraction of the
    edges is kept on every timestep until the next change.

    Args:
        days (int/arr/str/function): the day or array of days to isolate contacts, or a function ``days(interv, sim)`` returning them
        changes (float/arr): the changes in the number of contacts (1 = no change, 0 = no contacts)
        layers (str/list): the layers in which to isolate contacts (if None, then all layers)
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = cv.clip_edges(25, 0.3) # On day 25, reduce overall contacts by 70% to 0.3
        interv = cv.clip_edges([14, 28], [0.7, 1], layers='s') # On day 14, remove 30% of school contacts, and on day 28, restore them
    """

    def __init__(self, days, changes, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.days = days
        self.changes = changes
        self.layers = layers
        self._orig = None     # {layer: the original edges, e.g. p1, p2, and beta}
        self._order = None    # {layer: the order in which edges are removed}
        self._keep = None     # The current fraction of edges to keep
        self._edge_rng = ss.random() # Used to choose which edges are removed
        return

    def init_post(self):
        super().init_post()
        self.days, self.changes = process_days_changes(self.sim, self.days, self.changes)
        nets = self.sim.networks
        self.layers = list(nets.keys()) if self.layers is None else sc.tolist(self.layers)
        self._orig = {}
        self._order = {}
        n_edges = [len(nets[lk]) for lk in self.layers]
        rands = np.split(self._edge_rng.rvs(sum(n_edges)), np.cumsum(n_edges)[:-1]) # One random number per edge, split by layer
        for lk, layer_rands in zip(self.layers, rands):
            self._orig[lk] = {key:val.copy() for key,val in nets[lk].edges.items()}
            self._order[lk] = np.argsort(layer_rands) # Edges are kept in this order, so e.g. the edges kept at 30% are a subset of those kept at 70%
        return

    def step(self):
        nets = self.sim.networks
        inds = find_day(self.days, self.ti, interv=self, sim=self.sim)
        if len(inds):
            self._keep = self.changes[inds[0]]
        if self._keep is None: # No changes yet
            return

        for lk in self.layers:
            edges = nets[lk].edges
            if nets[lk].pars.get('dynamic'): # A dynamic layer is recreated on each timestep, so keep a fraction of the current edges each time
                if self._keep < 1:
                    n_edges = len(edges['p1'])
                    n_keep = int(round(self._keep * n_edges))
                    sel = np.sort(np.argsort(self._edge_rng.rvs(n_edges))[:n_keep])
                    for key in edges.keys():
                        edges[key] = edges[key][sel]
            elif len(inds): # A static layer: on the days of the changes, keep a fraction of the original edges
                orig = self._orig[lk]
                alive = self.sim.people.alive.raw
                n_keep = int(round(self._keep * len(orig['p1'])))
                sel = np.sort(self._order[lk][:n_keep])
                sel = sel[alive[orig['p1'][sel]] & alive[orig['p2'][sel]]] # Don't restore the edges of agents who have died
                for key,val in orig.items():
                    edges[key] = val[sel]
        return

    def shrink(self):
        """ Remove the copies of the original edges, for saving """
        super().shrink()
        self._orig = None
        self._order = None
        return


class dynamic_pars(Intervention):
    """
    A generic intervention that modifies a set of parameters at specified points
    in time.

    The intervention takes a single argument, pars, which is a dictionary of which
    parameters to change, with following structure: keys are the parameters to change,
    then subkeys 'days' and 'vals' are either a scalar or list of when the change(s)
    should take effect and what the new value should be, respectively. If a value is
    a dict (e.g. for ``beta_layer``), it updates the parameter rather than replacing it.

    You can also pass parameters to change directly as keyword arguments.

    Args:
        pars (dict): described above
        kwargs (dict): parameters to change, or else passed to Intervention()

    **Examples**::

        interv = cv.dynamic_pars(n_imports=dict(days=10, vals=100))
        interv = cv.dynamic_pars({'beta':{'days':[14, 28], 'vals':[0.005, 0.015]}, 'rel_death_prob':{'days':30, 'vals':2.0}}) # Change beta, and make death more likely
        interv = cv.dynamic_pars(beta_layer=dict(days=20, vals=dict(s=0))) # Close schools on day 20
    """

    def __init__(self, pars=None, **kwargs):

        # Move the keyword arguments that aren't arguments of Intervention() to the pars dict
        pars = sc.mergedicts(pars)
        interv_keys = ['label', 'show_label', 'do_plot', 'line_args', 'name']
        for key in list(kwargs.keys()):
            if key not in interv_keys:
                pars[key] = kwargs.pop(key)
        super().__init__(**kwargs)

        # Ensure the days and values are lists of the same length
        self.par_changes = {} # Not self.pars, which holds the intervention's own parameters
        for parkey, spec in pars.items():
            for subkey in ['days', 'vals']:
                if subkey not in spec:
                    errormsg = f'Parameter {parkey} is missing subkey {subkey}'
                    raise sc.KeyNotFoundError(errormsg)
            days = sc.tolist(spec['days'])
            vals = spec['vals']
            if isinstance(vals, dict) or not sc.isiterable(vals): # A single value
                vals = [vals]
            vals = list(vals)
            if len(days) != len(vals):
                errormsg = f'Length of days ({len(days)}) does not match length of values ({len(vals)}) for parameter {parkey}'
                raise ValueError(errormsg)
            self.par_changes[parkey] = dict(days=days, vals=vals)
        self.days = sorted(set(day for spec in self.par_changes.values() for day in spec['days'] if sc.isnumber(day))) # For plotting; dates are added once the sim is initialized
        return

    def init_post(self):
        """ Convert any dates to days """
        super().init_post()
        for spec in self.par_changes.values():
            spec['days'] = [self.sim.day(day) for day in spec['days']]
        self.days = sorted(set(day for spec in self.par_changes.values() for day in spec['days']))
        return

    def step(self):
        """ Loop over the parameters, and then loop over the days, applying them if any are found """
        for parkey, spec in self.par_changes.items():
            for ind in find_day(spec['days'], self.ti, interv=self, sim=self.sim):
                val = spec['vals'][ind]
                if isinstance(val, dict):
                    val = sc.mergedicts(self.sim[parkey], val) # Update the parameter if a nested dict, as a new dict so the original can be restored
                self.sim[parkey] = val
        return


class sequence(Intervention):
    """
    This is an example of a meta-intervention which switches between a sequence of interventions.

    Args:
        days (list): the days on which to start applying each intervention
        interventions (list): the interventions to apply on those days
        kwargs (dict): passed to Intervention()

    **Example**::

        interv = cv.sequence(days=[10, 51], interventions=[
                    cv.test_num(daily_tests=100),
                    cv.test_prob(symp_prob=0.2, asymp_prob=0.002),
                ])
    """

    def __init__(self, days, interventions, **kwargs):
        super().__init__(**kwargs)
        days = sc.tolist(days)
        if len(days) != len(interventions):
            errormsg = f'The number of days ({len(days)}) must match the number of interventions ({len(interventions)})'
            raise ValueError(errormsg)
        self.days = days
        self.interventions = interventions
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        for i,intv in enumerate(self.interventions):  # initialise the child interventions
            intv.name = f'{self.name}_{i}_{intv.name}' # The children aren't in sim.interventions, so give them unique names here (e.g. not "sequence_1", the name of a second sequence)
            intv.init_pre(sim)
        return

    def init_post(self):
        super().init_post()
        self.days = [self.sim.day(day) for day in self.days] # Convert any dates to days
        for intv in self.interventions:
            intv.init_post()
        return

    def start_step(self):
        super().start_step()
        for intv in self.interventions: # The children aren't in the integration loop, so set their time index (which starts at 0, even if the sequence is added part-way through a run) and advance their random numbers here
            intv.t.ti = self.ti
            intv.start_step()
        return

    def step(self):
        ti = self.ti
        active = [i for i, d in enumerate(self.days) if d <= ti]  # most recently activated
        if active:
            self.interventions[active[-1]].step()
        return

    def finish_step(self):
        super().finish_step()
        for intv in self.interventions: # Likewise, advance their time index
            intv.finish_step()
        return

    def finalize(self):
        super().finalize()
        for intv in self.interventions:
            intv.finalize()
        return
