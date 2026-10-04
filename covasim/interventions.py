"""
Interventions for Covasim on the Starsim base.

``cv.Intervention(ss.Intervention)`` is the base (same public name as v3). The testing
interventions ``cv.test_num`` / ``cv.test_prob`` and ``cv.contact_tracing`` are ``ss.Intervention``
subclasses. They run at the intervention loop slot (after the cross-immunity connector, before
transmission); the testing ones select agents and call the ``cv.COVID.test()`` action (which schedules
diagnoses and drives the diagnosis/isolation/quarantine state machines in ``cv.COVID.step_state``),
and ``contact_tracing`` traces the newly-diagnosed agents' network contacts and schedules quarantine.

The selection draws use ``ss.bernoulli`` (CRN), replacing v3's global-RNG ``cvu.binomial``.
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
    except KeyError as E:
        available = ', '.join(mapping.keys())
        errormsg = f'Only interventions "{available}" are available in dictionary representation, not "{which}"'
        raise sc.KeyNotFoundError(errormsg) from E
    intervention = IntervClass(**pars)
    return intervention


class Intervention(cvc.V3Module, ss.Intervention):
    """Base class for Covasim interventions (same public name as v3; thin over ``ss.Intervention``)."""

    def init_post(self):
        super().init_post()
        # v3 accepted date strings (or datetimes) for start_day/end_day; convert them to day indices.
        for attr in ('start_day', 'end_day'):
            val = getattr(self, attr, None)
            if val is not None and not isinstance(val, (int, np.integer, float, np.floating)):
                setattr(self, attr, self.sim.day(val))
        return

    def _covid(self):
        """Resolve the COVID disease module this intervention acts on."""
        return self.sim.diseases['covid']

    def _active(self, ti):
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


class BaseTest(Intervention):
    """Shared logic for test_num and test_prob: quarantine policy, influenza-like illness, and swab delay (v3)."""

    def _init_testing(self, quar_policy, subtarget, ili_prev, swab_delay):
        self.quar_policy = quar_policy if quar_policy else 'start'
        self.subtarget   = subtarget
        self.ili_prev    = ili_prev
        self.pdf         = cvu.get_pdf(**sc.mergedicts(swab_delay)) # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        self._choose_ili = ss.choose_n() # Who has influenza-like illness on each day
        return

    def init_post(self):
        super().init_post()
        self.ili_prev = process_daily_data(self.ili_prev, self.sim, self.start_day)
        return

    def _symp_time(self, symp_inds):
        """Days since symptom onset, for the swab delay"""
        return (self.ti - np.asarray(self._covid().ti_symptomatic[symp_inds])).astype(int)

    def _ili_inds(self, symp_inds):
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
    Test each person with a per-day probability that depends on their symptom + quarantine state
    (the v3 ``cv.test_prob``). The number of tests is an output, not an input.

    Args:
        symp_prob (float): daily probability of testing a symptomatic, un-quarantined person.
        asymp_prob (float): daily probability of testing an asymptomatic, un-quarantined person.
        symp_quar_prob (float): testing probability for symptomatic quarantined people (default symp_prob).
        asymp_quar_prob (float): testing probability for asymptomatic quarantined people (default asymp_prob).
        quar_policy (str): policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget (dict): subtarget intervention to people with particular indices (see get_subtargets())
        ili_prev (float/arr): prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        sensitivity (float): test sensitivity (true-positive rate).
        loss_prob (float): probability of loss-to-follow-up (never diagnosed).
        test_delay (int): days from test to diagnosis.
        start_day (int): first day the intervention is active.
        end_day (int): last day the intervention is active (None = no end).
        swab_delay (dict): distribution for the delay from onset to swab; if this is present, it is used instead of test_delay
    """

    def __init__(self, symp_prob, asymp_prob=0.0, symp_quar_prob=None, asymp_quar_prob=None, quar_policy=None, subtarget=None, ili_prev=None,
                 sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs)
        self.symp_prob       = symp_prob
        self.asymp_prob      = asymp_prob
        self.symp_quar_prob  = symp_quar_prob  if symp_quar_prob  is not None else symp_prob
        self.asymp_quar_prob = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self.sensitivity     = sensitivity
        self.loss_prob       = loss_prob
        self.test_delay      = test_delay
        self.start_day       = start_day
        self.end_day         = end_day
        self._init_testing(quar_policy, subtarget, ili_prev, swab_delay)
        self._select = ss.bernoulli(p=0.0)  # per-agent test-probability draw (CRN)
        return

    def step(self):
        if not self._active(self.ti):
            return
        covid = self._covid()
        alive = self.sim.people.auids

        # Find probability of testing for symptomatic people, optionally with a swab delay
        symp_inds = covid.symptomatic.uids
        symp_prob = self.symp_prob
        if self.pdf:
            symp_time = self._symp_time(symp_inds)
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape)
            count[inv_count != 0] = 1/inv_count[inv_count != 0]
            symp_prob = np.ones(len(symp_time))
            inds = 1 > (symp_time*self.symp_prob)
            symp_prob[inds] = self.symp_prob/(1-symp_time[inds]*self.symp_prob)
            symp_prob = self.pdf.pdf(symp_time) * symp_prob * count[symp_time]

        # Define the groups of people, as boolean arrays by UID (much faster than set operations on the UIDs)
        n = len(covid.symptomatic.raw)
        ili_inds = self._ili_inds(symp_inds)
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
    Test a fixed number of people per day, preferentially testing the symptomatic (the v3 ``cv.test_num``).

    Args:
        daily_tests (int/arr): number of tests per day, can be int, array, or dataframe/series; if integer, use that number every day; if 'data' or another string, use that column from the sim's data
        symp_test (float): odds ratio of a symptomatic person testing (default: 100x more likely)
        quar_test (float): probability of a person in quarantine testing (default: no more likely)
        quar_policy (str): policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget (dict): subtarget intervention to people with particular indices (see get_subtargets())
        ili_prev (arr): prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        sensitivity (float): test sensitivity.
        loss_prob (float): probability of loss-to-follow-up.
        test_delay (int): days from test to diagnosis.
        start_day (int): first active day.
        end_day (int): last active day (None = no end).
        swab_delay (dict): distribution for the delay from onset to swab; if this is present, it is used instead of test_delay
    """

    def __init__(self, daily_tests, symp_test=100.0, quar_test=1.0, quar_policy=None, subtarget=None, ili_prev=None,
                 sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs)
        self.daily_tests = daily_tests
        self.symp_test   = symp_test
        self.quar_test   = quar_test
        self.sensitivity = sensitivity
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        self._init_testing(quar_policy, subtarget, ili_prev, swab_delay)
        self._choose_tests = ss.choose_n() # Who is tested
        return

    def init_post(self):
        super().init_post()
        self.daily_tests = process_daily_data(self.daily_tests, self.sim, self.start_day)
        return

    def step(self):
        ti = self.ti
        if not self._active(ti):
            return

        # Check that there are tests today, correcting for the population scale factor
        rel_t = ti - self.start_day
        if rel_t >= len(self.daily_tests):
            return
        n_tests = sc.randround(self.daily_tests[rel_t]/self.sim.current_scale)
        if not (n_tests and np.isfinite(n_tests)):
            return

        # Assign testing weights by UID, starting with equal weight for everyone alive
        covid = self._covid()
        alive = self.sim.people.auids
        test_probs = np.zeros(len(covid.symptomatic.raw))
        test_probs[alive] = 1.0

        # Handle symptomatic testing, optionally with a swab delay
        symp_inds = covid.symptomatic.uids
        symp_test = self.symp_test
        if self.pdf:
            symp_time = self._symp_time(symp_inds)
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape) # Initialize the count
            count[inv_count != 0] = 1/inv_count[inv_count != 0] # Update the counts where defined
            symp_test *= self.pdf.pdf(symp_time) * count[symp_time] # Put it all together
        test_probs[symp_inds] *= symp_test

        # Handle the other groups
        test_probs[self._ili_inds(symp_inds)] *= self.symp_test
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
            n_tests = sc.randround(n_tests*in_frac) # Recompute the number of tests

        # Choose who tests, without replacement, weighted by the testing probabilities
        eligible = ss.uids(test_probs.nonzero()[0])
        self._choose_tests.set(n=n_tests, weights=test_probs[eligible])
        chosen = self._choose_tests.filter(eligible)
        covid.test(chosen, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay, n_tests=n_requested)
        return


class contact_tracing(Intervention):
    """
    Trace and quarantine the contacts of diagnosed agents (the v3 ``cv.contact_tracing``).

    Has no effect without a testing intervention (it acts on the newly diagnosed). On each active day
    it finds the network contacts of the newly-diagnosed (or, if ``presumptive``, the newly-tested
    exposed), keeps a fraction ``trace_probs`` per layer, marks them ``known_contact``, and schedules
    a quarantine starting ``trace_time`` days later.

    Args:
        trace_probs (float/dict): per-layer probability of tracing a contact (default 1.0).
        trace_time (float/dict): per-layer days from diagnosis to notification (default 0).
        start_day (int): first active day.
        end_day (int): last active day (None = no end).
        presumptive (bool): trace on a positive test rather than waiting for diagnosis.
        capacity (int): max index cases traced per day (None = unlimited).
        quar_period (int): quarantine duration for notified contacts (default ``covid.pars.quar_period``).
    """

    def __init__(self, trace_probs=None, trace_time=None, start_day=0, end_day=None,
                 presumptive=False, capacity=None, quar_period=None, **kwargs):
        super().__init__(**kwargs)
        self.trace_probs = trace_probs
        self.trace_time  = trace_time
        self.start_day   = start_day
        self.end_day     = end_day
        self.presumptive = presumptive
        self.capacity    = capacity
        self.quar_period = quar_period
        self._trace = ss.bernoulli(p=1.0)  # per-contact trace draw (CRN)
        self._choose_capacity = ss.choose_n() # Who is traced, if there are more index cases than the capacity
        return

    def _per_layer(self):
        """Expand trace_probs / trace_time into per-network-layer dicts (defaults 1.0 / 0.0)."""
        netkeys = list(self.sim.networks.keys())
        tp = 1.0 if self.trace_probs is None else self.trace_probs
        tt = 0.0 if self.trace_time is None else self.trace_time
        if np.isscalar(tp):
            tp = {k: tp for k in netkeys}
        if np.isscalar(tt):
            tt = {k: tt for k in netkeys}
        return tp, tt

    def step(self):
        ti = self.ti
        if not self._active(ti):
            return
        covid = self._covid()
        qp = covid.pars.quar_period if self.quar_period is None else int(self.quar_period)
        # Select index cases: diagnosed this step (or, if presumptive, newly-tested still-exposed).
        if not self.presumptive:
            trace = (covid.date_diagnosed == ti).uids
        else:
            tested = (covid.date_tested == ti).uids
            trace = tested[np.asarray(covid.exposed[tested])] if len(tested) else tested
        if not len(trace):
            return
        # If there is a tracing capacity constraint, limit the number of agents that can be traced
        if self.capacity is not None:
            self._choose_capacity.set(n=int(self.capacity/self.sim.current_scale)) # Convert capacity into a number of agents
            trace = self._choose_capacity.filter(trace)
        trace_arr = np.asarray(trace)
        tp, tt = self._per_layer()
        for lkey, net in self.sim.networks.items():
            this_tp = float(tp.get(lkey, 0.0))
            this_tt = int(tt.get(lkey, 0.0))
            if this_tp == 0:
                continue
            contacts = ss.uids(net.find_contacts(trace_arr))
            if not len(contacts):
                continue
            self._trace.set(p=this_tp)
            keep = contacts[self._trace.rvs(contacts)]            # filter by per-layer trace prob
            if len(keep):
                keep = keep[~np.asarray(covid.dead[keep])]        # do not notify dead contacts
            if not len(keep):
                continue
            covid.known_contact[keep] = True
            covid.schedule_quarantine(keep, start_date=ti + this_tt, period=max(1, qp - this_tt))
        return


# %% Vaccination interventions -----------------------------------------------------------------

__all__ += ['BaseVaccination', 'vaccinate', 'vaccinate_prob', 'vaccinate_num', 'simple_vaccine']


def _check_doses(doses, interval):
    """Validate the dose count + dosing interval (v3 check_doses)."""
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
    """Resolve a v3 subtarget into ``(inds, vals)`` (the v3 ``get_subtargets`` helper).

    ``subtarget`` is a dict ``{'inds': ..., 'vals': ...}`` (or a callable ``f(sim)`` returning one).
    ``inds`` (the target UIDs) and ``vals`` (a per-agent value, e.g. a probability) may each be a
    callable ``f(sim)``; a scalar ``vals`` is broadcast over ``inds``. Returns ``(inds, vals)`` arrays
    (``vals`` may be None).
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


def _apply_subtarget_probs(probs, pool, subtarget, sim):
    """Override ``probs`` (aligned to ``pool`` UIDs) with the subtarget ``vals`` for matching UIDs."""
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
    Base class for vaccination (the v3 ``BaseVaccination``).

    Confers immunity by conferring/boosting neutralizing antibodies through the same NAb pipeline as natural infection
    (so it requires ``use_waning=True``; for the non-NAb path use ``cv.simple_vaccine``). Subclasses
    implement ``select_people()`` (the allocation strategy); this base handles vaccine-parameter
    parsing, registration into the disease module's vaccine registry, ``target_eff`` back-calculation,
    dose scheduling/booking, and the per-dose state + NAb update.

    Args:
        vaccine (str/dict): a predefined product ('pfizer'/'moderna'/'az'/'jj'/...) or a pars dict.
        label (str): vaccine label if ``vaccine`` is a dict.
        booster (bool): if True, target already-vaccinated people.
    """

    def __init__(self, vaccine, label=None, booster=False, subtarget=None, **kwargs):
        super().__init__(**kwargs)
        self.index = None       # set at init: this vaccine's index in the module registry
        self.label = label
        self.p = None           # vaccine parameters (dose pars + per-variant efficacy)
        self.booster = booster
        self.subtarget = subtarget  # v3 {'inds':..., 'vals':...} (or f(sim)) per-agent prob override
        self._doses = None      # per-agent doses given by THIS intervention (raw-UID indexed)
        self._parse_vaccine_pars(vaccine)
        return

    def _parse_vaccine_pars(self, vaccine):
        """Resolve a predefined product name or a pars dict into ``self.p`` (v3 _parse_vaccine_pars)."""
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

    def _register(self, covid):
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
            VE_symp = cvimm.calc_VE_symp(2 ** nabs, covid.pars.nab_eff)
            peak_nab = nabs[np.argmax(VE_symp > self.p['target_eff'][0])]
            self.p['nab_init'] = dict(dist='normal', par1=float(peak_nab), par2=2)
            if self.p['doses'] == 2:
                boosted = nabs[np.argmax(VE_symp > self.p['target_eff'][1])]
                self.p['nab_boost'] = float((2 ** boosted) / (2 ** peak_nab))
        _check_doses(int(self.p['doses']), self.p['interval'])
        covid.vaccine_pars[self.label] = self.p
        self.index = list(covid.vaccine_pars.keys()).index(self.label)
        covid.vaccine_map[self.index] = self.label
        self._doses = np.zeros(len(covid.rel_sus.raw), dtype=int)
        return

    def init_post(self):
        super().init_post()
        covid = self._covid()
        if not covid.pars.use_waning:
            raise RuntimeError(f'cv.{type(self).__name__} requires use_waning=True; else use cv.simple_vaccine().')
        self._register(covid)
        return

    def _vaccinate(self, covid, uids):
        """Apply a dose: skip dead / already-fully-dosed (by this vaccine), then update state + NAbs."""
        uids = uids[~np.asarray(covid.dead[uids])]
        if len(uids):
            uids = uids[self._doses[np.asarray(uids)] < int(self.p['doses'])]
        if len(uids):
            self._doses[np.asarray(uids)] += 1
            covid.vaccinate_agents(uids, self.label, self.index)
        return uids

    def select_people(self, covid):
        raise NotImplementedError

    def step(self):
        covid = self._covid()
        uids = self.select_people(covid)
        if uids is not None and len(uids):
            uids = self._vaccinate(covid, ss.uids(np.unique(np.asarray(uids))))
        return uids


class vaccinate_prob(BaseVaccination):
    """
    Probability-based vaccination (v3 ``vaccinate_prob``): on each day in ``days``, vaccinate eligible
    people with probability ``prob``; schedule their second dose ``interval`` days later.

    Args:
        vaccine (str/dict): product or pars dict.
        days (int/list): day index/indices on which first doses are offered.
        prob (float): per-eligible-person daily probability of a first dose (default 1.0).
        booster (bool): target the already-vaccinated.
    """

    def __init__(self, vaccine, days, label=None, prob=1.0, booster=False, **kwargs):
        super().__init__(vaccine, label=label, booster=booster, **kwargs)
        self.days = days
        self.prob = prob
        self._day_set = None
        self._second_dose = None  # {day: [uids arrays]}
        self._select = ss.bernoulli(p=0.0)
        return

    def init_post(self):
        super().init_post()
        self._day_set = set(int(round(d)) for d in sc.toarray(self.days))
        self._second_dose = {}
        return

    def select_people(self, covid):
        ti = covid.ti
        first = ss.uids()
        if ti in self._day_set:
            alive = covid.sim.people.auids
            vaccinated = np.asarray(covid.vaccinated[alive])
            eligible = alive[vaccinated] if self.booster else alive[~vaccinated]
            if len(eligible):
                # Per-agent probability: ``prob`` by default, overridden by any subtarget vals.
                probs = np.full(len(eligible), float(self.prob), dtype=float)
                probs = _apply_subtarget_probs(probs, eligible, self.subtarget, covid.sim)
                self._select.set(p=probs)
                first = eligible[self._select.rvs(eligible)]
                interval = self.p['interval']
                if interval is not None and len(first):  # schedule the second dose
                    nxt = ti + int(interval)
                    self._second_dose.setdefault(nxt, []).append(np.asarray(first))
        dose2 = self._second_dose.pop(ti, None)
        if dose2:
            arrs = ([np.asarray(first)] if len(first) else []) + dose2
            return ss.uids(np.unique(np.concatenate(arrs)))
        return first


class vaccinate_num(BaseVaccination):
    """
    Number-based vaccination (v3 ``vaccinate_num``): deliver ``num_doses`` doses/day in a priority
    ``sequence``, prioritising scheduled second doses.

    Args:
        vaccine (str/dict): product or pars dict.
        num_doses (int/dict/callable): doses per day (scalar, {day: n}, or f(sim) -> n).
        sequence (None/'age'/array/callable): vaccination priority order (default random).
        booster (bool): target the already-vaccinated.
    """

    def __init__(self, vaccine, num_doses, label=None, sequence=None, booster=False, **kwargs):
        super().__init__(vaccine, label=label, booster=booster, **kwargs)
        self.num_doses = num_doses
        self.sequence = sequence
        self._sequence = None
        self._scheduled = {}  # {day: set(uids)}
        self._random_order = ss.random() # The order in which to vaccinate people, if no sequence is given
        return

    def init_post(self):
        super().init_post()
        if isinstance(self.num_doses, dict):  # day-index keys (string dates are not supported)
            self.num_doses = {int(k): v for k, v in self.num_doses.items()}
        self._sequence = self._process_sequence(self.sequence)
        return

    def _process_sequence(self, sequence):
        covid = self._covid()
        alive = np.asarray(covid.sim.people.auids)
        if sequence is None:
            return alive[np.argsort(self._random_order.rvs(ss.uids(alive)))]
        if sequence == 'age':
            ages = np.asarray(covid.sim.people.age[ss.uids(alive)])
            return alive[np.argsort(-ages)]
        if callable(sequence):
            return np.asarray(sequence(covid.sim.people))
        return np.asarray(sequence)

    def _n_doses_today(self, ti):
        nd = self.num_doses
        if callable(nd):
            return int(nd(self.sim))
        if isinstance(nd, dict):
            return int(nd.get(ti, 0))
        return int(nd)

    def select_people(self, covid):
        ti = covid.ti
        n_doses = self._n_doses_today(ti)
        scheduled_today = self._scheduled.pop(ti, set())
        if n_doses <= 0:
            if scheduled_today:
                self._scheduled.setdefault(ti + 1, set()).update(scheduled_today)  # defer
            return ss.uids()
        n_agents = int(sc.randround(n_doses / float(covid.sim.pars.pop_scale)))
        dead = set(np.asarray(covid.dead.uids).tolist())
        # Second doses first (drop dead / fully-dosed-by-this-vaccine).
        scheduled = np.array([u for u in scheduled_today
                              if u not in dead and self._doses[u] < int(self.p['doses'])], dtype=int)
        if len(scheduled) > n_agents:
            self._scheduled.setdefault(ti + 1, set()).update(scheduled[n_agents:].tolist())
            return ss.uids(np.sort(scheduled[:n_agents]))
        # First doses, in priority sequence, among the (un)vaccinated as appropriate.
        vaccinated = np.asarray(covid.vaccinated[ss.uids(self._sequence)])
        elig_mask = vaccinated if self.booster else ~vaccinated
        seq_alive = np.array([u not in dead for u in self._sequence])
        first_pool = self._sequence[elig_mask & seq_alive]
        first_pool = first_pool[~np.isin(first_pool, scheduled)]
        if self.subtarget is not None:  # exclude subtarget UIDs with vals==0 (e.g. booster targeting)
            s_inds, s_vals = get_subtargets(self.subtarget, covid.sim)
            if s_vals is not None and len(s_inds):
                exclude = np.asarray(s_inds)[s_vals == 0]
                if len(exclude):
                    first_pool = first_pool[~np.isin(first_pool, exclude)]
        n_first = max(0, n_agents - len(scheduled))
        first = first_pool[:n_first]
        if int(self.p['doses']) > 1 and len(first):  # schedule second doses
            self._scheduled.setdefault(ti + int(self.p['interval']), set()).update(first.tolist())
        out = np.concatenate([scheduled, first]) if len(scheduled) or len(first) else np.array([], dtype=int)
        return ss.uids(np.sort(np.unique(out)))


def vaccinate(*args, **kwargs):
    """Wrapper: ``vaccinate_num`` if ``num_doses`` is given, else ``vaccinate_prob`` (v3 ``vaccinate``)."""
    if 'num_doses' in kwargs:
        return vaccinate_num(*args, **kwargs)
    return vaccinate_prob(*args, **kwargs)


class simple_vaccine(Intervention):
    """
    Simple (non-NAb) vaccine (the v3 ``simple_vaccine``): directly scales susceptibility and the
    symptomatic probability of vaccinated agents, rather than going through the NAb pipeline. Intended
    for ``use_waning=False``; preserves the v3 public API.

    Args:
        days (int/list): day(s) on which to vaccinate.
        prob (float): probability of being vaccinated on each applied day.
        rel_sus (float): relative susceptibility after vaccination (0 = perfect protection, 1 = none).
        rel_symp (float): relative symptomatic probability after vaccination (0 = perfect, 1 = none).
        cumulative (bool/list): per-dose efficacy weights; False=[1,0] (only the 1st dose helps),
            True=[1] (every dose at full efficacy), or an explicit list.
    """

    def __init__(self, days, prob=1.0, rel_sus=0.0, rel_symp=0.0, cumulative=False, subtarget=None, **kwargs):
        super().__init__(**kwargs)
        self.days = days
        self.prob = prob
        self.rel_sus = rel_sus
        self.rel_symp = rel_symp
        self.subtarget = subtarget  # v3 {'inds':..., 'vals':...} per-agent prob override
        if cumulative in [0, False]:
            cumulative = [1, 0]
        elif cumulative in [1, True]:
            cumulative = [1]
        self.cumulative = np.array(cumulative, dtype=float)
        self._day_set = None
        self._doses_by_this = None
        self._select = ss.bernoulli(p=0.0)
        return

    def init_post(self):
        super().init_post()
        self._day_set = set(int(round(d)) for d in sc.toarray(self.days))
        self._doses_by_this = np.zeros(len(self._covid().rel_sus.raw), dtype=int)
        return

    def step(self):
        ti = self.ti
        if ti not in self._day_set:
            return
        covid = self._covid()
        alive = covid.sim.people.auids
        probs = np.full(len(alive), float(self.prob), dtype=float)
        probs = _apply_subtarget_probs(probs, alive, self.subtarget, covid.sim)
        self._select.set(p=probs)
        vacc = alive[self._select.rvs(alive)]
        if not len(vacc):
            return
        va = np.asarray(vacc)
        # Per-dose efficacy weight (later doses may add nothing, per `cumulative`).
        eff_doses = np.minimum(self._doses_by_this[va], len(self.cumulative) - 1)
        vacc_eff = self.cumulative[eff_doses]
        rel_sus_eff  = (1.0 - vacc_eff) + vacc_eff * self.rel_sus
        rel_symp_eff = (1.0 - vacc_eff) + vacc_eff * self.rel_symp
        # Directly scale susceptibility + symptomatic probability (no NAb pipeline).
        covid.rel_sus[vacc]   = np.asarray(covid.rel_sus[vacc]) * rel_sus_eff
        covid.symp_prob[vacc] = np.asarray(covid.symp_prob[vacc]) * rel_symp_eff
        # Bookkeeping.
        prior = np.asarray(covid.vaccinated[vacc])
        self._doses_by_this[va] += 1
        covid.vaccinated[vacc] = True
        covid.doses[vacc] = np.asarray(covid.doses[vacc]) + 1
        covid.count_doses(vacc, prior)
        return


# %% Historical (pre-t=0) immunity (the v3 historical_* interventions) ------------------------------

__all__ += ['historical_vaccinate_prob', 'historical_wave', 'prior_immunity']


class historical_vaccinate_prob(vaccinate_prob):
    """
    Probability-based vaccination that may occur BEFORE t=0 (the v3 ``historical_vaccinate_prob``).

    Negative ``days`` are applied at initialisation: a fraction ``prob`` of agents is vaccinated as if
    on that (back-dated) day, so they start the sim with appropriately-decayed NAbs (via the NAb waning
    kinetic kernel, replayed from the event). Non-negative ``days`` behave like ``cv.vaccinate_prob``.
    Requires ``use_waning=True``. (Bounded port: a single back-dated dose per pre-t=0 day; multi-dose
    historical scheduling is approximated by the vaccine's per-dose peak NAb.)
    """

    def init_post(self):
        super().init_post()  # registers the vaccine, builds _day_set, checks use_waning
        covid = self._covid()
        for day in sorted(d for d in self._day_set if d < 0):  # imprint each pre-t=0 day now
            self._historical_dose(covid, day)
        self._day_set = set(d for d in self._day_set if d >= 0)  # leave in-sim days to step()
        return

    def _historical_dose(self, covid, day):
        alive = covid.sim.people.auids
        eligible = alive[~np.asarray(covid.vaccinated[alive])]
        if not len(eligible):
            return
        probs = _apply_subtarget_probs(np.full(len(eligible), float(self.prob)), eligible, self.subtarget, covid.sim)
        self._select.set(p=probs)
        chosen = eligible[self._select.rvs(eligible)]
        chosen = chosen[~np.asarray(covid.dead[chosen])]
        if not len(chosen):
            return
        self._doses[np.asarray(chosen)] += 1
        covid.vaccinate_agents(chosen, self.label, self.index)  # sets vaccinated/source/doses/peak NAb, and counts the doses (on day 0, as in v3)
        covid.imprint_historical_nab(chosen, day)               # decay the peak NAb from `day` to t=0
        return


class historical_wave(Intervention):
    """
    Imprint a historical (pre t=0) wave of infections in the population (the v3 ``historical_wave``).

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
        self.prob      = self._per_wave(prob, n_waves)
        self.dist      = self._per_wave(default_dist if dist is None else dist, n_waves)
        self.subtarget = self._per_wave(subtarget, n_waves)
        self.variants  = self._per_wave('wild' if variant is None else variant, n_waves)
        self._select = [ss.bernoulli(p=0.0) for wave in range(n_waves)] # Who is infected in each wave
        self._timing = [cvcovid.v3_durs(dict(wave=wave_dist))['dur_wave'] for wave_dist in self.dist] # When they are infected, in whole days
        self.day0_flows = {} # Counts of infections etc. before the start of the sim, which are added to the results on day 0
        self.day0_flows_variant = {}
        return

    @staticmethod
    def _per_wave(val, n_waves):
        """A list is one value per wave; otherwise, use the same value for every wave"""
        return list(val) if isinstance(val, list) else [val]*n_waves

    def init_post(self):
        super().init_post()
        sim = self.sim
        covid = self._covid()
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
        covid = self._covid()
        mapping = {label:ind for ind,label in covid.variant_map.items()}

        flows_variant_before = {key:val.copy() for key,val in covid._flow_variant.items()}
        flows = dict(reinfections=-covid._flow['reinfections'], symptomatic=0, severe=0, critical=0, recoveries=0)
        flows_variant = dict(new_symptomatic=np.zeros(covid.nv), new_severe=np.zeros(covid.nv))

        # Infect the people in each wave
        dates = [covid.ti_infected, covid.ti_exposed, covid.ti_infectious, covid.ti_symptomatic, covid.ti_severe, covid.ti_critical, covid.ti_recovered, covid.ti_dead, covid.ti_vl_switch]
        alive = sim.people.auids
        for wave, days_prior in enumerate(self.days_prior):
            if isinstance(days_prior, str): # Interpret as a date
                days_prior = sc.daydiff(days_prior, sim['start_day'])

            # Choose who is infected, and when (days relative to the start of the sim, so negative)
            probs = _apply_subtarget_probs(np.full(len(alive), self.prob[wave]), alive, self.subtarget[wave], sim)
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
        flows['reinfections'] += covid._flow['reinfections']
        self.day0_flows = flows
        self.day0_flows_variant = {key:val - flows_variant_before[key] + flows_variant.get(key, 0) for key,val in covid._flow_variant.items()}
        return

    def step(self):
        """As in v3, count the infections before the start of the sim in the results on day 0"""
        if self.ti == 0:
            covid = self._covid()
            for key,val in self.day0_flows.items():
                covid._flow[key] += val
            for key,val in self.day0_flows_variant.items():
                covid._flow_variant[key] += val
        return


def prior_immunity(*args, **kwargs):
    """Seed prior immunity (the v3 ``prior_immunity`` wrapper).

    Dispatches to ``historical_vaccinate_prob`` if a ``vaccine`` is given, else ``historical_wave``.
    """
    if 'vaccine' in kwargs or (args and isinstance(args[0], str)):
        return historical_vaccinate_prob(*args, **kwargs)
    return historical_wave(*args, **kwargs)


# %% Beta / parameter / meta interventions ----------------------------------------------------------

__all__ += ['change_beta', 'clip_edges', 'dynamic_pars', 'sequence']


def find_day(arr, t=None, interv=None, sim=None, which='first'):
    '''
    Find which days of an intervention match the current timestep (the v3 ``cv.find_day``).

    Args:
        arr (list/function): list of days in the intervention, or a boolean array; or a function ``arr(interv, sim)`` that returns these
        t (int): current simulation timestep
        which (str): what to return: 'first', 'last', or 'all' indices
        interv (intervention): the intervention object (usually self); only used if arr is callable
        sim (sim): the simulation object; only used if arr is callable

    Returns:
        inds (list): list of matching days; length zero or one unless which is 'all'
    '''
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


def process_days(sim, days):
    """Convert days (day indices, date strings, or dates) to an array of day indices; leave a callable as-is (v3 ``process_days``)."""
    if callable(days):
        return days
    return sc.toarray(sim.day(days))


def process_changes(changes, days):
    """Ensure the changes are an array matching the days (v3 ``process_changes``)."""
    changes = sc.toarray(changes).astype(float)
    if not callable(days):
        if len(changes) == 1:
            changes = np.full(len(days), changes[0])
        elif len(changes) != len(days):
            errormsg = f'Number of days supplied ({len(days)}) does not match number of changes ({len(changes)})'
            raise ValueError(errormsg)
    return changes


class change_beta(Intervention):
    """
    Change transmissibility (beta) by a factor on given days (the v3 ``cv.change_beta``).

    Args:
        days (int/list/function): day(s) on which to change beta, or a function ``days(interv, sim)`` returning them
        changes (float/list): the multiplicative change(s) (1 = no change, 0 = no transmission),
            applied to the ORIGINAL beta (not cumulative).
        layers (str/list): which network layers to change (default: all, by changing the overall beta).
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
        self.days = process_days(self.sim, self.days)
        self.changes = process_changes(self.changes, self.days)
        covid = self._covid()
        if self.layers is None:
            self.orig_betas = {'overall': ss.probperday(covid.pars.beta).value} # Per-day value, whether stored as a float or a rate
        else:
            self.orig_betas = {lk: covid._layer_par('beta_layer', lk) for lk in sc.tolist(self.layers)}
            if not isinstance(covid.pars.beta_layer, dict): # A scalar beta_layer: expand it so layers can be changed individually
                covid.pars.beta_layer = {lk: covid.pars.beta_layer for lk in self.sim.networks.keys()}
        return

    def step(self):
        covid = self._covid()
        for ind in find_day(self.days, self.ti, interv=self, sim=self.sim):
            for lk, orig in self.orig_betas.items():
                if lk == 'overall':
                    covid.pars.beta = ss.probperday(orig * self.changes[ind])
                else:
                    covid.pars.beta_layer[lk] = orig * self.changes[ind]
        return


class clip_edges(Intervention):
    """
    Reduce contacts by clipping a fraction of network edges on given days (the v3 ``cv.clip_edges``).

    Unlike change_beta (which scales transmissibility), this removes edges, so it also reduces the
    pool of traceable contacts. ``changes`` is the fraction of edges to KEEP (1 = all, 0 = none),
    applied to the original edge set (not cumulative); removed edges are restored when the fraction
    rises again.

    Args:
        days (int/list/function): day(s) on which to clip, or a function ``days(interv, sim)`` returning them
        changes (float/list): fraction of edges to keep on each day.
        layers (str/list): which layers to clip (default: all).
    """

    def __init__(self, days, changes, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.days = days
        self.changes = changes
        self.layers = layers
        self._orig = None     # {layer: the original edges, e.g. p1, p2, and beta}
        self._order = None    # {layer: the order in which edges are removed}
        self._edge_rng = ss.random() # Used to choose which edges are removed
        return

    def init_post(self):
        super().init_post()
        self.days = process_days(self.sim, self.days)
        self.changes = process_changes(self.changes, self.days)
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
        ti = self.ti
        inds = find_day(self.days, ti, interv=self, sim=self.sim)
        if not len(inds):
            return
        keep = self.changes[inds[0]]
        nets = self.sim.networks
        alive = self.sim.people.alive.raw
        for lk in self.layers:
            orig = self._orig[lk]
            n_keep = int(round(keep * len(orig['p1'])))
            sel = np.sort(self._order[lk][:n_keep])
            sel = sel[alive[orig['p1'][sel]] & alive[orig['p2'][sel]]] # Don't restore the edges of agents who have died
            edges = nets[lk].edges
            for key,val in orig.items():
                edges[key] = val[sel]
        return

    def shrink(self):
        ''' Remove the copies of the original edges, for saving '''
        super().shrink()
        self._orig = None
        self._order = None
        return


class dynamic_pars(Intervention):
    """
    Change parameters at specified days (the v3 ``cv.dynamic_pars``).

    Args:
        pars (dict): ``{parname: {'days': d_or_list, 'vals': v_or_list}}``; ``parname`` is resolved
            against the COVID module pars (then the sim pars), with an optional dotted path. Use
            ``cv.change_beta`` for the per-layer beta dict.
        kwargs: ``parname=dict(days=..., vals=...)`` entries may also be passed directly.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        pars = sc.mergedicts(pars, kwargs)
        self.par_changes = {}
        for parname, spec in pars.items():
            days = [int(round(d)) for d in sc.toarray(spec['days'])]
            vals = spec['vals']
            vals = list(vals) if sc.isiterable(vals) and not isinstance(vals, dict) else [vals] * len(days)
            self.par_changes[parname] = dict(zip(days, vals))
        return

    def init_post(self):
        super().init_post()
        return

    @staticmethod
    def _apply(sim, parname, value):
        covid = list(sim.diseases.values())[0]
        if parname in covid.pars:
            covid.pars[parname] = value
        elif parname in sim.pars:
            sim.pars[parname] = value
        else:
            covid.pars[parname] = value

    def step(self):
        ti = self.ti
        for parname, daymap in self.par_changes.items():
            if ti in daymap:
                self._apply(self.sim, parname, daymap[ti])
        return


class sequence(Intervention):
    """
    Switch between a sequence of interventions over time (the v3 ``cv.sequence``).

    Args:
        days (list): the day on which each intervention becomes the active one.
        interventions (list): the interventions, aligned to ``days``; on each step the most recently
            activated intervention is applied.
    """

    def __init__(self, days, interventions, **kwargs):
        super().__init__(**kwargs)
        assert len(sc.toarray(days)) == len(interventions), 'days and interventions must align'
        self.days = [int(round(d)) for d in sc.toarray(days)]
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
