"""
Interventions for Covasim on the Starsim base.

``cv.Intervention(ss.Intervention)`` is the base (same public name as v3). M5 restores the testing
interventions ``cv.test_num`` / ``cv.test_prob`` and ``cv.contact_tracing`` as ``ss.Intervention``
subclasses. They run at the intervention loop slot (after the cross-immunity connector, before
transmission); the testing ones select agents and call the ``cv.COVID.test()`` action (which schedules
diagnoses and drives the diagnosis/isolation/quarantine state machines in ``cv.COVID.step_state``),
and ``contact_tracing`` traces the newly-diagnosed agents' network contacts and schedules quarantine.

The selection draws use ``ss.bernoulli`` (CRN), replacing v3's global-RNG ``cvu.binomial``.
"""
import numpy as np
import starsim as ss

__all__ = ['Intervention', 'test_num', 'test_prob', 'contact_tracing']


def _find_contacts(net, trace_uids):
    """Return the UIDs that share an edge with any of ``trace_uids`` in network ``net`` (the v3
    ``Layer.find_contacts``), via the network's edge list. Excludes the traced agents themselves."""
    edges = net.edges
    p1 = np.asarray(edges.p1)
    p2 = np.asarray(edges.p2)
    if not len(p1):
        return ss.uids()
    arr = np.asarray(trace_uids)
    contacts = np.concatenate([p2[np.isin(p1, arr)], p1[np.isin(p2, arr)]])
    contacts = np.setdiff1d(np.unique(contacts), arr)  # unique partners, excluding the index cases
    return ss.uids(contacts)


class Intervention(ss.Intervention):
    """Base class for Covasim interventions (same public name as v3; thin over ``ss.Intervention``)."""

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


class test_prob(Intervention):
    """
    Test each person with a per-day probability that depends on their symptom + quarantine state
    (the v3 ``cv.test_prob``). The number of tests is an output, not an input.

    Args:
        symp_prob (float): daily probability of testing a symptomatic, un-quarantined person.
        asymp_prob (float): daily probability of testing an asymptomatic, un-quarantined person.
        symp_quar_prob (float): testing probability for symptomatic quarantined people (default symp_prob).
        asymp_quar_prob (float): testing probability for asymptomatic quarantined people (default asymp_prob).
        sensitivity (float): test sensitivity (true-positive rate).
        loss_prob (float): probability of loss-to-follow-up (never diagnosed).
        test_delay (int): days from test to diagnosis.
        start_day (int): first day the intervention is active.
        end_day (int): last day the intervention is active (None = no end).
    """

    def __init__(self, symp_prob, asymp_prob=0.0, symp_quar_prob=None, asymp_quar_prob=None,
                 sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, **kwargs):
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
        self._select = ss.bernoulli(p=0.0)  # per-agent test-probability draw (CRN)
        return

    def step(self):
        ti = self.ti
        if not self._active(ti):
            return
        covid = self._covid()
        alive = self.sim.people.auids
        symp = np.asarray(covid.symptomatic[alive])
        quar = np.asarray(covid.quarantined[alive])
        diag = np.asarray(covid.diagnosed[alive])
        # Per-agent daily test probability by (symptomatic, quarantined) state; diagnosed aren't retested.
        p = np.where(symp,
                     np.where(quar, self.symp_quar_prob, self.symp_prob),
                     np.where(quar, self.asymp_quar_prob, self.asymp_prob)).astype(float)
        p[diag] = 0.0
        self._select.set(p=p)
        test_uids = alive[self._select.rvs(alive)]
        if len(test_uids):
            covid.test(test_uids, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob,
                       test_delay=self.test_delay)
        return


class test_num(Intervention):
    """
    Test a fixed number of people per day, preferentially testing the symptomatic (the v3 ``cv.test_num``).

    Args:
        daily_tests (int/array): number of tests available per day (scalar or per-day array).
        symp_test (float): relative weight for testing symptomatic vs asymptomatic people.
        sensitivity (float): test sensitivity.
        loss_prob (float): probability of loss-to-follow-up.
        test_delay (int): days from test to diagnosis.
        start_day (int): first active day.
        end_day (int): last active day (None = no end).
    """

    def __init__(self, daily_tests, symp_test=100.0, sensitivity=1.0, loss_prob=0.0, test_delay=0,
                 start_day=0, end_day=None, **kwargs):
        super().__init__(**kwargs)
        self.daily_tests = daily_tests
        self.symp_test   = symp_test
        self.sensitivity = sensitivity
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        return

    def _n_tests(self, ti):
        """Number of tests available on day ``ti`` (scalar, or indexed from a per-day array)."""
        dt = self.daily_tests
        if np.isscalar(dt):
            return int(dt)
        arr = np.asarray(dt)
        return int(arr[ti]) if ti < len(arr) else 0

    def step(self):
        ti = self.ti
        if not self._active(ti):
            return
        n_tests = self._n_tests(ti)
        if n_tests <= 0:
            return
        covid = self._covid()
        alive = np.asarray(self.sim.people.auids)
        diag = np.asarray(covid.diagnosed[ss.uids(alive)])
        eligible = alive[~diag]
        if not len(eligible):
            return
        # Symptomatic-weighted selection without replacement, via a deterministic per-(seed, ti) stream.
        symp = np.asarray(covid.symptomatic[ss.uids(eligible)])
        weights = np.where(symp, self.symp_test, 1.0).astype(float)
        weights = weights / weights.sum()
        n = min(n_tests, len(eligible))
        try:
            base = int(self.sim.pars.rand_seed)
        except Exception:
            base = 0
        rng = np.random.default_rng([base, 90, ti])
        chosen = ss.uids(np.sort(rng.choice(eligible, size=n, replace=False, p=weights)))
        covid.test(chosen, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob,
                   test_delay=self.test_delay)
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
        # Capacity limit on index cases traced per day.
        if self.capacity is not None and len(trace) > self.capacity:
            try:
                base = int(self.sim.pars.rand_seed)
            except Exception:
                base = 0
            rng = np.random.default_rng([base, 91, ti])
            trace = ss.uids(np.sort(rng.choice(np.asarray(trace), int(self.capacity), replace=False)))
        trace_arr = np.asarray(trace)
        tp, tt = self._per_layer()
        for lkey, net in self.sim.networks.items():
            this_tp = float(tp.get(lkey, 0.0))
            this_tt = int(tt.get(lkey, 0.0))
            if this_tp == 0:
                continue
            contacts = _find_contacts(net, trace_arr)
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
