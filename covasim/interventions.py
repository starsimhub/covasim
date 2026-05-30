"""
Interventions for Covasim on the Starsim base.

``cv.Intervention(ss.Intervention)`` is the base (same public name as v3). M5 restores the testing
interventions ``cv.test_num`` / ``cv.test_prob`` as ``ss.Intervention`` subclasses; ``cv.contact_tracing``
follows in M5 Task 3. They run at the intervention loop slot (after the cross-immunity connector,
before transmission), select agents to test, and call the ``cv.COVID.test()`` action -- which schedules
diagnoses and drives the diagnosis/isolation/quarantine state machines in ``cv.COVID.step_state``.

The selection draws use ``ss.bernoulli`` (CRN), replacing v3's global-RNG ``cvu.binomial``.
"""
import numpy as np
import starsim as ss

__all__ = ['Intervention', 'test_num', 'test_prob']


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
