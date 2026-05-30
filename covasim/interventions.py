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
import sciris as sc
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


# %% Vaccination interventions (M6) -----------------------------------------------------------------

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


class BaseVaccination(Intervention):
    """
    Base class for vaccination (the v3 ``BaseVaccination``).

    Confers immunity by conferring/boosting neutralizing antibodies through the shared M4 NAb pipeline
    (so it requires ``use_waning=True``; for the non-NAb path use ``cv.simple_vaccine``). Subclasses
    implement ``select_people()`` (the allocation strategy); this base handles vaccine-parameter
    parsing, registration into the disease module's vaccine registry, ``target_eff`` back-calculation,
    dose scheduling/booking, and the per-dose state + NAb update.

    Args:
        vaccine (str/dict): a predefined product ('pfizer'/'moderna'/'az'/'jj'/...) or a pars dict.
        label (str): vaccine label if ``vaccine`` is a dict.
        booster (bool): if True, target already-vaccinated people.
    """

    def __init__(self, vaccine, label=None, booster=False, **kwargs):
        super().__init__(**kwargs)
        self.index = None       # set at init: this vaccine's index in the module registry
        self.label = label
        self.p = None           # vaccine parameters (dose pars + per-variant efficacy)
        self.booster = booster
        self._doses = None      # per-agent doses given by THIS intervention (raw-UID indexed)
        self._parse_vaccine_pars(vaccine)
        return

    def _parse_vaccine_pars(self, vaccine):
        """Resolve a predefined product name or a pars dict into ``self.p`` (v3 _parse_vaccine_pars)."""
        import covasim.parameters as cvpar
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
        import covasim.parameters as cvpar
        import covasim.immunity as cvimm
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
                self._select.set(p=self.prob)
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
        return

    def init_post(self):
        super().init_post()
        if isinstance(self.num_doses, dict):  # day-index keys (string dates unsupported in M6)
            self.num_doses = {int(k): v for k, v in self.num_doses.items()}
        self._sequence = self._process_sequence(self.sequence)
        return

    def _process_sequence(self, sequence):
        covid = self._covid()
        alive = np.asarray(covid.sim.people.auids)
        if sequence is None:
            try:
                base = int(self.sim.pars.rand_seed)
            except Exception:
                base = 0
            rng = np.random.default_rng([base, 92])
            order = alive.copy(); rng.shuffle(order)
            return order
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

    def __init__(self, days, prob=1.0, rel_sus=0.0, rel_symp=0.0, cumulative=False, **kwargs):
        super().__init__(**kwargs)
        self.days = days
        self.prob = prob
        self.rel_sus = rel_sus
        self.rel_symp = rel_symp
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
        self._select.set(p=self.prob)
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
        covid._vacc_flow['doses'] += len(vacc)
        covid._vacc_flow['vaccinated'] += int((~prior).sum())
        return


# %% Beta / parameter / meta interventions ----------------------------------------------------------

__all__ += ['change_beta', 'clip_edges', 'dynamic_pars', 'sequence']


def _day_change_map(days, changes):
    """Build a {day_index: change_value} map from parallel days/changes (scalars or lists)."""
    days = [int(round(d)) for d in sc.toarray(days)]
    changes = np.atleast_1d(np.asarray(sc.toarray(changes), dtype=float))
    if changes.size == 1:
        changes = np.full(len(days), changes.item())
    return {d: float(c) for d, c in zip(days, changes)}


class change_beta(Intervention):
    """
    Change transmissibility (beta) by a factor on given days (the v3 ``cv.change_beta``).

    Args:
        days (int/list): day(s) on which to change beta.
        changes (float/list): the multiplicative change(s) (1 = no change, 0 = no transmission),
            applied to the ORIGINAL beta (not cumulative).
        layers (str/list): which network layers to change (default: all of the disease's beta layers).
    """

    def __init__(self, days, changes, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.days = days
        self.changes = changes
        self.layers = layers
        self._map = None
        self._orig = None     # {layer: original per-day beta value (float)}
        return

    def init_post(self):
        super().init_post()
        covid = self._covid()
        self._map = _day_change_map(self.days, self.changes)
        beta = covid.pars.beta
        layers = list(beta.keys()) if self.layers is None else sc.tolist(self.layers)
        self._orig = {lk: float(beta[lk]) for lk in layers}
        return

    def step(self):
        ti = self.ti
        if ti not in self._map:
            return
        covid = self._covid()
        change = self._map[ti]
        for lk, orig in self._orig.items():
            covid.pars.beta[lk] = ss.probperday(orig * change)  # scale the original per-day beta
        return


class clip_edges(Intervention):
    """
    Reduce contacts by clipping a fraction of network edges on given days (the v3 ``cv.clip_edges``).

    Unlike change_beta (which scales transmissibility), this removes edges, so it also reduces the
    pool of traceable contacts. ``changes`` is the fraction of edges to KEEP (1 = all, 0 = none),
    applied to the original edge set (not cumulative); removed edges are restored when the fraction
    rises again.

    Args:
        days (int/list): day(s) on which to clip.
        changes (float/list): fraction of edges to keep on each day.
        layers (str/list): which layers to clip (default: all).
    """

    def __init__(self, days, changes, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.days = days
        self.changes = changes
        self.layers = layers
        self._map = None
        self._orig = None     # {layer: (p1, p2, beta) of the ORIGINAL full edge set}
        return

    def init_post(self):
        super().init_post()
        self._map = _day_change_map(self.days, self.changes)
        nets = self.sim.networks
        self.layers = list(nets.keys()) if self.layers is None else sc.tolist(self.layers)
        self._orig = {}
        for lk in self.layers:
            e = nets[lk].edges
            self._orig[lk] = (np.array(e.p1), np.array(e.p2), np.array(e.beta))
        return

    def step(self):
        ti = self.ti
        if ti not in self._map:
            return
        keep = self._map[ti]
        nets = self.sim.networks
        try:
            base = int(self.sim.pars.rand_seed)
        except Exception:
            base = 0
        for li, lk in enumerate(self.layers):
            p1, p2, beta = self._orig[lk]
            n = len(p1)
            if not n:
                continue
            n_keep = int(round(keep * n))
            rng = np.random.default_rng([base, 93, ti, li])
            sel = np.sort(rng.choice(n, size=n_keep, replace=False)) if n_keep < n else np.arange(n)
            edges = nets[lk].edges
            edges.p1 = ss.uids(p1[sel])
            edges.p2 = ss.uids(p2[sel])
            edges.beta = beta[sel]
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
        for intv in self.interventions:  # initialise the child interventions
            intv.init_pre(sim)
        return

    def init_post(self):
        super().init_post()
        for intv in self.interventions:
            intv.init_post()
        return

    def step(self):
        ti = self.ti
        active = [i for i, d in enumerate(self.days) if d <= ti]  # most recently activated
        if active:
            self.interventions[active[-1]].step()
        return
