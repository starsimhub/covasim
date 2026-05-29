"""
The COVID-19 disease module for Covasim on the Starsim base.

``cv.COVID(ss.Infection)`` is the single-variant COVID disease. M2 implements
Covasim's full natural-history prognosis tree for a single variant:

    susceptible -> exposed -> infectious -> (asymptomatic | mild | severe | critical)
                                                          -> recovered | dead

The whole trajectory (every branch outcome AND every transition date) is drawn
*once at infection* in ``set_prognoses`` (the Covasim v3 design, confirmed in the
quarantined ``_v2_legacy/people.py``); ``step_state`` only flips boolean flags when
``ti_<stage> <= ti`` and never re-draws a probability. Branch probabilities are the
age-conditional prognoses from ``parameters.get_prognoses`` (already conditional --
do not re-divide). Recovery confers permanent immunity (``use_waning=False``);
waning / NAbs / reinfection are M4. Deaths are requested via
``sim.people.request_death`` and finalized in ``step_die``.

Per-agent transmissibility (``viral_load`` x ``beta_dist``) is added in M2 Task 2;
multiple variants in M3. Transmission stays the stock CRN-safe
``ss.Infection.infect()`` -- no custom transmission code.
"""
import numpy as np
import starsim as ss

__all__ = ['COVID']


class COVID(ss.Infection):
    """Single-variant COVID-19 disease with the full natural-history prognosis tree.

    Args:
        beta: per-contact transmission probability per day (scalar or dict keyed by
            network layer); Covasim's default is ``pars['beta'] = 0.016``.
        init_prev: an ``ss.bernoulli`` for seeding, or an int exact ``pop_infected``
            count (cv.Sim passes the int), or ``None``.
        dur_*: the Covasim duration distributions (lognormal mean/std in days,
            ``parameters.py`` dur block).
        rel_symp_prob/rel_severe_prob/rel_crit_prob/rel_death_prob: global severity
            scalers (default 1.0).
        n_beds_hosp/n_beds_icu/no_hosp_factor/no_icu_factor: health-system capacity
            (default no constraint; the factors are inert unless beds are set).
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta         = ss.probperday(0.016),                                # Covasim pars['beta'] (parameters.py:62)
            init_prev    = None,                                                # seeding via pop_infected (cv.Sim) or an ss.bernoulli
            # Progression durations (parameters.py:85-95; v3 lognormal_int -> ss.lognorm_ex mean/std in days)
            dur_exp2inf  = ss.lognorm_ex(mean=ss.days(4.5),  std=ss.days(1.5)),  # exposed -> infectious
            dur_inf2sym  = ss.lognorm_ex(mean=ss.days(1.1),  std=ss.days(0.9)),  # infectious -> symptomatic
            dur_sym2sev  = ss.lognorm_ex(mean=ss.days(6.6),  std=ss.days(4.9)),  # symptomatic -> severe
            dur_sev2crit = ss.lognorm_ex(mean=ss.days(1.5),  std=ss.days(2.0)),  # severe -> critical
            dur_crit2die = ss.lognorm_ex(mean=ss.days(10.7), std=ss.days(4.8)),  # critical -> death
            # Recovery durations
            dur_asym2rec = ss.lognorm_ex(mean=ss.days(8.0),  std=ss.days(2.0)),  # asymptomatic -> recovered
            dur_mild2rec = ss.lognorm_ex(mean=ss.days(8.0),  std=ss.days(2.0)),  # mild -> recovered
            dur_sev2rec  = ss.lognorm_ex(mean=ss.days(18.1), std=ss.days(6.3)),  # severe (non-critical) -> recovered
            dur_crit2rec = ss.lognorm_ex(mean=ss.days(18.1), std=ss.days(6.3)),  # critical (survivor) -> recovered
            # Global severity scalers (parameters.py:98-101)
            rel_symp_prob   = 1.0,
            rel_severe_prob = 1.0,
            rel_crit_prob   = 1.0,
            rel_death_prob  = 1.0,
            # Health-system capacity (parameters.py:117-120); factors inert while beds are None
            n_beds_hosp    = None,
            n_beds_icu     = None,
            no_hosp_factor = 2.0,
            no_icu_factor  = 2.0,
        )
        self.update_pars(pars, **kwargs)

        # Full state set (reset=True drops ss.Infection's defaults; the SIR pattern).
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('infected',                   label='Infected (exposed or infectious)'),
            ss.BoolState('exposed',                    label='Exposed (infected, pre-infectious)'),
            ss.BoolState('symptomatic',                label='Symptomatic'),
            ss.BoolState('severe',                     label='Severe (needs hospitalization)'),
            ss.BoolState('critical',                   label='Critical (needs ICU)'),
            ss.BoolState('recovered',                  label='Recovered'),
            ss.BoolState('dead',                       label='Dead'),
            ss.FloatArr('rel_sus',     default=1.0,    label='Relative susceptibility'),
            ss.FloatArr('rel_trans',   default=1.0,    label='Relative transmissibility'),
            ss.FloatArr('ti_infected',                 label='Time index of infection'),
            ss.FloatArr('ti_exposed',                  label='Time index of becoming exposed'),
            ss.FloatArr('ti_infectious',               label='Time index of becoming infectious'),
            ss.FloatArr('ti_symptomatic',              label='Time index of becoming symptomatic'),
            ss.FloatArr('ti_severe',                   label='Time index of becoming severe'),
            ss.FloatArr('ti_critical',                 label='Time index of becoming critical'),
            ss.FloatArr('ti_recovered',                label='Time index of recovery'),
            ss.FloatArr('ti_dead',                     label='Time index of death'),
            # Per-agent age-conditional branch probabilities (filled in init_post)
            ss.FloatArr('symp_prob',   default=0.0,    label='P(symptomatic | infected)'),
            ss.FloatArr('severe_prob', default=0.0,    label='P(severe | symptomatic)'),
            ss.FloatArr('crit_prob',   default=0.0,    label='P(critical | severe)'),
            ss.FloatArr('death_prob',  default=0.0,    label='P(death | critical)'),
            reset = True,
        )

        # Scratch bernoullis for the branch draws; their per-agent p is set each call
        # (the hpvsim M02 pattern -- one stable per-Dist CRN slot per branch).
        self._symp_bern  = ss.bernoulli(p=0.5)
        self._sev_bern   = ss.bernoulli(p=0.5)
        self._crit_bern  = ss.bernoulli(p=0.5)
        self._death_bern = ss.bernoulli(p=0.5)
        return

    @property
    def infectious(self):
        """Agents who transmit: infected and past the latent period (I, not merely E)."""
        return self.infected & (self.ti_infectious <= self.ti)

    # --- prognoses ------------------------------------------------------------

    def _fill_prognoses(self):
        """Fill the per-agent age-conditional branch probabilities + susceptibility.

        Mirrors v3 ``people.set_prognoses`` (people.py:152-158): age-bin the conditional
        prognoses and the susceptibility odds-ratios. ``get_prognoses`` already returns
        CONDITIONAL probabilities (it calls ``relative_prognoses`` internally), so they
        are used directly. ``sus_ORs`` are age-dependent (NOT all 1.0), so ``rel_sus``
        is age-structured; ``trans_ORs`` are all 1.0, so ``rel_trans`` stays 1.0 until
        the viral-load/beta_dist factors land in M2 Task 2.
        """
        import covasim.parameters as cvpar  # lazy: covid.py imports before sim.py, after parameters.py
        progs = cvpar.get_prognoses(by_age=True)
        age = np.asarray(self.sim.people.age)
        inds = np.digitize(age, progs['age_cutoffs']) - 1
        self.symp_prob[:]   = progs['symp_probs'][inds]
        self.severe_prob[:] = progs['severe_probs'][inds] * progs['comorbidities'][inds]  # comorbidity folds into severe
        self.crit_prob[:]   = progs['crit_probs'][inds]
        self.death_prob[:]  = progs['death_probs'][inds]
        self.rel_sus[:]     = progs['sus_ORs'][inds]  # age-dependent susceptibility (v3 people.py:157)
        return

    def _hosp_full(self):
        """Whether hospital beds are exhausted (always False at the default n_beds_hosp=None)."""
        nb = self.pars.n_beds_hosp
        return nb is not None and np.count_nonzero(self.severe) > nb

    def _icu_full(self):
        """Whether ICU beds are exhausted (always False at the default n_beds_icu=None)."""
        nb = self.pars.n_beds_icu
        return nb is not None and np.count_nonzero(self.critical) > nb

    def set_prognoses(self, uids, sources=None):
        """Draw the full disease trajectory once at infection (the v3 pre-scheduled tree)."""
        super().set_prognoses(uids, sources)  # logs the infection
        ti = self.ti
        p = self.pars

        # Entry: exposed and infectious latency
        self.susceptible[uids] = False
        self.infected[uids]    = True
        self.exposed[uids]     = True
        self.ti_infected[uids] = ti
        self.ti_exposed[uids]  = ti
        self.ti_infectious[uids] = ti + p.dur_exp2inf.rvs(uids)
        # Reset all downstream dates (defensive; matches v3 "reset all other dates")
        for arr in (self.ti_symptomatic, self.ti_severe, self.ti_critical, self.ti_recovered, self.ti_dead):
            arr[uids] = np.nan

        # Branch 1: symptomatic? (asymptomatic agents recover and cannot die)
        self._symp_bern.set(p=p.rel_symp_prob * self.symp_prob[uids])
        is_symp = self._symp_bern.rvs(uids)
        symp  = uids[is_symp]
        asymp = uids[~is_symp]
        self.ti_recovered[asymp] = self.ti_infectious[asymp] + p.dur_asym2rec.rvs(asymp)

        # Branch 2: severe? (among symptomatic)
        self.ti_symptomatic[symp] = self.ti_infectious[symp] + p.dur_inf2sym.rvs(symp)
        self._sev_bern.set(p=p.rel_severe_prob * self.severe_prob[symp])
        is_sev = self._sev_bern.rvs(symp)
        sev  = symp[is_sev]
        mild = symp[~is_sev]
        self.ti_recovered[mild] = self.ti_symptomatic[mild] + p.dur_mild2rec.rvs(mild)

        # Branch 3: critical? (among severe; no_hosp_factor raises the risk if beds are full)
        self.ti_severe[sev] = self.ti_symptomatic[sev] + p.dur_sym2sev.rvs(sev)
        hosp_factor = p.no_hosp_factor if self._hosp_full() else 1.0
        self._crit_bern.set(p=p.rel_crit_prob * self.crit_prob[sev] * hosp_factor)
        is_crit = self._crit_bern.rvs(sev)
        crit    = sev[is_crit]
        noncrit = sev[~is_crit]
        self.ti_recovered[noncrit] = self.ti_severe[noncrit] + p.dur_sev2rec.rvs(noncrit)

        # Branch 4: die? (among critical; no_icu_factor raises the risk if ICU is full)
        self.ti_critical[crit] = self.ti_severe[crit] + p.dur_sev2crit.rvs(crit)
        icu_factor = p.no_icu_factor if self._icu_full() else 1.0
        self._death_bern.set(p=p.rel_death_prob * self.death_prob[crit] * icu_factor)
        is_dead = self._death_bern.rvs(crit)
        dead    = crit[is_dead]
        survive = crit[~is_dead]
        self.ti_recovered[survive] = self.ti_critical[survive] + p.dur_crit2rec.rvs(survive)
        self.ti_dead[dead] = self.ti_critical[dead] + p.dur_crit2die.rvs(dead)
        # (ti_recovered for `dead` stays NaN from the defensive reset -- death and recovery are exclusive)
        return

    def step_state(self):
        """Advance the state machine (before transmission); flip flags on ti thresholds, no re-draws."""
        ti = self.ti
        # exposed -> infectious (the infectious property derives from ti_infectious; clear `exposed`)
        self.exposed[(self.exposed & (self.ti_infectious <= ti)).uids] = False
        # infectious -> symptomatic -> severe -> critical (cumulative nested flags)
        self.symptomatic[(self.infected & (self.ti_symptomatic <= ti)).uids] = True
        self.severe[(self.infected & (self.ti_severe <= ti)).uids] = True
        self.critical[(self.infected & (self.ti_critical <= ti)).uids] = True
        # -> recovered (permanent immunity; clear the stage flags)
        rec = (self.infected & (self.ti_recovered <= ti)).uids
        self.infected[rec]     = False
        self.recovered[rec]    = True
        self.symptomatic[rec]  = False
        self.severe[rec]       = False
        self.critical[rec]     = False
        # -> dead (request the death; resolved in step_die after transmission)
        to_dead = (self.infected & (self.ti_dead <= ti)).uids
        if len(to_dead):
            self.sim.people.request_death(to_dead)
        return

    def step_die(self, uids):
        """Reset disease flags for agents who die this step (so they are not double-counted)."""
        super().step_die(uids)
        self.infected[uids]     = False
        self.exposed[uids]      = False
        self.symptomatic[uids]  = False
        self.severe[uids]       = False
        self.critical[uids]     = False
        self.recovered[uids]    = False
        self.susceptible[uids]  = False
        self.dead[uids]         = True
        return

    # --- results --------------------------------------------------------------

    def init_results(self):
        """Add an n_infectious result (infectious is a property, so not auto-counted).

        The full burden Results (new_*/cum_*/n_* for symptomatic/severe/critical/deaths)
        are added in M2 Task 3.
        """
        super().init_results()
        self.define_results(
            ss.Result('n_infectious', dtype=int, scale=True, label='Number infectious'),
        )
        return

    def update_results(self):
        super().update_results()
        self.results.n_infectious[self.ti] = int(np.count_nonzero(self.infectious))
        return

    # --- seeding --------------------------------------------------------------

    def init_post(self):
        """Fill the age-conditional prognoses, then seed initial infections.

        Covasim seeds an *exact* count (``pop_infected``), not a per-agent probability.
        If ``init_prev`` is an integer, seed exactly that many agents (deterministically
        from the sim seed); otherwise defer to ``ss.Infection`` (``ss.bernoulli`` / None).
        """
        self._fill_prognoses()  # must precede any set_prognoses call (seeding below)

        exact = self.pars.init_prev if isinstance(self.pars.init_prev, (int, np.integer)) else None
        if exact is None:
            return super().init_post()  # ss.bernoulli / None: stock seeding

        # Exact-count path: base setup with no stock seeding, then seed exactly `exact` agents.
        self.pars.init_prev = None
        super().init_post()
        self.pars.init_prev = exact
        auids = self.sim.people.auids
        n = min(int(exact), len(auids))
        try:
            base = int(self.sim.pars.rand_seed)
        except Exception:
            base = 0
        rng = np.random.default_rng(base*100 + 50)  # distinct stream from the network layers (offsets 0-4)
        chosen = ss.uids(np.sort(rng.choice(np.asarray(auids), size=n, replace=False)))
        self.set_prognoses(chosen, sources=-1)
        self.pars._n_initial_cases = len(chosen)
        return chosen
