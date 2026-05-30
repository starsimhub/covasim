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

Per-agent transmissibility is now time-varying: each agent draws a constant
overdispersion factor (``beta_dist``, mean 1.0) at init and a two-level viral-load
kernel (``viral_dist``) over its infectious period, written to ``rel_trans`` each
step before transmission. Multiple variants come in M3. Transmission stays the
stock CRN-safe ``ss.Infection.infect()`` -- no custom transmission code; M2 only
writes the per-agent ``rel_trans`` state it consumes.
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
            # Per-agent transmissibility shape (parameters.py:60-63); closes the residual transmission gap
            beta_dist    = dict(dist='neg_binomial', par1=1.0, par2=0.45, step=0.01),  # constant per-agent overdispersion (mean 1.0)
            viral_dist   = dict(frac_time=0.3, load_ratio=2, high_cap=4),               # time-varying viral load (two-level, mean-preserving)
            asymp_factor = 1.0,                                                         # transmissibility multiplier for asymptomatic agents
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
            ss.FloatArr('rel_sus',      default=1.0,   label='Relative susceptibility'),
            ss.FloatArr('rel_trans',    default=1.0,   label='Relative transmissibility (time-varying)'),
            ss.FloatArr('rel_trans_base', default=1.0, label='Per-agent baseline transmissibility (beta_dist draw)'),
            ss.FloatArr('ti_vl_switch',                label='Time index of the viral-load high->low switch'),
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

        # Per-agent constant transmissibility draw (trans_ORs x beta_dist), drawn once at init
        # (v3 people.py:159). trans_ORs are all 1.0, so this is the beta_dist overdispersion draw.
        bd = self.pars.beta_dist
        rate, disp, step = bd['par1'], bd['par2'], bd.get('step', 1)
        nbn_n = disp
        nbn_p = disp / (rate/step + disp)  # cv.n_neg_binomial parameterization (utils.py:430)
        try:
            seed = int(self.sim.pars.rand_seed)
        except Exception:
            seed = 0
        rng = np.random.default_rng(seed*100 + 60)  # distinct deterministic stream (network 0-4, seeding 50)
        draw = rng.negative_binomial(nbn_n, nbn_p, len(age)) * step
        self.rel_trans_base[:] = progs['trans_ORs'][inds] * draw

        # Cache the two-level viral-load constants (mean-preserving normalizer Z; v3 utils.py:39-84)
        vd = self.pars.viral_dist
        z = 1.0 + vd['frac_time'] * (vd['load_ratio'] - 1)
        self._vl_high = vd['load_ratio'] / z
        self._vl_low  = 1.0 / z
        return

    def _hosp_full(self):
        """Whether hospital beds are exhausted (always False at the default n_beds_hosp=None)."""
        nb = self.pars.n_beds_hosp
        return nb is not None and np.count_nonzero(self.severe) > nb

    def _icu_full(self):
        """Whether ICU beds are exhausted (always False at the default n_beds_icu=None)."""
        nb = self.pars.n_beds_icu
        return nb is not None and np.count_nonzero(self.critical) > nb

    @staticmethod
    def _dur(dist, uids):
        """Sample a duration and round to whole days, matching v3's ``lognormal_int``.

        Covasim's durations are integer-day; without rounding here, the integer-timestep
        threshold checks in step_state effectively *ceil* a continuous duration, lengthening
        the generation interval and flattening the epidemic peak relative to v3.
        """
        return np.round(dist.rvs(uids))

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
        self.ti_infectious[uids] = ti + self._dur(p.dur_exp2inf, uids)
        # Reset all downstream dates (defensive; matches v3 "reset all other dates")
        for arr in (self.ti_symptomatic, self.ti_severe, self.ti_critical, self.ti_recovered, self.ti_dead):
            arr[uids] = np.nan

        # Branch 1: symptomatic? (asymptomatic agents recover and cannot die)
        self._symp_bern.set(p=p.rel_symp_prob * self.symp_prob[uids])
        is_symp = self._symp_bern.rvs(uids)
        symp  = uids[is_symp]
        asymp = uids[~is_symp]
        self.ti_recovered[asymp] = self.ti_infectious[asymp] + self._dur(p.dur_asym2rec, asymp)

        # Branch 2: severe? (among symptomatic)
        self.ti_symptomatic[symp] = self.ti_infectious[symp] + self._dur(p.dur_inf2sym, symp)
        self._sev_bern.set(p=p.rel_severe_prob * self.severe_prob[symp])
        is_sev = self._sev_bern.rvs(symp)
        sev  = symp[is_sev]
        mild = symp[~is_sev]
        self.ti_recovered[mild] = self.ti_symptomatic[mild] + self._dur(p.dur_mild2rec, mild)

        # Branch 3: critical? (among severe; no_hosp_factor raises the risk if beds are full)
        self.ti_severe[sev] = self.ti_symptomatic[sev] + self._dur(p.dur_sym2sev, sev)
        hosp_factor = p.no_hosp_factor if self._hosp_full() else 1.0
        self._crit_bern.set(p=p.rel_crit_prob * self.crit_prob[sev] * hosp_factor)
        is_crit = self._crit_bern.rvs(sev)
        crit    = sev[is_crit]
        noncrit = sev[~is_crit]
        self.ti_recovered[noncrit] = self.ti_severe[noncrit] + self._dur(p.dur_sev2rec, noncrit)

        # Branch 4: die? (among critical; no_icu_factor raises the risk if ICU is full)
        self.ti_critical[crit] = self.ti_severe[crit] + self._dur(p.dur_sev2crit, crit)
        icu_factor = p.no_icu_factor if self._icu_full() else 1.0
        self._death_bern.set(p=p.rel_death_prob * self.death_prob[crit] * icu_factor)
        is_dead = self._death_bern.rvs(crit)
        dead    = crit[is_dead]
        survive = crit[~is_dead]
        self.ti_recovered[survive] = self.ti_critical[survive] + self._dur(p.dur_crit2rec, survive)
        self.ti_dead[dead] = self.ti_critical[dead] + self._dur(p.dur_crit2die, dead)
        # (ti_recovered for `dead` stays NaN from the defensive reset -- death and recovery are exclusive)

        # Precompute the per-agent viral-load high->low switch time (v3 compute_viral_load trans_point).
        # End of the infectious period = recovery or death date (whichever is set).
        ti_dead = self.ti_dead[uids]
        end = np.where(np.isnan(ti_dead), self.ti_recovered[uids], ti_dead)
        infect_days = end - self.ti_infectious[uids]
        vd = p.viral_dist
        trans_point = np.minimum(vd['frac_time'], vd['high_cap'] / infect_days)  # cap the high phase at high_cap days
        self.ti_vl_switch[uids] = self.ti_infectious[uids] + trans_point * infect_days
        return

    def step_state(self):
        """Advance the state machine (before transmission); flip flags on ti thresholds, no re-draws."""
        ti = self.ti
        # exposed -> infectious (the infectious property derives from ti_infectious; clear `exposed`)
        self.exposed[(self.exposed & (self.ti_infectious <= ti)).uids] = False
        # Stage onsets -- count only NEW transitions (for the flow Results); flags are cumulative-nested.
        new_symp = (self.infected & ~self.symptomatic & (self.ti_symptomatic <= ti)).uids
        self.symptomatic[new_symp] = True
        new_sev = (self.infected & ~self.severe & (self.ti_severe <= ti)).uids
        self.severe[new_sev] = True
        new_crit = (self.infected & ~self.critical & (self.ti_critical <= ti)).uids
        self.critical[new_crit] = True
        # -> recovered (permanent immunity; clear the stage flags)
        new_rec = (self.infected & (self.ti_recovered <= ti)).uids
        self.infected[new_rec]     = False
        self.recovered[new_rec]    = True
        self.symptomatic[new_rec]  = False
        self.severe[new_rec]       = False
        self.critical[new_rec]     = False
        # -> dead (request the death; resolved in step_die after transmission)
        new_dead = (self.infected & (self.ti_dead <= ti)).uids
        if len(new_dead):
            self.sim.people.request_death(new_dead)
        # Capture this step's flows for update_results
        self._flow = dict(symptomatic=len(new_symp), severe=len(new_sev), critical=len(new_crit),
                          recoveries=len(new_rec), deaths=len(new_dead))

        # Update per-agent transmissibility BEFORE this step's transmission (loop slot 5 precedes infect at 9):
        # rel_trans = beta_dist draw x viral_load(t) x asymp_factor (v3 compute_trans_sus, utils.py:90-93).
        inf = self.infectious.uids
        if len(inf):
            early = ti < np.asarray(self.ti_vl_switch[inf])           # high viral-load phase (front-loaded)
            vl = np.where(early, self._vl_high, self._vl_low)
            f_asymp = np.where(np.asarray(self.symptomatic[inf]), 1.0, self.pars.asymp_factor)
            self.rel_trans[inf] = self.rel_trans_base[inf] * vl * f_asymp
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
        """Define the daily-flow (new_*) and cumulative (cum_*) burden Results.

        Stocks (n_symptomatic/n_severe/n_critical/n_recovered/n_infected/n_dead) are
        AUTO-created and counted by Starsim from the BoolStates, so they are not defined
        here -- only n_infectious (a property, not a state) is manual. ss.Infection already
        provides prevalence / new_infections / cum_infections. All are scale=True so Starsim
        multiplies them by pop_scale at finalize.
        """
        super().init_results()

        def R(name, label):
            return ss.Result(name, dtype=int, scale=True, label=label)

        self.define_results(
            R('n_infectious',  'Number infectious'),  # infectious is a property, not auto-counted
            R('new_symptomatic', 'New symptomatic'),
            R('new_severe',      'New severe'),
            R('new_critical',    'New critical'),
            R('new_recoveries',  'New recoveries'),
            R('new_deaths',      'New deaths'),
            R('cum_symptomatic', 'Cumulative symptomatic'),
            R('cum_severe',      'Cumulative severe'),
            R('cum_critical',    'Cumulative critical'),
            R('cum_recoveries',  'Cumulative recoveries'),
            R('cum_deaths',      'Cumulative deaths'),
        )
        self._flow = dict(symptomatic=0, severe=0, critical=0, recoveries=0, deaths=0)
        return

    def update_results(self):
        super().update_results()  # auto-counts the BoolState stocks (n_symptomatic, n_severe, ...)
        ti = self.ti
        res = self.results
        res.n_infectious[ti] = int(np.count_nonzero(self.infectious))
        # Flows (captured this timestep in step_state)
        res.new_symptomatic[ti] = self._flow['symptomatic']
        res.new_severe[ti]      = self._flow['severe']
        res.new_critical[ti]    = self._flow['critical']
        res.new_recoveries[ti]  = self._flow['recoveries']
        res.new_deaths[ti]      = self._flow['deaths']
        return

    def finalize_results(self):
        """Cumulate the daily flows into the cum_* Results (the hpvsim M02 pattern)."""
        super().finalize_results()
        res = self.results
        res.cum_symptomatic[:] = np.cumsum(res.new_symptomatic[:])
        res.cum_severe[:]      = np.cumsum(res.new_severe[:])
        res.cum_critical[:]    = np.cumsum(res.new_critical[:])
        res.cum_recoveries[:]  = np.cumsum(res.new_recoveries[:])
        res.cum_deaths[:]      = np.cumsum(res.new_deaths[:])
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
