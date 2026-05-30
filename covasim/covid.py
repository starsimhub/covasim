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

    def __init__(self, pars=None, variants=None, **kwargs):
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
            # Waning immunity / NAbs (M4; parameters.py:69-78). use_waning=False keeps the M2/M3 static
            # path (no NAb state evolves) -- byte-identical. When True, cv.CrossImmunity advances NAb
            # kinetics and maps them to per-axis protection via calc_VE. Defaults are the v3 values.
            use_waning   = False,
            nab_init     = dict(dist='normal', par1=0, par2=2),  # log2(initial NAb) after natural infection
            nab_decay    = dict(form='nab_growth_decay', growth_time=21, decay_rate1=np.log(2)/50,
                                decay_time1=150, decay_rate2=np.log(2)/250, decay_time2=365),
            nab_boost    = 1.5,                                  # multiplicative boost to peak NAb on reinfection
            nab_eff      = dict(alpha_inf=1.08, alpha_inf_diff=1.812, beta_inf=0.967, alpha_symp_inf=-0.739,
                                beta_symp_inf=0.038, alpha_sev_symp=-0.014, beta_sev_symp=0.079),  # NAb->efficacy
            rel_imm_symp = dict(asymp=0.85, mild=1.0, severe=1.5),  # natural-immunity scaling by symptom severity
            trans_redux  = 0.59,                                 # transmissibility reduction for breakthrough infections
            # Testing / quarantine (M5; parameters.py:106-108). iso_factor/quar_factor reduce the
            # transmissibility of isolated/quarantined agents (scalar M5 approximation of v3's per-layer
            # factors -- the random-layer values; see the M5 spec Open Q A). They are inert until a
            # testing/tracing intervention puts agents into isolation/quarantine, so M1-M4 are unchanged.
            quar_period  = 14,    # days an agent quarantines
            iso_factor   = 0.2,   # transmissibility multiplier for isolated (diagnosed) agents
            quar_factor  = 0.3,   # transmissibility multiplier for quarantined agents
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
            # Scalar per-agent variant tags (the v3 *_variant scalars; NaN = "none"). A COVID host
            # has exactly ONE SEIR chain, so these are scalars, not 2D -- exclusivity is structural
            # (M3 design spec sec.1). For nv==1 every infected agent is wild (variant 0).
            ss.FloatArr('exposed_variant',    label='Variant index this agent is exposed/infected with'),
            ss.FloatArr('infectious_variant', label='Variant index this agent is infectious with'),
            ss.FloatArr('recovered_variant',  label='Variant index this agent last recovered from'),
            # Neutralizing-antibody (NAb) host-level state (M4; v3 nab_states). 1D per host (the NAb
            # pool is shared across all prior infections/vaccinations; cross-immunity weights it per
            # variant in the connector). Only evolves when use_waning=True (else all-default => inert).
            ss.FloatArr('peak_nab',        default=0.0, label='Peak neutralizing-antibody titre'),
            ss.FloatArr('nab',             default=0.0, label='Current neutralizing-antibody titre'),
            ss.FloatArr('t_nab_event',                  label='Time index of the last NAb-conferring event'),
            ss.FloatArr('n_breakthroughs', default=0.0, label='Number of breakthrough (post-NAb) infections'),
            # Testing / tracing / quarantine host state (M5; v3 states). All default False/NaN, so they
            # are inert until a testing or contact-tracing intervention sets the trigger dates =>
            # M1-M4 are byte-identical with no such intervention attached.
            ss.BoolState('tested',         label='Ever tested'),
            ss.BoolState('diagnosed',      label='Diagnosed (confirmed case)'),
            ss.BoolState('known_contact',  label='Known contact of a diagnosed case'),
            ss.BoolState('quarantined',    label='In quarantine'),
            ss.BoolState('isolated',       label='In isolation'),
            ss.FloatArr('date_tested',          label='Date last tested'),
            ss.FloatArr('date_pos_test',        label='Date of a test that will return positive'),
            ss.FloatArr('date_diagnosed',       label='Date diagnosed'),
            ss.FloatArr('date_quarantined',     label='Date entered quarantine'),
            ss.FloatArr('date_end_quarantine',  label='Date quarantine ends'),
            ss.FloatArr('date_end_isolation',   label='Date isolation ends'),
            # Vaccination host state (M6; v3 vacc_states). Vaccine immunity flows through the same NAb
            # pipeline as natural infection. Inert until a vaccination intervention is attached =>
            # M1-M5 byte-identical with no such intervention.
            ss.BoolState('vaccinated',     label='Vaccinated'),
            ss.FloatArr('doses',           default=0.0, label='Number of vaccine doses received'),
            ss.FloatArr('vaccine_source',               label='Index of the vaccine received'),
            ss.FloatArr('date_vaccinated',              label='Date last vaccinated'),
            reset = True,
        )

        # Scratch bernoullis for the branch draws; their per-agent p is set each call
        # (the hpvsim M02 pattern -- one stable per-Dist CRN slot per branch).
        self._symp_bern  = ss.bernoulli(p=0.5)
        self._sev_bern   = ss.bernoulli(p=0.5)
        self._crit_bern  = ss.bernoulli(p=0.5)
        self._death_bern = ss.bernoulli(p=0.5)

        # --- Variant axis (M3) ---------------------------------------------------
        # Design B: a SINGLE COVID module carries an internal variant dimension. nv==1
        # (no variants registered) is byte-identical to M2. cv.variant.initialize() and
        # cv.Sim(variants=...) grow these in M3 Task 2; the 5 per-variant keys come from
        # parameters.get_variant_pars. wild is always index 0.
        import covasim.parameters as cvpar  # lazy: covid.py imports before immunity.py
        self.nv = 1
        self.variant_map  = {0: 'wild'}                       # index -> label (wild always 0)
        self.variant_pars = {'wild': dict(cvpar.get_variant_pars(variant='wild'))}  # 5 keys, all 1.0
        # Register any additional variants (cv.variant objects, or string/dict sugar) into the single
        # module's variant axis NOW, before init_post allocates the nv-shaped immunity arrays. Each
        # grows nv / variant_map / variant_pars and gets a stable .index; wild stays index 0.
        self._variant_objs = []
        if variants is not None:
            from . import immunity as cvimm
            if isinstance(variants, (str, dict, cvimm.variant)):
                variants = [variants]
            for v in variants:
                if not isinstance(v, cvimm.variant):
                    v = cvimm.variant(v, days=0)  # bare name/dict => introduce at t0
                v.initialize(self)
                self._variant_objs.append(v)
        # 2D per-variant immunity arrays sus_imm/symp_imm/sev_imm, shape (nv, n_raw), allocated in
        # init_post once n_agents is known (plain ndarrays indexed by raw UID -- NOT growth-aware,
        # so M3 forbids births; see init_post). All-zero at nv==1 => no effect (M2 byte-identity).
        self.sus_imm = self.symp_imm = self.sev_imm = None
        # Per-target variant of THIS step's new cases, keyed by UID (set in the M3 infect() override;
        # read by set_prognoses). Empty => the stock single-variant path (everyone wild).
        self._new_case_variant = {}
        # Cross-immunity / reinfection switch. False (the M2 path) => recovered agents stay
        # susceptible=False (permanent immunity), preserving M2 byte-identically. cv.CrossImmunity
        # (M3 Task 3) flips this on when attached (nv>1), enabling reinfection + static cross-immunity.
        self.cross_immunity_active = False

        # NAb engine (M4). The init_nab Dist (log2 NAb after natural infection) is created LAST so the
        # M2/M3 branch bernoullis keep their CRN slots (=> use_waning=False stays byte-identical). Only
        # drawn when use_waning. nab_kin (the precomputed waning kernel) is built in init_post.
        self._nab_init = ss.normal(loc=self.pars.nab_init['par1'], scale=self.pars.nab_init['par2'])
        self.nab_kin = None

        # Testing (M5). The test bernoulli (sensitivity / loss-to-follow-up) is created LAST so the
        # other branch Dists keep their CRN slots; only drawn when covid.test() is called by a testing
        # intervention => no testing intervention => no draw => byte-identical to M1-M4. The pending-
        # quarantine schedule (start_day -> [(uid, end_day)]) is (re)set in init_post.
        self._test_bern = ss.bernoulli(p=1.0)
        self._pending_quarantine = {}

        # Vaccination (M6). Vaccine registry (label -> pars, index -> label), populated by vaccination
        # interventions at init. The vaccine NAb-init draw uses its OWN Dist (created last, .set per
        # vaccine), distinct from the natural _nab_init, so a dose drawn at the intervention slot does
        # not perturb the natural-infection NAb draw at the transmission slot => M4 stays byte-identical.
        self.vaccine_pars = {}
        self.vaccine_map  = {}
        self._vacc_nab_init = ss.normal(loc=0.0, scale=2.0)
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

    # Per-variant flow stems (the 4 v3 by_variant flows). new_infectious is counted at the
    # infectious transition (step_state); the other three at infection (set_prognoses).
    _FLOW_VARIANT_KEYS = ('new_infections', 'new_symptomatic', 'new_severe', 'new_infectious')

    def _ensure_flow_variant(self):
        """Lazily (re)allocate the per-variant flow accumulators, sized to the current ``nv``.

        ``init_post`` (seeding) can call ``set_prognoses`` before ``init_results`` runs, and ``nv``
        may have grown via ``cv.variant`` between ``__init__`` and ``init``; this keeps the dict
        present and correctly sized in both cases.
        """
        fv = getattr(self, '_flow_variant', None)
        if fv is None or fv['new_infections'].shape[0] != self.nv:
            self._flow_variant = {k: np.zeros(self.nv) for k in self._FLOW_VARIANT_KEYS}
        return

    def _update_peak_nab(self, uids, nab_pars=None, symp_scale=None):
        """Set/boost peak NAb at a NAb-conferring event (M4/M6; the v3 ``update_peak_nab``).

        Agents with prior NAbs (``peak_nab>0``) are boosted by ``nab_boost``; the rest draw an initial
        log2 NAb from ``nab_init``.

        ``nab_pars=None`` (natural infection): use the disease ``nab_init``/``nab_boost``, scale the
        initial NAb by symptom severity (``symp_scale`` = per-UID ``rel_imm_symp``) and normalise onto
        the vaccine scale by ``1 + alpha_inf_diff`` (unchanged M4 behaviour, drawn from ``_nab_init``).

        ``nab_pars`` given (a vaccine dose, M6): use the vaccine's ``nab_init``/``nab_boost`` with NO
        symptom scaling and NO natural normalisation, drawn from the separate ``_vacc_nab_init`` Dist.
        """
        p = self.pars
        prior = np.asarray(self.peak_nab[uids]) > 0
        prior_uids = uids[prior]
        no_prior_uids = uids[~prior]
        boost = p.nab_boost if nab_pars is None else nab_pars['nab_boost']
        if len(prior_uids):  # boost agents who already had NAbs (reinfection / prior dose)
            self.peak_nab[prior_uids] = self.peak_nab[prior_uids] * boost
        if len(no_prior_uids):  # draw an initial NAb for agents with no prior immunity
            if nab_pars is None:  # natural infection
                init = np.asarray(self._nab_init.rvs(no_prior_uids))
                norm = 1.0 + p.nab_eff['alpha_inf_diff']         # natural -> vaccine NAb-scale normalisation
                self.peak_nab[no_prior_uids] = (2.0 ** init) * symp_scale[~prior] * norm
            else:  # a vaccine dose: draw with the vaccine's nab_init (no symptom scaling/normalisation)
                ni = nab_pars['nab_init']
                self._vacc_nab_init.set(loc=ni['par1'], scale=ni['par2'])
                init = np.asarray(self._vacc_nab_init.rvs(no_prior_uids))
                self.peak_nab[no_prior_uids] = 2.0 ** init
        self.t_nab_event[uids] = self.ti
        return

    def _variant_of(self, uids, variant=None):
        """Per-UID variant index for a batch of new cases (aligned to ``uids`` order).

        ``variant`` set (int) => all these cases are that variant (seeds / mid-run imports).
        ``variant`` None => read the per-target variant recorded by the M3 infect() override
        (``_new_case_variant``, keyed by UID); empty dict => everyone wild (the M2 path).
        """
        u = np.asarray(uids)
        if variant is not None:
            return np.full(len(u), int(variant), dtype=int)
        if self._new_case_variant:
            return np.array([self._new_case_variant.get(int(x), 0) for x in u], dtype=int)
        return np.zeros(len(u), dtype=int)

    def set_prognoses(self, uids, sources=None, variant=None):
        """Draw the full disease trajectory once at infection (the v3 pre-scheduled tree).

        Variant-aware (M3): each branch probability is scaled by the per-variant
        ``rel_*_prob`` and reduced by this variant's cross-immunity (``symp_imm``/``sev_imm``),
        exactly as v3 ``people.infect`` (people.py:523,538). The per-UID multipliers fold
        into the SAME four bernoulli draws as M2, so at ``nv==1`` (all-wild factors 1.0,
        all-zero immunity) the draw sequence is byte-identical to M2.
        """
        super().set_prognoses(uids, sources)  # logs the infection
        ti = self.ti
        p = self.pars
        u = np.asarray(uids)
        self._ensure_flow_variant()

        # Per-UID variant + per-variant branch-probability multipliers (length-nv lookups).
        var_of = self._variant_of(uids, variant)
        labels = [self.variant_map[i] for i in range(self.nv)]
        relsymp_v  = np.array([self.variant_pars[l]['rel_symp_prob']   for l in labels])
        relsev_v   = np.array([self.variant_pars[l]['rel_severe_prob'] for l in labels])
        relcrit_v  = np.array([self.variant_pars[l]['rel_crit_prob']   for l in labels])
        reldeath_v = np.array([self.variant_pars[l]['rel_death_prob']  for l in labels])
        relsymp_u  = relsymp_v[var_of];  relsev_u   = relsev_v[var_of]
        relcrit_u  = relcrit_v[var_of];  reldeath_u = reldeath_v[var_of]
        # Per-UID cross-immunity reductions for this variant (NaN-free 0.0 at nv==1).
        sympimm_u = self.symp_imm[var_of, u] if self.symp_imm is not None else np.zeros(len(u))
        sevimm_u  = self.sev_imm[var_of, u]  if self.sev_imm  is not None else np.zeros(len(u))

        # Entry: exposed + infectious latency; tag the variant; clear `recovered` on reinfection
        # (matches v3 people.infect:497 -- prevents an n_recovered double-count; no-op at M2 since
        # transmission only ever yields susceptible targets, who are not recovered).
        self.susceptible[uids] = False
        self.infected[uids]    = True
        self.exposed[uids]     = True
        self.recovered[uids]   = False
        self.exposed_variant[uids] = var_of.astype(float)
        self.ti_infected[uids] = ti
        self.ti_exposed[uids]  = ti
        self.ti_infectious[uids] = ti + self._dur(p.dur_exp2inf, uids)
        # Edge case: if dur_exp2inf rounds to 0 the agent is infectious-by-property THIS step, after
        # step_state already ran -- tag infectious_variant now so the by_variant stock stays exact
        # (step_state still clears `exposed` and counts the new_infectious flow next step).
        imm = uids[np.asarray(self.ti_infectious[uids]) <= ti]
        if len(imm):
            self.infectious_variant[imm] = self.exposed_variant[imm]
        # Reset all downstream dates (defensive; matches v3 "reset all other dates")
        for arr in (self.ti_symptomatic, self.ti_severe, self.ti_critical, self.ti_recovered, self.ti_dead):
            arr[uids] = np.nan

        # Branch 1: symptomatic? (asymptomatic agents recover and cannot die)
        self._symp_bern.set(p=p.rel_symp_prob * relsymp_u * self.symp_prob[uids] * (1 - sympimm_u))
        is_symp = self._symp_bern.rvs(uids)
        symp  = uids[is_symp]
        asymp = uids[~is_symp]
        self.ti_recovered[asymp] = self.ti_infectious[asymp] + self._dur(p.dur_asym2rec, asymp)

        # Branch 2: severe? (among symptomatic)
        self.ti_symptomatic[symp] = self.ti_infectious[symp] + self._dur(p.dur_inf2sym, symp)
        self._sev_bern.set(p=p.rel_severe_prob * relsev_u[is_symp] * self.severe_prob[symp] * (1 - sevimm_u[is_symp]))
        is_sev = self._sev_bern.rvs(symp)
        sev  = symp[is_sev]
        mild = symp[~is_sev]
        self.ti_recovered[mild] = self.ti_symptomatic[mild] + self._dur(p.dur_mild2rec, mild)

        # Branch 3: critical? (among severe; no_hosp_factor raises the risk if beds are full)
        self.ti_severe[sev] = self.ti_symptomatic[sev] + self._dur(p.dur_sym2sev, sev)
        hosp_factor = p.no_hosp_factor if self._hosp_full() else 1.0
        self._crit_bern.set(p=p.rel_crit_prob * relcrit_u[is_symp][is_sev] * self.crit_prob[sev] * hosp_factor)
        is_crit = self._crit_bern.rvs(sev)
        crit    = sev[is_crit]
        noncrit = sev[~is_crit]
        self.ti_recovered[noncrit] = self.ti_severe[noncrit] + self._dur(p.dur_sev2rec, noncrit)

        # Branch 4: die? (among critical; no_icu_factor raises the risk if ICU is full)
        self.ti_critical[crit] = self.ti_severe[crit] + self._dur(p.dur_sev2crit, crit)
        icu_factor = p.no_icu_factor if self._icu_full() else 1.0
        self._death_bern.set(p=p.rel_death_prob * reldeath_u[is_symp][is_sev][is_crit] * self.death_prob[crit] * icu_factor)
        is_dead = self._death_bern.rvs(crit)
        dead    = crit[is_dead]
        survive = crit[~is_dead]
        self.ti_recovered[survive] = self.ti_critical[survive] + self._dur(p.dur_crit2rec, survive)
        self.ti_dead[dead] = self.ti_critical[dead] + self._dur(p.dur_crit2die, dead)
        # (ti_recovered for `dead` stays NaN from the defensive reset -- death and recovery are exclusive)

        # Per-variant flows, counted AT INFECTION (the v3 quirk: new_{infections,symptomatic,severe}
        # _by_variant are destined counts recorded in infect(); new_infectious_by_variant is counted
        # at the infectious transition instead, in step_state).
        self._flow_variant['new_infections']  += np.bincount(var_of,           minlength=self.nv)
        self._flow_variant['new_symptomatic'] += np.bincount(var_of[is_symp],  minlength=self.nv)
        self._flow_variant['new_severe']      += np.bincount(var_of[is_symp][is_sev], minlength=self.nv)

        # Precompute the per-agent viral-load high->low switch time (v3 compute_viral_load trans_point).
        # End of the infectious period = recovery or death date (whichever is set).
        ti_dead = self.ti_dead[uids]
        end = np.where(np.isnan(ti_dead), self.ti_recovered[uids], ti_dead)
        infect_days = end - self.ti_infectious[uids]
        vd = p.viral_dist
        trans_point = np.minimum(vd['frac_time'], vd['high_cap'] / infect_days)  # cap the high phase at high_cap days
        self.ti_vl_switch[uids] = self.ti_infectious[uids] + trans_point * infect_days

        # NAb acquisition + breakthrough transmissibility (M4; only under waning immunity). When
        # use_waning=False this whole block is skipped, so no NAb Dist is drawn => byte-identical to M3.
        if p.use_waning:
            # Per-UID symptom-severity scaling for the initial NAb (asymp/mild/severe -> rel_imm_symp).
            ris = p.rel_imm_symp
            symp_scale = np.full(len(u), ris['asymp'], dtype=float)
            symp_scale[is_symp] = ris['mild']
            sev_positions = np.nonzero(is_symp)[0][is_sev]  # uids-positions that reached the severe branch
            symp_scale[sev_positions] = ris['severe']
            # Breakthrough: the FIRST reinfection of an agent with prior NAbs gets reduced transmissibility
            # (v3 first-breakthrough rule); read peak_nab BEFORE _update_peak_nab sets/boosts it.
            prior_bt = uids[np.asarray(self.peak_nab[uids]) > 0]
            if len(prior_bt):
                first = prior_bt[np.asarray(self.n_breakthroughs[prior_bt]) == 0]
                if len(first):
                    self.rel_trans_base[first] = self.rel_trans_base[first] * p.trans_redux
                self.n_breakthroughs[prior_bt] = self.n_breakthroughs[prior_bt] + 1
            self._update_peak_nab(uids, symp_scale=symp_scale)
        return

    def infect(self):
        """Per-variant transmission (M3): loop variants with per-variant beta + cross-immunity.

        Overrides the single-beta ``ss.Infection.infect()`` (starsim ``diseases.py``), wrapping its
        network loop in ``for vi in range(nv)`` exactly as v3 ``sim.py:622-649``. Per variant it
        (a) masks sources by ``infectious_variant==vi``, (b) scales transmissibility by the variant's
        ``rel_beta``, and (c) folds this variant's cross-immunity ``(1 - sus_imm[vi])`` into ``rel_sus``.
        The CRN-load-bearing ``self.trans_rng.rvs(src, trg)`` call is preserved verbatim; ``multi_random``
        auto-jumps per call, so calling it once per variant gives each variant an independent draw.

        Exclusivity at dedup: candidates are appended in ascending ``vi`` order, then
        ``unique(return_index=True)`` keeps the first occurrence => the lowest-variant-index wins,
        reproducing v3's sequential-loop tie-break (v3 mutated ``susceptible=False`` inside the loop).
        The surviving per-target variant is recorded by UID in ``self._new_case_variant`` for
        ``set_prognoses`` (which runs via ``set_outcomes`` with no variant arg).

        At ``nv==1`` this is byte-identical to the stock single-beta path (one variant, ``rel_beta``
        1.0, all-zero ``sus_imm``, the same ``trans_rng`` call sequence).
        """
        ss_int = ss.dtypes.int
        nv = self.nv
        betamap = self.validate_beta()
        susc = self.susceptible  # the age-OR baseline x susceptible mask (cross-immunity folds in per variant)
        auids = np.asarray(self.sim.people.auids)  # active UIDs, aligned with the FloatArr active values
        # M5: quarantined agents are also LESS SUSCEPTIBLE (v3 quar_factor applies to transmissibility
        # AND susceptibility). Inert with no testing intervention (nobody quarantined => multiplier 1).
        quar_vals = np.asarray(self.quarantined.values)
        quar_mult = np.where(quar_vals, self.pars.quar_factor, 1.0) if quar_vals.any() else None
        new_cases, sources, networks, case_variant = [], [], [], []

        for vi in range(nv):
            label = self.variant_map[vi]
            rel_beta_v = float(self.variant_pars[label]['rel_beta'])
            # Source mask: agents infectious WITH this variant (== full infectious set at nv==1).
            inf_mask = self.infectious if nv == 1 else (self.infectious & (self.infectious_variant == vi))
            # Fold the variant's relative beta into rel_trans on the ACTIVE values (== scaling beta).
            # Done here, not on rel_trans.raw, so the uninitialized inactive raw slots from asnew are
            # never multiplied (a garbage*rel_beta overflow would otherwise trip COVASIM_WARNINGS=error).
            trans_vals = inf_mask * self.rel_trans
            if rel_beta_v != 1.0:
                trans_vals = trans_vals * rel_beta_v
            rel_trans = self.rel_trans.asnew(trans_vals)
            sus_vals = susc * self.rel_sus
            if self.cross_immunity_active:  # reduce susceptibility by this variant's cross-immunity
                # Fold sus_imm (incl. the M4 NAb-weighted value) even at nv==1, so use_waning=True
                # single-variant reinfection is protected. sus_imm is all-zero when no connector is
                # attached (M2/M3 nv==1, use_waning=False), so this is a no-op there => byte-identical.
                # Index by the active UIDs so only active values are touched (no garbage-slot arithmetic).
                sus_vals = sus_vals * (1.0 - self.sus_imm[vi][auids])
            if quar_mult is not None:  # M5: quarantined agents are less susceptible
                sus_vals = sus_vals * quar_mult
            rel_sus = self.rel_sus.asnew(sus_vals)

            for i, (nkey, route) in enumerate(self.sim.networks.items()):
                nk = ss.standardize_netkey(nkey)
                if isinstance(route, ss.Network):
                    if len(route):
                        edges = route.edges
                        p1p2b0 = [edges.p1, edges.p2, betamap[nk][0]]
                        p2p1b1 = [edges.p2, edges.p1, betamap[nk][1]]
                        for src, trg, beta in [p1p2b0, p2p1b1]:
                            if beta:
                                disease_beta = beta.to_prob(self.t.dt) if isinstance(beta, ss.Rate) else beta
                                beta_per_dt = route.net_beta(disease_beta=disease_beta, disease=self)
                                randvals = self.trans_rng.rvs(src, trg)  # CRN: preserve verbatim
                                target_uids, source_uids = self.compute_transmission(
                                    src, trg, rel_trans, rel_sus, beta_per_dt, randvals)
                                new_cases.append(target_uids)
                                sources.append(source_uids)
                                networks.append(np.full(len(target_uids), dtype=ss_int, fill_value=i))
                                case_variant.append(np.full(len(target_uids), dtype=ss_int, fill_value=vi))
                elif isinstance(route, ss.Route):  # mixing pools etc. (unused by Covasim, kept for parity)
                    disease_beta = betamap[nk][0].to_prob(self.t.dt) if isinstance(betamap[nk][0], ss.Rate) else betamap[nk][0]
                    target_uids = route.compute_transmission(rel_sus, rel_trans, disease_beta, disease=self)
                    new_cases.append(target_uids)
                    sources.append(np.full(len(target_uids), dtype=ss_int, fill_value=ss.dtypes.int_nan))
                    networks.append(np.full(len(target_uids), dtype=ss_int, fill_value=i))
                    case_variant.append(np.full(len(target_uids), dtype=ss_int, fill_value=vi))
                else:
                    errormsg = f'Cannot compute transmission via route {type(route)}; subclass ss.Route.'
                    raise TypeError(errormsg)

        # Finalize: dedup keeping the FIRST (lowest-vi) occurrence per target = v3 tie-break.
        if len(new_cases) and len(sources):
            new_cases = ss.uids.concatenate(new_cases)
            new_cases, inds = new_cases.unique(return_index=True)
            sources = ss.uids.concatenate(sources)[inds]
            networks = np.concatenate(networks)[inds]
            case_variant = np.concatenate(case_variant)[inds]
            # Record per-target variant keyed by UID (survives set_outcomes' congenital/age split).
            self._new_case_variant = {int(u): int(v) for u, v in zip(np.asarray(new_cases), case_variant)}
        else:
            new_cases = ss.uids()
            sources = ss.uids()
            networks = np.empty(0, dtype=ss_int)
            self._new_case_variant = {}
        return new_cases, sources, networks

    def import_variant(self, uids, variant):
        """Seed an external introduction of ``variant`` (int index) into ``uids`` (the v3 importation).

        Routes to ``set_prognoses`` with an explicit variant and bumps the ``n_imports`` Result. Used
        by ``cv.variant.apply()`` for mid-run introductions and for t0 seeding of added variants.
        """
        uids = ss.uids(uids)
        if not len(uids):
            return uids
        self.set_prognoses(uids, sources=None, variant=int(variant))
        if 'n_imports' in self.results:
            self.results['n_imports'][self.ti] += len(uids)
        return uids

    # --- testing / tracing / quarantine (M5) ----------------------------------

    def test(self, uids, test_sensitivity=1.0, loss_prob=0.0, test_delay=0):
        """Test agents and schedule positive diagnoses (the v3 ``People.test`` action).

        Called by the testing interventions (M5 Task 2), not directly by users. Infectious agents
        test positive with probability ``test_sensitivity``; not-already-diagnosed positives that are
        not lost to follow-up get ``date_diagnosed = ti + test_delay`` (and ``date_pos_test = ti``).
        Returns the UIDs that will be diagnosed.
        """
        uids = ss.uids(np.unique(np.asarray(uids)))
        if not len(uids):
            return uids
        ti = self.ti
        self.tested[uids] = True
        self.date_tested[uids] = ti
        self._test_flow['tests'] += len(uids)
        inf = uids[np.asarray(self.infectious[uids])]          # only infectious agents can test positive
        if not len(inf):
            return ss.uids()
        self._test_bern.set(p=test_sensitivity)
        pos = inf[self._test_bern.rvs(inf)]                    # true positives (sensitivity)
        not_diag = pos[np.isnan(np.asarray(self.date_diagnosed[pos]))]  # exclude already-diagnosed
        if not len(not_diag):
            return ss.uids()
        self._test_bern.set(p=1.0 - loss_prob)
        final = not_diag[self._test_bern.rvs(not_diag)]        # exclude loss-to-follow-up
        self.date_diagnosed[final] = ti + test_delay
        self.date_pos_test[final] = ti
        return final

    def schedule_quarantine(self, uids, start_date=None, period=None):
        """Schedule a quarantine request (the v3 ``People.schedule_quarantine``).

        Eligibility (not already dead/recovered/diagnosed/isolated) is re-checked when ``start_date``
        is reached, in ``_step_testing``. Defaults: ``start_date=ti``, ``period=quar_period``.
        """
        ti = self.ti
        start_date = ti if start_date is None else int(start_date)
        period = self.pars.quar_period if period is None else int(period)
        bucket = self._pending_quarantine.setdefault(start_date, [])
        for u in np.asarray(uids):
            bucket.append((int(u), start_date + period))
        return

    def vaccinate_agents(self, uids, label, index):
        """Apply a vaccine dose to ``uids`` (M6; the NAb side of the v3 ``BaseVaccination.vaccinate``).

        Sets the vaccination state and confers/boosts peak NAb via the vaccine's ``nab_init``/
        ``nab_boost`` (the shared M4 NAb pipeline). The intervention is responsible for selecting/
        de-duplicating ``uids`` (skipping dead / already-fully-dosed); this just applies the dose.
        ``label``/``index`` identify the vaccine in ``vaccine_pars``/``vaccine_map``.
        """
        uids = ss.uids(np.unique(np.asarray(uids)))
        if not len(uids):
            return uids
        prior_vacc = np.asarray(self.vaccinated[uids])  # already vaccinated => not a NEW vaccinee
        self.vaccinated[uids] = True
        self.vaccine_source[uids] = index
        self.doses[uids] = self.doses[uids] + 1
        self.date_vaccinated[uids] = self.ti
        self._update_peak_nab(uids, nab_pars=self.vaccine_pars[label])
        self._vacc_flow['doses'] += len(uids)
        self._vacc_flow['vaccinated'] += int((~prior_vacc).sum())
        return uids

    def _step_testing(self):
        """Diagnosis / quarantine / isolation state machines (the v3 People.check_* updates).

        Inert until a testing/tracing intervention sets the trigger dates or schedules a quarantine
        (all date arrays default NaN and the pending-quarantine dict is empty), so this is a no-op for
        M1-M4 -- no RNG, no state change.
        """
        ti = self.ti
        # check_diagnosed: positives become diagnosed when their (delayed) diagnosis date arrives.
        self.date_pos_test[((self.date_pos_test <= ti) & ~self.diagnosed).uids] = np.nan
        new_diag = ((self.date_diagnosed <= ti) & ~self.diagnosed).uids
        self.diagnosed[new_diag] = True
        self._test_flow['diagnoses'] += len(new_diag)
        # check_enter_iso: anyone diagnosed this step isolates until recovery.
        if len(new_diag):
            self.isolated[new_diag] = True
            self.date_end_isolation[new_diag] = self.ti_recovered[new_diag]
        # check_exit_iso: release agents whose isolation has ended.
        self.isolated[((self.date_end_isolation <= ti) & self.isolated).uids] = False
        # check_quar: process pending quarantine requests scheduled for today.
        pending = self._pending_quarantine.pop(ti, [])
        for ind, end_day in pending:
            if self.quarantined[ind]:
                self.date_end_quarantine[ind] = max(float(self.date_end_quarantine[ind]), end_day)
            elif not (self.dead[ind] or self.recovered[ind] or self.diagnosed[ind] or self.isolated[ind]):
                self.quarantined[ind] = True
                self.date_quarantined[ind] = ti
                self.date_end_quarantine[ind] = end_day
        # Diagnosed-today agents end quarantine (they move to isolation instead).
        self.date_end_quarantine[(self.quarantined & (self.date_diagnosed == ti)).uids] = ti
        # Release agents whose quarantine has ended.
        self.quarantined[((self.date_end_quarantine <= ti) & self.quarantined).uids] = False
        return

    def _introduce_variants(self):
        """Drive mid-run + t0 variant introductions (the v3 ``variant.apply``) on matched days."""
        for var in self._variant_objs:
            var.apply(self)
        return

    def step_state(self):
        """Advance the state machine (before transmission); flip flags on ti thresholds, no re-draws."""
        ti = self.ti
        self._ensure_flow_variant()
        # Reset the per-variant flow accumulators for THIS step first, so the import seeding below and
        # transmission's set_prognoses later both accumulate into a fresh array. new_infectious is
        # counted here; new_{infections,symptomatic,severe} are accumulated in set_prognoses.
        for v in self._flow_variant.values():
            v[:] = 0.0
        if not hasattr(self, '_test_flow'):
            self._test_flow = dict(tests=0, diagnoses=0)
        self._test_flow['tests'] = 0
        self._test_flow['diagnoses'] = 0
        if not hasattr(self, '_vacc_flow'):
            self._vacc_flow = dict(doses=0, vaccinated=0)
        self._vacc_flow['doses'] = 0
        self._vacc_flow['vaccinated'] = 0
        self._introduce_variants()  # seed any variant introductions scheduled for this day
        # exposed -> infectious: tag infectious_variant from exposed_variant (v3 check_infectious),
        # count new_infectious_by_variant, then clear the `exposed` flag. The gate is `exposed &
        # (ti_infectious<=ti)` (the agents transitioning THIS step) rather than the full infectious
        # property `infected & (ti_infectious<=ti)`: the latter would re-count already-infectious
        # agents every step (over-counting the new_infectious flow). Same-step infectiousness
        # (dur_exp2inf==0) is tagged in set_prognoses, and those agents are still `exposed` here so
        # they are counted exactly once. Every infectious agent thus carries a finite infectious_variant,
        # so the nv>1 source mask `infectious & (infectious_variant==vi)` == the full infectious set.
        new_inf = (self.exposed & (self.ti_infectious <= ti)).uids
        if len(new_inf):
            iv = self.exposed_variant[new_inf]
            self.infectious_variant[new_inf] = iv
            self._flow_variant['new_infectious'] += np.bincount(np.asarray(iv).astype(int), minlength=self.nv)
            self.exposed[new_inf] = False
        # Stage onsets -- count only NEW transitions (for the flow Results); flags are cumulative-nested.
        new_symp = (self.infected & ~self.symptomatic & (self.ti_symptomatic <= ti)).uids
        self.symptomatic[new_symp] = True
        new_sev = (self.infected & ~self.severe & (self.ti_severe <= ti)).uids
        self.severe[new_sev] = True
        new_crit = (self.infected & ~self.critical & (self.ti_critical <= ti)).uids
        self.critical[new_crit] = True
        # -> recovered (clear the stage flags; tag recovered_variant; clear active-infection variant
        # tags). Under cross_immunity_active (M3 nv>1) recovery restores susceptibility so the agent
        # can be reinfected by a heterologous variant; otherwise (M2 path) immunity is permanent.
        new_rec = (self.infected & (self.ti_recovered <= ti)).uids
        self.infected[new_rec]     = False
        self.recovered[new_rec]    = True
        self.symptomatic[new_rec]  = False
        self.severe[new_rec]       = False
        self.critical[new_rec]     = False
        if len(new_rec):
            self.recovered_variant[new_rec]  = self.exposed_variant[new_rec]
            self.infectious_variant[new_rec] = np.nan
            self.exposed_variant[new_rec]    = np.nan
            if self.cross_immunity_active:
                self.susceptible[new_rec] = True
        # -> dead (request the death; resolved in step_die after transmission)
        new_dead = (self.infected & (self.ti_dead <= ti)).uids
        if len(new_dead):
            self.sim.people.request_death(new_dead)
        # Capture this step's flows for update_results
        self._flow = dict(symptomatic=len(new_symp), severe=len(new_sev), critical=len(new_crit),
                          recoveries=len(new_rec), deaths=len(new_dead))

        # Testing/tracing/quarantine state machines (M5; inert with no testing intervention attached).
        self._step_testing()

        # Update per-agent transmissibility BEFORE this step's transmission (loop slot 5 precedes infect at 9):
        # rel_trans = beta_dist draw x viral_load(t) x asymp_factor (v3 compute_trans_sus, utils.py:90-93).
        inf = self.infectious.uids
        if len(inf):
            early = ti < np.asarray(self.ti_vl_switch[inf])           # high viral-load phase (front-loaded)
            vl = np.where(early, self._vl_high, self._vl_low)
            f_asymp = np.where(np.asarray(self.symptomatic[inf]), 1.0, self.pars.asymp_factor)
            self.rel_trans[inf] = self.rel_trans_base[inf] * vl * f_asymp
            # M5: isolated/quarantined agents transmit less (scalar approx of v3's per-layer iso/quar
            # factors). Inert with no testing intervention (no agent is isolated/quarantined => no-op).
            iso = self.isolated.uids
            if len(iso):
                self.rel_trans[iso] = self.rel_trans[iso] * self.pars.iso_factor
            quar = (self.quarantined & ~self.isolated).uids
            if len(quar):
                self.rel_trans[quar] = self.rel_trans[quar] * self.pars.quar_factor
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
        # Clear the scalar variant tags (v3 people.py:309-311); clean state + M4/M6 safety.
        self.exposed_variant[uids]    = np.nan
        self.infectious_variant[uids] = np.nan
        self.recovered_variant[uids]  = np.nan
        # Clear testing/quarantine flags on death (v3 check_death; inert when no testing is active).
        self.known_contact[uids] = False
        self.quarantined[uids]   = False
        self.isolated[uids]      = False
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
            R('n_imports',       'Number of imported infections'),  # bumped by import_variant (M3)
            R('new_tests',       'New tests'),        # M5: bumped by covid.test()
            R('cum_tests',       'Cumulative tests'),
            R('new_diagnoses',   'New diagnoses'),    # M5: bumped in _step_testing
            R('cum_diagnoses',   'Cumulative diagnoses'),
            R('new_doses',       'New vaccine doses'),     # M6: bumped by vaccinate_agents
            R('cum_doses',       'Cumulative vaccine doses'),
            R('new_vaccinated',  'Newly vaccinated people'),
            R('cum_vaccinated',  'Cumulative vaccinated people'),
        )
        # M4 immunity summaries (population means -> scale=False). Filled only under use_waning.
        self.define_results(
            ss.Result('pop_nabs',       dtype=float, scale=False, label='Population mean NAb level'),
            ss.Result('pop_protection', dtype=float, scale=False, label='Population mean protective immunity'),
        )
        self._flow = dict(symptomatic=0, severe=0, critical=0, recoveries=0, deaths=0)
        self._test_flow = dict(tests=0, diagnoses=0)  # M5 per-step test/diagnosis flow
        self._vacc_flow = dict(doses=0, vaccinated=0)  # M6 per-step dose/vaccination flow

        # The 12-key 2D by_variant sub-dict (v3 sim.results['variant']), shape (nv, npts), FLOAT dtype
        # (v3 result_float -- avoids truncation). Built as a NESTED ss.Results because define_results
        # forces shape=npts (1D); the auto-scaler does not descend into it, so the scale=True members
        # are scaled MANUALLY by pop_scale in finalize (M3 design spec sec.7; done at sim level, Task 4).
        nv, npts = self.nv, self.t.npts
        def RV(name, scale, label):
            return ss.Result(name, dtype=float, scale=scale, shape=(nv, npts), label=label,
                             module=self.label, timevec=self.t.timevec)
        variant_res = ss.Results(self.label)
        variant_res += RV('prevalence_by_variant',     False, 'Prevalence by variant')
        variant_res += RV('incidence_by_variant',      False, 'Incidence by variant')
        variant_res += RV('new_infections_by_variant',  True, 'New infections by variant')
        variant_res += RV('cum_infections_by_variant',  True, 'Cumulative infections by variant')
        variant_res += RV('new_symptomatic_by_variant', True, 'New symptomatic by variant')
        variant_res += RV('cum_symptomatic_by_variant', True, 'Cumulative symptomatic by variant')
        variant_res += RV('new_severe_by_variant',      True, 'New severe by variant')
        variant_res += RV('cum_severe_by_variant',      True, 'Cumulative severe by variant')
        variant_res += RV('new_infectious_by_variant',  True, 'New infectious by variant')
        variant_res += RV('cum_infectious_by_variant',  True, 'Cumulative infectious by variant')
        variant_res += RV('n_exposed_by_variant',       True, 'Number exposed by variant')
        variant_res += RV('n_infectious_by_variant',    True, 'Number infectious by variant')
        self.results['variant'] = variant_res
        self._ensure_flow_variant()
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
        res.new_tests[ti]       = self._test_flow['tests']      # M5 testing flows
        res.new_diagnoses[ti]   = self._test_flow['diagnoses']
        res.new_doses[ti]       = self._vacc_flow['doses']      # M6 vaccination flows
        res.new_vaccinated[ti]  = self._vacc_flow['vaccinated']

        # By-variant flows (accumulated this step) + stocks (counted from the live tags).
        vres = res['variant']
        fv = self._flow_variant
        vres['new_infections_by_variant'][:, ti]  = fv['new_infections']
        vres['new_symptomatic_by_variant'][:, ti] = fv['new_symptomatic']
        vres['new_severe_by_variant'][:, ti]      = fv['new_severe']
        vres['new_infectious_by_variant'][:, ti]  = fv['new_infectious']
        exp_uids = self.exposed.uids
        if len(exp_uids):
            ev = np.asarray(self.exposed_variant[exp_uids]); fin = np.isfinite(ev)
            if fin.any():
                vres['n_exposed_by_variant'][:, ti] = np.bincount(ev[fin].astype(int), minlength=self.nv)
        inf_uids = self.infectious.uids
        if len(inf_uids):
            iv = np.asarray(self.infectious_variant[inf_uids]); fin = np.isfinite(iv)
            if fin.any():
                vres['n_infectious_by_variant'][:, ti] = np.bincount(iv[fin].astype(int), minlength=self.nv)

        # M4 population immunity summaries (mean over alive agents): NAb level + wild-axis protection.
        if self.pars.use_waning:
            nab = np.asarray(self.nab)  # active (alive) values
            res.pop_nabs[ti] = float(nab.mean()) if len(nab) else 0.0
            auids = np.asarray(self.sim.people.auids)
            res.pop_protection[ti] = float(self.sus_imm[0][auids].mean()) if len(auids) else 0.0
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
        res.cum_tests[:]       = np.cumsum(res.new_tests[:])       # M5
        res.cum_diagnoses[:]   = np.cumsum(res.new_diagnoses[:])
        res.cum_doses[:]       = np.cumsum(res.new_doses[:])       # M6
        res.cum_vaccinated[:]  = np.cumsum(res.new_vaccinated[:])

        # By-variant cumulatives (cumsum along the time axis). The pop_scale scaling, the wild
        # cum_infections seed-offset, and the prevalence/incidence denominators are applied at the
        # sim level in cv.Sim.finalize (M3 design spec sec.7); here we only accumulate the flows.
        vres = res['variant']
        for stem in ('infections', 'symptomatic', 'severe', 'infectious'):
            vres[f'cum_{stem}_by_variant'][:] = np.cumsum(vres[f'new_{stem}_by_variant'][:], axis=1)
        return

    # --- seeding --------------------------------------------------------------

    def init_post(self):
        """Fill the age-conditional prognoses, then seed initial infections.

        Covasim seeds an *exact* count (``pop_infected``), not a per-agent probability.
        If ``init_prev`` is an integer, seed exactly that many agents (deterministically
        from the sim seed); otherwise defer to ``ss.Infection`` (``ss.bernoulli`` / None).
        """
        self._fill_prognoses()  # must precede any set_prognoses call (seeding below)

        # Allocate the 2D per-variant immunity arrays (the v3 imm_states), shape (nv, n_raw), indexed
        # by RAW uid (aligned with rel_sus.raw / FloatArr indexing). These are plain ndarrays, NOT
        # ss.Arr, so they do NOT auto-grow/reorder on UID churn -- M3 therefore requires a constant
        # population (no births). M4/M6 must make these growth-aware (M3 design spec sec.1, adversary #10).
        births = [m for m in self.sim.demographics.values()
                  if any(s in type(m).__name__.lower() for s in ('birth', 'pregnan'))]
        if births:
            raise NotImplementedError('cv.COVID variant immunity arrays are not growth-aware: births/'
                                      'pregnancy are unsupported in M3 (deferred to M4/M6).')
        n_raw = len(self.rel_sus.raw)
        self.sus_imm  = np.zeros((self.nv, n_raw))
        self.symp_imm = np.zeros((self.nv, n_raw))
        self.sev_imm  = np.zeros((self.nv, n_raw))
        self._ensure_flow_variant()

        # Precompute the NAb waning kernel once (M4); indexed by (ti - t_nab_event) in the connector.
        if self.pars.use_waning:
            import covasim.immunity as cvimm  # lazy: covid.py imports before immunity.py
            self.nab_kin = cvimm.precompute_waning(self.t.npts, self.pars.nab_decay)
        self._pending_quarantine = {}  # M5: start_day -> [(uid, end_day)] (reset for clean re-runs)

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
