"""
The COVID-19 disease module for Covasim on the Starsim base.

``cv.COVID(ss.Infection)`` is the COVID disease, including Covasim's full
natural-history prognosis tree, multiple variants, waning immunity, testing,
quarantine, and vaccination:

    susceptible -> exposed -> infectious -> (asymptomatic | mild | severe | critical)
                                                          -> recovered | dead

The whole trajectory (every branch outcome AND every transition date) is drawn
*once at infection* in ``set_prognoses`` (as in Covasim v3); ``step_state`` only flips boolean flags when
``ti_<stage> <= ti`` and never re-draws a probability. Branch probabilities are the
age-conditional prognoses from ``parameters.get_prognoses`` (already conditional --
do not re-divide). Without waning immunity (``use_waning=False``), recovery confers
permanent immunity; with it, neutralizing antibodies (NAbs) wane and agents can be
reinfected. Deaths are requested via ``sim.people.request_death`` and finalized in
``step_die``.

Per-agent transmissibility is time-varying: each agent draws a constant
overdispersion factor (``beta_dist``, mean 1.0) at init and a two-level viral-load
kernel (``viral_dist``) over its infectious period, written to ``rel_trans`` each
step before transmission. Transmission (``infect()``) follows ``ss.Infection.infect()``,
with a loop over variants and the per-layer beta, isolation, and quarantine factors.
"""
import numpy as np
import scipy.stats as sps
import sciris as sc
import starsim as ss

from . import parameters as cvpar
from . import immunity as cvimm

__all__ = ['COVID', 'v3_dist', 'v3_durs']


def v3_dist(d, unit=None, round=False):
    """
    Convert a v3 distribution, e.g. ``dict(dist='lognormal_int', par1=4.5, par2=1.5)``, to a Starsim
    distribution, e.g. ``ss.lognorm_ex(mean=4.5, std=1.5, round='nearest')``. The choices are those of
    v3's ``cv.sample()``: uniform, normal, normal_pos, normal_int, lognormal, lognormal_int, poisson, and
    neg_binomial (with an optional ``step``, as for ``beta_dist``).

    Args:
        d (dict): the v3 distribution, with keys ``dist``, ``par1``, ``par2``, and optionally ``step``
        unit (class): the time unit of the parameters for a duration, e.g. ``ss.days``; not used for poisson or neg_binomial
        round (bool/str): how to round the values, e.g. ``'nearest'``; the "_int" distributions are always rounded to the nearest integer

    Returns:
        A Starsim distribution

    Note: as in v3, draws from a neg_binomial with a ``step`` must be multiplied by the step.
    """
    name = d['dist']
    par1 = d.get('par1')
    par2 = d.get('par2')
    if name.endswith('_int'): # As in v3, round to the nearest integer
        round = 'nearest'
        name = name[:-len('_int')]
        if name == 'normal': # As in v3, normal_int is always positive
            name = 'normal_pos'
    if (unit is not None) and (par2 is not None):
        par1 = unit(par1)
        par2 = unit(par2)

    if name in ['unif', 'uniform']:
        dist = ss.uniform(low=par1, high=par2, round=round)
    elif name in ['norm', 'normal']:
        dist = ss.normal(loc=par1, scale=par2, round=round)
    elif name == 'normal_pos': # The absolute value of a normal distribution, i.e. a folded normal
        c = abs(d['par1']) / d['par2']
        dist = ss.Dist(dist=sps.foldnorm, distname='normal_pos', c=c, scale=par2, round=round)
    elif name in ['lognorm', 'lognormal']:
        dist = ss.lognorm_ex(mean=par1, std=par2, round=round)
    elif name == 'poisson':
        dist = ss.poisson(lam=d['par1'], round=round)
    elif name == 'neg_binomial': # Mean par1 and dispersion par2, as v3's cv.n_neg_binomial()
        step = d.get('step', 1)
        dist = ss.nbinom(n=d['par2'], p=d['par2']/(d['par1']/step + d['par2']), round=round)
    else:
        errormsg = f'Distribution "{d["dist"]}" is not supported; choices are uniform, normal, normal_pos, normal_int, lognormal, lognormal_int, poisson, and neg_binomial'
        raise ValueError(errormsg)
    return dist


def v3_durs(dur):
    """
    Convert v3 duration parameters to Starsim distributions, e.g. ``dur['exp2inf'] =
    dict(dist='lognormal_int', par1=4.5, par2=1.5)`` becomes ``dur_exp2inf = ss.lognorm_ex(...)``.
    Durations are always rounded to the nearest whole day, as with v3's ``lognormal_int``.

    Args:
        dur (dict): the v3 durations, keyed by e.g. 'exp2inf'

    Returns:
        A dict of Starsim distributions, keyed by e.g. 'dur_exp2inf'
    """
    out = {}
    for key, d in dur.items():
        out[f'dur_{key}'] = v3_dist(d, unit=ss.days, round='nearest')
    return out


class COVID(ss.Infection):
    """COVID-19 disease with the full natural-history prognosis tree, multiple variants, waning immunity, testing, and quarantine.

    Args:
        pars (dict): parameters to update (any of those below, plus the v3 parameters defined in ``__init__()``)
        variants (list): the variants in addition to wild, as ``cv.variant`` objects or names/dicts (introduced on day 0)
        beta: per-contact transmission probability per day (scalar or dict keyed by
            network layer); Covasim's default is ``pars['beta'] = 0.016``.
        init_prev: the initial infections, e.g. ``ss.choose_n(20)`` (``cv.Sim`` uses ``pop_infected``)
            or ``ss.bernoulli(0.01)``, or ``None``.
        dur_*: the Covasim duration distributions (lognormal mean/std in days,
            ``parameters.py`` dur block).
        rel_symp_prob/rel_severe_prob/rel_crit_prob/rel_death_prob: global severity
            scalers (default 1.0).
        n_beds_hosp/n_beds_icu/no_hosp_factor/no_icu_factor: health-system capacity
            (default no constraint; the factors are inert unless beds are set).
    """

    def __init__(self, pars=None, variants=None, **kwargs):
        super().__init__()
        v3 = cvpar.make_pars() # The default values of the parameters are defined in parameters.py
        self.define_pars(
            # Transmission
            beta         = ss.probperday(v3['beta']), # Per-contact transmission probability per day
            beta_layer   = v3['beta_layer']['a'], # Per-layer multiplier on beta: a dict keyed by network, or a scalar for all networks
            rel_beta     = v3['rel_beta'], # Overall multiplier on beta
            beta_dist    = v3['beta_dist'], # Constant per-agent overdispersion in transmissibility (mean 1.0)
            viral_dist   = v3['viral_dist'], # Time-varying viral load (two-level, mean-preserving)
            asymp_factor = v3['asymp_factor'], # Transmissibility multiplier for asymptomatic agents
            n_imports    = v3['n_imports'], # Expected number of imported (wild) infections per day
            init_prev    = None, # The initial infections, e.g. ss.choose_n(20) (as set by cv.Sim from pop_infected) or ss.bernoulli(0.01)
            frac_susceptible = v3['frac_susceptible'], # The fraction of agents who are initially susceptible; the rest start non-naive, without being infected

            # Durations (v3 dur['exp2inf'] etc.)
            **v3_durs(v3['dur']),

            # Severity
            prognoses       = None, # The prognoses by age; if None, use get_prognoses()
            prog_by_age     = v3['prog_by_age'], # Whether prognoses depend on age
            rel_symp_prob   = v3['rel_symp_prob'],
            rel_severe_prob = v3['rel_severe_prob'],
            rel_crit_prob   = v3['rel_crit_prob'],
            rel_death_prob  = v3['rel_death_prob'],

            # Health-system capacity; factors inert while beds are None
            n_beds_hosp    = v3['n_beds_hosp'],
            n_beds_icu     = v3['n_beds_icu'],
            no_hosp_factor = v3['no_hosp_factor'],
            no_icu_factor  = v3['no_icu_factor'],

            # Waning immunity and neutralizing antibodies (NAbs). This is off unless a cv.CrossImmunity
            # connector is also added, which cv.Sim does by default (as in v3, where use_waning=True).
            use_waning   = False,
            nab_init     = v3['nab_init'], # log2(initial NAb) after natural infection
            nab_decay    = v3['nab_decay'],
            nab_boost    = v3['nab_boost'], # Multiplicative boost to peak NAb on reinfection
            nab_eff      = v3['nab_eff'], # NAbs to efficacy
            rel_imm_symp = v3['rel_imm_symp'], # Natural-immunity scaling by symptom severity
            trans_redux  = v3['trans_redux'], # Transmissibility reduction for breakthrough infections

            # Testing and quarantine. The factors are dicts keyed by network, or scalars for all networks;
            # these defaults are the random-network values.
            quar_period  = v3['quar_period'], # Days an agent quarantines
            iso_factor   = v3['iso_factor']['a'], # Transmissibility multiplier for isolated (diagnosed) agents
            quar_factor  = v3['quar_factor']['a'], # Transmissibility and susceptibility multiplier for quarantined agents
        )
        self.update_pars(pars, **kwargs)

        # The disease states (replacing those of ss.Infection)
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('infected',                   label='Infected (exposed or infectious)'),
            ss.BoolState('preinfectious',              label='Preinfectious (infected, not yet infectious)'),
            ss.BoolState('symptomatic',                label='Symptomatic'),
            ss.BoolState('severe',                     label='Severe (needs hospitalization)'),
            ss.BoolState('critical',                   label='Critical (needs ICU)'),
            ss.BoolState('recovered',                  label='Recovered'),
            ss.BoolState('dead',                       label='Dead'),
            ss.BoolState('naive',         default=True,label='Naive (never infected)'),
            ss.FloatArr('rel_sus',        default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans',      default=1.0, label='Relative transmissibility (time-varying)'),
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
            # The variant of each agent's current or last infection (NaN if none). An agent has only
            # one infection at a time. With one variant, every infected agent is wild (variant 0).
            ss.FloatArr('exposed_variant',    label='Variant index this agent is exposed/infected with'),
            ss.FloatArr('infectious_variant', label='Variant index this agent is infectious with'),
            ss.FloatArr('recovered_variant',  label='Variant index this agent last recovered from'),
            # Neutralizing antibodies (NAbs), shared across all prior infections and vaccinations;
            # cv.CrossImmunity weights them by variant. Only used when use_waning=True.
            ss.FloatArr('peak_nab',        default=0.0, label='Peak neutralizing-antibody titre'),
            ss.FloatArr('nab',             default=0.0, label='Current neutralizing-antibody titre'),
            ss.FloatArr('t_nab_event',                  label='Time index of the last NAb-conferring event'),
            ss.FloatArr('n_breakthroughs', default=0.0, label='Number of breakthrough (post-NAb) infections'),
            # Testing, tracing, and quarantine states. These are inert until a testing or contact-tracing
            # intervention sets the trigger dates.
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
            # Vaccination states (v3 vacc_states). Vaccine immunity flows through the same NAb
            # pipeline as natural infection. Inert until a vaccination intervention is added.
            ss.BoolState('vaccinated',     label='Vaccinated'),
            ss.FloatArr('doses',           default=0.0, label='Number of vaccine doses received'),
            ss.FloatArr('vaccine_source',               label='Index of the vaccine received'),
            ss.FloatArr('date_vaccinated',              label='Date last vaccinated'),
            reset = True,
        )

        # Bernoulli distributions for the prognosis branches; their per-agent p is set each call
        self._symp_bern  = ss.bernoulli(p=0.5)
        self._sev_bern   = ss.bernoulli(p=0.5)
        self._crit_bern  = ss.bernoulli(p=0.5)
        self._death_bern = ss.bernoulli(p=0.5)
        self._beta_dist  = None # Per-agent transmissibility (overdispersion); created from beta_dist in init_pre()
        self._choose_nonsusceptible = ss.choose_n() # Agents who are initially non-susceptible (if frac_susceptible < 1)
        self._n_imports = ss.poisson(lam=0) # The number of imported infections on each step (if n_imports > 0)
        self._choose_imports = ss.choose_n() # Who those infections are

        # Variants -------------------------------------------------------------------
        # A single COVID module carries an internal variant dimension. cv.variant.initialize() and
        # cv.Sim(variants=...) add variants; the per-variant parameters come from
        # parameters.get_variant_pars. Wild is always index 0.
        self.nv = 1
        self.variant_map  = {0: 'wild'}                       # index -> label (wild always 0)
        self.variant_pars = {'wild': dict(cvpar.get_variant_pars(variant='wild'))}  # 5 keys, all 1.0
        # Register any additional variants (cv.variant objects, or names/dicts) here, since the number
        # of variants sets the number of columns of the immunity states below. Each grows nv,
        # variant_map, and variant_pars, and gets a stable .index; wild stays index 0.
        self._variant_objs = []
        if variants is not None:
            if isinstance(variants, (str, dict, cvimm.variant)):
                variants = [variants]
            for v in variants:
                if not isinstance(v, cvimm.variant):
                    v = cvimm.variant(v, days=0)  # bare name/dict => introduce at t0
                v.initialize(self)
                self._variant_objs.append(v)
        # Per-variant immunity (the v3 imm_states), one column per variant, e.g. sus_imm[uids, vi]; set by
        # cv.CrossImmunity, and all zero with one variant and no waning
        self.define_states(
            ss.FloatArr('sus_imm',  default=0.0, columns=self.nv, label='Immunity to infection, by variant'),
            ss.FloatArr('symp_imm', default=0.0, columns=self.nv, label='Immunity to symptomatic disease, by variant'),
            ss.FloatArr('sev_imm',  default=0.0, columns=self.nv, label='Immunity to severe disease, by variant'),
        )
        # The new cases from transmission on this step and the variant of each (set in infect(), used and cleared in set_prognoses())
        self._new_cases = None
        self._new_case_variants = None
        # Cross-immunity / reinfection switch. If False, recovered agents stay susceptible=False
        # (permanent immunity). cv.CrossImmunity turns this on when added, enabling reinfection and
        # cross-immunity.
        self.cross_immunity_active = False

        # NAbs. The distribution of log2 NAbs after natural infection (from nab_init) and the waning
        # kernel (nab_kin) are created in init_pre(), and only used when use_waning=True.
        self._nab_init = None
        self.nab_kin = None

        # Testing. The test bernoulli (sensitivity / loss-to-follow-up) is only drawn when covid.test() is
        # called by a testing intervention. The pending-quarantine schedule (start_day -> [(uid, end_day)])
        # is (re)set in init_post.
        self._test_bern = ss.bernoulli(p=1.0)
        self._pending_quarantine = {}

        # Vaccination. Vaccine registry (label -> pars, index -> label), populated by vaccination
        # interventions at init. The vaccine NAb-init draw uses its own Dist (.set per vaccine), distinct
        # from the natural _nab_init, so vaccination doesn't change the random numbers for natural infection.
        self.vaccine_pars = {}
        self.vaccine_map  = {}
        self._vacc_nab_init = ss.normal(loc=0.0, scale=2.0)

        self.n_initial_cases = 0 # Set in init_post()
        return

    def init_pre(self, sim):
        """
        Create the distributions and the NAb waning kernel from the current parameters, convert the
        variant introduction days to day indices, and create the transmission log (v3's people.infection_log;
        it's cheap, and used by cv.TransTree, cv.daily_stats, and sim.compute_r_eff()).
        """
        super().init_pre(sim)
        self.infection_log = ss.diseases.InfectionLog(disease=self.name, networks=sim.networks.keys())
        self._beta_dist = v3_dist(self.pars.beta_dist)
        self._nab_init = v3_dist(self.pars.nab_init)
        if self.pars.use_waning: # Created here so historical immunity interventions can extend it in their init_post()
            self.nab_kin = cvimm.precompute_waning(self.t.npts, self.pars.nab_decay)
        for var in self._variant_objs:
            var.process_days(sim)
        return

    def get_layer_par(self, key, layer):
        """Get the value of a per-layer parameter (beta_layer, iso_factor, quar_factor) for this layer."""
        par = self.pars[key]
        return par[layer] if isinstance(par, dict) else par

    def variant_par(self, key, uids):
        """Get a per-variant parameter (e.g. rel_symp_prob) for each of these agents, for the variant they are infected with"""
        values = np.array([self.variant_pars[self.variant_map[i]][key] for i in range(self.nv)])
        return values[self.exposed_variant[uids].astype(int)]

    @property
    def today(self):
        """ The day this step starts on (day 0 is the start day), which days given by the user (e.g. a variant's ``days=10``) refer to; the step's own index, ``self.ti``, is one more, since time point 0 is the start of the run """
        return self.ti - 1

    @property
    def exposed(self):
        """Agents who are infected, including those who are infectious (as in v3; "preinfectious" are those who are not yet infectious)"""
        return self.infected

    @property
    def infectious(self):
        """Agents who transmit: infected and past the latent period (I, not merely E)."""
        return self.infected & (self.ti_infectious <= self.ti)

    # Prognoses ----------------------------------------------------------------

    def init_prognoses(self):
        """Fill the per-agent age-conditional branch probabilities + susceptibility.

        As in v3 ``people.set_prognoses()``, age-bin the prognoses and the susceptibility odds ratios.
        ``get_prognoses`` already returns conditional probabilities (it calls ``relative_prognoses``
        internally), so they are used directly. ``sus_ORs`` are age-dependent, so ``rel_sus``
        is age-structured; ``trans_ORs`` are all 1.0, so per-agent transmissibility comes
        from the beta_dist and viral-load factors.
        """
        progs = self.pars.prognoses if self.pars.prognoses is not None else cvpar.get_prognoses(by_age=self.pars.prog_by_age)
        age = self.sim.people.states['age'] # The Starsim state, i.e. only agents who are alive, to match the other states
        inds = np.digitize(age, progs['age_cutoffs']) - 1
        self.symp_prob[:]   = progs['symp_probs'][inds]
        self.severe_prob[:] = progs['severe_probs'][inds] * progs['comorbidities'][inds]  # comorbidity folds into severe
        self.crit_prob[:]   = progs['crit_probs'][inds]
        self.death_prob[:]  = progs['death_probs'][inds]
        self.rel_sus[:]     = progs['sus_ORs'][inds]  # age-dependent susceptibility

        # Per-agent constant transmissibility (trans_ORs x beta_dist), drawn once at init, as in v3.
        # trans_ORs are all 1.0, so this is the beta_dist overdispersion draw.
        step = self.pars.beta_dist.get('step', 1) # As in v3, the neg_binomial draws are multiplied by the step
        draw = self._beta_dist.rvs(self.sim.people.auids) * step
        self.rel_trans_base[:] = progs['trans_ORs'][inds] * draw

        # Cache the two-level viral-load constants (mean-preserving normalizer Z, as in v3)
        vd = self.pars.viral_dist
        z = 1.0 + vd['frac_time'] * (vd['load_ratio'] - 1)
        self._vl_high = vd['load_ratio'] / z
        self._vl_low  = 1.0 / z
        return

    def update_peak_nab(self, uids, nab_pars=None, symp_scale=None):
        """Set/boost peak NAb at a NAb-conferring event (the v3 ``update_peak_nab``).

        Agents with prior NAbs (``peak_nab>0``) are boosted by ``nab_boost``; the rest draw an initial
        log2 NAb from ``nab_init``.

        ``nab_pars=None`` (natural infection): use the disease ``nab_init``/``nab_boost``, scale the
        initial NAb by symptom severity (``symp_scale`` = per-UID ``rel_imm_symp``) and normalise onto
        the vaccine scale by ``1 + alpha_inf_diff`` (drawn from ``_nab_init``).

        ``nab_pars`` given (a vaccine dose): use the vaccine's ``nab_init``/``nab_boost`` with NO
        symptom scaling and NO natural normalisation, drawn from the separate ``_vacc_nab_init`` Dist.
        """
        p = self.pars
        prior = self.peak_nab[uids] > 0
        prior_uids = uids[prior]
        no_prior_uids = uids[~prior]
        boost = p.nab_boost if nab_pars is None else nab_pars['nab_boost']
        if len(prior_uids):  # boost agents who already had NAbs (reinfection / prior dose)
            self.peak_nab[prior_uids] = self.peak_nab[prior_uids] * boost
        if len(no_prior_uids):  # draw an initial NAb for agents with no prior immunity
            if nab_pars is None:  # natural infection
                init = self._nab_init.rvs(no_prior_uids)
                norm = 1.0 + p.nab_eff['alpha_inf_diff']         # natural -> vaccine NAb-scale normalisation
                self.peak_nab[no_prior_uids] = (2.0 ** init) * symp_scale[~prior] * norm
            else:  # a vaccine dose: draw with the vaccine's nab_init (no symptom scaling/normalisation)
                ni = nab_pars['nab_init']
                if ni.get('dist', 'normal') != 'normal':
                    errormsg = f'Vaccine nab_init distributions must be normal, not "{ni["dist"]}"'
                    raise ValueError(errormsg)
                self._vacc_nab_init.set(loc=ni['par1'], scale=ni['par2'])
                init = self._vacc_nab_init.rvs(no_prior_uids)
                self.peak_nab[no_prior_uids] = 2.0 ** init
        self.t_nab_event[uids] = self.ti
        return

    def imprint_historical_nab(self, uids, event_day):
        """Set the current NAb of ``uids`` as if their NAb event happened on (negative) ``event_day``.

        Used for pre-t=0 immunity (``historical_vaccinate_prob`` / ``historical_wave``): replays the
        daily NAb updates of ``update_nab()`` (``nab += nab_kin[i] x peak``, clamped to ``[0, peak]``) from
        the back-dated event up to the previous step, so the agents start the sim with appropriately-decayed
        NAbs and continue along the same kinetic curve. ``peak_nab`` must already be set (by
        ``update_peak_nab``); this fills ``nab`` + ``t_nab_event``.
        """
        uids = ss.uids(uids)
        if not len(uids):
            return
        offset = int(self.today) - int(event_day)
        if offset < 0:
            offset = 0
        # Extend the kernel so it covers the whole sim for these agents, as in v3
        n_kin = offset + self.t.npts
        if (self.nab_kin is None) or (len(self.nab_kin) < n_kin):
            self.nab_kin = cvimm.precompute_waning(n_kin, self.pars.nab_decay)
        peak = self.peak_nab[uids]
        nab = np.zeros(len(uids))
        kin = self.nab_kin
        for i in range(offset):  # Replay the daily updates from the day of the event to the day before the current step
            nab = np.clip(nab + kin[i] * peak, 0.0, peak)
        self.nab[uids] = nab
        self.t_nab_event[uids] = int(event_day) + 1 # In steps, like the other dates: the step that starts on that day
        return

    def set_prognoses(self, uids, sources=None, variant=None):
        """Draw the full disease trajectory once at infection (the v3 pre-scheduled tree).

        Each branch probability is scaled by the per-variant ``rel_*_prob`` and reduced by
        this variant's cross-immunity (``symp_imm``/``sev_imm``), as in v3 ``people.infect()``.
        """
        super().set_prognoses(uids, sources)  # logs the infection
        ti = self.ti # Dated by the step whose start they happen at (the initial infections by the first step, as ss.Infection dates them)
        p = self.pars

        # The variant of each new case
        if variant is not None: # All these cases are this variant (seeds and imports)
            var_of = np.full(len(uids), int(variant), dtype=int)
        elif self._new_cases is not None: # The variant recorded for each new case by infect(); the new cases are sorted and unique
            var_of = self._new_case_variants[np.searchsorted(self._new_cases, uids)]
        else: # Everyone has the wild variant
            var_of = np.zeros(len(uids), dtype=int)

        # Add the day and variant to the transmission log; ss.Infection.step() adds the network for transmissions
        log_data = dict(day=ti, variant=var_of)
        if sources is None or np.isscalar(sources): # Seed infections and importations have no network
            log_data['network'] = 'seed_infection' if variant is None else 'importation'
        self.infection_log.add_data(uids, **log_data)
        # Per-UID cross-immunity reductions for this variant (all zero with one variant and no waning)
        sympimm_u = self.symp_imm[uids, var_of]
        sevimm_u  = self.sev_imm[uids, var_of]

        # Entry: exposed + infectious latency; tag the variant; as in v3, clear `recovered` and the
        # diagnosis on reinfection, so reinfected agents can be diagnosed again
        self.flows['reinfections'] += np.count_nonzero(self.ti_recovered.notnan[uids]) # People who were infected before
        self.susceptible[uids] = False
        self.naive[uids]       = False
        self.infected[uids]    = True
        self.preinfectious[uids] = True
        self.recovered[uids]   = False
        self.diagnosed[uids]   = False
        self.date_diagnosed[uids] = np.nan
        self.exposed_variant[uids] = var_of.astype(float)
        self.ti_infected[uids] = self.ti # Starsim's time of infection: 0 for the initial infections
        self.ti_exposed[uids]  = ti
        self.ti_infectious[uids] = ti + p.dur_exp2inf.rvs(uids)
        # Edge case: if dur_exp2inf rounds to 0 the agent is infectious-by-property THIS step, after
        # step_state already ran -- tag infectious_variant now so the by_variant stock stays exact
        # (step_state still clears `preinfectious` and counts the new_infectious flow next step).
        imm = uids[self.ti_infectious[uids] <= ti]
        if len(imm):
            self.infectious_variant[imm] = self.exposed_variant[imm]
        # Reset all downstream dates (defensive; matches v3 "reset all other dates")
        for arr in (self.ti_symptomatic, self.ti_severe, self.ti_critical, self.ti_recovered, self.ti_dead):
            arr[uids] = np.nan

        # Branch 1: symptomatic? (asymptomatic agents recover and cannot die)
        self._symp_bern.set(p=p.rel_symp_prob * self.variant_par('rel_symp_prob', uids) * self.symp_prob[uids] * (1 - sympimm_u))
        is_symp = self._symp_bern.rvs(uids)
        symp  = uids[is_symp]
        asymp = uids[~is_symp]
        self.ti_recovered[asymp] = self.ti_infectious[asymp] + p.dur_asym2rec.rvs(asymp)

        # Branch 2: severe? (among symptomatic)
        self.ti_symptomatic[symp] = self.ti_infectious[symp] + p.dur_inf2sym.rvs(symp)
        self._sev_bern.set(p=p.rel_severe_prob * self.variant_par('rel_severe_prob', symp) * self.severe_prob[symp] * (1 - sevimm_u[is_symp]))
        is_sev = self._sev_bern.rvs(symp)
        sev  = symp[is_sev]
        mild = symp[~is_sev]
        self.ti_recovered[mild] = self.ti_symptomatic[mild] + p.dur_mild2rec.rvs(mild)

        # Branch 3: critical? (among severe; no_hosp_factor raises the risk if beds are full)
        self.ti_severe[sev] = self.ti_symptomatic[sev] + p.dur_sym2sev.rvs(sev)
        hosp_factor = 1.0
        if (p.n_beds_hosp is not None) and (np.count_nonzero(self.severe) > p.n_beds_hosp): # Hospital beds are full (never the case by default, since n_beds_hosp=None)
            hosp_factor = p.no_hosp_factor
        self._crit_bern.set(p=p.rel_crit_prob * self.variant_par('rel_crit_prob', sev) * self.crit_prob[sev] * hosp_factor)
        is_crit = self._crit_bern.rvs(sev)
        crit    = sev[is_crit]
        noncrit = sev[~is_crit]
        self.ti_recovered[noncrit] = self.ti_severe[noncrit] + p.dur_sev2rec.rvs(noncrit)

        # Branch 4: die? (among critical; no_icu_factor raises the risk if ICU is full)
        self.ti_critical[crit] = self.ti_severe[crit] + p.dur_sev2crit.rvs(crit)
        icu_factor = 1.0
        if (p.n_beds_icu is not None) and (np.count_nonzero(self.critical) > p.n_beds_icu): # ICU beds are full (never the case by default, since n_beds_icu=None)
            icu_factor = p.no_icu_factor
        self._death_bern.set(p=p.rel_death_prob * self.variant_par('rel_death_prob', crit) * self.death_prob[crit] * icu_factor)
        is_dead = self._death_bern.rvs(crit)
        dead    = crit[is_dead]
        survive = crit[~is_dead]
        self.ti_recovered[survive] = self.ti_critical[survive] + p.dur_crit2rec.rvs(survive)
        self.ti_dead[dead] = self.ti_critical[dead] + p.dur_crit2die.rvs(dead)
        # (ti_recovered for `dead` stays NaN from the defensive reset -- death and recovery are exclusive)

        # Count the new infections by variant; the other flows by variant are counted when they happen, in step_state()
        self.flows_by_variant['new_infections']  += np.bincount(var_of,           minlength=self.nv)

        # Precompute the per-agent viral-load high->low switch time (as in v3's compute_viral_load()).
        # End of the infectious period = recovery or death date (whichever is set).
        ti_dead = self.ti_dead[uids]
        end = np.where(np.isnan(ti_dead), self.ti_recovered[uids], ti_dead)
        infect_days = end - self.ti_infectious[uids]
        vd = p.viral_dist
        high_days = np.minimum(vd['frac_time'] * infect_days, vd['high_cap'])  # cap the high phase at high_cap days (v3 had an extra day in some cases, due to rounding; see the migration rules)
        self.ti_vl_switch[uids] = self.ti_infectious[uids] + high_days

        # NAb acquisition + breakthrough transmissibility (only with waning immunity)
        if p.use_waning:
            # Per-UID symptom-severity scaling for the initial NAb (asymp/mild/severe -> rel_imm_symp).
            ris = p.rel_imm_symp
            symp_scale = np.full(len(uids), ris['asymp'], dtype=float)
            symp_scale[is_symp] = ris['mild']
            sev_positions = np.nonzero(is_symp)[0][is_sev]  # uids-positions that reached the severe branch
            symp_scale[sev_positions] = ris['severe']
            # Breakthrough: the FIRST reinfection of an agent with prior NAbs gets reduced transmissibility
            # (v3 first-breakthrough rule); read peak_nab BEFORE update_peak_nab sets/boosts it.
            prior_bt = uids[self.peak_nab[uids] > 0]
            if len(prior_bt):
                first = prior_bt[self.n_breakthroughs[prior_bt] == 0]
                if len(first):
                    self.rel_trans_base[first] = self.rel_trans_base[first] * p.trans_redux
                self.n_breakthroughs[prior_bt] = self.n_breakthroughs[prior_bt] + 1
            self.update_peak_nab(uids, symp_scale=symp_scale)
        return

    def transmit(self):
        """
        Transmission by variant, calling ``ss.Infection.transmit_route()`` for each variant and network (as in
        v3 ``sim.step()``). For each variant, only agents infectious with that variant transmit, scaled by the
        variant's ``rel_beta``, and susceptibility is reduced by immunity to that variant (``sus_imm``). Each
        network has its own beta multiplier (``beta_layer``), and isolation and quarantine factors.

        If an agent is infected by more than one variant on the same step, the lowest variant index wins
        (since v3 infected with each variant in turn). The new cases and the variant of each are stored in
        ``self._new_cases`` and ``self._new_case_variants`` for ``set_prognoses()``.
        """
        betamap = self.validate_beta()
        dt = self.t.dt
        auids = self.sim.people.auids
        iso  = self.isolated.values # Isolated agents transmit less, and quarantined agents both transmit and acquire less (v3 compute_trans_sus)
        quar = self.quarantined.values
        any_iso, any_quar = iso.any(), quar.any()
        new_cases, sources, networks, case_variant = [], [], [], []

        for vi in range(self.nv):
            label = self.variant_map[vi]
            rel_beta = self.variant_pars[label]['rel_beta'] * self.pars.rel_beta # The variant's and the overall relative beta
            inf_mask = self.infectious if self.nv == 1 else (self.infectious & (self.infectious_variant == vi)) # Agents infectious with this variant
            trans_vals = inf_mask * self.rel_trans * rel_beta # Only active values are used, so there's no arithmetic on uninitialized inactive slots
            sus_vals = self.susceptible * self.rel_sus
            if self.cross_immunity_active: # Reduce susceptibility by immunity to this variant (zero if no connector)
                sus_vals = sus_vals * (1.0 - self.sus_imm[auids, vi])

            for i, (nkey, route) in enumerate(self.sim.networks.items()):
                beta_layer = self.get_layer_par('beta_layer', nkey)
                betas = [b.to_prob(dt) if isinstance(b, ss.Rate) else b for b in betamap[ss.utils.standardize_netkey(nkey)]]
                betas = [b*beta_layer for b in betas] # Linear in beta_layer, as in v3 (multiplying an ss.Rate isn't)
                layer_trans, layer_sus = trans_vals, sus_vals
                if any_iso:
                    layer_trans = layer_trans * np.where(iso, self.get_layer_par('iso_factor', nkey), 1.0)
                if any_quar:
                    f_quar = np.where(quar, self.get_layer_par('quar_factor', nkey), 1.0)
                    layer_trans = layer_trans * f_quar
                    layer_sus = layer_sus * f_quar
                rel_trans = self.rel_trans.asnew(layer_trans)
                rel_sus = self.rel_sus.asnew(layer_sus)
                targets, srcs, nets = self.transmit_route(i, route, betas, rel_trans, rel_sus)
                new_cases.append(targets)
                sources.append(srcs)
                networks.append(nets)
                case_variant.append(np.full(len(targets), vi))

        # Remove duplicates, keeping the first (i.e. lowest variant index), and record the variant of each new case
        if not len(new_cases): # No networks
            return ss.uids(), ss.uids(), np.empty(0, dtype=ss.dtypes.int)
        new_cases, inds = ss.uids.concatenate(new_cases).unique(return_index=True)
        sources = ss.uids.concatenate(sources)[inds]
        networks = np.concatenate(networks)[inds]
        self._new_cases = new_cases
        self._new_case_variants = np.concatenate(case_variant)[inds]
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

    # Testing / tracing / quarantine --------------------------------------

    def test(self, uids, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, n_tests=None):
        """Test agents and schedule positive diagnoses (the v3 ``People.test`` action).

        Called by the testing interventions, not directly by users. Infectious agents
        test positive with probability ``test_sensitivity``; not-already-diagnosed positives that are
        not lost to follow-up get ``date_diagnosed = ti + test_delay`` (and ``date_pos_test = ti``).
        Returns the UIDs that will be diagnosed. ``n_tests``, if supplied, is the number of tests to
        record instead (e.g. cv.test_num records the number requested, as in v3).
        """
        uids = ss.uids(np.unique(uids))
        if n_tests is None:
            n_tests = len(uids) * self.full_pop_factor
        self.test_flows['tests'] += n_tests
        if not len(uids):
            return uids
        ti = self.ti
        self.tested[uids] = True
        self.date_tested[uids] = ti
        inf = uids[self.infectious[uids]]          # only infectious agents can test positive
        if not len(inf):
            return ss.uids()
        self._test_bern.set(p=test_sensitivity)
        pos = inf[self._test_bern.rvs(inf)]                    # true positives (sensitivity)
        not_diag = pos[np.isnan(self.date_diagnosed[pos])]  # exclude already-diagnosed
        if not len(not_diag):
            return ss.uids()
        self._test_bern.set(p=1.0 - loss_prob)
        final = not_diag[self._test_bern.rvs(not_diag)]        # exclude loss-to-follow-up
        self.date_diagnosed[final] = ti + test_delay
        self.date_pos_test[final] = ti
        self.test_flows['diagnoses'] += len(final) # As in v3, diagnoses are counted on the day of the test, since that's how most data are reported
        return final

    def schedule_quarantine(self, uids, start_date=None, period=None):
        """Schedule a quarantine request (the v3 ``People.schedule_quarantine``).

        Eligibility (not already dead/recovered/diagnosed/isolated) is re-checked when ``start_date``
        is reached, in ``step_testing()``, which runs after the interventions. Requests for a day that
        has already been processed (e.g. made by an analyzer) are processed on the next step.
        Defaults: ``start_date=ti``, ``period=quar_period``.
        """
        ti = self.ti
        start_date = ti if start_date is None else int(start_date)
        period = self.pars.quar_period if period is None else int(period)
        bucket = self._pending_quarantine.setdefault(start_date, [])
        bucket.append((ss.uids(uids), start_date + period))
        return

    def vaccinate_agents(self, uids, label, index):
        """Apply a vaccine dose to ``uids`` (the NAb side of the v3 ``BaseVaccination.vaccinate``).

        Sets the vaccination state and confers/boosts peak NAb via the vaccine's ``nab_init``/
        ``nab_boost`` (the same NAb pipeline as natural infection). The intervention is responsible for selecting/
        de-duplicating ``uids`` (skipping dead / already-fully-dosed); this just applies the dose.
        ``label``/``index`` identify the vaccine in ``vaccine_pars``/``vaccine_map``.
        """
        uids = ss.uids(np.unique(uids))
        if not len(uids):
            return uids
        prior_vacc = self.vaccinated[uids]  # already vaccinated => not a NEW vaccinee
        self.vaccinated[uids] = True
        self.vaccine_source[uids] = index
        self.doses[uids] = self.doses[uids] + 1
        self.date_vaccinated[uids] = self.ti
        self.update_peak_nab(uids, nab_pars=self.vaccine_pars[label])
        self.count_doses(uids, prior_vacc)
        return uids

    @property
    def full_pop_factor(self):
        """
        With dynamic rescaling, tests and vaccine doses are counted as if they were given to the whole population,
        not only the agents being simulated, since the people not included are also tested and vaccinated (as in v3)
        """
        return self.sim.pars.pop_scale / self.sim.current_scale

    def count_doses(self, uids, prior_vacc):
        """Count the vaccine doses given to these agents, and the number newly vaccinated (``prior_vacc`` is whether each was already vaccinated)"""
        self.vacc_flows['doses'] += len(uids) * self.full_pop_factor
        self.vacc_flows['vaccinated'] += np.count_nonzero(~prior_vacc) * self.full_pop_factor
        return

    def step_testing(self):
        """Diagnosis / quarantine / isolation state machines (the v3 ``People.update_states_post()``).

        As in v3, this runs after the interventions and before transmission, so a test with no delay leads
        to diagnosis, isolation, and the quarantine of traced contacts on the same day. Inert until a
        testing/tracing intervention sets the trigger dates or schedules a quarantine (all date arrays
        default NaN and the pending-quarantine dict is empty); no random numbers are used.
        """
        ti = self.ti
        # check_diagnosed: positives become diagnosed when their (delayed) diagnosis date arrives.
        self.date_pos_test[((self.date_pos_test <= ti) & ~self.diagnosed).uids] = np.nan
        new_diag = ((self.date_diagnosed <= ti) & ~self.diagnosed).uids
        self.diagnosed[new_diag] = True
        # check_enter_iso: anyone diagnosed this step isolates until recovery.
        self.test_flows['isolated'] += len(new_diag)
        if len(new_diag):
            self.isolated[new_diag] = True
            self.date_end_isolation[new_diag] = self.ti_recovered[new_diag]
        # check_quar: process pending quarantine requests scheduled for today (or earlier, if they were made after that day was processed)
        pending = []
        for day in sorted(self._pending_quarantine.keys()):
            if day <= ti:
                pending += self._pending_quarantine.pop(day)
        if len(pending):
            # If an agent has several requests, use the latest end day
            inds = np.concatenate([uids for uids,end_day in pending])
            end_days = np.concatenate([np.full(len(uids), end_day) for uids,end_day in pending])
            latest_end = np.full(inds.max()+1, -np.inf)
            np.maximum.at(latest_end, inds, end_days)
            uids = ss.uids(np.unique(inds))
            end_days = latest_end[uids]

            # Extend the quarantine of agents already in quarantine
            in_quar = self.quarantined[uids]
            extend = uids[in_quar]
            self.date_end_quarantine[extend] = np.maximum(self.date_end_quarantine[extend], end_days[in_quar])

            # Quarantine the others, unless they're dead, recovered, diagnosed or isolated
            new = uids[~in_quar]
            new_end_days = end_days[~in_quar]
            ineligible = self.dead[new] | self.recovered[new] | self.diagnosed[new] | self.isolated[new]
            new = new[~ineligible]
            self.quarantined[new] = True
            self.date_quarantined[new] = ti
            self.date_end_quarantine[new] = new_end_days[~ineligible]
        # Diagnosed-today agents end quarantine (they move to isolation instead).
        self.date_end_quarantine[(self.quarantined & (self.date_diagnosed == ti)).uids] = ti
        # Release agents whose quarantine has ended.
        self.quarantined[((self.date_end_quarantine <= ti) & self.quarantined).uids] = False
        return

    def seed_imports(self):
        """Infect a Poisson-distributed number of susceptible agents with the wild variant each step (v3 importations)."""
        n_imports = self.pars.n_imports
        if n_imports <= 0:
            return
        self._n_imports.set(lam=n_imports/self.sim.current_scale) # As in v3, the expected number of agents falls as the population scale rises
        n = int(self._n_imports.rvs(1)[0])
        if n > 0:
            self._choose_imports.set(n=n)
            chosen = self._choose_imports.filter(self.susceptible.uids)
            self.import_variant(chosen, 0)
        return

    def step_state(self):
        """Advance the state machine (before transmission); flip flags on ti thresholds, no re-draws."""
        ti = self.ti
        # Reset this step's flows first, so the variant introductions and importations below and the
        # transmission later are all counted. new_infections_by_variant is counted in set_prognoses(),
        # and the other flows here.
        for v in self.flows_by_variant.values():
            v[:] = 0.0
        for flows in [self.flows, self.test_flows]:
            for key in flows:
                flows[key] = 0
        for var in self._variant_objs: # Seed any variant introductions scheduled for this day (the v3 variant.apply)
            var.apply(self)
        # preinfectious -> infectious: tag infectious_variant from exposed_variant (v3 check_infectious),
        # count new_infectious_by_variant, then clear the `preinfectious` flag. The gate is `preinfectious &
        # (ti_infectious<=ti)` (the agents transitioning THIS step) rather than the full infectious
        # property `infected & (ti_infectious<=ti)`: the latter would re-count already-infectious
        # agents every step (over-counting the new_infectious flow). Same-step infectiousness
        # (dur_exp2inf==0) is tagged in set_prognoses, and those agents are still `preinfectious` here so
        # they are counted exactly once. Every infectious agent thus carries a finite infectious_variant,
        # so the nv>1 source mask `infectious & (infectious_variant==vi)` == the full infectious set.
        new_inf = (self.preinfectious & (self.ti_infectious <= ti)).uids
        if len(new_inf):
            iv = self.exposed_variant[new_inf]
            self.infectious_variant[new_inf] = iv
            self.flows_by_variant['new_infectious'] += np.bincount(iv.astype(int), minlength=self.nv)
            self.preinfectious[new_inf] = False
        # Stage onsets -- count only NEW transitions (for the flow Results); flags are cumulative-nested.
        new_symp = (self.infected & ~self.symptomatic & (self.ti_symptomatic <= ti)).uids
        self.symptomatic[new_symp] = True
        new_sev = (self.infected & ~self.severe & (self.ti_severe <= ti)).uids
        self.severe[new_sev] = True
        self.flows_by_variant['new_symptomatic'] += np.bincount(self.exposed_variant[new_symp].astype(int), minlength=self.nv) # v3 counted these by variant on the day of infection rather than when they happened
        self.flows_by_variant['new_severe']      += np.bincount(self.exposed_variant[new_sev].astype(int), minlength=self.nv)
        new_crit = (self.infected & ~self.critical & (self.ti_critical <= ti)).uids
        self.critical[new_crit] = True
        # -> recovered (clear the stage flags; tag recovered_variant; clear active-infection variant
        # tags). If cross_immunity_active, recovery restores susceptibility so the agent can be
        # reinfected; otherwise immunity is permanent.
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
            self.infected[new_dead] = False # As in v3, people don't transmit on the day they die
        # Capture this step's flows for update_results(); reinfections are counted in set_prognoses()
        self.flows['symptomatic']  = len(new_symp)
        self.flows['severe']       = len(new_sev)
        self.flows['critical']     = len(new_crit)
        self.flows['recoveries']   = len(new_rec)
        self.flows['deaths']       = len(new_dead)
        self.flows['known_deaths'] = np.count_nonzero(self.diagnosed[new_dead])

        # check_exit_iso: release agents whose isolation has ended (as in v3, before the interventions)
        self.isolated[((self.date_end_isolation <= ti) & self.isolated).uids] = False

        # Background importation, as in v3; inert when n_imports == 0 (the default)
        self.seed_imports()

        # Update per-agent transmissibility before this step's transmission:
        # rel_trans = beta_dist draw x viral_load(t) x asymp_factor (as in v3's compute_trans_sus()).
        inf = self.infectious.uids
        if len(inf):
            early = ti < self.ti_vl_switch[inf]           # high viral-load phase (front-loaded)
            vl = np.where(early, self._vl_high, self._vl_low)
            f_asymp = np.where(self.symptomatic[inf], 1.0, self.pars.asymp_factor)
            self.rel_trans[inf] = self.rel_trans_base[inf] * vl * f_asymp # Isolation and quarantine are applied per layer in infect()
        return

    def step(self):
        """
        As in v3, process today's diagnoses and quarantines (after the interventions), then transmission,
        and then update the NAbs (at the end of the step, after the day's infections and vaccinations).
        """
        self.step_testing()
        out = super().step()
        self._new_cases = None # Used by set_prognoses() for this step's transmission only
        self._new_case_variants = None
        if self.pars.use_waning:
            self.update_nab()
        return out

    def update_nab(self):
        """
        Step NAb levels forward along the precomputed waning kernel (v3 ``update_nab()``):
        ``nab += nab_kin[ti - t_nab_event] * peak_nab``, clamped to ``[0, peak_nab]``.
        """
        uids = (self.peak_nab > 0).uids
        if not len(uids):
            return
        t_since = (self.ti - self.t_nab_event[uids]).astype(int)
        if t_since.max() >= len(self.nab_kin):
            errormsg = f'The NAb waning kernel has {len(self.nab_kin)} entries, but {t_since.max()+1} are needed; for NAb events before the start of the sim, use imprint_historical_nab()'
            raise ValueError(errormsg)
        peak = self.peak_nab[uids]
        nab = self.nab[uids] + self.nab_kin[t_since] * peak
        self.nab[uids] = np.clip(nab, 0.0, peak)
        return

    def step_die(self, uids):
        """Reset disease flags for agents who die this step (so they are not double-counted)."""
        super().step_die(uids)
        self.infected[uids]     = False
        self.preinfectious[uids] = False
        self.symptomatic[uids]  = False
        self.severe[uids]       = False
        self.critical[uids]     = False
        self.recovered[uids]    = False
        self.susceptible[uids]  = False
        self.dead[uids]         = True
        # Clear the variant tags (as in v3)
        self.exposed_variant[uids]    = np.nan
        self.infectious_variant[uids] = np.nan
        self.recovered_variant[uids]  = np.nan
        # Clear testing/quarantine flags on death (v3 check_death; inert when no testing is active).
        self.known_contact[uids] = False
        self.quarantined[uids]   = False
        self.isolated[uids]      = False
        return

    # Results ------------------------------------------------------------------

    def init_results(self):
        """
        Define the results: the daily flows (new_*), cumulative counts (cum_*), stocks that aren't
        states (e.g. n_infectious), rates, immunity summaries, and results by variant. The stocks of
        the states (e.g. symptomatic) and new_infections and cum_infections come from ss.Infection.
        Counts are scaled by the population scale.
        """
        super().init_results()

        def R(name, label):
            return ss.Result(name, dtype=int, scale=True, label=label)

        self.define_results(
            R('n_infectious',  'Number infectious'),  # infectious is a property, not auto-counted
            R('new_infectious',  'New infectious'),
            R('new_symptomatic', 'New symptomatic'),
            R('new_severe',      'New severe'),
            R('new_critical',    'New critical'),
            R('new_recoveries',  'New recoveries'),
            R('new_deaths',      'New deaths'),
            R('cum_infectious',  'Cumulative infectious'),
            R('cum_symptomatic', 'Cumulative symptomatic'),
            R('cum_severe',      'Cumulative severe'),
            R('cum_critical',    'Cumulative critical'),
            R('cum_recoveries',  'Cumulative recoveries'),
            R('cum_deaths',      'Cumulative deaths'),
            R('new_known_deaths', 'New known deaths'), # Deaths among people who had been diagnosed
            R('cum_known_deaths', 'Cumulative known deaths'),
            R('n_known_dead',    'Number of confirmed deaths'), # As in v3, the same as cum_known_deaths
            R('n_removed',       'Number removed'), # Recovered or dead
            R('new_reinfections', 'New reinfections'),
            R('cum_reinfections', 'Cumulative reinfections'),
            R('n_exposed',       'Number exposed'), # As in v3, exposed means infected, including infectious
            R('n_imports',       'Number of imported infections'),  # bumped by import_variant
            R('new_tests',       'New tests'),        # bumped by covid.test()
            R('cum_tests',       'Cumulative tests'),
            R('new_diagnoses',   'New diagnoses'),    # bumped in step_testing
            R('cum_diagnoses',   'Cumulative diagnoses'),
            R('new_doses',       'New vaccine doses'),     # bumped by vaccinate_agents
            R('cum_doses',       'Cumulative vaccine doses'),
            R('new_vaccinated',  'Newly vaccinated people'),
            R('cum_vaccinated',  'Cumulative vaccinated people'),
            R('new_quarantined', 'Newly quarantined people'),  # quarantine flow (the stock, quarantined, is auto-counted)
            R('cum_quarantined', 'Cumulative quarantined people'),
            R('new_isolated',    'Newly isolated people'), # Everyone who is diagnosed is isolated
            R('cum_isolated',    'Cumulative isolated people'),
        )
        # Derived/rate Results (population-level, scale=False): test yield + effective reproduction number.
        self.define_results(
            ss.Result('test_yield', dtype=float, scale=False, label='Test yield (diagnoses/tests)'),
            ss.Result('r_eff',      dtype=float, scale=False, label='Effective reproduction number'),
            ss.Result('incidence',       dtype=float, scale=False, label='Incidence'),
            ss.Result('rel_test_yield',  dtype=float, scale=False, label='Relative testing yield'),
            ss.Result('frac_vaccinated', dtype=float, scale=False, label='Proportion vaccinated'),
            ss.Result('doubling_time',   dtype=float, scale=False, label='Doubling time'),
        )
        # immunity summaries (population means -> scale=False). Filled only under use_waning.
        self.define_results(
            ss.Result('pop_nabs',       dtype=float, scale=False, label='Population mean NAb level'),
            ss.Result('pop_protection', dtype=float, scale=False, label='Population mean protective immunity'),
            ss.Result('pop_symp_protection', dtype=float, scale=False, label='Population symptomatic protection'),
        )
        self.flows = dict(symptomatic=0, severe=0, critical=0, recoveries=0, deaths=0, known_deaths=0, reinfections=0)
        self.test_flows = dict(tests=0, diagnoses=0, isolated=0)  # per-step testing flows
        self.vacc_flows = dict(doses=0, vaccinated=0)  # per-step dose/vaccination flow
        self.flows_by_variant = {key: np.zeros(self.nv) for key in ['new_infections', 'new_symptomatic', 'new_severe', 'new_infectious']}

        # Results by variant (the v3 sim.results['variant']), with one column per variant, e.g. results.new_infections_by_variant.wild
        labels = [self.variant_map[i] for i in range(self.nv)]
        def RV(name, scale, label):
            return ss.Result(name, dtype=float, scale=scale, columns=labels, label=label)
        self.define_results(
            RV('prevalence_by_variant',      False, 'Prevalence by variant'),
            RV('incidence_by_variant',       False, 'Incidence by variant'),
            RV('new_infections_by_variant',  True,  'New infections by variant'),
            RV('cum_infections_by_variant',  True,  'Cumulative infections by variant'),
            RV('new_symptomatic_by_variant', True,  'New symptomatic by variant'),
            RV('cum_symptomatic_by_variant', True,  'Cumulative symptomatic by variant'),
            RV('new_severe_by_variant',      True,  'New severe by variant'),
            RV('cum_severe_by_variant',      True,  'Cumulative severe by variant'),
            RV('new_infectious_by_variant',  True,  'New infectious by variant'),
            RV('cum_infectious_by_variant',  True,  'Cumulative infectious by variant'),
            RV('n_exposed_by_variant',       True,  'Number exposed by variant'),
            RV('n_infectious_by_variant',    True,  'Number infectious by variant'),
        )
        return

    def update_results(self):
        super().update_results()  # auto-counts the BoolState stocks (symptomatic, severe, ...; sim.results.n_symptomatic etc. as in v3)
        ti = self.ti
        res = self.results
        res.n_infectious[ti] = int(np.count_nonzero(self.infectious))
        # Flows (captured this timestep in step_state)
        res.new_infectious[ti]  = self.flows_by_variant['new_infectious'].sum()
        res.new_symptomatic[ti] = self.flows['symptomatic']
        res.new_severe[ti]      = self.flows['severe']
        res.new_critical[ti]    = self.flows['critical']
        res.new_recoveries[ti]  = self.flows['recoveries']
        res.new_deaths[ti]      = self.flows['deaths']
        res.new_known_deaths[ti] = self.flows['known_deaths']
        res.new_reinfections[ti] = self.flows['reinfections']
        res.n_exposed[ti] = int(np.count_nonzero(self.infected))
        res.new_tests[ti]       = self.test_flows['tests']      # testing flows
        res.new_diagnoses[ti]   = self.test_flows['diagnoses']
        res.new_isolated[ti]    = self.test_flows['isolated'] # As in v3, people are isolated when they are diagnosed
        res.new_doses[ti]       = self.vacc_flows['doses']      # vaccination flows
        res.new_vaccinated[ti]  = self.vacc_flows['vaccinated']
        self.vacc_flows = dict(doses=0, vaccinated=0) # Reset here rather than at the start of the step, so historical doses (given when the sim is initialized) are counted on day 0, as in v3
        # Quarantine flow + test yield (the quarantine stock, quarantined, is auto-counted by Starsim).
        res.new_quarantined[ti] = int(np.count_nonzero(self.date_quarantined.raw == ti))
        n_tests = self.test_flows['tests']
        res.test_yield[ti]      = (self.test_flows['diagnoses'] / n_tests) if n_tests > 0 else 0.0

        # By-variant flows (accumulated this step) and stocks (counted from the variant of each agent)
        fv = self.flows_by_variant
        res.new_infections_by_variant[ti]  = fv['new_infections']
        res.new_symptomatic_by_variant[ti] = fv['new_symptomatic']
        res.new_severe_by_variant[ti]      = fv['new_severe']
        res.new_infectious_by_variant[ti]  = fv['new_infectious']
        for key, variants in [['n_exposed_by_variant', self.exposed_variant[self.infected.uids]],
                              ['n_infectious_by_variant', self.infectious_variant[self.infectious.uids]]]:
            variants = variants[np.isfinite(variants)].astype(int)
            res[key][ti] = np.bincount(variants, minlength=self.nv)

        # Population immunity summaries (means over alive agents, and over all variants for protection, as in v3)
        if self.pars.use_waning:
            nab = self.nab.values  # active (alive) values
            res.pop_nabs[ti] = float(nab.mean()) if len(nab) else 0.0
            auids = self.sim.people.auids
            res.pop_protection[ti] = float(self.sus_imm[auids].mean()) if len(auids) else 0.0
            res.pop_symp_protection[ti] = float(self.symp_imm[auids].mean()) if len(auids) else 0.0
        return

    def finalize_results(self):
        """Cumulate the daily flows into the cum_* results, and finalize the by-variant results."""
        super().finalize_results()
        res = self.results
        stems = ['infectious', 'symptomatic', 'severe', 'critical', 'recoveries', 'deaths', 'known_deaths',
                 'tests', 'diagnoses', 'isolated', 'doses', 'vaccinated', 'quarantined', 'reinfections']
        for stem in stems:
            res[f'cum_{stem}'][:] = np.cumsum(res[f'new_{stem}'][:])
        res.n_known_dead[:] = res.cum_known_deaths[:]
        count_recov = 1 - self.pars.use_waning # As in v3, with waning, people who have recovered are susceptible again, so aren't counted as removed
        res.n_removed[:] = count_recov*res.cum_recoveries[:] + res.cum_deaths[:]

        # By-variant cumulatives (after scaling, since the scale can vary over time)
        for stem in ['infections', 'symptomatic', 'severe', 'infectious']:
            res[f'cum_{stem}_by_variant'][:] = np.cumsum(res[f'new_{stem}_by_variant'][:], axis=0)

        # As in v3, cumulative infections include the initial infections
        n_seed = self.n_initial_cases * np.atleast_1d(self.sim.result_scale(self))[0] # The scale at the start of the sim
        res.cum_infections.values = res.cum_infections.values + n_seed # Not in-place, since it's an integer array and the scale may not be
        res.cum_infections_by_variant[:, 0] += n_seed

        # Derived results (v3 compute_states and compute_yield)
        n_alive = np.asarray(self.sim.results.alive, dtype=float)
        n_susc = np.asarray(res.susceptible, dtype=float)
        def divide(a, b):
            a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
            return np.divide(a, b, out=np.zeros_like(a), where=b > 0)
        res.incidence[:] = divide(res.new_infections, n_susc)
        res.frac_vaccinated[:] = divide(res.vaccinated, n_alive)
        denom = divide(res.n_infectious, n_alive - res.cum_diagnoses) # Alive + undiagnosed people might test; infectious people will test positive
        res.rel_test_yield[:] = divide(res.test_yield, denom)
        self.compute_doubling()

        # Effective reproduction number (the v3 'daily' method): mean infectious duration times the
        # daily non-imported new infections divided by the number infectious, then smoothed (with the
        # initial seed period averaged so it isn't dominated by the seeds). A diagnostic, not a driver.
        self.compute_r_eff()

        # Compute the by-variant rates (v3 used the new infections for the prevalence by variant, rather than the number infected)
        res.incidence_by_variant[:]  = divide(res.new_infections_by_variant, n_susc[:, None])
        res.prevalence_by_variant[:] = divide(res.n_exposed_by_variant, n_alive[:, None])
        return

    def shrink(self):
        """Shrink the module for saving, including the per-variant immunity arrays and other per-agent data"""
        super().shrink()
        for attr in ['nab_kin', 'infection_log', '_new_cases', '_new_case_variants', '_pending_quarantine']:
            if hasattr(self, attr):
                setattr(self, attr, None)
        return

    def compute_doubling(self, window=3, max_doubling_time=30):
        """
        Calculate the doubling time using an exponential approximation (the v3 ``Sim.compute_doubling``).
        Compares infections at time t to infections at time t-window, and uses that to compute the
        doubling time. For example, if there are 100 cumulative infections on day 12 and 200 on day 19,
        the doubling time is 7 days.

        Args:
            window (float): the size of the window used (larger values are more accurate but less precise)
            max_doubling_time (float): doubling time could be infinite, so this places a bound on it
        """
        cum_infections = np.asarray(self.results['cum_infections'], dtype=float)
        infections_now = cum_infections[window:]
        infections_prev = cum_infections[:-window]
        use = (infections_prev > 0) & (infections_now > infections_prev)
        doubling_time = window * np.log(2) / np.log(infections_now[use] / infections_prev[use])
        self.results['doubling_time'][:] = np.nan
        self.results['doubling_time'][window:][use] = np.minimum(doubling_time, max_doubling_time)
        return self.results['doubling_time'].values

    def compute_r_eff(self, smoothing=2):
        """Fill ``results['r_eff']`` using the v3 'daily' method (as in v3 ``Sim.compute_r_eff()``)."""
        res = self.results
        npts = self.t.npts
        # Mean infectious duration over agents who reached an outcome (recovery or death).
        ti_inf = np.asarray(self.ti_infectious.raw, dtype=float)
        ti_rec = np.asarray(self.ti_recovered.raw, dtype=float)
        ti_dead = np.asarray(self.ti_dead.raw, dtype=float)
        outcome = np.where(np.isfinite(ti_rec), ti_rec, ti_dead)  # whichever occurred
        has = np.isfinite(outcome) & np.isfinite(ti_inf)
        mean_inf = float((outcome[has] - ti_inf[has]).mean()) if has.any() else 0.0

        new_inf = np.asarray(res['new_infections'], dtype=float) - np.asarray(res['n_imports'], dtype=float)
        n_inf = np.asarray(res['n_infectious'], dtype=float)
        raw = mean_inf * np.divide(new_inf, n_inf, out=np.zeros(npts), where=n_inf > 0)

        # Average over the initial seed period, then smooth (as in v3), guarding short arrays
        if npts >= 3:
            seed_period = 0
            for dist in [self.pars.dur_exp2inf, self.pars.dur_asym2rec]: # Approximate the duration of the seed infections by the median durations
                seed_period += float(dist.ppf(np.array([0.5]))[0])
            initial = int(min(npts, seed_period))
            for i in range(initial):
                raw[i] = raw[i:initial].mean()
            values = sc.smooth(raw, smoothing)
            values[:smoothing] = raw[:smoothing] # avoid edge artefacts from the smoothing kernel
            values[-smoothing:] = raw[-smoothing:]
        else:
            values = raw
        res.r_eff[:] = values
        return

    # Seeding ------------------------------------------------------------------

    def init_post(self):
        """Fill the age-conditional prognoses, then seed the initial infections (e.g. ``init_prev=ss.choose_n(20)``)."""
        self.init_prognoses()  # must precede any set_prognoses call (seeding below)
        self._pending_quarantine = {}  # start_day -> [(uid, end_day)] (reset for clean re-runs)

        initial_cases = super().init_post()
        self.n_initial_cases = len(initial_cases) if initial_cases is not None else 0 # As in v3, these are added to cum_infections
        for flow in self.flows_by_variant.values(): # As for new_infections, the seeds are not new infections at time point 0, but are added to the cumulative counts in finalize_results()
            flow[:] = 0

        # Make some of the other agents non-naive, so they can't be infected (the v3 people.make_nonnaive()).
        # Unlike v3, which chose them first and so removed some of the seed infections, they are chosen from the agents who weren't seeded.
        frac = self.pars.frac_susceptible
        if frac < 1:
            self._choose_nonsusceptible.set(n=int(round((1 - frac) * len(self.sim.people))))
            nonsusceptible = self._choose_nonsusceptible.filter(self.susceptible.uids)
            self.susceptible[nonsusceptible] = False
            self.naive[nonsusceptible] = False
        return initial_cases

    def make_naive(self, uids, skip_states=None):
        """
        Reset agents to never having been infected, for dynamic rescaling (the v3 ``people.make_naive()``).
        As in v3, vaccination is kept, including the immunity of vaccinated agents, as are the per-agent
        prognoses and transmissibility.
        """
        if skip_states is None:
            skip_states = ['rel_sus', 'rel_trans', 'rel_trans_base', 'symp_prob', 'severe_prob', 'crit_prob', 'death_prob',
                           'vaccinated', 'doses', 'vaccine_source', 'date_vaccinated']
        imm_states = ['sus_imm', 'symp_imm', 'sev_imm', 'peak_nab', 'nab', 't_nab_event']
        super().make_naive(uids, skip_states=skip_states + imm_states)
        non_vx = uids[~self.vaccinated[uids]]
        for name in imm_states:
            getattr(self, name).set(non_vx)
        return
