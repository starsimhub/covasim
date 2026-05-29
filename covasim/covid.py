"""
The COVID-19 disease module for Covasim on the Starsim base.

``cv.COVID(ss.Infection)`` is the single-variant COVID disease. M1 implements the
minimal natural history -- susceptible -> exposed -> infectious -> recovered (SEIR)
-- with transmission + recovery only:

    - On infection, an agent becomes ``exposed`` (``infected`` but not yet
      ``infectious``); after the ``dur_exp2inf`` latent period it becomes
      ``infectious``; after the ``dur_asym2rec`` infectious period it ``recovered``.
    - Only ``infectious`` (not merely ``exposed``) agents transmit: the
      ``infectious`` property is overridden to the I-subset, and stock
      ``ss.Infection.infect()`` gates transmission on ``self.infectious``.
    - Recovery confers permanent immunity (``use_waning=False`` semantics):
      recovered agents never return to susceptible. Waning / NAbs / reinfection
      are added in M4.

Symptoms, severity, critical illness, and death (the rest of Covasim's prognosis
tree) are added in M2; per-agent transmissibility heterogeneity (``beta_dist``)
and multiple variants come in M2/M3. Transmission is the stock CRN-safe
``ss.Infection.infect()`` -- no custom transmission code.
"""
import numpy as np
import starsim as ss

__all__ = ['COVID']


class COVID(ss.Infection):
    """Single-variant COVID-19 disease: S->E->I->R transmission + recovery.

    Args:
        beta: per-contact transmission probability per day (scalar or dict keyed by
            network layer); Covasim's default is ``pars['beta'] = 0.016``.
        init_prev: an ``ss.bernoulli`` for seeding initial infections, or ``None`` if
            the sim seeds an exact ``pop_infected`` count (cv.Sim, M1 Task 3).
        dur_exp2inf: exposed -> infectious latency (Covasim ``dur.exp2inf``,
            lognormal mean 4.5 / std 1.5 days).
        dur_asym2rec: infectious -> recovered duration (Covasim ``dur.asym2rec``,
            lognormal mean 8.0 / std 2.0 days; M1 is the asymptomatic-only path).
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta         = ss.probperday(0.016),                              # Covasim pars['beta'] (parameters.py:62)
            init_prev    = None,                                              # seeding via pop_infected (cv.Sim) or an ss.bernoulli here
            dur_exp2inf  = ss.lognorm_ex(mean=ss.days(4.5), std=ss.days(1.5)),  # E->I latency (Covasim dur.exp2inf)
            dur_asym2rec = ss.lognorm_ex(mean=ss.days(8.0), std=ss.days(2.0)),  # I->R duration (Covasim dur.asym2rec)
        )
        self.update_pars(pars, **kwargs)

        # Redefine the full state set (reset=True drops ss.Infection's defaults; the SIR pattern).
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('infected',                   label='Infected (exposed or infectious)'),
            ss.BoolState('exposed',                    label='Exposed (infected, pre-infectious)'),
            ss.BoolState('recovered',                  label='Recovered'),
            ss.FloatArr('rel_sus',     default=1.0,    label='Relative susceptibility'),
            ss.FloatArr('rel_trans',   default=1.0,    label='Relative transmissibility'),
            ss.FloatArr('ti_infected',                 label='Time index of infection'),
            ss.FloatArr('ti_exposed',                  label='Time index of becoming exposed'),
            ss.FloatArr('ti_infectious',               label='Time index of becoming infectious'),
            ss.FloatArr('ti_recovered',                label='Time index of recovery'),
            reset = True,
        )
        return

    @property
    def infectious(self):
        """Agents who transmit: infected and past the latent period (I, not merely E)."""
        return self.infected & (self.ti_infectious <= self.ti)

    def set_prognoses(self, uids, sources=None):
        """On infection: become exposed, and schedule E->I and I->R (asymptomatic-only)."""
        super().set_prognoses(uids, sources)  # logs the infection
        ti = self.ti
        self.susceptible[uids] = False
        self.infected[uids]    = True
        self.exposed[uids]     = True
        self.ti_infected[uids] = ti
        self.ti_exposed[uids]  = ti
        # Sample each duration once per timestep (CRN-safe via uids).
        self.ti_infectious[uids] = ti + self.pars.dur_exp2inf.rvs(uids)
        self.ti_recovered[uids]  = self.ti_infectious[uids] + self.pars.dur_asym2rec.rvs(uids)
        return

    def step_state(self):
        """Advance E->I and I->R at the start of the timestep (before transmission)."""
        ti = self.ti
        # Exposed -> infectious (the infectious property derives from ti_infectious; just clear `exposed`)
        new_infectious = (self.exposed & (self.ti_infectious <= ti)).uids
        self.exposed[new_infectious] = False
        # Infectious -> recovered (permanent immunity: do NOT return to susceptible)
        recovered = (self.infected & (self.ti_recovered <= ti)).uids
        self.infected[recovered]  = False
        self.recovered[recovered] = True
        return

    def init_results(self):
        """Add an n_infectious result (infectious is a property, so not auto-counted)."""
        super().init_results()
        self.define_results(
            ss.Result('n_infectious', dtype=int, scale=True, label='Number infectious'),
        )
        return

    def update_results(self):
        super().update_results()
        self.results.n_infectious[self.ti] = int(np.count_nonzero(self.infectious))
        return

    def init_post(self):
        """Seed initial infections.

        Covasim seeds an *exact* count of initial infections (``pop_infected``), not a
        per-agent probability. If ``init_prev`` is an integer, seed exactly that many
        agents (chosen deterministically from the sim seed); otherwise defer to
        ``ss.Infection`` (an ``ss.bernoulli`` fraction, or ``None``).
        """
        exact = self.pars.init_prev if isinstance(self.pars.init_prev, (int, np.integer)) else None
        if exact is None:
            return super().init_post()  # ss.bernoulli / None: stock seeding

        # Exact-count path: run the base setup (with no stock seeding), then seed exactly `exact` agents.
        self.pars.init_prev = None      # suppress the stock bernoulli seeding in ss.Infection.init_post
        super().init_post()             # runs the required Module/Disease init_post setup
        self.pars.init_prev = exact     # restore for the record
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
