'''
Tests for immune waning, variants, and vaccine intervention.
'''

#%% Imports and settings
import sciris as sc
import covasim as cv
import pandas as pd
import pylab as pl
import numpy as np
import pytest

do_plot = 1
cv.options(interactive=False) # Assume not running interactively

# Shared parameters across simulations
base_pars = sc.objdict(
    pop_size = 1e3,
    verbose  = -1,
)


#%% Define the tests

def test_states():
    ''' Test state consistency against state_diagram.xlsx '''

    filename = 'state_diagram.xlsx'
    sheets   = ['Without waning', 'With waning']
    indexcol = 'In ↓ you can be →'

    # Load state diagram
    dfs = sc.odict()
    for sheet in sheets:
        dfs[sheet] = pd.read_excel(filename, sheet_name=sheet)
        dfs[sheet] = dfs[sheet].set_index(indexcol)

    # Create and run simulation
    for use_waning in [False, True]:
        sc.heading(f'Testing state consistency with waning = {use_waning}')
        df = dfs[use_waning] # Different states are possible with or without waning

        # Parameters chosen to be midway through the sim so as few states as possible are empty
        pars = dict(
            pop_size = 1e3,
            pop_infected = 20,
            n_days = 70,
            use_waning = use_waning,
            verbose = 0,
            interventions = [
                cv.test_prob(symp_prob=0.4, asymp_prob=0.01),
                cv.contact_tracing(trace_probs=0.1),
                cv.simple_vaccine(days=60, prob=0.1)
            ]
        )
        sim = cv.Sim(pars).run()
        ppl = sim.people

        # Check states
        errormsg = ''
        states = df.columns.values.tolist()
        for s1 in states:
            for s2 in states:
                if s1 != s2:
                    relation = df.loc[s1, s2] # e.g. df.loc['susceptible', 'exposed']
                    print(f'Checking {s1:13s} → {s2:13s} = {relation:2n} ... ', end='')
                    inds     = cv.true(ppl[s1])
                    n_inds   = len(inds)
                    vals2    = ppl[s2][inds]
                    is_true  = cv.true(vals2)
                    is_false = cv.false(vals2)
                    n_true   = len(is_true)
                    n_false  = len(is_false)
                    if relation == 1 and n_true != n_inds:
                        errormsg = f'Being {s1}=True implies {s2}=True, but only {n_true}/{n_inds} people are'
                        print(f'× {n_true}/{n_inds} error!')
                    elif relation == -1 and n_false != n_inds:
                        errormsg = f'Being {s1}=True implies {s2}=False, but only {n_false}/{n_inds} people are'
                        print(f'× {n_false}/{n_inds} error!')
                    else:
                        print(f'✓ {n_true}/{n_inds}')
                    if errormsg:
                        raise RuntimeError(errormsg)

    return


def test_waning(do_plot=False):
    sc.heading('Testing with and without waning')
    msims = dict()

    for rescale in [0,1]:
        print(f'Checking with rescale = {rescale}...')

        # Define parameters specific to this test
        pars = dict(
            n_days    = 90,
            beta      = 0.01,
        )

        # Optionally include rescaling
        if rescale:
            pars.update(
                pop_scale      = 10,
                rescale_factor = 2.0, # Use a large rescale factor to make differences more obvious
            )

        # Run the simulations and pull out the results
        s0 = cv.Sim(base_pars, **pars, use_waning=False, label='No waning').run()
        s1 = cv.Sim(base_pars, **pars, use_waning=True, label='With waning', analyzers=cv.nab_histogram()).run()
        res0 = s0.summary
        res1 = s1.summary
        msim = cv.MultiSim([s0,s1])
        msims[rescale] = msim

        # Check results
        for key in ['n_susceptible', 'cum_infections', 'cum_reinfections', 'pop_nabs', 'pop_protection', 'pop_symp_protection']:
            v0 = res0[key]
            v1 = res1[key]
            print(f'Checking {key:20s} ... ', end='')
            assert v1 > v0, f'Expected {key} to be higher with waning ({v1}) than without ({v0})'
            print(f'✓ ({v1} > {v0})')

        # Optionally plot
        if do_plot:
            msim.plot('overview-variant', rotation=30)

    return msims


def test_variants(do_plot=False):
    sc.heading('Testing variants...')
    nabs = []
    b117 = cv.variant('b117',         days=10, n_imports=20)
    p1   = cv.variant('beta',   days=20, n_imports=20)
    cust = cv.variant(label='Custom', days=40, n_imports=20, variant={'rel_beta': 2, 'rel_symp_prob': 1.6})
    sim  = cv.Sim(base_pars, use_waning=True, variants=[b117, p1, cust], analyzers=lambda sim: nabs.append(sim.people.nab.copy()))
    sim.run()

    if do_plot:
        nabs = np.array(nabs).sum(axis=1)
        pl.figure()
        pl.plot(nabs)
        pl.show()
        sim.plot('overview-variant')

    return sim


def test_vaccines(do_plot=False):
    sc.heading('Testing vaccines...')

    nabs = []
    p1 = cv.variant('beta',   days=20, n_imports=20)
    pfizer = cv.vaccinate(vaccine='pfizer', days=30)
    sim  = cv.Sim(base_pars, use_waning=True, variants=p1, interventions=pfizer, analyzers=lambda sim: nabs.append(sim.people.nab.copy()))
    sim.run()
    sim.shrink()

    if do_plot:
        nabs = np.array(nabs).sum(axis=1)
        pl.figure()
        pl.plot(nabs)
        pl.show()
        sim.plot('overview-variant')

    return sim


def test_vaccines_sequential(do_plot=False):
    sc.heading('Testing sequential vaccine...')

    n_days = 60
    p1 = cv.variant('beta', days=20, n_imports=20)
    num_doses = {i:(i**2)*(i%2==0) for i in np.arange(n_days)} # Test subtarget and fluctuating doses

    n_doses = []
    subtarget = dict(inds=np.arange(int(base_pars.pop_size//2)), vals=0.1)
    pfizer = cv.vaccinate_num(vaccine='pfizer', sequence='age', num_doses=num_doses, subtarget=subtarget)
    sim  = cv.Sim(base_pars, n_days=n_days, rescale=False, use_waning=True, variants=p1, interventions=pfizer, analyzers=lambda sim: n_doses.append(sim.people.doses.copy()))
    sim.run()

    n_doses = np.array(n_doses)

    if do_plot:
        fully_vaccinated = (n_doses == 2).sum(axis=1)
        first_dose = (n_doses == 1).sum(axis=1)
        pl.stackplot(sim.tvec, first_dose, fully_vaccinated)

        # Stacked bars by 10 year age

        # At the end of the simulation
        df = pd.DataFrame(n_doses.T)
        df['age_bin'] = np.digitize(sim.people.age,np.arange(0,100,10))
        df['fully_vaccinated'] = df[60]==2
        df['first_dose'] = df[60]==1
        df['unvaccinated'] = df[60]==0
        out = df.groupby('age_bin').sum()
        out[["unvaccinated", "first_dose","fully_vaccinated"]].plot(kind="bar", stacked=True)

        # Part-way through the simulation
        df = pd.DataFrame(n_doses.T)
        df['age_bin'] = np.digitize(sim.people.age,np.arange(0,100,10))
        df['fully_vaccinated'] = df[40]==2
        df['first_dose'] = df[40]==1
        df['unvaccinated'] = df[40]==0
        out = df.groupby('age_bin').sum()
        out[["unvaccinated", "first_dose","fully_vaccinated"]].plot(kind="bar", stacked=True)

    return sim


def test_two_vaccines(do_plot=False):
    sc.heading('Testing two vaccines...')

    p1 = cv.variant('beta',   days=20, n_imports=0)

    nabs = []
    vac1 = cv.vaccinate_num(vaccine='pfizer', sequence=[0], num_doses=1)
    vac2 = cv.vaccinate_num(vaccine='jj', sequence=[1], num_doses=1)

    sim  = cv.Sim(base_pars, n_days=1000, pop_size=2, pop_infected=0, variants=p1, interventions=[vac1, vac2], analyzers=lambda sim: nabs.append(sim.people.nab.copy()))

    # No infections, so suppress warnings
    with cv.options.context(warnings='print'):
        sim.run()
        print('↑↑↑ Should print warning about no infections')

    if do_plot:
        nabs = np.array(nabs).sum(axis=1)
        pl.figure()
        pl.plot(nabs)
        pl.show()

    return sim


def test_vaccine_target_eff():

    sc.heading('Testing vaccine with pre-specified efficacy...')
    target_eff_1 = 0.7
    target_eff_2 = 0.95

    default_pars = cv.parameters.get_vaccine_dose_pars(default=True)
    test_pars = dict(doses=2, interval=21, target_eff=[target_eff_1, target_eff_2])
    vacc_pars = sc.mergedicts(default_pars, test_pars)

    # construct analyzer to select placebo arm
    class placebo_arm(cv.Analyzer):
        def __init__(self, day, trial_size, **kwargs):
            super().__init__(**kwargs)
            self.day = day
            self.trial_size = trial_size
            return

        def initialize(self, sim=None):
            self.placebo_inds = []
            self.initialized = True
            return

        def apply(self, sim):
            if sim.t == self.day:
                eligible = cv.true(~np.isfinite(sim.people.date_exposed) & ~sim.people.vaccinated)
                self.placebo_inds = eligible[cv.choose(len(eligible), min(self.trial_size, len(eligible)))]
            return

    pars = dict(
        rand_seed  = 1, # Note: results may be sensitive to the random seed
        pop_size   = 20_000,
        beta       = 0.01,
        n_days     = 90,
        verbose    = -1,
        use_waning = True,
    )

    # Define vaccine arm
    trial_size = 4_000
    start_trial = 20

    def subtarget(sim):
        ''' Select people who are susceptible '''
        if sim.t == start_trial:
            eligible = cv.true(~np.isfinite(sim.people.date_exposed))
            inds = eligible[cv.choose(len(eligible), min(trial_size // 2, len(eligible)))]
        else:
            inds = []
        return {'vals': [1.0 for ind in inds], 'inds': inds}

    # Initialize
    vx = cv.vaccinate_prob(vaccine=vacc_pars, days=[start_trial], label='target_eff', prob=0.0, subtarget=subtarget)
    sim = cv.Sim(
        pars          = pars,
        interventions = vx,
        analyzers     = placebo_arm(day=start_trial, trial_size=trial_size // 2)
    )

    # Run
    sim.run()

    print('Vaccine efficiency:')
    results = sc.objdict()
    vacc_inds = cv.true(sim.people.vaccinated)  # Find trial arm indices, those who were vaccinated
    placebo_inds = sim['analyzers'][0].placebo_inds
    assert (len(set(vacc_inds).intersection(set(placebo_inds))) == 0)  # Check that there is no overlap
    # Calculate vaccine efficacy against infection
    VE_inf = 1 - (np.isfinite(sim.people.date_exposed[vacc_inds]).sum() /
                  np.isfinite(sim.people.date_exposed[placebo_inds]).sum())
    # Calculate vaccine efficacy against symptoms
    VE_symp = 1 - (np.isfinite(sim.people.date_symptomatic[vacc_inds]).sum() /
                   np.isfinite(sim.people.date_symptomatic[placebo_inds]).sum())
    # Calculate vaccine efficacy against severe disease
    VE_sev = 1 - (np.isfinite(sim.people.date_severe[vacc_inds]).sum() /
                  np.isfinite(sim.people.date_severe[placebo_inds]).sum())
    results['inf'] = VE_inf
    results['symp'] = VE_symp
    results['sev'] = VE_sev
    print(f'Against: infection: {VE_inf * 100:0.2f}%, symptoms: {VE_symp * 100:0.2f}%, severity: {VE_sev * 100:0.2f}%')

    # Check that actual efficacy is within 6 %age points of target
    errormsg = f'Expected VE to be about {target_eff_2}, but it is {VE_symp}. Check different random seeds; this test is highly sensitive.'
    assert round(abs(VE_symp-target_eff_2),2)<=0.1, errormsg

    nab_init = sim['vaccine_pars']['target_eff']['nab_init']
    boost = sim['vaccine_pars']['target_eff']['nab_boost']
    print(f'Initial NAbs: {nab_init}')
    print(f'Boost: {boost}')

    return sim


def test_decays(do_plot=False):
    sc.heading('Testing decay parameters...')

    n = 300
    x = pl.arange(n)

    pars = sc.objdict(

        nab_growth_decay = dict(
            func = cv.immunity.nab_growth_decay,
            length = n,
            growth_time = 21,
            decay_rate1 = 0.007,
            decay_time1 = 47,
            decay_rate2 = .002,
            decay_time2 = 106,
        ),

        nab_decay = dict(
            func = cv.immunity.nab_decay,
            length = n,
            decay_rate1 = 0.05,
            decay_time1 = 100,
            decay_rate2 = 0.002,
        ),

        exp_decay = dict(
            func = cv.immunity.exp_decay,
            length = n,
            init_val = 0.8,
            half_life = 100,
            delay = 20,
        ),

        linear_decay = dict(
            func = cv.immunity.linear_decay,
            length = n,
            init_val = 0.8,
            slope = 0.01,
        ),

        linear_growth = dict(
            func = cv.immunity.linear_growth,
            length = n,
            slope = 0.01,
        ),
    )

    # Calculate all the decays
    res = sc.objdict()
    for key,par in pars.items():
        func = par.pop('func')
        res[key] = func(**par)

    if do_plot:
        pl.figure(figsize=(12,8))
        for key,y in res.items():
            pl.semilogy(x, np.cumsum(y), label=key, lw=3, alpha=0.7)
        pl.legend()
        pl.show()

    res.x = x

    return res


def test_historical():
    pfizer = cv.prior_immunity(vaccine='pfizer', days=np.arange(-30, 0), prob=0.007)
    wave = cv.historical_wave(120, 0.05)
    sim1 = cv.Sim(base_pars, interventions=pfizer).run()
    sim2 = cv.Sim(base_pars, interventions=wave).run()
    with pytest.raises(RuntimeError):
        cv.Sim(base_pars, pop_scale=5, interventions=wave).run()
    with pytest.raises(ValueError):
        cv.Sim(base_pars, interventions=cv.historical_wave(120, 0.05, variant='invalid')).run()
    return sim1, sim2


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    sim1  = test_states()
    msim  = test_waning(do_plot=do_plot)
    sim2  = test_variants(do_plot=do_plot)
    sim3  = test_vaccines(do_plot=do_plot)
    sim4  = test_vaccines_sequential(do_plot=do_plot)
    sim5  = test_two_vaccines(do_plot=do_plot)
    sim6  = test_vaccine_target_eff()
    res   = test_decays(do_plot=do_plot)
    sims7 = test_historical()

    sc.toc(T)
    print('Done.')
