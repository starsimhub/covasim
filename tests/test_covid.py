"""
Tests for the COVID module: disease progression and transmission
"""

#%% Imports and settings
import numpy as np
import sciris as sc
import covasim as cv

do_plot = 1
cv.options.set(interactive=False) # Assume not running interactively


#%% Define the tests

def test_prognoses():
    sc.heading('Testing prognoses')

    # Infect half the population and check what will happen to them
    n = 10_000
    sim = cv.Sim(pop_size=2*n, pop_infected=0, verbose=0)
    sim.initialize()
    ppl = sim.people
    ppl.infect(np.arange(n))

    infected    = cv.defined(ppl.date_exposed)
    symptomatic = cv.defined(ppl.date_symptomatic)
    severe      = cv.defined(ppl.date_severe)
    critical    = cv.defined(ppl.date_critical)
    dead        = cv.defined(ppl.date_dead)
    recovered   = cv.defined(ppl.date_recovered)

    # Everyone becomes infectious, and each stage of disease has fewer people than the one before
    assert len(cv.defined(ppl.date_infectious)) == len(infected) == n
    assert n > len(symptomatic) > len(severe) > len(critical) > len(dead) > 0

    # Each stage happens after the one before
    assert np.all(ppl.date_infectious[symptomatic] <= ppl.date_symptomatic[symptomatic])
    assert np.all(ppl.date_symptomatic[severe]     <= ppl.date_severe[severe])
    assert np.all(ppl.date_severe[critical]        <= ppl.date_critical[critical])
    assert np.all(ppl.date_critical[dead]          <= ppl.date_dead[dead])

    # Everyone either recovers or dies, and only people who become critical die
    assert len(recovered) + len(dead) == n
    assert len(np.intersect1d(recovered, dead)) == 0
    assert np.all(np.isin(dead, critical))

    # Older people are more likely to die
    assert ppl.age[dead].mean() > ppl.age[recovered].mean()

    return sim


def test_transmission(do_plot=False):
    sc.heading('Testing transmission')

    pars = dict(pop_size=5000, pop_infected=50, n_days=60, use_waning=False, verbose=0)
    s0 = cv.Sim(pars, beta=0, label='No transmission').run()
    s1 = cv.Sim(pars, label='Default').run()
    s2 = cv.Sim(pars, beta=0.03, label='High transmission').run()

    # More transmission means more infections; with none, only the initial infections
    assert s0.summary['cum_infections'] == pars['pop_infected']
    assert s0.summary['cum_infections'] < s1.summary['cum_infections'] < s2.summary['cum_infections']

    # The v3 results match the Starsim result, except that cum_infections also includes the initial infections
    res = s2.diseases.covid.results
    assert np.array_equal(s2.results['new_infections'], res.infection)
    assert s2.summary['cum_infections'] == res.infection.cum[-1] + pars['pop_infected']

    # People aren't infectious as soon as they are exposed
    ppl = s1.people
    assert np.all(ppl.date_infectious[ppl.infectious] > ppl.date_exposed[ppl.infectious])

    # Without waning immunity, people who have recovered can't be infected again
    assert ppl.recovered.sum() > 0
    assert not (ppl.recovered & ppl.susceptible).any()

    # Some people are much more infectious than others, but on average it's 1
    rel_trans = s1.diseases.covid.rel_trans_base
    assert np.isclose(rel_trans.mean(), 1, atol=0.1)
    assert rel_trans.std() > 0.5
    s3 = cv.Sim(pars, n_days=1, beta_dist=dict(dist='uniform', par1=0.9, par2=1.1)).init()
    assert s3.diseases.covid.rel_trans_base.min() >= 0.9 # Any v3 distribution can be used for beta_dist

    # With testing, people who are reinfected can be diagnosed again
    tp = cv.test_prob(symp_prob=1.0, asymp_prob=1.0, test_delay=0)
    s4 = cv.Sim(pars, pop_size=2000, pop_infected=1000, n_days=90, beta=0.03, use_waning=True, interventions=tp,
                variants=cv.variant('beta', days=30, n_imports=200)).run()
    covid = s4.diseases.covid
    assert s4.summary['cum_reinfections'] > 0
    inf = covid.infectious.uids
    assert not np.any(covid.date_diagnosed[inf] < covid.ti_infected[inf]) # No one has a diagnosis left over from an earlier infection

    if do_plot:
        cv.MultiSim([s0, s1, s2]).plot()

    return s1



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    sim1 = test_prognoses()
    sim2 = test_transmission(do_plot=do_plot)

    sc.toc(T)
    print('Done.')
