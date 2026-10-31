"""
Tests for variants and cross-immunity

See also test_immunity.py for tests of waning immunity and vaccines.
"""

#%% Imports and settings
import numpy as np
import sciris as sc
import covasim as cv

do_plot = 1
cv.options.set(interactive=False) # Assume not running interactively

base_pars = dict(
    pop_size = 5000,
    pop_infected = 50,
    n_days = 60,
    verbose = 0,
)


#%% Define the tests

def test_variant_results(do_plot=False):
    sc.heading('Testing results by variant')

    alpha = cv.variant('alpha', days=10, n_imports=20)
    delta = cv.variant('delta', days='2020-03-21', n_imports=30) # Days can also be dates; this is day 20
    sim = cv.Sim(base_pars, variants=[alpha, delta])
    sim.run()
    res = sim.results

    # The wild type is always the first variant
    assert sim['n_variants'] == 3
    assert sim['variant_map'] == {0:'wild', 1:'alpha', 2:'delta'}
    assert sim['immunity'].shape == (3, 3) # The cross-immunity matrix

    # Each variant is imported on the right day, in the results at the end of that day (index 0 is the start of day 0)
    assert res['n_imports'][11] == 20
    assert res['n_imports'][21] == 30
    assert res['n_imports'][:].sum() == 50

    # Results by variant have one row per day and one column per variant, and can be accessed by name
    cum_infections = res['variant']['cum_infections_by_variant']
    assert cum_infections.shape == (sim.npts, 3)
    assert np.all(cum_infections[-1] > 0) # Every variant has infected people
    assert cum_infections.delta[-1] == cum_infections[-1, 2]

    # Results by variant add up to the totals
    for key in ['new_infections', 'new_symptomatic', 'new_severe', 'n_exposed', 'n_infectious', 'prevalence', 'incidence']:
        by_variant = res['variant'][f'{key}_by_variant']
        assert np.allclose(by_variant[:].sum(axis=1), res[key]), f'{key} by variant should sum to the total'

    # People are only infected with one variant at a time
    ppl = sim.people
    assert np.array_equal(ppl.exposed_variant[ppl.infectious], ppl.infectious_variant[ppl.infectious])

    if do_plot:
        sim.plot('variants')

    return sim


def test_variant_pars():
    sc.heading('Testing variant parameters')

    # A variant can be supplied as a string, in which case it's there from the start
    sim = cv.Sim(base_pars, variants='beta')
    sim.run()
    assert sim['variant_map'] == {0:'wild', 1:'beta'}
    assert sim.results['variant']['cum_infections_by_variant'].beta[-1] > 0

    # Custom variants that are more transmissible infect more people
    n_infections = []
    for rel_beta in [0.5, 2.5]:
        custom = cv.variant({'rel_beta':rel_beta}, label='custom', days=0, n_imports=50)
        sim = cv.Sim(base_pars, pop_infected=0, variants=custom)
        sim.run()
        n_infections.append(sim.results['variant']['cum_infections_by_variant'].custom[-1])
    assert n_infections[0] < n_infections[1]

    return sim


def test_cross_immunity():
    sc.heading('Testing cross-immunity')

    # Without waning immunity, people who have recovered from one variant can only be infected with a different one
    sims = sc.objdict()
    for label,cross_imm in dict(low=0.1, high=0.9).items():
        immunity = np.array([[1.0, cross_imm], [cross_imm, 1.0]]) # Protection against each variant from each other variant
        variant = cv.variant({'rel_beta':1.5}, label='new', days=25, n_imports=30)
        sim = cv.Sim(base_pars, n_days=100, use_waning=False, variants=variant, connectors=cv.CrossImmunity(immunity=immunity))
        sim.run()
        sims[label] = sim

    # With less cross-immunity, more people are reinfected
    assert sims.low.summary['cum_reinfections'] > sims.high.summary['cum_reinfections'] > 0
    assert sims.low.results['variant']['cum_infections_by_variant'].new[-1] > sims.high.results['variant']['cum_infections_by_variant'].new[-1]

    # With only one variant, no one is reinfected
    sim = cv.Sim(base_pars, n_days=100, use_waning=False)
    sim.run()
    assert sim.summary['cum_reinfections'] == 0

    return sims



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    sim1 = test_variant_results(do_plot=do_plot)
    sim2 = test_variant_pars()
    sims = test_cross_immunity()

    sc.toc(T)
    print('Done.')
