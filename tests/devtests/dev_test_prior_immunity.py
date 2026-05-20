import covasim as cv
import numpy as np
import pytest

###################################################
## Vaccination examples
###################################################

def example_estimate_prob():
    import matplotlib.pyplot as plt

    # check single day estimation campaign:
    duration, coverage = (1, 0.3)
    p = cv.historical_vaccinate_prob.estimate_prob(duration=duration, coverage=coverage)
    assert(np.isclose(p, coverage))
    print('single day campaign:', coverage, '\simeq', p)

    # plot campaign coverages as a function of daily probability and campaign duration
    durations = np.arange(1, 60)
    probs = [0.0008, 0.008, 0.08, 0.8]
    plt.figure()
    for ix, prob in enumerate(probs):
        coverage = cv.historical_vaccinate_prob.NB_cdf(durations-1, 1 - prob)
        plt.plot(durations, coverage, label=prob)
    plt.legend()
    plt.xlabel('Duration of Campaign')
    plt.ylabel('Fraction of population vaccinated')
    plt.show()


def examplev0():
    pfizer = cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-30, 0), prob=0.007)
    cv.Sim(interventions=pfizer, use_waning=True).run().plot()


def examplev1():
    # length of our base campaign
    duration = 30
    # estimate per-day probability needed for a coverage of 30%
    prob = cv.historical_vaccinate_prob.estimate_prob(duration=duration, coverage=0.30)
    print('using per-day probability of ', prob)

    pfizer = cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-duration, 0), prob=prob)
    sim = cv.Sim(interventions=pfizer, use_waning=True)

    sim.run()

    to_plot = cv.get_default_plots(kind='sim')
    to_plot['Total counts'] += ['cum_vaccinated']
    to_plot['Daily counts'] += ['new_doses']
    sim.plot(to_plot=to_plot)


def examplev2():
    pars = {'use_waning': True}
    variants = [cv.variant('b117', days=30, n_imports=10)]
    sim = cv.Sim(pars=pars, variants=variants)

    # length of our base campaign
    duration = 30
    # estimate per-day probability needed for a coverage of 30%
    prob = cv.historical_vaccinate_prob.estimate_prob(duration=duration, coverage=0.30)
    print('using per-day probability of ', prob)

    # estimate per-day probability needed for a coverage of 30%
    prob2 = cv.historical_vaccinate_prob.estimate_prob(duration=2*duration, coverage=0.30)

    scenarios = {
        'base':{
            'name': 'baseline',
            'pars': {}
        },
        'scen1':{
            'name': 'historical_vaccinate',
            'pars': {
                'interventions':[cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-duration, 0), prob=prob)]
            }
        },
        'scen2': {
            'name': 'vaccinate',
            'pars': {
                'interventions': [cv.vaccinate_prob(vaccine='pfizer',days=np.arange(0, 30), prob=prob)]
            }
        },
        'scen3': {
            'name': 'historical_vaccinate into sim',
            'pars': {
                'interventions': [cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-30, 30), prob=prob2)]
            }
        },
    }

    scens = cv.Scenarios(sim=sim, scenarios=scenarios)

    scens.run()

    scens.plot()


def examplev3():
    pars = {'use_waning': True}
    variants = [cv.variant('b117', days=30, n_imports=10)]
    sim = cv.Sim(pars=pars, variants=variants)

    # length of our base campaign
    duration = 30
    # estimate per-day probability needed for a coverage of 30%
    prob = cv.historical_vaccinate_prob.estimate_prob(duration=duration, coverage=0.30)
    print('using per-day probability of ', prob)

    scenarios = {
        'scen1':{
            'name': 'both doses',
            'pars': {
                'interventions':[cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-duration, 0), prob=prob)]
            }
        },
        'scen3': {
            'name': 'first dose only',
            'pars': {
                'interventions': [cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-duration, 0), prob=prob, compliance=[1.0, 0.0])]
            }
        },
    }

    scens = cv.Scenarios(sim=sim, scenarios=scenarios)
    scens.run()
    to_plot = cv.get_default_plots(kind='scenarios')
    to_plot.pop(2)
    to_plot.update({'Cumulative doses': ['cum_vaccinated', 'cum_doses']})
    scens.plot(to_plot=to_plot)


def examplev4():
    pfizer = cv.historical_vaccinate_prob(vaccine='pfizer', days=[-360], prob=0.5)
    sim = cv.Sim(pars={'n_days':1}, interventions=pfizer, use_waning=True,
                 analyzers=cv.nab_histogram(days=[0], edges=np.linspace(-4,2,12+1)))
    sim.run()

    sim['analyzers'][0].plot()


###################################################
# Example wave examples
###################################################

def examplew0():
    cv.Sim(use_waning=True, interventions=[cv.historical_wave(120, 0.05)]).run().plot()


def examplew1():
    # run single sim
    pars = {'use_waning': True, 'rand_seed':1}
    variants = [cv.variant('delta', days=15, n_imports=10)]

    sim = cv.Sim(pars=pars, variants=variants)
    sim['interventions'] += [cv.historical_wave(variant='wild', prob=[0.05, 0.05], days_prior=[150, 50])]
    sim.run()
    sim.plot();
    sim.plot('variants')


def examplew2():
    pars = {'use_waning': True, 'n_days':180}
    sim = cv.Sim(pars=pars)
    scenarios = {
        'base':{
            'name': 'baseline',
            'pars': {}
        },
        'scen1':{
            'name': '1 wave',
            'pars': {
                'interventions':[cv.historical_wave(prob=0.25, days_prior=50)]
            }
        },
        'scen2': {
            'name': '2 waves',
            'pars': {
                'interventions': [cv.historical_wave(prob=[0.20, 0.05], days_prior=[360, 50])]
            }
        }
    }

    metapars = cv.make_metapars()
    metapars.update({'n_runs':8})
    scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars=metapars)

    scens.run()

    scens.plot()

def examplew3():
    pars = {'use_waning': True, 'n_days':1}
    sim = cv.Sim(pars=pars)
    sim['interventions'] += [cv.historical_wave(prob=[0.05, 0.05], days_prior=[100, 50])]

    sim['analyzers'] += [cv.nab_histogram(days=[0])]
    sim.run()
    sim.plot('variants')
    sim['analyzers'][0].plot()


def examplew4():
    with pytest.raises(ValueError):
        cv.Sim(use_waning=True, interventions=[cv.historical_wave(120, 0.05, variant='delta')]).run().plot()

###################################################
# Example prior immunity
###################################################

def examplep0():
    intv = cv.prior_immunity(vaccine='pfizer', days=[-30], prob=0.7)
    cv.Sim(pars={'use_waning':True}, interventions=intv).run().plot()


def examplep1():
    intv = cv.prior_immunity(120, 0.05)
    cv.Sim(pars={'use_waning':True}, interventions=intv).run().plot()


if __name__ == "__main__":

    ## PRIOR IMMUNITY EXAMPLE
    # use prior_immunity to add historical vaccination
    examplep0()

    # use prior_immunity to add historical_wave
    examplep1()

    ## VACCINATION EXAMPLES

    # basic example
    examplev0()
    # single vaccine campaign example
    examplev1()

    # compare vaccinate and historical vaccinate
    examplev2()
    # compare vaccinate and historical vaccinate
    examplev3()

    # example using NAb histogram
    examplev4()
    # examples using estimate_prob
    example_estimate_prob()


    ## PREVIOUS WAVE EXAMPLES
    # basic example
    examplew0()

    # 2 wave example
    examplew1()

    # multi-wave comparison
    examplew2()

    # example using NAb histogram
    examplew3()

    # Testing imprinting variant that is not circulatnig
    examplew4()