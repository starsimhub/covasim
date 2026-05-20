'''
Testing the effect of testing interventions in Covasim
'''

#%% Imports and settings
import os
import sciris as sc
import covasim as cv

do_plot   = 1
do_show   = 1
do_save   = 0
debug     = 1
keep_sims = 0
fig_paths = [f'results/testing_scen_{i}.png' for i in range(7)]


def test_interventions(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test of testing interventions')


    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1
    base_pars = {
      'pop_size': 1000,
      'pop_type': 'hybrid',
      }

    base_sim = cv.Sim(base_pars) # create sim object
    n_people = base_sim['pop_size']
    npts = base_sim.npts

    # Define overall testing assumptions
    # Remember that this is the daily % of the population that gets tested. S Korea (one of the highest-testing places) tested
    # an average of 10000 people/day over March, or 270,000 in total. This is ~200 people per million every day (0.02%)....
    max_optimistic_testing = 0.1 # ... which means that this is an artificially high number, for testing purposes only!!
    optimistic_daily_tests = [max_optimistic_testing*n_people]*npts # Very best-case scenario for asymptomatic testing

    # Define the scenarios
    scenarios = {
        'baseline': {
          'name':'Status quo, no testing',
          'pars': {
              'interventions': None,
              }
          },
        'test_skorea': {
          'name':'Assuming South Korea testing levels of 0.02% daily (untargeted); isolate positives',
          'pars': {
              'interventions': cv.test_num(daily_tests=optimistic_daily_tests)
              }
          },
        'floating': {
            'name': 'Test with constant probability based on symptoms',
            'pars': {
                'interventions': cv.test_prob(symp_prob=max_optimistic_testing, asymp_prob=0.0)
                }
        },
        'sequence': {
            'name': 'Historical switching to probability',
            'pars': {
                'interventions': cv.sequence(days=[10, 51], interventions=[
                    cv.test_num(daily_tests=optimistic_daily_tests),
                    cv.test_prob(symp_prob=0.2, asymp_prob=0.002),
                ])
            }
        },

    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    to_plot = ['cum_infections', 'n_infectious', 'new_tests', 'new_diagnoses']
    fig_args = dict(figsize=(20, 24))

    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=fig_path, interval=7, fig_args=fig_args, to_plot=to_plot)

    return scens


def test_turnaround(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test impact of reducing delay time for getting test results')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1
    base_pars = {
      'pop_size': 1000,
      'pop_type': 'hybrid',
      }

    base_sim = cv.Sim(base_pars) # create sim object
    n_people = base_sim['pop_size']
    npts = base_sim.npts

    # Define overall testing assumptions
    testing_prop = 0.1 # Assumes we could test 10% of the population daily (!!)
    daily_tests = [testing_prop*n_people]*npts # Number of daily tests

    # Define the scenarios
    scenarios = {
        f'{d}dayturnaround': {
            'name':f'Symptomatic testing with {d} days to get results',
            'pars': {
                'interventions': cv.test_num(daily_tests=daily_tests, test_delay=d)
            }
        } for d in range(1, 3+1, 2)
    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    to_plot = ['cum_infections', 'n_infectious', 'new_tests', 'new_diagnoses']
    fig_args = dict(figsize=(20, 24))

    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=fig_path, interval=7, fig_args=fig_args, to_plot=to_plot)

    return scens


def test_tracedelay(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test impact of reducing delay time for finding contacts of positives')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1
    base_pars = {
      'pop_size': 1000,
      'pop_type': 'hybrid',
      }

    base_sim = cv.Sim(base_pars) # create sim object
    base_sim['n_days'] = 50
    base_sim['beta'] = 0.03 # Increase beta

    n_people = base_sim['pop_size']
    npts = base_sim.npts


    # Define overall testing assumptions
    testing_prop = 0.1 # Assumes we could test 10% of the population daily (way too optimistic!!)
    daily_tests = [testing_prop*n_people]*npts # Number of daily tests

    # Define the scenarios
    scenarios = {
        'lowtrace': {
            'name': 'Poor contact tracing',
            'pars': {
                'quar_eff': {'h': 1, 's': 0.5, 'w': 0.5, 'c': 0.25},
                'quar_period': 7,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 0, 's': 0, 'w': 0, 'c': 0},
                        trace_time  = {'h': 1, 's': 7,   'w': 7,   'c': 7})]
            }
        },
        'modtrace': {
            'name': 'Moderate contact tracing',
            'pars': {
                'quar_eff': {'h': 0.75, 's': 0.25, 'w': 0.25, 'c': 0.1},
                'quar_period': 10,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.5, 'c': 0.1},
                        trace_time  = {'h': 0,  's': 3,  'w': 3,   'c': 8})]
            }
        },
        'hightrace': {
            'name': 'Fast contact tracing',
            'pars': {
                'quar_eff': {'h': 0.5, 's': 0.1, 'w': 0.1, 'c': 0.1},
                'quar_period': 14,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.8, 'c': 0.2},
                        trace_time  = {'h': 0, 's': 1,   'w': 1,   'c': 5})]
            }
        },
        'alltrace': {
            'name': 'Same-day contact tracing',
            'pars': {
                'quar_eff': {'h': 0.0, 's': 0.0, 'w': 0.0, 'c': 0.0},
                'quar_period': 21,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 1, 'w': 1, 'c': 1},
                        trace_time  = {'h': 0, 's': 1, 'w': 1, 'c': 2})]
            }
        },
    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        to_plot = [
            'cum_infections',
            'cum_recoveries',
            'new_infections',
            'n_quarantined',
            'new_quarantined'
        ]
        fig_args = dict(figsize=(24,16))
        scens.plot(do_save=do_save, do_show=do_show, to_plot=to_plot, fig_path=fig_path, n_cols=2, fig_args=fig_args)

    return scens



def test_beta_edges(do_plot=False, do_show=True, do_save=False, fig_path=None):

    pars = dict(
        pop_size=1000,
        pop_infected=20,
        pop_type='hybrid',
        )

    start_day = 25 # Day to start the intervention
    end_day   = 40 # Day to end the intervention
    change    = 0.3 # Amount of change

    sims = sc.objdict()
    sims.b = cv.Sim(pars) # Beta intervention
    sims.e = cv.Sim(pars) # Edges intervention

    beta_interv = cv.change_beta(days=[start_day, end_day], changes=[change, 1.0])
    edge_interv = cv.clip_edges(start_day=start_day, end_day=end_day, change=change, verbose=True)
    sims.b.update_pars(interventions=beta_interv)
    sims.e.update_pars(interventions=edge_interv)

    for sim in sims.values():
        sim.run()
        if do_plot:
            sim.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)
            sim.plot_result('r_eff')

    return sims


def test_beds(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test of bed capacity estimation')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 2
    verbose = 1

    basepars = {'pop_size': 1000}
    metapars = {'n_runs': n_runs}

    sim = cv.Sim()

    # Define the scenarios
    scenarios = {
        'baseline': {
          'name': 'No bed constraints',
          'pars': {
              'pop_infected': 100
          }
        },
        'bedconstraint': {
            'name': 'Only 50 beds available',
            'pars': {
                'pop_infected': 100,
                'n_beds': 50,
            }
        },
        'bedconstraint2': {
            'name': 'Only 10 beds available',
            'pars': {
                'pop_infected': 100,
                'n_beds': 10,
            }
        },
    }

    scens = cv.Scenarios(sim=sim, basepars=basepars, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        to_plot = sc.odict({
            'Cumulative deaths':   'cum_deaths',
            'People needing beds / beds': 'bed_capacity',
            'Number of cases requiring hospitalization': 'n_severe',
            'Number of cases requiring ICU': 'n_critical',
        })
        scens.plot(to_plot=to_plot, do_save=do_save, do_show=do_show, fig_path=fig_path)

    return scens


def test_borderclosure(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test effect of border closures')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 2
    verbose = 1

    basepars = {'pop_size': 1000}
    basepars = {'n_imports': 5}
    metapars = {'n_runs': n_runs}

    sim = cv.Sim()

    # Define the scenarios
    scenarios = {
        'baseline': {
            'name': 'No border closures',
            'pars': {
            }
        },
        'borderclosures_day10': {
            'name': 'Close borders on day 10',
            'pars': {
                'interventions': [cv.dynamic_pars({'n_imports': {'days': 10, 'vals': 0}})]
            }
        },
    }

    scens = cv.Scenarios(sim=sim, basepars=basepars, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)

    return scens



def test_presumptive_quar(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test impact of presumptive quarantine')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1
    base_pars = {
      'pop_size'    : 10000,
      'pop_infected':    10,
      'pop_type'    : 'synthpops',
      'n_days'      : 150,
      'beta'        : 0.02,
      'quar_period' : 14,
      'quar_factor' : {'h': 0.5, 's': 0.1, 'w': 0.1, 'c': 0.1}, # Bidirectional becaues on transmit and receive, e.g. in home
      'iso_factor'  : {'h': 0.5, 's': 0.1, 'w': 0.1, 'c': 0.1}, # Worried about diagnosed while in quarantine - double impact!
    }

    base_sim = cv.Sim(base_pars) # create sim object

    # DEFINE INTERVENTIONS
    test_delay = 2

    testing = cv.test_prob(symp_prob=0.03, asymp_prob=0.001, symp_quar_prob=0.5, asymp_quar_prob=0.1, test_delay=test_delay)
    isokwargs = {
        'start_day'  : 30,
        'trace_probs': {'h': 0, 's': 0, 'w': 0, 'c': 0},
        'trace_time' : {'h': 0, 's': 0,   'w': 0,   'c': 0}
    }
    ctkwargs = {
        'start_day'  : 30,
        'trace_probs': {'h': 0.9, 's': 0.7, 'w': 0.7, 'c': 0.2},
        'trace_time' : {'h': 0, 's': 2,   'w': 2,   'c': 3}
    }

    baseline = {
        'name': 'Baseline',
        'pars': {
            'interventions': [ testing, ]
        }
    }

    contact_tracing = {
        'name': 'Contact tracing',
        'pars': {
            'interventions': [
                testing,
                cv.contact_tracing(**ctkwargs)]
        }
    }

    quar_on_test = {
        'name': 'Quarantine on test',
        'pars': {
            'interventions': [
                testing,
                cv.contact_tracing(**isokwargs, presumptive=True, test_delay=test_delay)]
        }
    }

    # Only if trace_time < test_delay will presumptive tracing follow contacts of negatives to quarantine
    presumptive_ct = {
        'name': 'Contact tracing (presumptive)',
        'pars': {
            'interventions': [
                testing,
                cv.contact_tracing(**ctkwargs, presumptive=True, test_delay=test_delay)]
        }
    }

    # Define the scenarios
    scenarios = {
        'Baseline': baseline,
        'Quarantine on test': quar_on_test,
        'Trace on diagnosis': contact_tracing,
        'Trace on test': presumptive_ct,
    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        to_plot = {
            'Number of people currently infectious': [
                'n_infectious',
            ],
            'Number of people in quarantine': [
                'n_quarantined',
            ],
            'Number of newly quarantined': [
                'new_quarantined',
            ],
            'Number diagnosed': [
                'n_diagnosed',
            ],
            'Cum inf': [
                'cum_infections',
            ],
            'Re': [
                'r_eff',
            ],
        }

        fig_args = dict(figsize=(24,16))
        scens.plot(do_save=do_save, do_show=do_show, to_plot=to_plot, fig_path=fig_path, n_cols=2, fig_args=fig_args)

    return scens


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    scens1 = test_interventions(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_paths[0])
    scens2 = test_turnaround(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_paths[1])
    scens3 = test_tracedelay(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_paths[2])
    sims = test_beta_edges(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_paths[3])
    bed_scens = test_beds(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_paths[4])
    border_scens = test_borderclosure(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_paths[5])
    scens4 = test_presumptive_quar(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_paths[6])

    for path in fig_paths:
        if os.path.exists(path):
            print(f'Removing {path}')
            os.remove(path)

    sc.toc()


print('Done.')
