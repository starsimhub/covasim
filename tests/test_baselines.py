"""
Test that the current version of Covasim exactly matches
the baseline results.
"""

import numpy as np
import sciris as sc
import covasim as cv

do_plot = 1
do_save = 0
baseline_filename  = sc.thisdir(__file__, 'baseline.json')
benchmark_filename = sc.thisdir(__file__, 'benchmark.json')
parameters_filename = sc.thisdir(cv.__file__, 'regression', f'pars_v{cv.__version__}.json')
cv.options.set(interactive=False) # Assume not running interactively


def make_sim(use_defaults=False, do_plot=False, **kwargs):
    '''
    Define a default simulation for testing the baseline -- use hybrid and include
    interventions to increase coverage. If run directly (not via pytest), also
    plot the sim by default.
    '''

    # Define the interventions
    tp = cv.test_prob(start_day=20, symp_prob=0.1, asymp_prob=0.01)
    vx = cv.vaccinate_prob('pfizer', days=30, prob=0.1)
    cb = cv.change_beta(days=40, changes=0.5)
    ct = cv.contact_tracing(trace_probs=0.3, start_day=50)

    # Define the parameters
    pars = dict(
        use_waning    = True,         # Whether or not to use waning and NAb calculations
        pop_size      = 20e3,         # Population size
        pop_infected  = 100,          # Number of initial infections -- use more for increased robustness
        pop_type      = 'hybrid',     # Population to use -- "hybrid" is random with household, school,and work structure
        n_days        = 60,           # Number of days to simulate
        verbose       = 0,            # Don't print details of the run
        rand_seed     = 2,            # Set a non-default seed
        interventions = [cb, tp, ct, vx], # Include the most common interventions
    )
    pars = sc.mergedicts(pars, kwargs)

    # Create the sim
    if use_defaults:
        sim = cv.Sim()
    else:
        sim = cv.Sim(pars)

    # Optionally plot
    if do_plot:
        s2 = sim.copy()
        s2.run()
        s2.plot()

    return sim


def save_baseline():
    '''
    Refresh the baseline results. This function is not called during standard testing,
    but instead is called by the update_baseline script.
    '''

    print('Updating baseline values...')

    # Export default parameters
    s1 = make_sim(use_defaults=True)
    s1.export_pars(filename=parameters_filename) # If not different from previous version, can safely delete

    # Export results
    s2 = make_sim(use_defaults=False)
    s2.run()
    s2.to_json(filename=baseline_filename, keys='summary')

    print('Done.')

    return


def test_baseline():
    ''' Compare the current default sim against the saved baseline '''

    # Load existing baseline
    baseline = sc.loadjson(baseline_filename)
    old = baseline['summary']

    # Calculate new baseline
    new = make_sim()
    new.run()

    # Compute the comparison
    cv.diff_sims(old, new, die=True)

    return new


def test_benchmark(do_save=do_save, repeats=1, verbose=True):
    ''' Compare benchmark performance '''

    if verbose: print('Running benchmark...')
    previous = sc.loadjson(benchmark_filename)

    t_inits = []
    t_runs  = []

    def normalize_performance():
        ''' Normalize performance across CPUs -- simple Numpy calculation '''
        t_bls = []
        bl_repeats = 3
        n_outer = 10
        n_inner = 1e6
        for r in range(bl_repeats):
            t0 = sc.tic()
            for i in range(n_outer):
                a = np.random.random(int(n_inner))
                b = np.random.random(int(n_inner))
                a*b
            t_bl = sc.toc(t0, output=True)
            t_bls.append(t_bl)
        t_bl = min(t_bls)
        reference = 0.112 # Benchmarked on an Intel i9-8950HK CPU @ 2.90GHz
        ratio = reference/t_bl
        return ratio


    # Test CPU performance before the run
    r1 = normalize_performance()

    # Do the actual benchmarking
    for r in range(repeats):

        # Create the sim
        sim = make_sim(verbose=0)

        # Time initialization
        t0 = sc.tic()
        sim.initialize()
        t_init = sc.toc(t0, output=True)

        # Time running
        t0 = sc.tic()
        sim.run()
        t_run = sc.toc(t0, output=True)

        # Store results
        t_inits.append(t_init)
        t_runs.append(t_run)

    # Test CPU performance after the run
    r2 = normalize_performance()
    ratio = (r1+r2)/2
    t_init = min(t_inits)*ratio
    t_run  = min(t_runs)*ratio

    # Construct json
    n_decimals = 3
    json = {'time': {
                'initialize': round(t_init, n_decimals),
                'run':        round(t_run,  n_decimals),
                },
            'parameters': {
                'pop_size': sim['pop_size'],
                'pop_type': sim['pop_type'],
                'n_days':   sim['n_days'],
                },
            'cpu_performance': ratio,
            }

    if verbose:
        print('Previous benchmark:')
        sc.pp(previous)

        print('\nNew benchmark:')
        sc.pp(json)
    else:
        brief = sc.dcp(json['time'])
        brief['cpu_performance'] = json['cpu_performance']
        sc.pp(brief)

    if do_save:
        sc.savejson(filename=benchmark_filename, obj=json, indent=2)

    if verbose:
        print('Done.')

    return json



if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    json = test_benchmark(do_save=do_save, repeats=5) # Run this first so benchmarking is available even if results are different
    new  = test_baseline()
    make_sim(do_plot=do_plot)

    print('\n'*2)
    sc.toc(T)
    print('Done.')
