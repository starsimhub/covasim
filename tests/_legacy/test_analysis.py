'''
Tests for the analyzers and other analysis tools.
'''

import numpy as np
import sciris as sc
import covasim as cv
import pytest


#%% General settings

do_plot = 1 # Whether to plot when run interactively
cv.options.set(interactive=False) # Assume not running interactively

pars = dict(
    pop_size = 1000,
    verbose = 0,
)


#%% Define tests

def fname(label):
    ''' Name a figure -- complicated because needs a doubly nested dictionary '''
    return dict(fig_args=dict(num=label))


def test_snapshot():
    sc.heading('Testing snapshot analyzer')
    sim = cv.Sim(pars, analyzers=cv.snapshot('2020-04-04', '2020-04-14'))
    sim.run()
    snapshot = sim.get_analyzer()
    people1 = snapshot.snapshots[0]            # Option 1
    people2 = snapshot.snapshots['2020-04-04'] # Option 2
    people3 = snapshot.get('2020-04-14')       # Option 3
    people4 = snapshot.get(34)                 # Option 4
    people5 = snapshot.get()                   # Option 5

    assert people1 == people2, 'Snapshot options should match but do not'
    assert people3 != people4, 'Snapshot options should not match but do'
    return people5


def test_age_hist():
    sc.heading('Testing age histogram')

    day_list = ["2020-03-20", "2020-04-20"]
    age_analyzer = cv.age_histogram(days=day_list)
    sim = cv.Sim(pars, analyzers=age_analyzer)
    sim.run()

    # Checks to see that compute windows returns correct number of results
    sim.make_age_histogram() # Show post-hoc example
    agehist = sim.get_analyzer()
    agehist.compute_windows()
    agehist.get() # Not used, but check get
    agehist.get(day_list[1])
    assert len(agehist.window_hists) == len(day_list), "Number of histograms should equal number of days"

    # Check plot()
    if do_plot:
        plots = agehist.plot(windows=True, **fname('Age histogram'))
        assert len(plots) == len(day_list), "Number of plots generated should equal number of days"

    # Check daily age histogram
    daily_age = cv.daily_age_stats()
    sim = cv.Sim(pars, analyzers=daily_age)
    sim.run()

    return agehist


def test_daily_age():
    sc.heading('Testing daily age analyzer')
    sim = cv.Sim(pars, analyzers=cv.daily_age_stats())
    sim.run()
    daily_age = sim.get_analyzer()
    if do_plot:
        daily_age.plot(**fname('Daily age 1'))
        daily_age.plot(total=True, **fname('Daily age 2'))
    return daily_age


def test_daily_stats():
    sc.heading('Testing daily stats analyzer')
    ds = cv.daily_stats(days=['2020-04-04'], save_inds=True)
    sim = cv.Sim(pars, n_days=40, analyzers=ds)
    sim.run()
    daily = sim.get_analyzer()
    if do_plot:
        daily.plot(**fname('Daily stats'))
    return daily


def test_nab_hist():
    sc.heading('Testing NAb histogram analyzer')

    sim = cv.Sim(pars, analyzers=cv.nab_histogram())
    sim.run()
    nab_hist = sim.get_analyzer()
    if do_plot:
        nab_hist.plot(**fname('NAb histogram'))
    return nab_hist



def test_fit():
    sc.heading('Testing fitting function')

    # Create a testing intervention to ensure some fit to data
    tp = cv.test_prob(0.1)

    sim = cv.Sim(pars, rand_seed=1, interventions=tp, datafile="example_data.csv")
    sim.run()

    # Checking that Fit can handle custom input
    custom_inputs = {'custom_data':{'data':np.array([1,2,3]), 'sim':np.array([1,2,4]), 'weights':[2.0, 3.0, 4.0]}}
    fit1 = sim.compute_fit(custom=custom_inputs, compute=True)

    # Test that different seed will change compute results
    sim2 = cv.Sim(pars, rand_seed=2, interventions=tp, datafile="example_data.csv")
    sim2.run()
    fit2 = sim2.compute_fit(custom=custom_inputs)

    assert fit1.mismatch != fit2.mismatch, "Differences between fit and data remains unchanged after changing sim seed"

    # Test custom analyzers
    actual = np.array([1,2,4])
    predicted = np.array([1,2,3])

    def simple(actual, predicted, scale=2):
        return np.sum(abs(actual - predicted))*scale

    gof1 = cv.compute_gof(actual, predicted, normalize=False, as_scalar='sum')
    gof2 = cv.compute_gof(actual, predicted, estimator=simple, scale=1.0)
    assert gof1 == gof2
    with pytest.raises(Exception):
        cv.compute_gof(actual, predicted, skestimator='not an estimator')
    with pytest.raises(Exception):
        cv.compute_gof(actual, predicted, estimator='not an estimator')

    if do_plot:
        fit1.plot(**fname('Fit'))

    return fit1


def test_calibration():
    sc.heading('Testing calibration')

    pars = dict(
        verbose = 0,
        start_day = '2020-02-05',
        pop_size = 1e3,
        pop_scale = 4,
        interventions = [cv.test_prob(symp_prob=0.1)],
    )

    sim = cv.Sim(pars, datafile='example_data.csv')

    calib_pars = dict(
        beta      = [0.013, 0.005, 0.020],
        test_prob = [0.01, 0.00, 0.30]
    )

    def set_test_prob(sim, calib_pars):
        tp = sim.get_intervention(cv.test_prob)
        tp.symp_prob = calib_pars['test_prob']
        return sim

    calib = sim.calibrate(calib_pars=calib_pars, custom_fn=set_test_prob, n_trials=10, n_workers=1)
    calib.plot_sims(to_plot=['cum_deaths', 'cum_diagnoses'], **fname('Calibration sims'))
    calib.plot_trend()

    assert calib.after.fit.mismatch < calib.before.fit.mismatch

    return calib


def test_transtree():
    sc.heading('Testing transmission tree')

    sim = cv.Sim(pars, pop_size=100)
    sim.run()

    transtree = sim.make_transtree()
    print(len(transtree))
    if do_plot:
        transtree.plot(**fname('Transmission tree'))
        transtree.animate(animate=False)
        transtree.plot_histograms(**fname('Transmission histograms'))

    # Try networkx, but don't worry about failures
    try:
        tt = sim.make_transtree(to_networkx=True)
        tt.r0()
    except ImportError as E:
        print(f'Could not test conversion to networkx ({str(E)})')

    return transtree



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    snapshot  = test_snapshot()
    agehist   = test_age_hist()
    daily_age = test_daily_age()
    daily     = test_daily_stats()
    nab_hist  = test_nab_hist()
    fit       = test_fit()
    calib     = test_calibration()
    transtree = test_transtree()

    print('\n'*2)
    sc.toc(T)
    print('Done.')
