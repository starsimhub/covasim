'''
Tests of the numerical utilities for the model.
'''

#%% Imports and settings
import pytest
import numpy as np
import numba as nb
import pylab as pl
import sciris as sc
import covasim as cv

do_plot = 0
cv.options.set(interactive=False) # Assume not running interactively


#%% Define the tests

def test_rand():

    default_seed = 1

    @nb.njit
    def numba_rand():
        return np.random.rand()

    # Check that two consecutive numbers don't match
    cv.set_seed(default_seed)
    a = np.random.rand()
    b = np.random.rand()
    v = numba_rand()
    w = numba_rand()
    assert a != b
    assert v != w

    # Check that after resetting the seed, they do
    cv.set_seed(default_seed)
    c = np.random.rand()
    x = numba_rand()
    assert a == c
    assert v == x

    # Check that resetting with no argument, they don't again
    cv.set_seed()
    d = np.random.rand()
    y = numba_rand()
    cv.set_seed()
    e = np.random.rand()
    z = numba_rand()
    assert d != e
    assert y != z

    return a


def test_samples(do_plot=False, verbose=True):
    sc.heading('Samples distribution')

    n = 200_000
    nbins = 100

    # Warning, must match utils.py!
    choices = [
        'uniform',
        'normal',
        'lognormal',
        'normal_pos',
        'normal_int',
        'lognormal_int',
        'poisson',
        'neg_binomial'
        ]

    if do_plot:
        pl.figure(figsize=(20,14))

    # Run the samples
    nchoices = len(choices)
    nsqr, _ = sc.get_rows_cols(nchoices)
    results = sc.objdict()
    mean = 11
    std = 7
    low = 3
    high = 9
    normal_dists = ['normal', 'normal_pos', 'normal_int', 'lognormal', 'lognormal_int']
    for c,choice in enumerate(choices):
        kw = {}
        if choice in normal_dists:
            par1 = mean
            par2 = std
        elif choice == 'neg_binomial':
            par1 = mean
            par2 = 1.2
            kw['step'] = 0.1
        elif choice == 'poisson':
            par1 = mean
            par2 = 0
        elif choice == 'uniform':
            par1 = low
            par2 = high
        else:
            errormsg = f'Choice "{choice}" not implemented'
            raise NotImplementedError(errormsg)

        # Compute
        results[choice] = cv.sample(dist=choice, par1=par1, par2=par2, size=n, **kw)

        # Optionally plot
        if do_plot:
            pl.subplot(nsqr, nsqr, c+1)
            plotbins = np.unique(results[choice]) if (choice=='poisson' or '_int' in choice) else nbins
            pl.hist(x=results[choice], bins=plotbins, width=0.8)
            pl.title(f'dist={choice}, par1={par1}, par2={par2}')

    with pytest.raises(NotImplementedError):
        cv.sample(dist='not_found')

    # Do statistical tests
    tol = 1/np.sqrt(n/50/len(choices)) # Define acceptable tolerance -- broad to avoid false positives

    def isclose(choice, tol=tol, **kwargs):
        key = list(kwargs.keys())[0]
        ref = list(kwargs.values())[0]
        npfunc = getattr(np, key)
        value = npfunc(results[choice])
        msg = f'Test for {choice:14s}: expecting {key:4s} = {ref:8.4f} ± {tol*ref:8.4f} and got {value:8.4f}'
        if verbose:
            print(msg)
        assert np.isclose(value, ref, rtol=tol), msg
        return True

    # Normal
    for choice in normal_dists:
        isclose(choice, mean=mean)
        if all([k not in choice for k in ['_pos', '_int']]): # These change the variance
            isclose(choice, std=std)

    # Negative binomial
    isclose('neg_binomial', mean=mean)

    # Poisson
    isclose('poisson', mean=mean)
    isclose('poisson', var=mean)

    # Uniform
    isclose('uniform', mean=(low+high)/2)

    return results



def test_choose():
    sc.heading('Choose people')
    x1 = cv.choose(10, 5)
    with pytest.raises(Exception):
        cv.choose_w(10, 5) # Requesting mroe people than are available
    print(f'Uniform sample from 0-9: {x1}')
    return x1


def test_choose_w():
    sc.heading('Choose weighted people')
    n = 100
    samples = 5
    lin = np.arange(n)
    lin = lin/lin.sum()
    x0 = cv.choose_w([0.01]*n, samples)
    x1 = cv.choose_w(lin, samples)
    x2 = cv.choose_w([1, 0, 0, 0, 0], 1)
    x3 = cv.choose_w([0.5, 0.5, 0, 0, 0], 1)
    assert x2[0] == 0
    assert x3[0] in [0,1]
    assert len(x0) == len(x1) == samples
    with pytest.raises(Exception):
        cv.choose_w([0.5, 0.5], 10) # Requesting mroe people than are available
    print(f'Uniform sample 0-99: x0 = {x0}, mean {x0.mean()}')
    print(f'Weighted sample 0-99: x1 = {x1}, mean {x1.mean()}')
    print(f'All weight on 0: x2 = {x2}')
    print(f'All weight on 0 or 1: x3 = {x3}')
    return x1


def test_indexing():

    # Definitions
    farr = np.array([1.5,0,0,1,1,0]) # Float array
    barr = np.array(farr, dtype=bool) # Boolean array
    darr = np.array([0,np.nan,1,np.nan,0,np.nan]) # Defined/undefined array
    inds = np.array([0,10,20,30,40,50]) # Indices
    inds2 = np.array([1,2,3,4]) # Skip first and last index

    # Test true, false, defined, and undefined
    assert cv.true(farr).tolist()      == [0,3,4]
    assert cv.false(farr).tolist()     == [1,2,5]
    assert cv.defined(darr).tolist()   == [0,2,4]
    assert cv.undefined(darr).tolist() == [1,3,5]

    # Test with indexing
    assert cv.itrue(barr, inds).tolist()      == [0,30,40]
    assert cv.ifalse(barr, inds).tolist()     == [10,20,50]
    assert cv.idefined(darr, inds).tolist()   == [0,20,40]
    assert cv.iundefined(darr, inds).tolist() == [10,30,50]

    # Test with double indexing
    assert cv.itruei(barr, inds2).tolist()      == [3,4]
    assert cv.ifalsei(barr, inds2).tolist()     == [1,2]
    assert cv.idefinedi(darr, inds2).tolist()   == [2,4]
    assert cv.iundefinedi(darr, inds2).tolist() == [1,3]

    return


def test_doubling_time():

    sim = cv.Sim(pop_size=1000)
    sim.run(verbose=0)

    d = sc.objdict()

    # Test doubling time
    d.t1 = cv.get_doubling_time(sim, interval=[3,sim['n_days']+10], verbose=2) # should reset end date to sim['n_days']
    d.t2 = cv.get_doubling_time(sim, start_day=3,end_day=sim['n_days'])
    d.t3 = cv.get_doubling_time(sim, interval=[3,sim['n_days']], exp_approx=True)
    d.t4 = cv.get_doubling_time(sim, start_day=3, end_day=sim['n_days'], moving_window=4) # should return array
    d.t5 = cv.get_doubling_time(sim, series=np.power(1.03, range(100)), interval=[3,30], moving_window=3) # Should be a series with values = 23.44977..
    d.t6 = cv.get_doubling_time(sim, start_day=9, end_day=20, moving_window=1, series="cum_infections") # Should recast window to 2 then return a series with 100s in it
    with pytest.raises(ValueError):
        d.t7 = cv.get_doubling_time(sim, start_day=3, end_day=20, moving_window=4, series="cum_deaths") # Should fail, no growth in deaths

    print('NOTE: this test prints some warnings; these are intended.')
    return d


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    rnd1    = test_rand()
    samples = test_samples(do_plot=do_plot)
    people1 = test_choose()
    people2 = test_choose_w()
    inds    = test_indexing()
    dt      = test_doubling_time()

    print('\n'*2)
    sc.toc(T)
    print('Done.')