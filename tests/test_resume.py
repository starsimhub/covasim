'''
Tests for resuming a simulation partway, as well as reproducing two simulations with
different initialization states and after saving to disk.
'''

#%% Imports and settings
import os
import pytest
import numpy as np
import sciris as sc
import covasim as cv

# Simulation and test parameters
pars = dict(pop_size=1e3, verbose=0)


#%% Define the tests

def test_resuming():
    sc.heading('Test that resuming a run works')

    s0 = cv.Sim(pars, start_day='2020-01-01')
    s1 = s0.copy()
    s0.run()

    # Cannot run the same simulation multiple times
    with pytest.raises(cv.AlreadyRunError):
        s0.run()

    # If until=0 then no timesteps will be taken
    with pytest.raises(cv.AlreadyRunError):
        s1.run(until=0, reset_seed=False)
    assert s1.initialized # It should still have been initialized though
    with pytest.raises(RuntimeError):
        s1.compute_summary(require_run=True) # Not ready yet

    s1.run(until='2020-01-31', reset_seed=False)
    with pytest.raises(cv.AlreadyRunError):
        s1.run(until=30, reset_seed=False) # Error if running up to the same value
    with pytest.raises(cv.AlreadyRunError):
        s1.run(until=20, reset_seed=False) # Error if running until a previous timestep
    with pytest.raises(cv.AlreadyRunError):
        s1.run(until=1000, reset_seed=False) # Error if running until the end of the sim

    s1.run(until=45, reset_seed=False)
    s1.run(reset_seed=False)
    with pytest.raises(cv.AlreadyRunError):
        s1.finalize() # Can't re-finalize a finalized sim

    assert np.all(s0.results['cum_infections'].values == s1.results['cum_infections']) # Results should be identical

    return s1


def test_reset_seed():
    sc.heading('Test that resetting a simulation works')

    until = 30
    s0 = cv.Sim(pars)
    s1 = s0.copy()
    s0.run()

    s1.run(until=until)
    s1.run(reset_seed=True)

    assert     np.all(s0.results['cum_infections'][:until] == s1.results['cum_infections'][:until]) # Results for the first 30 days should be the same
    assert not np.all(s0.results['cum_infections'][until:] == s1.results['cum_infections'][until:]) # Results should be different

    return s1


def test_reproducibility():
    sc.heading('Test that sims are reproducible')

    fn = 'save-load-test.sim' # Name of the test file to save
    key = 'cum_infections'

    #The results of the sim shouldn't be affected by what you do or don't do prior to sim.run()
    s1 = cv.Sim(pars)
    s1.initialize()
    s2 = s1.copy()
    s1.run()
    s2.run()
    r1 = s1.summary[key]
    r2 = s2.summary[key]
    assert r1 == r2

    # If you run a sim and save it, you should be able to re-run it on load
    s3 = cv.Sim(pars, pop_infected=44)
    s3.run()
    s3.save(fn)
    s4 = cv.load(fn)
    s4.initialize()
    s4.run()
    r3 = s3.summary[key]
    r4 = s4.summary[key]
    assert r3 == r4
    if os.path.exists(fn): # Tidy up -- after the assert to allow inspection if it fails
        os.remove(fn)

    # Running a sim and resetting people should result in the same result; otherwise they should differ
    s5 = cv.Sim(pars)
    s5.run()
    r5 = s5.summary[key]
    s5.initialize(reset=True)
    s5.run()
    r6 = s5.summary[key]
    s5.initialize(reset=False)
    with pytest.raises(cv.AlreadyRunError):
        s5.run()
    s5.people.t = s5.t # Manually reset
    s5.run()
    r7 = s5.summary[key]
    assert r5 == r6
    assert r5 != r7

    return s4


def test_step(): # If being run via pytest, turn off
    sc.heading('Test stepping')

    # Create and run a basic simulation
    s1 = cv.Sim(pars)
    s1.run(verbose=0)

    # Test that step works
    s2 = cv.Sim(pars)
    s2.initialize()
    for n in range(s2.npts):
        s2.step()
    s2.finalize()

    # Compare results
    key = 'cum_infections'
    assert (s1.results[key][:] == s2.results[key][:]).all(), 'Next values do not match'

    return s2


def test_stopping(): # If being run via pytest, turn off
    sc.heading('Test stopping')

    # Run a sim with very short time limit
    s1 = cv.Sim(pars, timelimit=0)
    s1.run()

    # Run a sim with a stopping function
    def stopping_func(sim): return True
    s2 = cv.Sim(pars, stopping_func=stopping_func)
    s2.run()
    s2.finalize()

    return s1


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim1 = test_resuming()
    sim2 = test_reset_seed()
    sim3 = test_reproducibility()
    sim4 = test_step()
    sim5 = test_stopping()

    print('\n'*2)
    sc.toc(T)
    print('Done.')