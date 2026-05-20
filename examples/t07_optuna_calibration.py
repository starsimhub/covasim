'''
Example for running built-in calibration with Optuna
'''

import sciris as sc
import covasim as cv

# Create default simulation
pars = sc.objdict(
    pop_size       = 10_000,
    start_day      = '2020-02-01',
    end_day        = '2020-04-11',
    beta           = 0.015,
    rel_death_prob = 1.0,
    interventions  = cv.test_num(daily_tests='data'),
    verbose        = 0,
)
sim = cv.Sim(pars=pars, datafile='example_data.csv')

# Parameters to calibrate -- format is best, low, high
calib_pars = dict(
    beta           = [pars.beta, 0.005, 0.20],
    rel_death_prob = [pars.rel_death_prob, 0.5, 3.0],
)

if __name__ == '__main__':

    # Run the calibration
    n_trials = 25
    n_workers = 4
    calib = sim.calibrate(calib_pars=calib_pars, n_trials=n_trials, n_workers=n_workers)

    # Plot the results
    calib.plot(to_plot=['cum_tests', 'cum_diagnoses', 'cum_deaths'])