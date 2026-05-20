'''
Example for running custom calibration with Optuna
'''

import covasim as cv

pars = dict(
    verbose = 0,
    start_day = '2020-02-05',
    pop_size = 1000,
    pop_scale = 4,
    interventions = cv.test_prob(symp_prob=0.01),
)

sim = cv.Sim(pars, datafile='example_data.csv')

calib_pars = dict(
    beta      = [0.013, 0.005, 0.020],
    test_prob = [0.01, 0.00, 0.30]
)

# Define the custom calibration function here
def set_test_prob(sim, calib_pars):
    tp = sim.get_intervention(cv.test_prob)
    tp.symp_prob = calib_pars['test_prob']
    return sim


if __name__ == '__main__':

    # Run the calibration
    calib = sim.calibrate(calib_pars=calib_pars, custom_fn=set_test_prob, n_trials=5)
    calib.plot(to_plot=['cum_deaths', 'cum_diagnoses'])