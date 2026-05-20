import covasim as cv
import sciris as sc

cv.check_version('0.27.9')

do_plot = 1

datafile = sc.thisdir(__file__, '../../example_data.csv')

pars = sc.objdict(
    diag_factor = 1.0,
    n_days=30,
    pop_infected=100
    )


sim = cv.Sim(pars, datafile=datafile)

case = 2
testprobdict = {
    0: cv.test_prob(symp_prob=0, asymp_prob=0, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0), # works, no diagnoses
    1: cv.test_prob(symp_prob=1, asymp_prob=0, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0), # works, ~half diagnosed
    2: cv.test_prob(symp_prob=0, asymp_prob=1, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0), # works, most diagnosed (during asymptomatic period)
    3: cv.test_prob(symp_prob=1, asymp_prob=1, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0), # works, most diagnosed, slight mismatch due to new infections that day
    4: cv.test_prob(symp_prob=1, asymp_prob=1, test_sensitivity=0.0, loss_prob=0.0, test_delay=0, start_day=0), # works, no diagnoses
    5: cv.test_prob(symp_prob=1, asymp_prob=1, test_sensitivity=0.5, loss_prob=0.0, test_delay=0, start_day=0), # works, ~80% diagnosed
    6: cv.test_prob(symp_prob=1, asymp_prob=1, test_sensitivity=1.0, loss_prob=0.5, test_delay=0, start_day=0), # works, ~80% diagnosed
    7: cv.test_prob(symp_prob=1, asymp_prob=1, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0), # works, most diagnosed, slight mismatch due to new infections that day
    8: cv.test_prob(symp_prob=1, asymp_prob=1, test_sensitivity=1.0, loss_prob=0.0, test_delay=5, start_day=0), # works, most diagnosed, with delay, should be about one set of tick marks behind
    9: cv.test_prob(symp_prob=1, asymp_prob=1, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=7), # works, most diagnosed, with delay
}

sim.update_pars(interventions=testprobdict[case])

to_plot = sc.dcp(cv.default_sim_plots)
to_plot['Total counts'].append('cum_infectious')
to_plot['Daily counts']+= ['new_tests', 'new_infectious']
sim.run(do_plot=do_plot, to_plot=to_plot)