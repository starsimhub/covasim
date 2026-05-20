import covasim as cv

n_days = 90
n_tests = 100
pop_size = 100
delay = 5
tn = cv.test_num(daily_tests=[n_tests]*n_days, test_delay=delay)
# tp = cv.test_prob(symp_prob=0.1, test_delay=delay)


sim = cv.Sim(interventions=tn, pop_size=pop_size, n_days=n_days, diag_factor=1.0)
sim.run()
sim.plot()