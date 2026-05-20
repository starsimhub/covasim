import covasim as cv

debug = False

my_parameters = cv.parameters.make_pars()

tweaking = {
    'n_days': 60
}

test_prob_start_day = 35
prob_test_perfect_symptomatic = cv.test_prob(symp_prob=1.0, symp_quar_prob=1.0, asymp_quar_prob=1.0,
                                          sensitivity=1.0, test_delay=1, start_day=test_prob_start_day)

my_parameters['interventions'] = prob_test_perfect_symptomatic

test_stuff_simulation = cv.Sim(my_parameters)
test_stuff_simulation.run()

test_stuff_results = test_stuff_simulation.to_json(tostring=False)
if debug:
    test_stuff_simulation.to_json(filename="DEBUG_test_stuff_simulation.json")

assert test_stuff_results["results"]["n_symptomatic"][test_prob_start_day] > 0  # If there are symptomatics
assert sum(test_stuff_results["results"]["new_tests"][test_prob_start_day:test_prob_start_day + 5]) > 0  # then there should be tests
print("OK")
