import json
from covasim import Sim, parameters, test_prob, contact_tracing, sequence

def test_basic_contact_trace():

    tweaking = {
        'n_days': 60,
        'pop_size': 2000,
        'pop_type': 'hybrid'
    }

    start_days = []
    interventions = []

    test_prob_start_day = 35
    prob_test_perfect_symptomatic = test_prob(symp_prob=1.0, symp_quar_prob=1.0, asymp_quar_prob=1.0,
                                              test_sensitivity=1.0, test_delay=1, start_day=test_prob_start_day)
    start_days.append(test_prob_start_day)
    interventions.append(prob_test_perfect_symptomatic)

    contact_trace_start_day = 35
    all_contacts = {
        'h': 1.0,
        'w': 1.0,
        's': 1.0,
        'c': 1.0
    }
    single_day_delays = {
        'h': 1,
        'w': 1,
        's': 1,
        'c': 1
    }
    brutal_contact_trace = contact_tracing(trace_probs=all_contacts,
                                           trace_time=single_day_delays,
                                           start_day=contact_trace_start_day)
    start_days.append(contact_trace_start_day)
    interventions.append(brutal_contact_trace)

    tweaking['interventions'] = interventions

    test_stuff_simulation = Sim(pars=tweaking)
    test_stuff_simulation.run()

    test_stuff_results = test_stuff_simulation.to_json(tostring=False)
    with open("DEBUG_test_intervention_list_simulation.json","w") as outfile:
        json.dump(test_stuff_results, outfile, indent=4, sort_keys=True)
        pass

    assert test_stuff_results["results"]["n_symptomatic"][test_prob_start_day] > 0  # If there are symptomatics
    assert sum(test_stuff_results["results"]["new_tests"][test_prob_start_day:test_prob_start_day + 5]) > 0  # then there should be tests
    assert sum(test_stuff_results["results"]["new_diagnoses"][test_prob_start_day + 1:test_prob_start_day + 6]) > 0 # and some diagnoses
    assert sum(test_stuff_results["results"]["new_diagnoses"][test_prob_start_day + 2:test_prob_start_day + 7]) > 0 # and therefore some quarantined
    pass


if __name__ == "__main__":
    test_basic_contact_trace()
    pass

