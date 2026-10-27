"""
Tests for testing, contact tracing, quarantine and isolation

See also test_interventions.py, which runs all of the interventions.
"""

#%% Imports and settings
import numpy as np
import sciris as sc
import covasim as cv

do_plot = 1
cv.options.set(interactive=False) # Assume not running interactively

base_pars = dict(
    pop_size = 5000,
    pop_infected = 100,
    pop_type = 'hybrid',
    n_days = 60,
    verbose = 0,
)


#%% Define the tests

def test_testing():
    sc.heading('Testing the testing interventions')

    start_day = 10
    daily_tests = 100

    # Without testing, no one is diagnosed
    sim = cv.Sim(base_pars).run()
    assert sim.summary['cum_tests'] == 0
    assert sim.summary['cum_diagnoses'] == 0

    # Testing by probability: the more likely people are to test, the more are diagnosed
    low  = cv.Sim(base_pars, interventions=cv.test_prob(symp_prob=0.05, start_day=start_day)).run()
    high = cv.Sim(base_pars, interventions=cv.test_prob(symp_prob=0.5,  start_day=start_day)).run()
    assert low.results['new_tests'][:start_day].sum() == 0 # No tests before the start day
    assert 0 < low.summary['cum_diagnoses'] < high.summary['cum_diagnoses']

    # Testing by number: never more than the number of tests available
    sim = cv.Sim(base_pars, interventions=cv.test_num(daily_tests=daily_tests, start_day=start_day)).run()
    new_tests = sim.results['new_tests']
    assert new_tests[:start_day].sum() == 0
    assert new_tests[start_day:].max() == daily_tests
    assert sim.summary['cum_diagnoses'] > 0

    # Only people who have been infected are diagnosed
    ppl = sim.people
    assert len(cv.undefined(ppl.date_exposed[ppl.diagnosed])) == 0

    return sim


def test_isolation():
    sc.heading('Testing diagnosis, isolation and quarantine')

    sim = cv.Sim(base_pars, pop_infected=500)
    sim.run(until=10)
    ppl = sim.people
    day = sim.ti

    # Test everyone who is infectious: with a perfect test, they are all diagnosed when they get their results
    delay = 2
    infectious = cv.true(ppl.infectious)
    ppl.test(infectious, test_sensitivity=1.0, test_delay=delay)
    assert np.all(ppl.tested[infectious])
    assert np.all(ppl.date_diagnosed[infectious] == day + delay)

    # Quarantine some people who are susceptible
    period = 7
    susceptible = cv.true(ppl.susceptible & ~ppl.recovered)[:50] # Not people who have recovered and are susceptible again, since they aren't quarantined
    ppl.schedule_quarantine(susceptible, start_date=day+delay, period=period)

    # Once the results are back, people who are diagnosed and still infected are isolated
    sim.run(until=day+delay+2)
    reinfected = ppl.date_exposed[infectious] > day # Reinfection resets the diagnosis
    assert np.all(ppl.diagnosed[infectious] | reinfected)
    assert ppl.isolated.sum() > 0
    assert np.all(ppl.quarantined[susceptible])

    # They leave quarantine at the end of the quarantine period
    sim.run(until=day+delay+period+1)
    assert not np.any(ppl.quarantined[susceptible])

    # People who are isolated transmit less
    n_infections = []
    for iso_factor in [0, 1]:
        tp = cv.test_prob(symp_prob=0.3, start_day=5)
        sim = cv.Sim(base_pars, iso_factor=dict(h=iso_factor, s=iso_factor, w=iso_factor, c=iso_factor), interventions=tp).run()
        n_infections.append(sim.summary['cum_infections'])
    assert n_infections[0] < n_infections[1]

    return sim


def test_tracing(do_plot=False):
    sc.heading('Testing contact tracing')

    start_day = 10
    def make_sim(trace_probs=None, label=None):
        """ Make a sim with testing, with or without contact tracing """
        interventions = [cv.test_prob(symp_prob=0.3, asymp_prob=0.01, start_day=start_day)]
        if trace_probs is not None:
            interventions += [cv.contact_tracing(trace_probs=trace_probs, start_day=start_day)]
        sim = cv.Sim(base_pars, interventions=interventions, label=label)
        return sim

    no_tracing = make_sim(label='No tracing').run()
    tracing = make_sim(trace_probs=0.8, label='Tracing').run()

    # People are only quarantined if there is contact tracing, and tracing reduces infections
    assert no_tracing.summary['cum_quarantined'] == 0
    assert tracing.summary['cum_quarantined'] > 0
    assert tracing.summary['cum_infections'] < no_tracing.summary['cum_infections']
    assert tracing.summary['cum_isolated'] == tracing.summary['cum_diagnoses'] # Everyone diagnosed is isolated
    assert tracing.summary['n_known_dead'] == tracing.summary['cum_known_deaths'] # Deaths among people who were diagnosed

    # Contact tracing does nothing without testing, since no one is diagnosed
    sim = cv.Sim(base_pars, interventions=cv.contact_tracing(trace_probs=0.8, start_day=start_day)).run()
    assert sim.summary['cum_quarantined'] == 0

    # A derived class can override the steps of tracing
    class no_notification(cv.contact_tracing):
        def notify_contacts(self, sim, contacts):
            self.n_contacts = getattr(self, 'n_contacts', 0) + sum(len(uids) for uids in contacts.values())
    sim = cv.Sim(base_pars, interventions=[cv.test_prob(symp_prob=0.3, start_day=start_day), no_notification()]).run()
    assert sim.get_intervention(no_notification).n_contacts > 0 # The override was called...
    assert sim.summary['cum_quarantined'] == 0 # ...instead of quarantining the contacts

    if do_plot:
        cv.MultiSim([no_tracing, tracing]).plot(to_plot=['cum_infections', 'cum_diagnoses', 'n_quarantined'])

    return tracing



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    sim1 = test_testing()
    sim2 = test_isolation()
    sim3 = test_tracing(do_plot=do_plot)

    sc.toc(T)
    print('Done.')
