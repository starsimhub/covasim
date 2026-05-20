'''
Demonstrate simple analyzer usage
'''

import covasim as cv


# Age histograms
sim = cv.Sim(interventions=cv.test_prob(0.5), analyzers=cv.age_histogram())
sim.run()
agehist = sim.get_analyzer() # Only one analyzer so we can retrieve it like this
agehist.plot()


# Transmission trees
tt = sim.make_transtree()
fig1 = tt.plot()
fig2 = tt.plot_histograms()


# A custom analyzer
def check_88(sim):
    people_who_are_88 = sim.people.age.round() == 88 # Find everyone who's aged 88 (to the nearest year)
    people_exposed = sim.people.exposed # Find everyone who's infected with COVID
    people_who_are_88_with_covid = cv.true(people_who_are_88 * people_exposed) # Multiplication is the same as logical "and"
    n = len(people_who_are_88_with_covid) # Count how many people there are
    if n:
        print(f'Oh no! {n} people aged 88 have covid on timestep {sim.t} {"🤯"*n}')
    return

sim = cv.Sim(n_days=120, analyzers=check_88, verbose=0)
sim.run()