'''
A more detailed benchmarking test that uses additional interventions.
'''

import numpy as np
import sciris as sc
import covasim as cv

# Define the interventions
tp = cv.test_prob(start_day=20, symp_prob=0.1, asymp_prob=0.01)
vx = cv.vaccinate_prob('pfizer', days=np.arange(30,100), prob=0.01)
cb = cv.change_beta(days=40, changes=0.5)
ct = cv.contact_tracing(trace_probs=0.3, start_day=50)

# Define the parameters
pars = dict(
    use_waning    = True,         # Whether or not to use waning and NAb calculations
    pop_size      = 50e3,         # Population size
    pop_infected  = 100,          # Number of initial infections -- use more for increased robustness
    pop_type      = 'hybrid',     # Population to use -- "hybrid" is random with household, school,and work structure
    n_days        = 100,          # Number of days to simulate
    verbose       = 0,            # Don't print details of the run
    rand_seed     = 2,            # Set a non-default seed
    interventions = [cb, tp, ct, vx], # Include the most common interventions
)

# Create the sim
sim = cv.Sim(pars)

# Comment out so the line you want to run is last
follow = [
    sim.run, # 100%
        # sim.initialize, # 25%, step 2: 12%
        #     sim.init_people, # 97%
        #         cv.population.make_people, # 88%
        #             cv.population.make_randpop, # 53%
        #                 cv.population.make_hybrid_contacts, # 99%
        #                     cv.population.make_microstructured_contacts, # 49% -- potential for improvement here
        #             cv.people.People.__init__, # 47%
        #                 cv.people.People.add_contacts, # 99%
        #                     cv.people.People.make_edgelist, # Most of the time -- now skipped
        # sim.step, # 75%, step 2: 83%, step 3: 64%
            # cv.immunity.check_immunity, # 67%, step 3: 9%
            # cb.apply, # 13% interventions; step 3: 35% # < 1%
            # tp.apply, # 5%
            # ct.apply, # 14%
            #     ct.identify_contacts, # 99%
            # vx.apply, # 3%
            # cv.people.infect, # 13% infections; step 3: 34%
][-1]

sc.profile(run=sim.run, follow=follow)