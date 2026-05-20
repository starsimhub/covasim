'''
Illustrate dynamic layers
'''

import covasim as cv
import numpy as np
import sciris as sc

class CustomLayer(cv.Layer):
    ''' Create a custom layer that updates daily based on supplied contacts '''

    def __init__(self, layer, contact_data):
        ''' Convert an existing layer to a custom layer and store contact data '''
        for k,v in layer.items():
            self[k] = v
        self.contact_data = contact_data
        return

    def update(self, people):
        ''' Update the contacts '''
        pop_size = len(people)
        n_new = self.contact_data[people.t] # Pull out today's contacts
        self['p1']   = np.array(cv.choose_r(max_n=pop_size, n=n_new), dtype=cv.default_int) # Choose with replacement
        self['p2']   = np.array(cv.choose_r(max_n=pop_size, n=n_new), dtype=cv.default_int) # Paired contact
        self['beta'] = np.ones(n_new, dtype=cv.default_float) # Per-contact transmission (just 1.0)
        return


# Define some simple parameters
pars = sc.objdict(
    pop_size = 1000,
    n_days = 90,
)

# Set up and run the simulation
base_sim = cv.Sim(pars, label='Default simulation')
sim = cv.Sim(pars, dynam_layer=True, label='Dynamic layers')
sim.initialize()

# Update to custom layer
for key in sim.layer_keys():
    contact_data = np.random.randint(pars.pop_size*10, pars.pop_size*20, size=pars.n_days+1) # Generate a number of contacts for today
    sim.people.contacts[key] = CustomLayer(sim.people.contacts[key], contact_data=contact_data)


# Run and plot
if __name__ == '__main__':
    msim = cv.parallel(base_sim, sim)
    msim.plot()