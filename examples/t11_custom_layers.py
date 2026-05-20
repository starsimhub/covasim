'''
Demonstration of custom population layers
'''

import numpy as np
import covasim as cv

# Create the first sim
orig_sim = cv.Sim(pop_type='hybrid', n_days=120, label='Default hybrid population')
orig_sim.initialize() # Initialize the population

# Create the second sim
sim = orig_sim.copy()

# Define the new layer, 'transport'
n_people = len(sim.people)
n_contacts_per_person = 0.5
n_contacts = int(n_contacts_per_person*n_people)
contacts_p1 = cv.choose(max_n=n_people, n=n_contacts)
contacts_p2 = cv.choose(max_n=n_people, n=n_contacts)
beta = np.ones(n_contacts)
layer = cv.Layer(p1=contacts_p1, p2=contacts_p2, beta=beta) # Create the new layer

# Add this layer in and re-initialize the sim
sim.people.contacts.add_layer(transport=layer)
sim.reset_layer_pars() # Automatically add layer 'q' to the parameters using default values
sim.initialize() # Reinitialize
sim.label = f'Transport layer with {n_contacts_per_person} contacts/person'

# Run and compare
if __name__ == '__main__':
    msim = cv.parallel([orig_sim, sim])
    msim.plot()