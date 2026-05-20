'''
Demonstrate visualization of the population contact structure using NetworkX
'''

import numpy as np
import networkx as nx
import covasim as cv

G = nx.Graph()

# Settings
pop_size = 100
pop_type = 'hybrid'
layers   = ['h', 's'] # Choose contact layer: can be one or more of 'h' (households), 's' (schools), 'w' (work), or 'c' (community)

# Create the simulation and population (we don't need to run it)
sim = cv.Sim(pop_size=pop_size, pop_type=pop_type) # Create sim
sim.initialize() # Initialize population

# Add nodes
for node in np.arange(pop_size):
    G.add_node(node)

# Add edges
for layer in layers:
    contacts = sim.people.contacts[layer] # e.g., pull out only household contacts
    for p1,p2 in zip(contacts['p1'], contacts['p2']): # Iterate over the contacts
        G.add_edge(p1, p2)

# Plot
nx.draw_shell(G)