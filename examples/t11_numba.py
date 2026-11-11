'''
Illustrate different Numba options
'''

import covasim as cv

# Create a standard 32-bit simulation
sim32 = cv.Sim(label='32-bit, single-threaded (default)', verbose=0)
sim32.run().brief()

# Use 64-bit instead of 32
cv.options.set(precision=64)
sim64 = cv.Sim(label='64-bit, single-threaded', verbose=0)
sim64.run().brief()

# Reset to defaults
cv.options.set('defaults')
sim32b = cv.Sim(label='32-bit, single-threaded (restored)', verbose=0)
sim32b.run().brief()

# Plot
msim = cv.MultiSim([sim32, sim64, sim32b])
msim.plot()