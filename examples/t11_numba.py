'''
Illustrate different Numba options
'''

import covasim as cv

# Create a standard 32-bit simulation
sim32 = cv.Sim(label='32-bit, single-threaded (default)', verbose='brief')
sim32.run()

# Use 64-bit instead of 32
cv.options.set(precision=64)
sim64 = cv.Sim(label='64-bit, single-threaded', verbose='brief')
sim64.run()

# Use parallel threading
cv.options.set(numba_parallel=True)
sim_par = cv.Sim(label='64-bit, multi-threaded', verbose='brief')
sim_par.run()

# Reset to defaults
cv.options.set('defaults')
sim32b = cv.Sim(label='32-bit, single-threaded (restored)', verbose='brief')
sim32b.run()

# Plot
msim = cv.MultiSim([sim32, sim64, sim_par, sim32b])
msim.plot()