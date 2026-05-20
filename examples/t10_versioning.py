'''
Demonstrate versioning
'''

import covasim as cv

# Check the version
cv.check_save_version('1.5.0', filename='my-version.gitinfo', die=False) # Old version, will raise a warning

# Run and plot the sim
sim = cv.Sim(verbose=0)
sim.run()
sim.plot()

# Save the figure
filename = 'my-figure.png'
cv.savefig(filename) # Save including version information

# Retrieve and print information
print('Figure version info:')
cv.get_png_metadata(filename)
