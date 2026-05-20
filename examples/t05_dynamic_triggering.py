'''
Demonstrate dynamically triggered interventions
'''

import covasim as cv
import numpy as np

def inf_thresh(self, sim, thresh=500):
    ''' Dynamically define on and off days for a beta change -- use self like it's a method '''

    # Meets threshold, activate
    if sim.people.infectious.sum() > thresh:
        if not self.active:
            self.active = True
            self.t_on = sim.t
            self.plot_days.append(self.t_on)

    # Does not meet threshold, deactivate
    else:
        if self.active:
            self.active = False
            self.t_off = sim.t
            self.plot_days.append(self.t_off)

    return [self.t_on, self.t_off]

# Set up the intervention
on = 0.2 # Beta less than 1 -- intervention is on
off = 1.0 # Beta is 1, i.e. normal -- intervention is off
changes = [on, off]
plot_args = dict(label='Dynamic beta', show_label=True, line_args={'c':'blue'})
cb = cv.change_beta(days=inf_thresh, changes=changes, **plot_args)

# Set custom properties
cb.t_on = np.nan
cb.t_off = np.nan
cb.active = False
cb.plot_days = []

# Run the simulation and plot
sim = cv.Sim(interventions=cb)
sim.run().plot()