'''
Test Plotly plotting outside of the webapp.
'''

import plotly.io as pio
import covasim as cv

pio.renderers.default = "browser" # Or can use a Jupyter notebook

ce = cv.clip_edges(**{'start_day': 10, 'change': 0.5})
sim = cv.Sim(pop_size=100, n_days=60, datafile='../example_data.csv', interventions=ce, verbose=0)
sim.run()

f1list = cv.plotly_sim(sim, do_show=True)
f2     = cv.plotly_people(sim, do_show=True)
f3     = cv.plotly_animate(sim, do_show=True)
