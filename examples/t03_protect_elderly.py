'''
Example of a custom intervention and a very quick scenario analysis
'''

import covasim as cv

def protect_elderly(sim):
    if sim.t == sim.day('2021-04-01'):
        elderly = sim.people.age>70
        sim.people.rel_sus[elderly] = 0.0

pars = {'start_day':'2021-03-01', 'n_days':120}
s1 = cv.Sim(pars, label='Default')
s2 = cv.Sim(pars, label='Protect the elderly', interventions=protect_elderly)

if __name__ == '__main__':
    cv.parallel(s1, s2).plot(to_plot=['cum_deaths', 'cum_infections'])