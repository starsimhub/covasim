'''
Demonstrate that even interventions can be exported
'''

import covasim as cv

intervs = [
    cv.change_beta(30, 0.5),
    cv.test_prob(symp_prob=0.1, start_day=20, do_plot=False),
    ]

s1 = cv.Sim(interventions=intervs)
pars = s1.export_pars()
s1.run()

s2 = cv.Sim(pars)
s2.run()

assert s1.summary == s2.summary