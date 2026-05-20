'''
Demonstrate flexibility in date handling
'''

import sciris as sc
import covasim as cv

intervs1 = [
    cv.change_beta(30, 0.5),
    cv.test_prob(symp_prob=0.1, start_day=20),
    ]

s1 = cv.Sim(start_day='2020-03-01', interventions=intervs1)
s1.run()


intervs2 = [
    cv.change_beta('2020-03-31', 0.5),
    cv.test_prob(symp_prob=0.1, start_day='2020-03-21'),
    ]

s2 = cv.Sim(start_day='2020-03-01', interventions=intervs2)
s2.run()

assert s1.summary == s2.summary

intervs3 = [
    cv.change_beta(days=['2020-03-31', sc.readdate('2020-04-07'), 50], changes=[0.5, 0.3, 0.0]),
    ]

s3 = cv.Sim(start_day='2020-03-01', interventions=intervs3)
s3.run()
s3.plot()