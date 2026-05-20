'''
Demonstrate the compare method of multisims
'''

import covasim as cv

option = 2

s0 = cv.Sim(label='Low beta', beta=0.012)
s1 = cv.Sim(label='Normal beta')
s2 = cv.Sim(label='High beta', beta=0.018)

if option == 1:
    msim = cv.MultiSim(sims=[s0, s1, s2])
    msim.run()
elif option == 2:
    msim = cv.MultiSim(sims=s1)
    msim.run(n_runs=10)

df = msim.compare(output=True)
print(df)

msim.plot_compare()

msim.plot_result('r_eff')

