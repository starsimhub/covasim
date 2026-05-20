'''
Test MultiSim merging and splitting
'''

import covasim as cv; cv

m1 = cv.MultiSim(cv.Sim(label='sim1'), initialize=True)
m2 = cv.MultiSim(cv.Sim(label='sim2'), initialize=True)
m3 = cv.MultiSim.merge(m1, m2)
m3.run()
m1b, m2b = m3.split()

msim = cv.MultiSim(cv.Sim(), n_runs=6)
msim.run()
m1, m2 = msim.split(inds=[[0,2,4], [1,3,5]])
mlist1 = msim.split(chunks=[2,4]) # Equivalent to inds=[[0,1], [2,3,4,5]]
mlist2 = msim.split(chunks=3) # Equivalent to inds=[[0,1,2], [3,4,5]]