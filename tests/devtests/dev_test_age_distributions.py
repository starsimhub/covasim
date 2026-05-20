'''
Plot age distributions for Sweden and Somalia as a test.
'''

import covasim as cv
import pylab as pl

loc1 = 'Sweden'
loc2 = 'Somalia'
loc3 = 'USA-Washington'
loc4 = 'USA-Wisconsin'

sim1 = cv.Sim(location=loc1)
sim2 = cv.Sim(location=loc2)
sim3 = cv.Sim(location=loc3)
sim4 = cv.Sim(location=loc4)

sim1.initialize()
sim2.initialize()
sim3.initialize()
sim4.initialize()

ages1 = sim1.people.age
ages2 = sim2.people.age
ages3 = sim3.people.age
ages4 = sim4.people.age

n = 20
fig = pl.figure()

pl.subplot(2,2,1)
pl.hist(ages1, n)
pl.title(loc1)

pl.subplot(2,2,2)
pl.hist(ages2, n)
pl.title(loc2)

pl.subplot(2,2,3)
pl.hist(ages3, n)
pl.title(loc3)

pl.subplot(2,2,4)
pl.hist(ages4, n)
pl.title(loc4)


pl.xlabel('Age')
pl.ylabel('Count')
pl.show()
