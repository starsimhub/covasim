import covasim as cv

fn = 'tmp.sim'

s1 = cv.Sim()
s1.run()

s2 = cv.Sim()
s2.run(until=30)
s2.save(fn)

s3 = cv.load(fn)
s3.run()

assert s3.summary == s1.summary
