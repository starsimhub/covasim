'''
Check that runs are repeatable with restore_pars = True, and are not repeatable
with restore_pars = False.
'''

import covasim as cv

# Use an intervention that dynamically modifies the parameters
pars = dict(
    interventions = cv.change_beta(days=30, changes=0.5)
    )

# Create and run the first sim, restoring the parameters
s1 = cv.Sim(pars)
s1.run(restore_pars=True)

# Create and run the second, and do not restore the parameters
s2 = cv.Sim(s1.pars)
s2.run(restore_pars=False)

# Run the third with the modified parameters
s3 = cv.Sim(s2.pars)
s3.run()

for i,s in enumerate([s1, s2, s3]):
    print(f'\n\nFor run {i+1}:')
    print(s.summary)

assert s1.summary == s2.summary
assert s2.summary != s3.summary

print('Tests passed.')