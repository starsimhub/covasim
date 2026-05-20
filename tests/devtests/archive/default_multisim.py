import covasim as cv

n_runs = 11
repeats = 3

for i in range(repeats):
    sim = cv.Sim(rand_seed=59448+i*92348)
    msim = cv.MultiSim(sim, n_runs=n_runs)
    msim.run()
    msim.reduce()
    msim.plot()
