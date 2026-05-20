'''
Illustration of running Covasim via Dask
'''

import dask
from dask.distributed import Client
import numpy as np
import covasim as cv


def run_sim(index, beta):
    ''' Run a standard simulation '''
    sim = cv.Sim(beta=beta, label=f'Sim {index}, beta={beta}')
    sim.run()
    return sim


if __name__ == '__main__':

    # Run settings
    n = 8
    n_workers = 4
    betas = np.sort(np.random.random(n))

    # Create and queue the Dask jobs
    client = Client(n_workers=n_workers)
    queued = []
    for i,beta in enumerate(betas):
        run = dask.delayed(run_sim)(i, beta)
        queued.append(run)

    # Run and process the simulations
    sims = list(dask.compute(*queued))
    msim = cv.MultiSim(sims)
    msim.plot(color_by_sim=True)