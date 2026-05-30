"""
M8 demo: multi-seed uncertainty quantification with cv.MultiSim.

Runs N seeds of a sim, reduces to a median trajectory + 10/90% quantile band, and plots the
infectious curve and the cumulative burden with uncertainty. Run:

    python examples/m8_uq_sweep.py
"""
import numpy as np
import covasim as cv


def run(n_runs=10, pop_size=20_000, n_days=120, save='/tmp/m8_uq_sweep.png'):
    base = cv.Sim(pop_size=pop_size, pop_infected=100, pop_type='hybrid', n_days=n_days, verbose=0)
    msim = cv.MultiSim(base, n_runs=n_runs).run(verbose=0).reduce(quantiles=[0.1, 0.9])

    ni = msim.results['n_infectious']
    print(f'{n_runs}-seed UQ: n_infectious peak median={int(ni.best.max())}, '
          f'10-90% band at peak=[{int(ni.low[np.argmax(ni.best)])}, {int(ni.high[np.argmax(ni.best)])}]')

    if save:
        import matplotlib
        matplotlib.use('agg')
        msim.plot(keys=['n_infectious', 'cum_infections', 'cum_severe', 'cum_deaths']).savefig(save, dpi=110)
        print(f'Saved {save}')
    return msim


if __name__ == '__main__':
    run()
