"""M10 save/load tests.

cv.Sim is an ss.Sim, so it saves/loads via the Starsim/sciris pickle path. Two Covasim-specific
points are covered here:
  - sim.save() saves the FULL sim by default (cv.COVID carries large legitimate state, so the stock
    ss.Sim shrink-on-save trips Starsim's size check); a loaded sim reproduces the results exactly.
  - cv.load / cv.Sim.load load v4 (Starsim-based) objects natively, bypassing the v3 migration path.
"""
import os
import tempfile
import numpy as np
import sciris as sc
import covasim as cv


def _run(seed=3):
    sim = cv.Sim(pop_size=4000, pop_infected=40, pop_type='hybrid', n_days=25, rand_seed=seed, verbose=0)
    sim.run()
    return sim


def _cum_inf(sim):
    return float(np.asarray(sim.diseases.covid.results['cum_infections']).max())


def test_sim_save_load_roundtrip(tmp_path):
    """sim.save() (no shrink arg) writes the full sim; every load path reproduces the results."""
    sim = _run()
    ci = _cum_inf(sim)
    fn = str(tmp_path / 'test.sim')
    sim.save(fn)                       # out-of-the-box, no shrink kwarg
    assert os.path.getsize(fn) > 0
    for loader, label in [(cv.Sim.load, 'cv.Sim.load'),
                          (lambda f: cv.load(f, verbose=False), 'cv.load'),
                          (sc.load, 'sc.load')]:
        s = loader(fn)
        assert isinstance(s, cv.Sim), f'{label} returns a cv.Sim'
        assert _cum_inf(s) == ci, f'{label} reproduces cum_infections'
        assert 'cum_deaths' in s.results, f'{label} keeps the bridged top-level results'


def test_loaded_sim_is_usable(tmp_path):
    """A loaded sim can still be plotted (the bridged + module results survive the round-trip)."""
    import matplotlib
    matplotlib.use('agg')
    sim = _run()
    fn = str(tmp_path / 'test2.sim')
    sim.save(fn)
    s = cv.Sim.load(fn)
    fig = s.plot()
    assert fig is not None and len(fig.axes) >= 1


def test_multisim_save_load(tmp_path):
    """cv.MultiSim round-trips via cv.save/cv.load (it has no version attr, so it loads as-is)."""
    msim = cv.MultiSim(cv.Sim(pop_size=2000, pop_infected=20, n_days=15, verbose=0), n_runs=3)
    msim.run()
    fn = str(tmp_path / 'test.msim')
    cv.save(fn, msim)
    m2 = cv.load(fn, verbose=False)
    assert isinstance(m2, cv.MultiSim) and len(m2.sims) == 3
