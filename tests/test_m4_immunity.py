"""M4 waning-immunity + NAb tests.

Grows across M4 Tasks 1-3. Task 1 (this batch) covers the ported NAb-engine *functions*
(the waning kernel + calc_VE) -- pure-numpy ports of v3 _v2_legacy/immunity.py, wired into
the live engine in Tasks 2-3.
"""
import numpy as np
import starsim as ss
import covasim as cv
import covasim.immunity as cvimm

NAB_DECAY = dict(form='nab_growth_decay', growth_time=21, decay_rate1=np.log(2) / 50,
                 decay_time1=150, decay_rate2=np.log(2) / 250, decay_time2=365)
NAB_EFF = dict(alpha_inf=1.08, alpha_inf_diff=1.812, beta_inf=0.967, alpha_symp_inf=-0.739,
               beta_symp_inf=0.038, alpha_sev_symp=-0.014, beta_sev_symp=0.079)


# --- waning kernel -----------------------------------------------------------

def test_waning_kernel_growth_then_decay():
    """The default nab_growth_decay kernel rises to peak over growth_time, then decays."""
    k = np.asarray(cvimm.precompute_waning(180, dict(NAB_DECAY)))
    assert len(k) == 180
    gt = NAB_DECAY['growth_time']
    # Cumulative NAb (relative to peak) reaches 1.0 at the end of the growth phase.
    assert abs(k[:gt].sum() - 1.0) < 1e-9, 'NAb grows to peak (cumsum 1.0) over growth_time'
    assert (k[:gt] > 0).all(), 'all-positive increments during growth'
    assert (k[gt + 5:] < 0).all(), 'negative increments (decay) after growth'


def test_waning_kernel_matches_v3_shape():
    """nab_decay and exp_decay forms run and produce finite kernels of the right length."""
    k_decay = np.asarray(cvimm.precompute_waning(100, dict(form='nab_decay', decay_rate1=0.01,
                                                           decay_time1=150, decay_rate2=0.001)))
    assert len(k_decay) == 100 and np.isfinite(k_decay).all()
    k_exp = np.asarray(cvimm.precompute_waning(100, dict(form='exp_decay', init_val=1.0, half_life=180)))
    assert len(k_exp) == 100 and np.isfinite(k_exp).all()


# --- calc_VE -----------------------------------------------------------------

def test_calc_VE_zero_is_zero():
    """calc_VE(0) is exactly 0 on every axis (no spurious protection for non-immune agents)."""
    for ax in ('sus', 'symp', 'sev'):
        assert cvimm.calc_VE(0.0, ax, NAB_EFF) == 0.0, f'calc_VE(0,{ax}) must be 0'
        assert np.array_equal(cvimm.calc_VE(np.zeros(5), ax, NAB_EFF), np.zeros(5))


def test_calc_VE_monotonic_and_bounded():
    """calc_VE is monotincreasing in NAb and stays in [0, 1)."""
    nabs = np.linspace(0, 20, 200)
    for ax in ('sus', 'symp', 'sev'):
        ve = cvimm.calc_VE(nabs, ax, NAB_EFF)
        assert np.all(np.diff(ve) >= -1e-12), f'calc_VE({ax}) must be non-decreasing'
        assert ve.min() >= 0.0 and ve.max() < 1.0, f'calc_VE({ax}) must be in [0,1)'


def test_calc_VE_invalid_axis():
    import pytest
    with pytest.raises(ValueError):
        cvimm.calc_VE(1.0, 'nope', NAB_EFF)


# === Task 2: NAb state + acquisition/boosting on cv.COVID ===

def test_use_waning_false_no_nab_state():
    """With use_waning=False (default) no NAb state evolves and no connector is attached."""
    sim = cv.Sim(pop_size=8000, pop_infected=30, pop_type='random', n_days=60, rand_seed=1, verbose=0)
    sim.run()
    d = sim.diseases.covid
    assert d.pars.use_waning is False
    assert len(sim.connectors) == 0
    assert d.nab_kin is None
    assert not (np.asarray(d.peak_nab) > 0).any(), 'no peak_nab set when use_waning=False'


def test_use_waning_true_acquires_nabs():
    """With use_waning=True, infected agents acquire a peak NAb and the kernel is precomputed."""
    sim = cv.Sim(pop_size=20000, pop_infected=100, pop_type='random', n_days=80, rand_seed=2,
                 use_waning=True, verbose=0)
    sim.run()
    d = sim.diseases.covid
    assert d.nab_kin is not None and len(d.nab_kin) == d.t.npts
    peak = np.asarray(d.peak_nab)
    # Everyone ever-infected (alive) has a positive peak NAb; never-infected have 0.
    ever_inf = (np.asarray(d.recovered) | np.asarray(d.infected))
    assert (peak[ever_inf] > 0).all(), 'every ever-infected agent has peak_nab>0'
    assert peak.mean() > 0


def test_peak_nab_scales_with_severity():
    """Initial peak NAb scales with symptom severity (severe > mild > asymptomatic; rel_imm_symp)."""
    covid = cv.COVID(init_prev=None, use_waning=True)
    sim = ss.Sim(people=cv.People(20000), diseases=covid, networks='random',
                 start=ss.date('2020-03-01'), dur=ss.days(1), dt=ss.days(1),
                 rand_seed=1, verbose=0, copy_inputs=False)
    sim.init()
    uids = sim.people.auids
    covid.set_prognoses(uids)
    peak = np.asarray(covid.peak_nab)
    tsym = np.asarray(covid.ti_symptomatic); tsev = np.asarray(covid.ti_severe)
    u = np.asarray(uids)
    asymp = u[np.isnan(tsym[u])]
    mild  = u[~np.isnan(tsym[u]) & np.isnan(tsev[u])]
    severe = u[~np.isnan(tsev[u])]
    # Mean peak NAb increases with severity (rel_imm_symp asymp=0.85 < mild=1.0 < severe=1.5).
    assert peak[asymp].mean() < peak[mild].mean() < peak[severe].mean(), \
        f'peak NAb should rise with severity: asymp={peak[asymp].mean():.2f} mild={peak[mild].mean():.2f} sev={peak[severe].mean():.2f}'


def test_nab_boost_on_reinfection():
    """A second infection boosts an agent's existing peak NAb by nab_boost (no fresh draw)."""
    covid = cv.COVID(init_prev=None, use_waning=True)
    sim = ss.Sim(people=cv.People(2000), diseases=covid, networks='random',
                 start=ss.date('2020-03-01'), dur=ss.days(1), dt=ss.days(1),
                 rand_seed=1, verbose=0, copy_inputs=False)
    sim.init()
    uids = sim.people.auids[:500]
    covid.set_prognoses(uids)
    peak1 = np.asarray(covid.peak_nab)[np.asarray(uids)].copy()
    # Re-infect the same agents: prior NAb (peak>0) => boosted by nab_boost, not redrawn.
    covid.set_prognoses(uids)
    peak2 = np.asarray(covid.peak_nab)[np.asarray(uids)]
    assert np.allclose(peak2, peak1 * covid.pars.nab_boost), 'reinfection boosts peak_nab by nab_boost'
