"""M4 waning-immunity + NAb tests.

Grows across M4 Tasks 1-3. Task 1 (this batch) covers the ported NAb-engine *functions*
(the waning kernel + calc_VE) -- pure-numpy ports of v3 _v2_legacy/immunity.py, wired into
the live engine in Tasks 2-3.
"""
import numpy as np
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
