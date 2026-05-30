"""Tests for the ported-in v4 features (tutorial validation follow-up).

Covers: vaccination subtarget (vaccinate_prob / vaccinate_num / simple_vaccine), location-based age
distributions, custom-analyzer apply() support, and routed custom nab_decay parameters.
"""
import numpy as np
import covasim as cv


def _ages_of_vaccinated(sim):
    d = sim.diseases.covid
    return np.asarray(sim.people.age[d.vaccinated.uids])


def test_vaccinate_prob_subtarget_by_age():
    """vaccinate_prob with a subtarget vaccinates only the targeted (over-65) agents."""
    def over65(sim):
        return dict(inds=cv.true(np.asarray(sim.people.age) >= 65), vals=0.9)
    sim = cv.Sim(pop_size=8000, pop_infected=80, pop_type='hybrid', n_days=40, rand_seed=1,
                 use_waning=True, verbose=0,
                 interventions=cv.vaccinate_prob('pfizer', days=20, prob=0.0, subtarget=over65))
    sim.run()
    ages = _ages_of_vaccinated(sim)
    assert len(ages) > 0 and ages.min() >= 65, 'only over-65 vaccinated'


def test_vaccinate_num_subtarget_filter():
    """vaccinate_num subtarget with vals=0 excludes those agents (the booster-targeting pattern)."""
    def exclude_young(sim):
        return dict(inds=cv.true(np.asarray(sim.people.age) < 65), vals=0)
    sim = cv.Sim(pop_size=8000, pop_infected=80, pop_type='hybrid', n_days=40, rand_seed=1,
                 use_waning=True, verbose=0,
                 interventions=cv.vaccinate_num('pfizer', num_doses=300, subtarget=exclude_young))
    sim.run()
    ages = _ages_of_vaccinated(sim)
    assert len(ages) > 0 and ages.min() >= 65, 'under-65 excluded by subtarget vals=0'


def test_simple_vaccine_subtarget_runs():
    """simple_vaccine accepts an age-based subtarget (the tut_interventions pattern) and reduces spread."""
    def by_age(sim):
        age = np.asarray(sim.people.age)
        return dict(inds=sim.people.uid, vals=np.where(age < 50, 0.1, np.where(age < 75, 0.5, 0.9)))
    base = cv.Sim(pop_size=8000, pop_infected=80, pop_type='hybrid', n_days=40, rand_seed=1, verbose=0)
    base.run()
    sv = cv.simple_vaccine(days=15, rel_sus=0.5, rel_symp=0.1, subtarget=by_age)
    sim = cv.Sim(pop_size=8000, pop_infected=80, pop_type='hybrid', n_days=40, rand_seed=1, verbose=0,
                 interventions=sv)
    sim.run()
    b = float(np.asarray(base.diseases.covid.results['cum_infections']).max())
    s = float(np.asarray(sim.diseases.covid.results['cum_infections']).max())
    assert s < b, 'subtargeted vaccination should reduce cumulative infections'


def test_location_age_distribution():
    """location= draws ages from the country pyramid (Japan older, Bangladesh younger than default)."""
    default = cv.Sim(pop_size=8000, n_days=10, verbose=0); default.init()
    japan   = cv.Sim(pop_size=8000, n_days=10, location='japan', verbose=0); japan.init()
    bang    = cv.Sim(pop_size=8000, n_days=10, location='Bangladesh', verbose=0); bang.init()
    md = np.asarray(default.people.age).mean()
    mj = np.asarray(japan.people.age).mean()
    mb = np.asarray(bang.people.age).mean()
    assert mj > md > mb, f'expected Japan({mj:.1f}) > default({md:.1f}) > Bangladesh({mb:.1f})'


def test_custom_analyzer_apply():
    """A custom cv.Analyzer with a v3-style apply(sim) is called each step."""
    class recorder(cv.Analyzer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps = []
        def apply(self, sim):
            self.steps.append(int(sim.diseases.covid.infectious.sum()))
    sim = cv.Sim(pop_size=5000, pop_infected=50, n_days=20, verbose=0, analyzers=recorder(label='rec'))
    sim.run()
    rec = sim.get_analyzer('rec')
    assert len(rec.steps) == sim.diseases.covid.t.npts, 'apply() called every step'
    assert max(rec.steps) > 0


def test_historical_vaccinate_prob():
    """Pre-t=0 vaccination imprints decayed NAbs at t=0; older events leave lower NAbs than recent ones."""
    def run(day):
        sim = cv.Sim(n_days=5, pop_size=5000, pop_infected=20, use_waning=True, rand_seed=1, verbose=0,
                     interventions=cv.historical_vaccinate_prob(vaccine='pfizer', days=[day], prob=0.5))
        sim.run()
        d = sim.diseases.covid
        return d.vaccinated.uids, np.asarray(d.nab[d.vaccinated.uids])
    vacc, nab360 = run(-360)
    _, nab30 = run(-30)
    assert len(vacc) > 1500, '~50% vaccinated historically'
    assert nab360.mean() > 0, 'historical NAbs are decayed but nonzero at t=0'
    assert nab30.mean() > nab360.mean(), 'a 30-day-old event leaves higher NAbs than a 360-day-old one'


def test_prior_immunity_wave():
    """cv.prior_immunity(days, prob) seeds a prior natural-infection wave (recovered + NAbs at t=0)."""
    sim = cv.Sim(n_days=10, pop_size=5000, pop_infected=20, use_waning=True, rand_seed=1, verbose=0,
                 interventions=cv.prior_immunity(120, 0.05))
    sim.run()
    d = sim.diseases.covid
    assert int(np.asarray(d.recovered.raw).sum()) > 100, '~5% recovered from the prior wave'
    assert int((np.asarray(d.nab.raw) > 0).sum()) > 100, 'prior-wave agents carry NAbs'


def test_nab_histogram_edges_alias():
    """nab_histogram accepts the v3 `edges=` alias for `bins`."""
    import numpy as np
    sim = cv.Sim(n_days=20, pop_size=4000, pop_infected=50, use_waning=True, verbose=0,
                 analyzers=cv.nab_histogram(days=[15], edges=np.linspace(-4, 2, 13)))
    sim.run()
    assert len(sim.get_analyzer().hists) == 1


def test_nab_decay_custom_params_routed():
    """A custom nab_decay (passed as a kwarg) is routed to the COVID module and used."""
    sim = cv.Sim(pop_size=5000, pop_infected=50, n_days=40, use_waning=True, verbose=0,
                 nab_decay=dict(form='nab_growth_decay', growth_time=21, decay_rate1=0.07,
                                decay_time1=47, decay_rate2=0.02, decay_time2=106))
    sim.run()
    assert sim.diseases.covid.pars.nab_decay['growth_time'] == 21
    assert float(np.asarray(sim.diseases.covid.results['pop_nabs']).max()) > 0
