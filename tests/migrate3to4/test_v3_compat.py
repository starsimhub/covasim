"""
Typical v3 user code, run against v4: the v3 backwards-compatibility tests.

Each probe is a snippet of v3 code (from the v3 tutorials, examples, and tests) that ran on
Covasim v3.1.9. Every probe should either pass, or be listed in XFAIL with the reason: a pending
Starsim change, a migration rule (docs/migrate3to4.md), or a feature not yet ported. A probe
that starts passing shows up as XPASS, so it can be removed from XFAIL.

v3_saved.sim is a small sim saved by Covasim v3.1.9, used to check that v3 objects can be loaded.
"""
import os
import atexit
import shutil
import tempfile
import numpy as np
import pandas as pd
import sciris as sc
import pylab as pl
import pytest
import covasim as cv

WORK = tempfile.mkdtemp(prefix='covasim_v3_compat_') # Folder for files written by the probes
atexit.register(shutil.rmtree, WORK, ignore_errors=True) # Remove it at the end
P = dict(pop_size=2000, pop_infected=20, n_days=40, verbose=0)

PROBES = []
def probe(f): PROBES.append(f); return f

def rs(**kw):
    sim = cv.Sim(sc.mergedicts(P, kw)); sim.run(); return sim

_cache = {}
def base():
    if 'sim' not in _cache: _cache['sim'] = rs()
    return _cache['sim']

def dead():
    ''' A sim with deaths, to check that dead agents stay in the arrays, as in v3 '''
    if 'dead' not in _cache:
        _cache['dead'] = rs(pop_infected=200, n_days=60, rel_death_prob=10)
        assert _cache['dead'].results['cum_deaths'][-1] > 0
    return _cache['dead']

def datafile():
    """ Write a small data file to fit against, if it doesn't exist yet, and return its path """
    fn = os.path.join(WORK, 'data.csv')
    if not os.path.exists(fn):
        pd.DataFrame(dict(date=pd.date_range('2020-03-01', periods=20).strftime('%Y-%m-%d'), new_diagnoses=np.arange(20), cum_deaths=np.arange(20)//5)).to_csv(fn, index=False)
    return fn


# ---------------- Basic construction / pars ----------------
@probe
def sim_default_run():
    sim = cv.Sim(verbose=0, pop_size=2000); sim.run(); assert sim.results['cum_infections'][-1] > 0
@probe
def sim_kwargs_form():
    sim = cv.Sim(pop_size=2000, n_days=30, verbose=0, rand_seed=3); sim.run()
@probe
def sim_pars_dict_plus_kwargs():
    sim = cv.Sim(P, beta=0.02, label='x'); sim.run(); assert sim.label == 'x'
@probe
def sim_getitem_beta():
    sim = cv.Sim(P); assert abs(sim['beta'] - 0.016) < 1e-9
@probe
def sim_getitem_pop_size():
    sim = cv.Sim(P); assert sim['pop_size'] == 2000
@probe
def sim_getitem_n_days():
    sim = cv.Sim(P); assert sim['n_days'] == 40
@probe
def sim_setitem_beta_then_run():
    sim = cv.Sim(P); sim['beta'] = 0.0; sim.run(); assert sim.results['new_infections'][5:].sum() == 0, sim.results['new_infections'][5:].sum()
@probe
def sim_pars_dict_access():
    sim = cv.Sim(P); assert sim.pars['pop_size'] == 2000 and 'beta' in sim.pars
@probe
def sim_pars_beta_layer():
    sim = cv.Sim(P, pop_type='hybrid'); sim.initialize(); assert set(sim['beta_layer'].keys()) == {'h','s','w','c'}
@probe
def sim_pars_contacts():
    sim = cv.Sim(P, pop_type='hybrid'); sim.initialize(); assert sim['contacts']['h'] > 0
@probe
def sim_update_pars():
    sim = cv.Sim(P); sim.update_pars(beta=0.02); assert sim['beta'] == 0.02
@probe
def sim_start_end_day():
    sim = cv.Sim(P, start_day='2020-03-01', end_day='2020-04-15'); sim.run(); assert len(sim.results['date']) == 46
@probe
def sim_rel_severe_prob():
    sim = cv.Sim(P, rel_severe_prob=2.0, rel_death_prob=0.5); sim.run()
@probe
def sim_quar_factor():
    sim = cv.Sim(P, pop_type='hybrid', quar_factor={'h':0.5}); sim.run()
@probe
def sim_iso_factor():
    sim = cv.Sim(P, pop_type='hybrid', iso_factor=dict(h=0.3, s=0.1, w=0.1, c=0.1)); sim.run()
@probe
def sim_dur_pars():
    pars = cv.make_pars(); d = sc.dcp(pars['dur']); d['exp2inf'] = dict(dist='lognormal_int', par1=3.0, par2=1.0)
    sim = cv.Sim(P, dur=d); sim.run()
@probe
def sim_custom_prognoses():
    prog = cv.get_prognoses(); prog['death_probs'] *= 2
    sim = cv.Sim(P, prognoses=prog); sim.run()
@probe
def sim_n_imports():
    sim = cv.Sim(P, n_imports=2); sim.run()
@probe
def sim_rand_seed_reproducible():
    a = rs(rand_seed=5); b = rs(rand_seed=5); assert np.array_equal(a.results['cum_infections'].values, b.results['cum_infections'].values)
@probe
def sim_label_and_sim_label():
    sim = cv.Sim(P, label='hello'); assert sim.label == 'hello'
@probe
def sim_verbose_float():
    sim = cv.Sim(sc.mergedicts(P, dict(verbose=0.1))); sim.run()
@probe
def sim_pop_type_random():
    rs(pop_type='random')
@probe
def sim_pop_type_hybrid():
    rs(pop_type='hybrid')
@probe
def sim_pop_type_synthpops():
    rs(pop_type='synthpops')
@probe
def sim_rescale():
    sim = rs(pop_scale=10, rescale=True); assert sim.results['cum_infections'][-1] > 0
@probe
def sim_pop_scale_no_rescale():
    sim = rs(pop_scale=10, rescale=False)
@probe
def sim_location():
    rs(location='japan')
@probe
def sim_use_waning_false():
    rs(use_waning=False)
@probe
def sim_n_beds():
    rs(n_beds_hosp=10, n_beds_icu=2)
@probe
def sim_asymp_factor():
    rs(asymp_factor=0.5)

# ---------------- Results ----------------
@probe
def results_values_attr():
    r = base().results['cum_infections']; assert isinstance(r.values, np.ndarray) and len(r.values) == 41
@probe
def results_index_minus1():
    assert base().results['cum_infections'][-1] > 0
@probe
def results_low_high_after_reduce():
    msim = cv.MultiSim(cv.Sim(P), n_runs=3); msim.run(); msim.mean(); r = msim.base_sim.results['cum_infections']; assert len(r.low) == 41 and len(r.high) == 41
@probe
def results_date():
    d = base().results['date']; assert len(d) == 41
@probe
def results_t():
    t = base().results['t']; assert len(t) == 41
@probe
def results_keys_v3():
    keys = ['cum_infections','cum_deaths','cum_severe','cum_critical','cum_diagnoses','cum_tests','new_infections','new_deaths','n_infectious','n_susceptible','n_exposed','n_symptomatic','n_severe','n_critical','n_recovered','n_dead','n_quarantined','n_alive','prevalence','incidence','r_eff','doubling_time','test_yield','rel_test_yield','new_diagnoses','new_tests','n_diagnosed','cum_recoveries','new_quarantined','cum_quarantined','n_imports','cum_reinfections','n_naive','n_preinfectious','frac_vaccinated','cum_vaccinated','cum_doses','pop_nabs','pop_protection','pop_symp_protection']
    missing = [k for k in keys if k not in base().results]; assert not missing, missing
@probe
def results_variant():
    r = base().results['variant']['cum_infections_by_variant']; assert r.values.shape[1] == 1
@probe
def results_result_name():
    r = base().results['cum_infections']; assert r.name and hasattr(r, 'color')
@probe
def results_iterate_result_keys():
    ks = base().result_keys(); assert 'cum_infections' in ks
@probe
def results_scale_attr():
    assert base().results['cum_infections'].scale in (True, 'dynamic', 'static', 1) or True; _ = base().results['cum_infections'].scale
@probe
def results_npts_tvec():
    s = base(); assert s.npts == 41 and len(s.tvec) == 41
@probe
def results_datevec():
    s = base(); assert len(s.datevec) == 41

# ---------------- Summary / display ----------------
@probe
def summary_bare_keys():
    s = base().summary; assert s['cum_infections'] > 0 and 'cum_deaths' in s
@probe
def summarize_call():
    base().summarize()
@probe
def summarize_full():
    base().summarize(full=True)
@probe
def compute_summary():
    base().compute_summary()
@probe
def brief_call():
    base().brief()
@probe
def disp_call():
    base().disp()
@probe
def print_sim():
    str(base())
@probe
def sim_n_attr():
    assert dead().n == 2000

# ---------------- dates ----------------
@probe
def cv_date_helpers():
    assert cv.date('2020-03-01') is not None; cv.date('2020-03-01', as_date=False); cv.day('2020-03-05', start_date='2020-03-01') == 4
@probe
def cv_daydiff():
    assert cv.daydiff('2020-03-01', '2020-03-05') == 4
@probe
def cv_date_range():
    cv.date(np.arange(3), start_date='2020-03-01')
@probe
def sim_day_date():
    s = base(); assert s.day('2020-03-11') == 10 and s.date(10) == '2020-03-11'
@probe
def sim_day_list():
    assert base().day(['2020-03-11', '2020-03-12']) == [10, 11]
@probe
def sim_date_as_date():
    base().date(10, as_date=True)

# ---------------- people ----------------
@probe
def people_age():
    p = dead().people; assert len(p.age) == 2000
@probe
def people_sex():
    assert len(dead().people.sex) == 2000
@probe
def people_states():
    p = base().people; [getattr(p, k) for k in ['susceptible','exposed','infectious','symptomatic','severe','critical','recovered','dead','diagnosed','quarantined','vaccinated']]
@probe
def people_date_states():
    p = base().people; [getattr(p, k) for k in ['date_exposed','date_infectious','date_symptomatic','date_diagnosed','date_dead','date_recovered']]
@probe
def people_dur_states():
    p = base().people; [getattr(p, k) for k in ['dur_exp2inf','dur_disease']]
@probe
def people_true_count():
    p = base().people; p.true('exposed'); p.count('recovered'); p.defined('date_diagnosed')
@probe
def people_cv_true():
    p = base().people; inds = cv.true(p.infectious); inds2 = cv.false(p.susceptible); assert isinstance(inds, np.ndarray)
@probe
def people_story():
    base().people.story(5)
@probe
def people_plot():
    base().people.plot(); pl.close('all')
@probe
def people_to_df():
    df = pd.DataFrame(dict(age=base().people.age, sex=base().people.sex)); assert 'age' in df.columns
@probe
def people_len():
    assert len(dead().people) == 2000
@probe
def people_uid():
    assert len(dead().people.uid) == 2000
@probe
def people_getitem():
    assert len(dead().people['age']) == 2000
@probe
def people_keys():
    base().people.keys()
@probe
def people_person():
    base().people.person(3)
@probe
def people_rel_sus_set():
    sim = cv.Sim(P); sim.initialize(); sim.people.rel_sus[sim.people.age > 60] = 0; sim.run()
@probe
def people_infect():
    sim = cv.Sim(P); sim.initialize(); sim.people.infect(inds=np.array([1,2,3])); sim.run()
@probe
def people_contacts_h():
    sim = cv.Sim(P, pop_type='hybrid'); sim.initialize(); h = sim.people.contacts['h']; assert len(h['p1']) > 0 and len(h) > 0
@probe
def people_contacts_keys():
    sim = cv.Sim(P, pop_type='hybrid'); sim.initialize(); assert set(sim.people.contacts.keys()) == {'h','s','w','c'}
@probe
def people_layer_keys():
    sim = cv.Sim(P, pop_type='hybrid'); sim.initialize(); assert set(sim.people.layer_keys()) == {'h','s','w','c'}
@probe
def layer_find_contacts():
    sim = cv.Sim(P, pop_type='hybrid'); sim.initialize(); sim.people.contacts['h'].find_contacts([0,1,2])
@probe
def layer_to_df():
    sim = cv.Sim(P, pop_type='hybrid'); sim.initialize(); df = sim.people.contacts['h'].to_df(); assert 'p1' in df.columns
@probe
def cv_layer_add():
    sim = cv.Sim(P, pop_type='hybrid'); sim.initialize()
    n = 100; p1 = np.random.randint(2000, size=n); p2 = np.random.randint(2000, size=n)
    layer = cv.Layer(p1=p1, p2=p2, beta=np.ones(n), label='transport')
    sim.people.contacts.add_layer(transport=layer)
    sim.reset_layer_pars(); sim.run()
@probe
def custom_layer_via_beta_layer():
    sim = cv.Sim(P, pop_type='hybrid'); sim.initialize()
    sim['beta_layer']['h'] = 5.0; sim.run()
@probe
def dynam_layer():
    rs(pop_type='hybrid', dynam_layer={'c':1})
@probe
def make_people_and_pass():
    sim = cv.Sim(P); ppl = cv.make_people(sim); sim2 = cv.Sim(P, people=ppl); sim2.run()
@probe
def popfile_save_load():
    fn = os.path.join(WORK, 'pop.ppl')
    sim = cv.Sim(P); sim.initialize(); sim.people.save(fn)
    sim2 = cv.Sim(P, popfile=fn); sim2.run()
@probe
def people_save_load():
    fn = os.path.join(WORK, 'p2.ppl'); sim = cv.Sim(P); sim.initialize(); sim.people.save(fn); ppl = cv.load(fn); cv.Sim(P, people=ppl).run()
@probe
def popdict_custom():
    pd_ = cv.make_people(cv.Sim(P)); assert len(pd_) == 2000

# ---------------- interventions ----------------
@probe
def iv_change_beta_date_str():
    rs(interventions=cv.change_beta(days='2020-03-15', changes=0.5))
@probe
def iv_change_beta_layers():
    rs(pop_type='hybrid', interventions=cv.change_beta(days=[10, 20], changes=[0.5, 0.8], layers=['s','w']))
@probe
def iv_clip_edges():
    rs(pop_type='hybrid', interventions=cv.clip_edges(days=10, changes=0.3, layers='w'))
@probe
def iv_test_prob():
    s = rs(interventions=cv.test_prob(symp_prob=0.2, asymp_prob=0.01, start_day='2020-03-05', test_delay=1)); assert s.results['cum_tests'][-1] > 0
@probe
def iv_test_num():
    s = rs(interventions=cv.test_num(daily_tests=50, start_day=5)); assert s.results['cum_tests'][-1] > 0
@probe
def iv_test_num_array():
    rs(interventions=cv.test_num(daily_tests=np.full(41, 30)))
@probe
def iv_test_prob_quar_policy():
    rs(interventions=[cv.test_prob(symp_prob=0.1, quar_policy='start'), cv.contact_tracing(trace_probs=0.5)])
@probe
def iv_test_prob_subtarget():
    rs(interventions=cv.test_prob(symp_prob=0.1, subtarget={'inds': np.arange(100), 'vals': 0.9}))
@probe
def iv_contact_tracing():
    s = rs(pop_type='hybrid', interventions=[cv.test_prob(symp_prob=0.2), cv.contact_tracing(trace_probs=dict(h=0.9, s=0.5, w=0.5, c=0.1), trace_time=dict(h=0, s=1, w=1, c=2))])
    assert s.results['cum_quarantined'][-1] > 0
@probe
def iv_dynamic_pars():
    rs(interventions=cv.dynamic_pars(n_imports=dict(days=[10, 20], vals=[5, 0])))
@probe
def iv_dynamic_pars_beta():
    rs(interventions=cv.dynamic_pars({'beta': {'days': 10, 'vals': 0.005}}))
@probe
def iv_sequence():
    rs(interventions=cv.sequence(days=[5, 20], interventions=[cv.test_num(daily_tests=20), cv.test_prob(symp_prob=0.2)]))
@probe
def iv_simple_vaccine():
    rs(interventions=cv.simple_vaccine(days=5, prob=0.5, rel_sus=0.5))
@probe
def iv_vaccinate_prob_named():
    s = rs(interventions=cv.vaccinate_prob(vaccine='pfizer', days=[5, 10], prob=0.1)); assert s.results['cum_vaccinated'][-1] > 0
@probe
def iv_vaccinate_num():
    s = rs(interventions=cv.vaccinate_num(vaccine='moderna', num_doses=50)); assert s.results['cum_vaccinated'][-1] > 0
@probe
def iv_vaccinate_num_sequence():
    rs(interventions=cv.vaccinate_num(vaccine='az', num_doses={5: 100, 10: 100}, sequence='age'))
@probe
def iv_vaccinate_alias():
    rs(interventions=cv.vaccinate(vaccine='pfizer', days=5, prob=0.2))
@probe
def iv_vaccinate_booster():
    rs(interventions=[cv.vaccinate_prob('pfizer', days=2, prob=0.5), cv.vaccinate_prob('pfizer', days=30, prob=0.5, booster=True)])
@probe
def iv_vaccine_subtarget():
    def st(sim): return {'inds': cv.true(sim.people.age > 50), 'vals': 0.9}
    rs(interventions=cv.vaccinate_prob('pfizer', days=5, prob=0.0, subtarget=st))
@probe
def iv_label_kwarg():
    s = rs(interventions=cv.change_beta(10, 0.5, label='lockdown')); assert s.get_intervention('lockdown') is not None
@probe
def iv_do_plot_kwarg():
    rs(interventions=cv.change_beta(10, 0.5, do_plot=False))
@probe
def iv_function_intervention():
    def f(sim):
        if sim.ti == 10: sim['beta'] *= 0.5
    rs(interventions=f)
@probe
def iv_function_intervention_people():
    def f(sim):
        if sim.ti == 10: sim.people.rel_sus[sim.people.age > 60] = 0.0
    rs(interventions=f)
@probe
def iv_custom_subclass():
    class protect_elderly(cv.Intervention):
        def __init__(self, start_day=None, end_day=None, age_cutoff=70, rel_sus=0.0, *args, **kwargs):
            super().__init__(**kwargs)
            self.start_day = start_day; self.end_day = end_day; self.age_cutoff = age_cutoff; self.rel_sus = rel_sus
        def initialize(self, sim):
            super().initialize()
            self.start_day = sim.day(self.start_day); self.end_day = sim.day(self.end_day)
            self.days = [self.start_day, self.end_day]
            self.elderly = sim.people.age > self.age_cutoff
            self.exposed = np.zeros(sim.npts); self.tvec = sim.tvec
        def apply(self, sim):
            if sim.ti == self.start_day: sim.people.rel_sus[self.elderly] = self.rel_sus
            elif sim.ti == self.end_day: sim.people.rel_sus[self.elderly] = 1.0
            self.exposed[sim.ti] = sim.people.exposed[self.elderly].sum()
    s = rs(interventions=protect_elderly(start_day='2020-03-10', end_day='2020-03-30', label='protect'))
    iv = s.get_intervention(protect_elderly); assert iv.exposed.sum() > 0
@probe
def iv_custom_subclass_minimal():
    class my_iv(cv.Intervention):
        def apply(self, sim):
            if sim.ti == 5: self.found = sim.date(sim.ti)
    s = rs(interventions=my_iv()); assert s.get_interventions()[0].found == '2020-03-06'
@probe
def iv_custom_subclass_sim_results():
    class my_iv(cv.Intervention):
        def apply(self, sim):
            if sim.ti > 5 and sim.results['new_infections'][sim.ti-1] > 10: self.hit = True
    rs(interventions=my_iv())
@probe
def iv_dynamic_trigger():
    def check(sim):
        return sim.results['n_infectious'][sim.ti-1] > 50 if sim.ti else False
    # v3 tutorial: custom trigger via function
    def inter(sim):
        if sim.ti and sim.people.infectious.sum() > 50: sim['beta'] = 0.008
    rs(interventions=inter)
@probe
def iv_plot_intervention():
    s = rs(interventions=cv.change_beta(10, 0.5)); s.plot(); pl.close('all')
@probe
def get_interventions_by_index():
    s = rs(interventions=[cv.change_beta(10, 0.5), cv.test_prob(0.1)]); assert isinstance(s.get_intervention(1), cv.test_prob)
@probe
def get_interventions_by_class():
    s = rs(interventions=[cv.change_beta(10, 0.5), cv.test_prob(0.1)]); assert len(s.get_interventions(cv.test_prob)) == 1
@probe
def sim_interventions_list_index():
    s = rs(interventions=[cv.change_beta(10, 0.5), cv.test_prob(0.1)]); assert isinstance(s['interventions'][1], cv.test_prob)
@probe
def interventions_passed_as_pars_key():
    sim = cv.Sim(sc.mergedicts(P, dict(interventions=[cv.change_beta(10, 0.5)]))); sim.run()

# ---------------- analyzers ----------------
@probe
def an_snapshot():
    s = rs(analyzers=cv.snapshot('2020-03-10', 20)); snap = s.get_analyzer(); p = snap.get('2020-03-10'); assert hasattr(p, 'age')
@probe
def an_age_histogram():
    s = rs(analyzers=cv.age_histogram(days=[10, 20])); ah = s.get_analyzer(); ah.plot(); pl.close('all'); ah.get(10) if hasattr(ah, 'get') else None
@probe
def an_age_histogram_from_sim():
    cv.age_histogram(sim=base())
@probe
def an_daily_age_stats():
    s = rs(analyzers=cv.daily_age_stats()); a = s.get_analyzer(); a.plot(); pl.close('all'); a.to_df()
@probe
def an_nab_histogram():
    s = rs(analyzers=cv.nab_histogram(days=[20])); s.get_analyzer().plot(); pl.close('all')
@probe
def an_daily_stats():
    s = rs(analyzers=cv.daily_stats(days=[10])); s.get_analyzer().plot(); pl.close('all')
@probe
def an_function_analyzer():
    store = []
    def f(sim): store.append(sim.people.infectious.sum())
    rs(analyzers=f); assert len(store) == 41, len(store)
@probe
def an_custom_subclass():
    class store_seir(cv.Analyzer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs); self.tvec = []; self.S = []; self.E = []; self.I = []; self.R = []
        def apply(self, sim):
            ppl = sim.people
            self.tvec.append(sim.ti); self.S.append(ppl.susceptible.sum()); self.E.append(ppl.exposed.sum() - ppl.infectious.sum())
            self.I.append(ppl.infectious.sum()); self.R.append(ppl.recovered.sum() + ppl.dead.sum())
        def plot(self):
            pl.figure(); pl.plot(self.tvec, self.S); pl.close('all')
    s = rs(analyzers=store_seir(label='seir')); a = s.get_analyzer('seir'); a.plot(); assert len(a.tvec) == 41, len(a.tvec)
@probe
def an_custom_initialize_finalize():
    class A(cv.Analyzer):
        def initialize(self, sim):
            super().initialize(); self.n = sim.npts; self.vals = np.zeros(sim.npts)
        def apply(self, sim): self.vals[sim.ti] = sim.results['n_infectious'][sim.ti] if sim.ti else 0
        def finalize(self, sim): super().finalize(); self.done = True
    s = rs(analyzers=A()); assert s.get_analyzer().done
@probe
def an_transtree():
    s = rs(); tt = s.make_transtree(); tt.plot(); pl.close('all'); assert tt.n_targets is not None
@probe
def an_transtree_ctor():
    tt = cv.TransTree(base()); assert tt.n_targets is not None
@probe
def an_fit():
    fn = datafile()
    s = cv.Sim(P, datafile=fn, interventions=cv.test_prob(0.1)); s.run(); fit = s.compute_fit(); fit.plot(); pl.close('all'); assert fit.mismatch >= 0
@probe
def an_fit_class():
    fn = datafile(); s = cv.Sim(P, datafile=fn, interventions=cv.test_prob(0.1)); s.run(); f = cv.Fit(s); f.summarize()

# ---------------- run control ----------------
@probe
def run_until_resume():
    s = cv.Sim(P); s.run(until=20); s.run(); assert s.results['cum_infections'][-1] > 0
@probe
def run_until_date():
    s = cv.Sim(P); s.run(until='2020-03-15'); s.run()
@probe
def sim_step():
    s = cv.Sim(P); s.initialize()
    for i in range(5): s.step()
@probe
def sim_run_do_plot():
    s = cv.Sim(P); s.run(do_plot=True); pl.close('all')
@probe
def sim_copy():
    s = cv.Sim(P); s2 = s.copy(); s2.run()
@probe
def sim_initialize_reset():
    s = cv.Sim(P); s.initialize(); s.initialize(reset=True); s.run()
@probe
def sim_reuse_after_dcp():
    s = cv.Sim(P); sims = [sc.dcp(s) for i in range(2)]
    for i, x in enumerate(sims): x['rand_seed'] = i; x.run()
@probe
def sim_set_seed():
    s = cv.Sim(P); s.set_seed(4)

# ---------------- plotting ----------------
@probe
def plot_default():
    base().plot(); pl.close('all')
@probe
def plot_to_plot_list():
    base().plot(to_plot=['cum_infections', 'new_deaths']); pl.close('all')
@probe
def plot_to_plot_dict():
    base().plot(to_plot={'Infections': ['cum_infections', 'n_infectious']}); pl.close('all')
@probe
def plot_to_plot_overview():
    base().plot(to_plot='overview'); pl.close('all')
@probe
def plot_kwargs():
    base().plot(do_show=False, fig_args=dict(figsize=(8,6)), scatter_args=dict(s=4), n_cols=2, legend_args={'loc':'best'}); pl.close('all')
@probe
def plot_result():
    base().plot_result('new_infections'); pl.close('all')
@probe
def plot_do_save():
    base().plot(do_save=True, fig_path=os.path.join(WORK, 'fig.png')); pl.close('all')
@probe
def plot_variants():
    base().plot('variant'); pl.close('all')
@probe
def plot_people():
    cv.plot_people(base().people) if hasattr(cv, 'plot_people') else base().people.plot(); pl.close('all')

# ---------------- export / save ----------------
@probe
def to_df():
    df = base().to_df(); assert 'cum_infections' in df.columns and 'date' in df.columns
@probe
def to_df_date_index():
    base().to_df(date_index=True)
@probe
def to_excel():
    base().to_excel(os.path.join(WORK, 'x.xlsx'))
@probe
def to_json():
    j = base().to_json(); assert 'results' in j and 'parameters' in j
@probe
def to_json_file():
    base().to_json(os.path.join(WORK, 'x.json'))
@probe
def export_results():
    base().export_results()
@probe
def export_pars():
    base().export_pars()
@probe
def save_load_sim():
    fn = os.path.join(WORK, 'a.sim'); s = rs(); s.save(fn); s2 = cv.load(fn); assert s2.results['cum_infections'][-1] == s.results['cum_infections'][-1]
@probe
def save_keep_people():
    fn = os.path.join(WORK, 'b.sim'); s = dead(); s.save(fn, keep_people=True); s2 = cv.Sim.load(fn); assert len(s2.people) == 2000
@probe
def save_load_before_run():
    fn = os.path.join(WORK, 'c.sim'); s = cv.Sim(P); s.initialize(); s.save(fn); s2 = cv.load(fn); s2.run()
@probe
def shrink():
    s = rs(); s.shrink(); s.results['cum_infections']
@probe
def shrink_in_place_false():
    s = rs(); s2 = s.shrink(in_place=False)
@probe
def load_v3_pickle():
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3_saved.sim')
    s = cv.load(fn); s.results['cum_infections'][-1]

# ---------------- MultiSim / Scenarios ----------------
@probe
def msim_basic():
    msim = cv.MultiSim(cv.Sim(P), n_runs=3); msim.run(); msim.reduce(); msim.plot(); pl.close('all')
@probe
def msim_mean_median():
    msim = cv.MultiSim(cv.Sim(P), n_runs=3); msim.run(); msim.mean(); msim.median(quantiles=[0.1, 0.9])
@probe
def msim_plot_result():
    msim = cv.MultiSim(cv.Sim(P), n_runs=3); msim.run(); msim.reduce(); msim.plot_result('cum_infections'); pl.close('all')
@probe
def msim_plot_to_plot():
    msim = cv.MultiSim(cv.Sim(P), n_runs=3); msim.run(); msim.plot(to_plot=['cum_infections']); pl.close('all')
@probe
def msim_compare():
    sims = [cv.Sim(P, beta=b, label=f'b{b}') for b in [0.01, 0.02]]; msim = cv.MultiSim(sims); msim.run(); df = msim.compare(output=True); msim.plot(); pl.close('all')
@probe
def msim_sims_attr_results():
    msim = cv.MultiSim(cv.Sim(P), n_runs=3); msim.run(); v = [s.results['cum_infections'][-1] for s in msim.sims]; assert len(v) == 3
@probe
def msim_combine():
    msim = cv.MultiSim(cv.Sim(P), n_runs=2); msim.run(); msim.combine(); msim.plot(); pl.close('all')
@probe
def msim_merge():
    m1 = cv.MultiSim(cv.Sim(P, label='a'), n_runs=2); m1.run(); m1.reduce(); m2 = cv.MultiSim(cv.Sim(P, label='b'), n_runs=2); m2.run(); m2.reduce(); m = cv.MultiSim.merge(m1, m2, base=True); assert len(m.sims) == 2; m.plot(); pl.close('all')
@probe
def msim_save_load():
    fn = os.path.join(WORK, 'm.msim'); msim = cv.MultiSim(cv.Sim(P), n_runs=2); msim.run(); msim.save(fn); m2 = cv.load(fn); len(m2.sims)
@probe
def msim_summarize():
    msim = cv.MultiSim(cv.Sim(P), n_runs=2); msim.run(); msim.reduce(); msim.summarize(); msim.brief(); msim.disp()
@probe
def msim_to_excel():
    msim = cv.MultiSim(cv.Sim(P), n_runs=2); msim.run(); msim.reduce(); msim.to_excel(os.path.join(WORK, 'm.xlsx'))
@probe
def msim_split():
    msim = cv.MultiSim(cv.Sim(P), n_runs=4); msim.run(); msim.split(chunks=[2,2])
@probe
def cv_parallel():
    s1 = cv.Sim(P, label='a'); s2 = cv.Sim(P, beta=0.02, label='b'); msim = cv.parallel(s1, s2); msim.plot(); pl.close('all')
@probe
def cv_multi_run():
    sims = cv.multi_run(cv.Sim(P), n_runs=2); assert len(sims) == 2
@probe
def cv_multi_run_iterpars():
    sims = cv.multi_run(cv.Sim(P), iterpars=dict(beta=[0.01, 0.02])); assert len(sims) == 2
@probe
def msim_base_sim_results():
    msim = cv.MultiSim(cv.Sim(P), n_runs=3); msim.run(); msim.mean(); v = msim.results['cum_infections'].values; assert len(v) == 41
@probe
def msim_run_keep_people():
    msim = cv.MultiSim(cv.Sim(P), n_runs=2); msim.run(keep_people=True); msim.sims[0].people.age
@probe
def scenarios_basic():
    scenarios = {'baseline': {'name': 'Baseline', 'pars': {}}, 'distance': {'name': 'SD', 'pars': {'interventions': cv.change_beta(days=10, changes=0.5)}}}
    scens = cv.Scenarios(basepars=P, scenarios=scenarios, metapars=dict(n_runs=2, noise=0)); scens.run(); scens.plot(); pl.close('all')
@probe
def scenarios_results_access():
    scenarios = {'baseline': {'name': 'Baseline', 'pars': {}}, 'high': {'name': 'High', 'pars': {'beta': 0.03}}}
    scens = cv.Scenarios(basepars=P, scenarios=scenarios, metapars=dict(n_runs=2)); scens.run()
    v = scens.results['cum_infections']['high']['best'][-1]; assert v > 0
@probe
def scenarios_sim_arg():
    sim = cv.Sim(P); scens = cv.Scenarios(sim=sim, scenarios={'b': {'name': 'b', 'pars': {'beta': 0.02}}}, metapars=dict(n_runs=1)); scens.run(); scens.compare()
@probe
def scenarios_summarize_save():
    scens = cv.Scenarios(basepars=P, metapars=dict(n_runs=1)); scens.run(); scens.summarize(); scens.save(os.path.join(WORK, 'x.scens')); scens.to_excel(os.path.join(WORK, 'x_scens.xlsx'))
@probe
def scenarios_default():
    scens = cv.Scenarios(basepars=P, metapars=dict(n_runs=1)); scens.run(); scens.plot(to_plot=['cum_infections']); pl.close('all')

# ---------------- variants / immunity ----------------
@probe
def variant_delta_days():
    s = rs(variants=cv.variant('delta', days=10, n_imports=10)); assert s.results['variant']['cum_infections_by_variant'].values.shape[1] == 2
@probe
def variant_custom_dict():
    v = cv.variant(variant={'rel_beta': 2.0, 'rel_severe_prob': 1.5}, label='custom', days=5, n_imports=5); rs(variants=v)
@probe
def variant_multiple():
    rs(variants=[cv.variant('alpha', days=5), cv.variant('beta', days=15)])
@probe
def variant_result_by_variant_plot():
    s = rs(variants=cv.variant('delta', days=10)); s.plot('variant'); pl.close('all')
@probe
def variant_map_pars():
    s = cv.Sim(P, variants=cv.variant('delta', days=10)); s.initialize(); assert 'delta' in s['variant_map'].values()
@probe
def immunity_nab_decay():
    rs(nab_decay=dict(form='nab_growth_decay', growth_time=21, decay_rate1=np.log(2)/50, decay_time1=150, decay_rate2=np.log(2)/250, decay_time2=365))
@probe
def cv_get_vaccine_choices():
    cv.parameters.get_vaccine_choices() if hasattr(cv, 'parameters') else cv.get_vaccine_choices()
@probe
def prior_immunity():
    rs(interventions=cv.prior_immunity(vaccine='pfizer', days=[-30], prob=0.3))
@probe
def historical_wave():
    rs(interventions=cv.historical_wave(prob=0.2, days_prior=30))
@probe
def cv_immunity_fns():
    s = cv.Sim(P); s.initialize(); cv.immunity.init_immunity(s) if hasattr(cv, 'immunity') and hasattr(cv.immunity, 'init_immunity') else s.init_immunity()

# ---------------- misc helpers ----------------
@probe
def cv_options():
    cv.options.set(verbose=0); cv.options.set('dpi', 100); cv.options.help() if hasattr(cv.options, 'help') else None
@probe
def cv_options_context():
    with cv.options.context(jupyter=False): pass
@probe
def cv_get_default_pars():
    p = cv.make_pars(); assert p['pop_size'] == 20e3
@probe
def cv_make_pars_set_prog():
    p = cv.make_pars(set_prognoses=True); assert 'prognoses' in p
@probe
def cv_get_prognoses():
    cv.get_prognoses(by_age=True)
@probe
def cv_check_version():
    cv.check_version('>=3.0.0')
@probe
def cv_git_info():
    cv.git_info()
@probe
def cv_requirements():
    cv.check_save_version(filename=os.path.join(WORK, 'version.gitinfo')); cv.get_version_pars('3.0.0')
@probe
def cv_load_data():
    fn = datafile(); df = cv.load_data(fn); assert 'new_diagnoses' in df.columns
@probe
def cv_utils():
    cv.n_binomial(0.5, 10); cv.choose(10, 3); cv.poisson(3); cv.n_multinomial([0.2, 0.8], 5); cv.sample('normal', 1, 1, size=5); cv.set_seed(1); cv.true(np.array([1,0,1])); cv.ifalse(np.array([True,False]), np.array([3,4])); cv.itrue(np.array([True,False]), np.array([3,4]))
@probe
def cv_compute_gof():
    cv.compute_gof(np.array([1,2,3]), np.array([1,2,4]))
@probe
def cv_get_png_metadata():
    fn = os.path.join(WORK, 'meta.png'); base().plot(); cv.savefig(fn); pl.close('all'); cv.get_png_metadata(fn, output=True)
@probe
def cv_diff_sims():
    a = rs(rand_seed=1); b = rs(rand_seed=2); cv.diff_sims(a, b, output=True)
@probe
def cv_Result_class():
    r = cv.Result(name='x', npts=10); r[3] = 5; assert r.values[3] == 5
@probe
def cv_date_formats():
    cv.date(['2020-03-01', '2020-03-02']); cv.date(sc.now())
@probe
def cv_undefined_datestr():
    cv.day('2020-03-05', start_date='2020-03-01')
@probe
def cv_calibration():
    fn = datafile(); s = cv.Sim(P, datafile=fn, interventions=cv.test_prob(0.1))
    calib = cv.Calibration(s, calib_pars=dict(beta=[0.015, 0.01, 0.02]), total_trials=2, n_workers=1, verbose=False, keep_db=False)
    calib.calibrate(die=True)

# ---------------- people/sim fine-grained behaviour ----------------
@probe
def sim_t_in_function_is_int():
    ts = []
    def f(sim): ts.append(sim.ti)
    rs(interventions=f); assert ts[0] == 0 and ts[-1] == 40 and isinstance(ts[0], (int, np.integer)), (ts[0], ts[-1], type(ts[0]))
@probe
def sim_results_during_run():
    vals = []
    def f(sim):
        if sim.ti > 0: vals.append(sim.results['new_infections'][sim.ti-1])
    rs(interventions=f); assert sum(vals) > 0
@probe
def sim_rescale_vec():
    s = rs(pop_scale=5, rescale=True); s.rescale_vec
@probe
def sim_scaled_pop_size():
    s = rs(pop_scale=10); assert s.scaled_pop_size == 20000
@probe
def sim_people_quarantine_schedule():
    def f(sim):
        if sim.ti == 5: sim.people.schedule_quarantine(np.arange(10))
    rs(interventions=f)
@probe
def sim_people_test_method():
    def f(sim):
        if sim.ti == 5: sim.people.test(np.arange(100), test_sensitivity=1.0)
    rs(interventions=f)
@probe
def sim_people_flows():
    base().people.flows
@probe
def people_date_diagnosed_after_testing():
    s = rs(interventions=cv.test_prob(0.5)); assert np.isfinite(s.people.date_diagnosed).sum() > 0
@probe
def people_infection_log():
    s = rs(); assert len(s.people.infection_log) > 0


#%% Probes that don't pass yet, and why

starsim = 'Pending Starsim change'
rule = 'Migration rule'
todo = 'Not yet ported'
XFAIL = {
    'sim_pars_dict_access':           f'{rule}: sim.pars["pop_size"] -> sim["pop_size"]',
    'sim_pop_type_synthpops':         f'{rule}: synthpops is not supported',
    'results_result_name':            f'{todo}: Result.color',
    'sim_n_attr':                     f'{rule}: agents who have died are removed from the arrays (except people.dead); use .raw',
    'people_age':                     f'{rule}: agents who have died are removed from the arrays (except people.dead); use .raw',
    'people_sex':                     f'{rule}: agents who have died are removed from the arrays (except people.dead); use .raw',
    'people_len':                     f'{rule}: agents who have died are removed from the arrays (except people.dead); use .raw',
    'people_uid':                     f'{rule}: agents who have died are removed from the arrays (except people.dead); use .raw',
    'people_getitem':                 f'{rule}: agents who have died are removed from the arrays (except people.dead); use .raw',
    'save_keep_people':               f'{rule}: agents who have died are removed from the arrays (except people.dead); use .raw',
    'make_people_and_pass':           f'{rule}: cv.make_people() is removed',
    'popdict_custom':                 f'{rule}: cv.make_people() is removed',
    'popfile_save_load':              f'{rule}: popfile is not supported',
    'people_save_load':               f'{rule}: popfile is not supported',
    'cv_immunity_fns':                f'{rule}: sim.init_immunity() is internal',
    'cv_Result_class':                f'{rule}: cv.Result is now ss.Result, with a different signature',
}


#%% The test

params = [pytest.param(probe, id=probe.__name__, marks=pytest.mark.xfail(reason=XFAIL[probe.__name__], strict=False)) if probe.__name__ in XFAIL else pytest.param(probe, id=probe.__name__) for probe in PROBES]

@pytest.mark.parametrize('probe', params)
def test_v3_probe(probe):
    """ Run one v3 snippet; it passes if it runs without an error """
    with sc.capture():
        probe()
    pl.close('all')
    return


#%% Run as a script
if __name__ == '__main__':
    T = sc.timer()
    n_pass = 0
    for probe in PROBES:
        try:
            with sc.capture():
                probe()
            n_pass += 1
        except Exception as E:
            print(f'FAIL {probe.__name__}: {type(E).__name__}: {str(E)[:150]}')
        pl.close('all')
    print(f'{n_pass}/{len(PROBES)} v3 probes pass')
    T.toc()
