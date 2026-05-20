'''
Test layer key usage and validation
'''

import covasim as cv
import sciris as sc
import pytest

pars = dict(
    pop_size = 1000
    )

T = sc.tic()


sc.heading('Basic test')
sb = cv.Sim(pars)
sb.run()
assert sb.layer_keys() == ['a']


sc.heading('Hybrid test')
sh = cv.Sim(pars, pop_type='hybrid')
sh.run()


sc.heading('Hybrid update_pars test')
shu = cv.Sim(pars)
shu.update_pars(pop_type='hybrid')
shu.run()
assert shu['pop_type'] == 'hybrid'
assert shu.layer_keys() == ['h', 's', 'w', 'c']
assert shu.people.layer_keys() == ['h', 's', 'w', 'c']


sc.heading('Update beta_layer test')
sub = cv.Sim(pars, beta_layer={'a':0.5})
sub.run()
assert sub['beta_layer']['a'] == 0.5


sc.heading('Update beta_layer test 2')
sub2 = cv.Sim(pars)
sub2.update_pars(beta_layer={'a':0.5})
sub2.run()
assert sub2['beta_layer']['a'] == 0.5
with pytest.raises(sc.KeyNotFoundError): # q is not ok
    sub2.update_pars(beta_layer={'q':0.5})
    sub2.run()


sc.heading('SynthPops test')
ssp = cv.Sim(pars, pop_type='synthpops')
ssp.run()


sc.heading('SynthPops + LTCF test')
sf = cv.Sim(pop_size=2000, pop_type='synthpops')
sf.popdict = cv.make_synthpop(sf, with_facilities=True, layer_mapping={'LTCF':'f'})
cv.save('synth.pop', sf.popdict)
with pytest.raises(sc.KeyNotFoundError): # Layer key mismatch, f is not initialized
    sf.run()
sf.reset_layer_pars()
sf.run()
assert 'f' in sf.layer_keys()
assert sf['beta_layer']['f'] == 1.0


sc.heading('Population save test')
sps = cv.Sim(pars, pop_type='hybrid', popfile='test.pop', save_pop=True)
sps.run()


sc.heading('Population load test')
spl = cv.Sim(pars, pop_type='hybrid', popfile='test.pop', load_pop=True)
spl.run()


sc.heading('SynthPops + LTCF load test')
with pytest.raises(ValueError): # Wrong population size
    cv.Sim(pars, pop_type=None, popfile='synth.pop', load_pop=True)
sspl = cv.Sim(pop_size=2000, pop_type=None, popfile='synth.pop', load_pop=True, beta_layer=dict(h=10, s=2, w=2, c=2, f=1))
sspl.run()
assert sspl['beta_layer']['h'] == 10


sc.toc(T)
print('Done.')