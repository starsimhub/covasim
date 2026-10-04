'''
Test the script for migrating v3 code to v4
'''

import sciris as sc
import covasim as cv

v3_code = '''
class store_severe(cv.Analyzer):
    def apply(self, sim):
        if sim.t == 10:
            sim.pars['beta'] *= 0.5
        self.severe = sc.findinds(sim.people.severe)
        self.n = len(sim.people)
        self.where = sc.findinds(self.vals, 1)
tt = sim.results.transtree
gt = sim.results['gen_time']
'''

v4_code = '''
class store_severe(cv.Analyzer):
    def apply(self, sim):
        if sim.ti == 10:
            sim['beta'] *= 0.5
        self.severe = cv.true(sim.people.severe)
        self.n = len(sim.people)
        self.where = sc.findinds(self.vals, 1)
tt = sim.transtree
gt = sim.gen_time
'''


def test_migrate_code():
    sc.heading('Testing migrating code...')
    new, changes, to_check = cv.migrate3to4.migrate_code(v3_code)
    assert new == v4_code
    assert len(changes) == 5
    assert [lineno for lineno,line,description in to_check] == [7] # len(sim.people) only counts agents who are alive
    return new


def test_migrate_file(tmp_path):
    sc.heading('Testing migrating a file...')
    filename = tmp_path / 'v3_script.py'
    sc.savetext(filename, v3_code)
    cv.migrate3to4.migrate(tmp_path) # By default, don't change the file
    assert sc.loadtext(filename) == v3_code
    out = cv.migrate3to4.migrate(tmp_path, apply=True)
    assert sc.loadtext(filename) == v4_code
    assert len(out) == 1
    return out


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    new = test_migrate_code()

    sc.toc(T)
    print('Done.')
