"""
Check that v4 gives the same results as v3, on average

Each test runs several seeds of a v4 sim, and compares the mean of each result
with the mean from v3.1.8 (stored in the v3_*.json files). Results can't be compared
seed-by-seed, since v3 and v4 use different random number streams. A result
fails if it differs by more than z_threshold standard errors.

To regenerate the v3 results, see the README.
"""

#%% Imports and settings
import sys
import pytest
import sciris as sc

sys.path.insert(0, str(sc.thispath().parent)) # So the migrate3to4 folder can be imported
import migrate3to4.short_summary as mss
from migrate3to4.parity import parity_gate
from migrate3to4.compare import compute_drift
from migrate3to4 import anchor_transmission, anchor_natural_history, anchor_variants, anchor_waning, anchor_testing, anchor_vaccination

n_seeds = 10 # Number of v4 seeds to run
n_v3_seeds = 30 # Number of v3 seeds in the stored results
not_results = ['_seed', '_total_pop', 'n_alive'] # Keys in the summaries that aren't results


#%% Define the tests

def check_parity(name, anchor, summary, v3_name=None, z_threshold=5.0, skip=None):
    """
    Compare v4 with v3 for the random and hybrid populations

    Args:
        name        (str):    the name of the feature being checked
        anchor      (module): the module with the make_sim() function for this feature
        summary     (func):   the function that gets the results to compare from a sim
        v3_name     (str):    the name of the v3 results file, if different from the feature
        z_threshold (float):  how many standard errors v4 can differ from v3 by; 5 allows differences of a few percent, which are expected since the random numbers differ
        skip        (list):   results that are known to differ, so aren't checked
    """
    sc.heading(f'Checking v3 vs. v4: {name}')
    skip = set(sc.mergelists(skip, not_results))
    v4_rows = None

    for pop_type in ['random', 'hybrid']:
        filename = sc.thispath() / f'v3_{v3_name or name}_{pop_type}_seeds_n{n_v3_seeds}.json'
        if not filename.exists():
            pytest.skip(f'v3 results not found: {filename}')
        v3_rows = sc.loadjson(filename)
        v4_rows = [summary(anchor.make_sim(pop_type=pop_type, rand_seed=seed).run()) for seed in range(n_seeds)]
        failures = parity_gate(v4_rows, v3_rows, z_threshold=z_threshold, skip_keys=skip)
        assert not failures, f'{name} ({pop_type}) differs from v3 by more than {z_threshold} standard errors for (result, z-score): {failures}'

    return v4_rows


@pytest.mark.slow
def test_transmission():
    return check_parity('transmission', anchor_transmission, mss.build_summary_transmission, z_threshold=3.0)


@pytest.mark.slow
def test_natural_history():
    return check_parity('natural_history', anchor_natural_history, mss.build_summary_natural_history)


@pytest.mark.slow
def test_variants():
    # Without waning immunity, v4 uses the cross-immunity values directly, whereas v3 multiplies
    # them by each person's NAbs. So only the wild type and the overall peak are expected to match;
    # the other variants match with waning immunity (see test_waning)
    matches = ['cum_infections_wild', 'peak_n_infectious', 'peak_prevalence']
    skip = [key for key in mss.METRIC_KEYS_VARIANTS if key not in matches]
    return check_parity('variants', anchor_variants, mss.build_summary_variants, skip=skip)


@pytest.mark.slow
def test_waning():
    return check_parity('waning', anchor_waning, mss.build_summary_variants, v3_name='variants') # The v3 variant results are with waning immunity


@pytest.mark.slow
def test_testing():
    # The number of tests is within 2% of v3, but varies so little between seeds that this is many standard errors
    return check_parity('testing', anchor_testing, mss.build_summary_testing, skip=['cum_tests'])


@pytest.mark.slow
def test_vaccination():
    return check_parity('vaccination', anchor_vaccination, mss.build_summary_vaccination)


def test_parity_gate():
    sc.heading('Testing the function that compares v3 and v4')

    def rows(values):
        """ Make one summary per seed """
        return [dict(cum_infections=value, _seed=seed) for seed,value in enumerate(values)]

    v3      = rows([100, 102, 98, 101])
    similar = rows([101, 99, 103, 100])
    higher  = rows([200, 202, 198, 201])
    assert parity_gate(similar, v3, skip_keys={'_seed'}) == [] # Similar results pass
    failures = parity_gate(higher, v3, skip_keys={'_seed'})
    assert failures[0][0] == 'cum_infections' # Different results fail, and say which result

    # If there is no variation between seeds, the results have to be the same
    assert parity_gate(rows([5, 5, 5]), rows([5, 5, 5]), skip_keys={'_seed'}) == []
    assert parity_gate(rows([7, 7, 7]), rows([5, 5, 5]), skip_keys={'_seed'}) == [('cum_infections', float('inf'))]

    return failures


def test_compute_drift():
    sc.heading('Testing the function that compares a single v3 and v4 run')

    v3 = dict(a=100, b=50, c=0, d=1)
    v4 = dict(a=120, b=49, c=1) # Results that are missing from v4 are skipped
    drift = {row['key']:row for row in compute_drift(v3, v4, threshold=0.10)}
    assert list(drift.keys()) == ['a', 'b', 'c']
    assert drift['a']['rel_diff'] == 0.20
    assert drift['b']['rel_diff'] == -0.02
    assert drift['c']['rel_diff'] is None # Can't calculate a relative difference from zero...
    assert [row['over_threshold'] for row in drift.values()] == [True, False, True] # ...so it counts as over the threshold

    return drift



#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    rows1 = test_transmission()
    rows2 = test_natural_history()
    rows3 = test_variants()
    rows4 = test_waning()
    rows5 = test_testing()
    rows6 = test_vaccination()
    failures = test_parity_gate()
    drift = test_compute_drift()

    sc.toc(T)
    print('Done.')
