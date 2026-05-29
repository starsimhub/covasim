"""Continuous-runnability invariant: the v4 stub Sim constructs and runs.

M0 ships no real port, but cv.v4.Sim().run() must return results so the
invariant (MIGRATION_PLAN Implementation conventions item 1) holds from day one.
"""
import covasim as cv


def test_v4_stub_runs():
    sim = cv.v4.Sim(n_agents=100)
    sim.run()
    assert sim.results is not None, 'v4 stub sim produced no results'
    return sim


def test_v4_does_not_rebind_v3_sim():
    # Backwards compatibility: the existing v3.1.8 cv.Sim must be untouched.
    assert cv.Sim.__module__ == 'covasim.sim', \
        f'cv.Sim should still be the v3.1.8 Sim, got module {cv.Sim.__module__}'
    assert cv.v4.Sim.__module__ == 'covasim._v4', \
        f'cv.v4.Sim should be the stub, got module {cv.v4.Sim.__module__}'
    return


if __name__ == '__main__':
    test_v4_stub_runs()
    test_v4_does_not_rebind_v3_sim()
    print('v4 stub Sim ran successfully and did not disturb cv.Sim.')
