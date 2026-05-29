"""
Tests of simulation parameters from
../../covasim/README.md
"""

import covasim as cv
import unittest
from unittest_support import CovaTest


class DiseaseMortalityTests(CovaTest):

    def test_default_death_prob_one(self):
        """
        Infect lots of people with cfr one and short time to die
        duration. Verify that everyone dies, no recoveries.
        """
        pop_size = 200
        n_days = 90
        sim = cv.Sim(pop_size=pop_size, pop_infected=pop_size, n_days=n_days)
        for key in ['rel_symp_prob', 'rel_severe_prob', 'rel_crit_prob', 'rel_death_prob']:
            sim[key] = 1e6
        sim.run()
        assert sim.summary.cum_deaths == pop_size

    def test_default_death_prob_zero(self):
        """
        Infect lots of people with cfr zero and short time to die
        duration. Verify that no one dies.
        Depends on default_cfr_one
        """
        total_agents = 500
        self.everyone_dies(num_agents=total_agents)
        prob_dict = {'rel_death_prob': 0.0}
        self.set_sim_prog_prob(prob_dict)
        self.run_sim()
        deaths_at_timestep_ch = self.get_full_result_ch('new_deaths')
        deaths_cumulative_ch = self.get_full_result_ch('cum_deaths')
        death_chs = [deaths_at_timestep_ch,deaths_cumulative_ch]
        for c in death_chs:
            for t in range(len(c)):
                self.assertEqual(c[t], 0, msg=f"There should be no deaths with critical to death probability 0.0. Channel {c} had bad data at t: {t}")
        cumulative_recoveries = self.get_day_final_ch_value('cum_recoveries')
        self.assertGreaterEqual(cumulative_recoveries, 200, msg="Should be lots of recoveries")
        pass

    def test_default_death_prob_scaling(self):
        """
        Infect lots of people with cfr zero and short time to die
        duration. Verify that no one dies.
        Depends on default_cfr_one
        """
        total_agents = 500
        self.everyone_dies(num_agents=total_agents)
        death_probs = [0.01, 0.05, 0.10, 0.15]
        old_cum_deaths = 0
        for death_prob in death_probs:
            prob_dict = {'rel_death_prob': death_prob}
            self.set_sim_prog_prob(prob_dict)
            self.run_sim()
            cum_deaths = self.get_day_final_ch_value('cum_deaths')
            self.assertGreaterEqual(cum_deaths, old_cum_deaths, msg="Should be more deaths with higer ratio")
            old_cum_deaths = cum_deaths

# Run unit tests if called as a script
if __name__ == '__main__':
    unittest.TestCase.run = lambda self,*args,**kw: unittest.TestCase.debug(self)
    unittest.main()