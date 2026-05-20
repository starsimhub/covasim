"""
Tests of simulation parameters from
../../covasim/README.md
"""

import unittest
from unittest_support import CovaTest, SpecialSims
Hightrans = SpecialSims.Hightransmission

class DiseaseTransmissionTests(CovaTest):
    """
    Tests of the parameters involved in transmission
    pre requisites simulation parameter tests
    """

    def test_beta_zero(self):
        """
        Test that with beta at zero, no transmission
        Start with high transmission sim
        """
        self.set_smallpop_hightransmission()
        beta_zero = {'beta': 0}
        self.run_sim(beta_zero)
        exposed_today_ch = self.get_full_result_ch('cum_infections')
        prev_exposed = exposed_today_ch[0]
        self.assertEqual(prev_exposed, Hightrans.pop_infected, msg="Make sure we have some initial infections")
        for t in range(1, len(exposed_today_ch)):
            today_exposed = exposed_today_ch[t]
            self.assertLessEqual(today_exposed, prev_exposed, msg=f"The exposure counts should do nothing but decline. At time {t}: {today_exposed} at {t-1}: {prev_exposed}.")
            prev_exposed = today_exposed

        infections_ch = self.get_full_result_ch('new_infections')
        for t in range(len(infections_ch)):
            today_infectious = infections_ch[t]
            self.assertEqual(today_infectious, 0, msg=f"With beta 0, there should be no infections. At ts: {t} got {today_infectious}.")

# Run unit tests if called as a script
if __name__ == '__main__':
    unittest.TestCase.run = lambda self,*args,**kw: unittest.TestCase.debug(self)
    unittest.main()