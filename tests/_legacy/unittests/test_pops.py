from unittest_support import CovaTest
import unittest

class PopulationTypeTests(CovaTest):

    def test_different_pop_types(self):
        pop_types = ['random', 'hybrid']  #, 'synthpops']
        results = {}
        short_sample = {
            'pop_size': 1000,
            'n_days': 10,
            'pop_infected': 50
        }
        for poptype in pop_types:
            self.run_sim(short_sample, population_type=poptype)
            results[poptype] = self.simulation_result['results']
            pass
        self.assertEqual(len(results), len(pop_types))
        for k in results:
            these_results = results[k]
            self.assertIsNotNone(these_results)
            day_0_susceptible = these_results['n_susceptible'][0]
            day_0_exposed = these_results['cum_infections'][0]
            self.assertEqual(day_0_susceptible + day_0_exposed, short_sample['pop_size'], msg=f"Day 0 population should be as specified in params. Poptype {k} was different.")
            self.assertGreater(these_results['cum_infections'][-1], these_results['cum_infections'][0], msg=f"Should see infections increase. Pop type {k} didn't do that.")
            self.assertGreater(these_results['cum_symptomatic'][-1], these_results['cum_symptomatic'][0], msg=f"Should see symptomatic counts increase. Pop type {k} didn't do that.")

# Run unit tests if called as a script
if __name__ == '__main__':
    unittest.TestCase.run = lambda self,*args,**kw: unittest.TestCase.debug(self)
    unittest.main()