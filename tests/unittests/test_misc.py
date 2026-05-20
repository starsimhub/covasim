"""

"""

import unittest
import pandas as pd
from unittest_support import CovaTest
from covasim import Sim, parameters
import os

class MiscellaneousFeatureTests(CovaTest):
    def setUp(self):
        super().setUp()
        self.sim = Sim(pop_size=500)
        self.pars = parameters.make_pars()
        self.is_debugging = False

    def test_xslx_generation(self):
        super().tearDown()
        self.is_debugging = False
        root_filename = "DEBUG_test_xlsx_generation"
        excel_filename = f"{root_filename}.xlsx"
        if os.path.isfile(excel_filename):
            os.unlink(excel_filename)
        test_infected_value = 31
        params_dict = {
            'pop_size': 500,
            'pop_infected': test_infected_value
        }
        self.run_sim(params_dict)
        self.sim.to_excel(filename=root_filename)
        simulation_df = pd.ExcelFile(excel_filename)
        expected_sheets = ['Results','Parameters']
        for sheet in expected_sheets:
            self.assertIn(sheet, simulation_df.sheet_names)
        params_df = simulation_df.parse('Parameters')
        observed_infected_param = params_df.loc[params_df['Parameter'] == 'pop_infected', 'Value'].values[0]
        self.assertEqual(observed_infected_param, test_infected_value,
                         msg="Should be able to parse the pop_infected parameter from the results sheet")
        results_df = simulation_df.parse('Results')
        observed_day_0_exposed = results_df.loc[results_df['t'] == 0, 'n_exposed'].values[0]
        self.assertGreaterEqual(observed_day_0_exposed, test_infected_value,
                         msg="Should be able to parse the day 0 n_exposed value from the results sheet.")
        if not self.is_debugging:
            os.unlink(excel_filename)

    def test_set_pars_invalid_key(self):
        with self.assertRaises(KeyError) as context:
            self.sim['n_infectey'] = 10
        error_message = str(context.exception)
        self.assertIn('n_infectey', error_message)
        self.assertIn('pop_infected', error_message)

    def test_update_pars_invalid_key(self):
        invalid_key = {
            'dooty_doo': 5
        }
        with self.assertRaises(KeyError) as context:
            self.sim.update_pars(invalid_key)
        error_message = str(context.exception)
        self.assertIn('dooty_doo', error_message)

# Run unit tests if called as a script
if __name__ == '__main__':
    unittest.TestCase.run = lambda self,*args,**kw: unittest.TestCase.debug(self)
    unittest.main()