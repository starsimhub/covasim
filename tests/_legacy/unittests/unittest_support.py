"""
Classes that provide test content for tests later,
and easier configuration of tests to make tests
easier to red.

Test implementation is agnostic to model implementation
by design.
"""

import unittest
import json
import os
import numpy as np
import covasim as cv


class SpecialSims:
    class Microsim:
        n = 10
        pop_infected = 1
        contacts = 2
        n_days = 10

    class Hightransmission:
        n = 500
        pop_infected = 10
        n_days = 30
        contacts = 3
        beta = 0.4
        serial = 2
        dur = 3

    class HighMortality:
        n = 1000
        cfr_by_age = False
        default_cfr = 0.2
        timetodie = 6


class CovaTest(unittest.TestCase):
    def setUp(self):
        self.is_debugging = False

        self.sim_pars = None
        self.sim_progs = None
        self.sim = None
        self.simulation_result = None
        self.interventions = None
        self.expected_result_filename = f"DEBUG_{self.id()}.json"
        if os.path.isfile(self.expected_result_filename):
            os.unlink(self.expected_result_filename)


    def tearDown(self):
        if not self.is_debugging:
            if os.path.isfile(self.expected_result_filename):
                os.unlink(self.expected_result_filename)


    # region configuration methods
    def set_sim_pars(self, params_dict=None):
        """
        Overrides all of the default sim parameters
        with the ones in the dictionary
        Args:
            params_dict: keys are param names, values are expected values to use

        Returns:
            None, sets self.simulation_params

        """
        if not self.sim_pars:
            self.sim_pars = cv.make_pars(set_prognoses=True, prog_by_age=True)
        if params_dict:
            self.sim_pars.update(params_dict)


    def set_sim_prog_prob(self, params_dict):
        """
        Allows for testing prognoses probability as absolute rather than relative.
        NOTE: You can only call this once per test or you will overwrite your stuff.
        """
        supported_probabilities = [
            'rel_symp_prob',
            'rel_severe_prob',
            'rel_crit_prob',
            'rel_death_prob'
        ]
        if not self.sim_pars:
            self.set_sim_pars()


        if not self.sim_progs:
            self.sim_progs = cv.get_prognoses(self.sim_pars['prog_by_age'])

        for k in params_dict:
            prognosis_in_question = None
            expected_prob = params_dict[k]
            if   k == 'rel_symp_prob':    prognosis_in_question = 'symp_probs'
            elif k == 'rel_severe_prob':  prognosis_in_question = 'severe_probs'
            elif k == 'rel_crit_prob':    prognosis_in_question = 'crit_probs'
            elif k == 'rel_death_prob':   prognosis_in_question = 'death_probs'
            else:
                raise KeyError(f"Key {k} not found in {supported_probabilities}.")
            old_probs = self.sim_progs[prognosis_in_question]
            self.sim_progs[prognosis_in_question] = np.array([expected_prob] * len(old_probs))



    def set_duration_distribution_parameters(self, duration_in_question,
                                             par1, par2):
        if not self.sim_pars:
            self.set_sim_pars()

        duration_node = self.sim_pars["dur"]
        duration_node[duration_in_question] = {
            "dist": "normal",
            "par1": par1,
            "par2": par2
        }
        params_dict = {"dur": duration_node}
        self.set_sim_pars(params_dict=params_dict)

    def run_sim(self, params_dict=None, write_results_json=False, population_type=None):
        if not self.sim_pars or params_dict: # If we need one, or have one here
            self.set_sim_pars(params_dict=params_dict)
        self.sim_pars['interventions'] = self.interventions
        self.sim = cv.Sim(pars=self.sim_pars, datafile=None)
        if not self.sim_progs:
            self.sim_progs = cv.get_prognoses(self.sim_pars['prog_by_age'])

        self.sim['prognoses'] = self.sim_progs
        if population_type:
            self.sim.update_pars(pop_type=population_type)
        self.sim.run(verbose=0)
        self.simulation_result = self.sim.to_json(tostring=False)
        if write_results_json or self.is_debugging:
            with open(self.expected_result_filename, 'w') as outfile:
                json.dump(self.simulation_result, outfile, indent=4, sort_keys=True)


    def get_full_result_ch(self, channel):
        result_data = self.simulation_result["results"][channel]
        return result_data

    def get_day_zero_ch_value(self, channel='n_susceptible'):
        """

        Args:
            channel: timeseries channel to report ('n_susceptible')

        Returns: day zero value for channel

        """
        result_data = self.get_full_result_ch(channel=channel)
        return result_data[0]

    def get_day_final_ch_value(self, channel):
        channel = self.get_full_result_ch(channel=channel)
        return channel[-1]

    def intervention_set_changebeta(self, days_array, multiplier_array, layers = None):
        self.interventions = cv.change_beta(days=days_array, changes=multiplier_array, layers=layers)


    def intervention_set_test_prob(self, symptomatic_prob=0, asymptomatic_prob=0, asymptomatic_quarantine_prob=0, symp_quar_prob=0, test_sensitivity=1.0, loss_prob=0.0, test_delay=1, start_day=0):
        self.interventions = cv.test_prob(symp_prob=symptomatic_prob, asymp_prob=asymptomatic_prob, asymp_quar_prob=asymptomatic_quarantine_prob, symp_quar_prob=symp_quar_prob, sensitivity=test_sensitivity, loss_prob=loss_prob, test_delay=test_delay, start_day=start_day)


    def intervention_set_contact_tracing(self, start_day, trace_probabilities=None, trace_times=None):

        if not trace_probabilities:
            trace_probabilities = {'h': 1, 's': 1, 'w': 1, 'c': 1}

        if not trace_times:
            trace_times = {'h': 1, 's': 1, 'w': 1, 'c': 1}
        self.interventions = cv.contact_tracing(trace_probs=trace_probabilities, trace_time=trace_times, start_day=start_day)


    def intervention_build_sequence(self, day_list, intervention_list):
        my_sequence = cv.sequence(days=day_list, interventions=intervention_list)
        self.interventions = my_sequence
    # endregion

    # region specialized simulation methods
    def set_microsim(self):
        Micro = SpecialSims.Microsim
        microsim_parameters = {
            'pop_size' : Micro.n,
            'pop_infected': Micro.pop_infected,
            'n_days': Micro.n_days
        }
        self.set_sim_pars(microsim_parameters)


    def set_everyone_infected(self, agent_count=1000):
        everyone_infected = {
            'pop_size': agent_count,
            'pop_infected': agent_count
        }
        self.set_sim_pars(params_dict=everyone_infected)


    def set_everyone_infectious_same_day(self, num_agents, days_to_infectious=1, num_days=60):
        """
        Args:
            num_agents: number of agents to create and infect
            days_to_infectious: days until all agents are infectious (1)
            num_days: days to simulate (60)
        """
        self.set_everyone_infected(agent_count=num_agents)
        prob_dict = {
            'rel_symp_prob': 0
        }
        self.set_sim_prog_prob(prob_dict)
        test_config = {
            'n_days': num_days
        }
        self.set_duration_distribution_parameters(
            duration_in_question='exp2inf',
            par1=days_to_infectious,
            par2=0
        )
        self.set_sim_pars(params_dict=test_config)


    def set_everyone_symptomatic(self, num_agents, constant_delay:int=None):
        """
        Cause all agents in the simulation to begin infected
        And proceed to symptomatic (but not severe or death)
        Args:
            num_agents: Number of agents to begin with
        """
        self.set_everyone_infectious_same_day(num_agents=num_agents,
                                              days_to_infectious=0)
        prob_dict = {
            'rel_symp_prob': 1.0,
            'rel_severe_prob': 0
        }
        self.set_sim_prog_prob(prob_dict)
        if constant_delay is not None:
            self.set_duration_distribution_parameters(
                duration_in_question='inf2sym',
                par1=constant_delay,
                par2=0
            )

    def everyone_dies(self, num_agents):
        """
        Cause all agents in the simulation to begin infected and die.
        Args:
            num_agents: Number of agents to simulate
        """
        self.set_everyone_infectious_same_day(num_agents=num_agents)
        prob_dict = {
            'rel_symp_prob': 1,
            'rel_severe_prob': 1,
            'rel_crit_prob': 1,
            'rel_death_prob': 1
        }
        self.set_sim_prog_prob(prob_dict)

    def set_everyone_severe(self, num_agents, constant_delay:int=None):
        self.set_everyone_symptomatic(num_agents=num_agents, constant_delay=constant_delay)
        prob_dict = {
            'rel_severe_prob': 1.0,
            'rel_crit_prob': 0.0
        }
        self.set_sim_prog_prob(prob_dict)
        if constant_delay is not None:
            self.set_duration_distribution_parameters(
                duration_in_question='sym2sev',
                par1=constant_delay,
                par2=0
            )

    def set_everyone_critical(self, num_agents, constant_delay:int=None):
        """
        Causes all agents to become critically ill day 1
        """
        self.set_everyone_severe(num_agents=num_agents, constant_delay=constant_delay)
        prob_dict = {
            'rel_crit_prob': 1.0,
            'rel_death_prob': 0.0
        }
        self.set_sim_prog_prob(prob_dict)
        if constant_delay is not None:
            self.set_duration_distribution_parameters(
                duration_in_question='sev2crit',
                par1=constant_delay,
                par2=0
            )


    def set_smallpop_hightransmission(self):
        """
        Creates a small population with lots of transmission
        """
        Hightrans = SpecialSims.Hightransmission
        hightrans_parameters = {
            'pop_size' : Hightrans.n,
            'pop_infected': Hightrans.pop_infected,
            'n_days': Hightrans.n_days,
            'beta' : Hightrans.beta
        }
        self.set_sim_pars(hightrans_parameters)

class TestSupportTests(CovaTest):
    def test_run_vanilla_simulation(self):
        """
        Runs an uninteresting but predictable
        simulation, makes sure that results
        are created and json parsable
        """
        self.assertIsNone(self.sim)
        self.run_sim(write_results_json=True)
        json_file_found = os.path.isfile(self.expected_result_filename)
        self.assertTrue(json_file_found, msg=f"Expected {self.expected_result_filename} to be found.")


    def test_everyone_infected(self):
        """
        All agents start infected
        """

        total_agents = 500
        self.set_everyone_infected(agent_count=total_agents)
        self.run_sim()
        exposed_ch = 'cum_infections'
        day_0_exposed = self.get_day_zero_ch_value(exposed_ch)
        self.assertEqual(day_0_exposed, total_agents)


    def test_run_small_hightransmission_sim(self):
        """
        Runs a small simulation with lots of transmission
        Verifies that there are lots of infections in
        a short time.
        """
        self.assertIsNone(self.sim_pars)
        self.assertIsNone(self.sim)
        self.set_smallpop_hightransmission()
        self.run_sim()

        self.assertIsNotNone(self.sim)
        self.assertIsNotNone(self.sim_pars)
        exposed_today_ch = self.get_full_result_ch('cum_infections')
        prev_exposed = exposed_today_ch[0]
        for t in range(1, 10):
            today_exposed = exposed_today_ch[t]
            self.assertGreaterEqual(today_exposed, prev_exposed, msg=f"The first 10 days should have increasing exposure counts. At time {t}: {today_exposed} at {t-1}: {prev_exposed}.")
            prev_exposed = today_exposed

        infections_ch = self.get_full_result_ch('new_infections')
        self.assertGreaterEqual(sum(infections_ch), 150, msg="Should have at least 150 infections")





