"""
Tests of simulation parameters from
../../covasim/README.md
"""
import unittest
from unittest_support import CovaTest

class DiseaseProgressionTests(CovaTest):
    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    def test_exposure_to_infectiousness_delay_scaling(self):
        """
        Set exposure to infectiousness early simulation, mid simulation,
        late simulation. Set std_dev to zero. Verify move to infectiousness
        moves later as delay is longer.
        Depends on delay deviation test
        """
        total_agents = 500
        self.set_everyone_infected(total_agents)
        sim_dur = 60
        exposed_delays = [1, 2, 5, 15, 20, 25, 30]  # Keep values in order
        std_dev = 0
        for exposed_delay in exposed_delays:
            self.set_duration_distribution_parameters(
                duration_in_question='exp2inf',
                par1=exposed_delay,
                par2=std_dev
            )
            prob_dict = {'rel_symp_prob': 0}
            self.set_sim_prog_prob(prob_dict)
            serial_delay = {'n_days': sim_dur}
            self.run_sim(serial_delay)
            infectious_ch = self.get_full_result_ch('new_infectious')
            agents_on_infectious_day = infectious_ch[exposed_delay]
            if self.is_debugging:
                print(f"Delay: {exposed_delay}")
                print(f"Agents turned: {agents_on_infectious_day}")
                print(f"Infectious channel {infectious_ch}")
                pass
            for t in range(len(infectious_ch)):
                current_infectious = infectious_ch[t]
                if t < exposed_delay:
                    self.assertEqual(current_infectious, 0, msg=f"All {total_agents} should turn infectious at t: {exposed_delay} instead got {current_infectious} at t: {t}")
                elif t == exposed_delay:
                    self.assertEqual(infectious_ch[exposed_delay], total_agents, msg=f"With stddev 0, all {total_agents} agents should turn infectious on day {exposed_delay}, instead got {agents_on_infectious_day}. ")
        pass

    def test_mild_infection_duration_scaling(self):
        """
        Make sure that all initial infected cease being infected
        on following day. Std_dev 0 will help here
        """
        total_agents = 500
        exposed_delay = 1
        self.set_everyone_infectious_same_day(num_agents=total_agents, days_to_infectious=exposed_delay)
        prob_dict = {'rel_symp_prob': 0.0}
        self.set_sim_prog_prob(prob_dict)
        infectious_durations = [1, 2, 5, 10, 20] # Keep values in order
        for TEST_dur in infectious_durations:
            recovery_day = exposed_delay + TEST_dur
            self.set_duration_distribution_parameters(
                duration_in_question='asym2rec',
                par1=TEST_dur,
                par2=0
            )
            self.run_sim()
            recoveries_ch = self.get_full_result_ch('new_recoveries')
            recoveries_on_recovery_day = recoveries_ch[recovery_day]
            if self.is_debugging:
                print(f"Delay: {recovery_day}")
                print(f"Agents turned: {recoveries_on_recovery_day}")
                print(f"Recoveries channel {recoveries_ch}")
            self.assertEqual(recoveries_ch[recovery_day], total_agents, msg=f"With stddev 0, all {total_agents} agents should turn infectious on day {recovery_day}, instead got {recoveries_on_recovery_day}. ")

        pass

    def test_time_to_die_duration_scaling(self):
        total_agents = 500
        self.set_everyone_critical(num_agents=500, constant_delay=0)
        prob_dict = {'rel_death_prob': 1.0}
        self.set_sim_prog_prob(prob_dict)

        time_to_die_durations = [1, 2, 5, 10, 20]
        time_to_die_stddev = 0

        for TEST_dur in time_to_die_durations:
            self.set_duration_distribution_parameters(
                duration_in_question='crit2die',
                par1=TEST_dur,
                par2=time_to_die_stddev
            )
            self.run_sim()
            deaths_today_ch = self.get_full_result_ch('new_deaths')
            for t in range(len(deaths_today_ch)):
                curr_deaths = deaths_today_ch[t]
                if t < TEST_dur:
                    self.assertEqual(curr_deaths, 0, msg=f"With std 0, all {total_agents} agents should die on t: {TEST_dur}. Got {curr_deaths} at t: {t}")
                elif t == TEST_dur:
                    self.assertEqual(curr_deaths, total_agents, msg=f"With std 0, all {total_agents} agents should die at t: {TEST_dur}, got {curr_deaths} instead.")
                else:
                    self.assertEqual(curr_deaths, 0, msg=f"With std 0, all {total_agents} agents should die at t: {TEST_dur}, got {curr_deaths} at t: {t}")
        pass

if __name__ == '__main__':
    unittest.TestCase.run = lambda self,*args,**kw: unittest.TestCase.debug(self)
    unittest.main()