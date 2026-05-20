'''
This file contains all the code for a single run of the cruise ship flavor of
Covasim.

Based heavily on LEMOD-FP (https://github.com/amath-idm/lemod_fp).

Version: 2020mar20
'''

#%% Imports
import numba as nb
import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import parameters as cova_pars


# Specify all externally visible functions this file defines
__all__ = ['Person', 'Sim']


#%% Define classes

class Person(cv.Person):
    '''
    Class for a single person.
    '''
    def __init__(self, age=0, sex=0, crew=False, contacts=0):
        self.uid      = str(pl.randint(0,1e9)) # Unique identifier for this person
        self.age      = float(age) # Age of the person (in years)
        self.sex      = sex # Female (0) or male (1)
        self.crew     = crew # Wehther the person is a crew member
        self.contacts = contacts # Determine how many contacts they have

        # Define state
        self.alive       = True
        self.susceptible = True
        self.exposed     = False
        self.infectious  = False
        self.diagnosed   = False
        self.recovered   = False

        # Keep track of dates
        self.date_exposed    = None
        self.date_infectious = None
        self.date_diagnosed  = None
        self.date_recovered  = None
        return


class Sim(cv.BaseSim):
    '''
    The Sim class handles the running of the simulation: the number of children,
    number of time points, and the parameters of the simulation.
    '''

    def __init__(self, pars=None, datafile=None):
        if pars is None:
            pars = cova_pars.make_pars()
        super().__init__(pars) # Initialize and set the parameters as attributes
        self.data = cova_pars.load_data(datafile)
        self.set_seed(self['rand_seed'])
        self.init_results()
        self.init_people()
        self.interventions = {}
        return


    def init_results(self):
        ''' Initialize results '''
        self.results_keys = [
            'n_susceptible',
            'n_exposed',
            'n_infectious',
            'n_recovered',
            'infections',
            'tests',
            'diagnoses',
            'recoveries',
            'cum_exposed',
            'cum_tested',
            'cum_diagnosed',
            'evac_diagnoses',]
        self.results = {}
        for key in self.results_keys:
            self.results[key] = np.zeros(int(self.npts))
        self.results['t'] = np.arange(int(self.npts))
        self.results['transtree'] = {} # For storing the transmission tree
        self.results['ready'] = False
        return

    def init_people(self, seed_infections=1):
        ''' Create the people '''
        self.people = sc.odict() # Dictionary for storing the people
        self.off_ship = sc.odict() # For people who've been moved off the ship
        guests = [0]*self['n_guests']
        crew   = [1]*self['n_crew']
        for is_crew in crew+guests: # Loop over each person
            age,sex = cova_pars.get_age_sex(is_crew)
            if is_crew:
                contacts = self['contacts_crew']
            else:
                contacts = self['contacts_guest']
            person = Person(age=age, sex=sex, crew=is_crew, contacts=contacts) # Create the person
            self.people[person.uid] = person # Save them to the dictionary

        # Create the seed infections
        for i in range(seed_infections):
            person = self.people[i]
            person.susceptible = False
            person.exposed = True
            person.infectious = True
            person.date_exposed = 0
            person.date_infectious = 0

        return


    def summary_stats(self):
        ''' Compute the summary statistics to display at the end of a run '''
        keys = ['n_susceptible', 'n_exposed', 'n_infectious']
        summary = {}
        for key in keys:
            summary[key] = self.results[key][-1]
        return summary


    def run(self, seed_infections=1, verbose=None, calc_likelihood=False, do_plot=False, **kwargs):
        ''' Run the simulation '''

        T = sc.tic()

        # Reset settings and results
        if verbose is None:
            verbose = self['verbose']
        self.init_results()
        self.init_people(seed_infections=seed_infections) # Actually create the people
        daily_tests = self.data['new_tests'] # Number of tests each day, from the data
        evacuated = self.data['evacuated'] # Number of people evacuated

        # Main simulation loop
        for t in range(self.npts):

            # Print progress
            if verbose>=1:
                string = f'  Running day {t:0.0f} of {self["n_days"]}...'
                if verbose>=2:
                    sc.heading(string)
                else:
                    print(string)

            test_probs = {} # Store the probability of each person getting tested

            # Update each person
            for person in self.people.values():

                # Count susceptibles
                if person.susceptible:
                    self.results['n_susceptible'][t] += 1
                    continue # Don't bother with the rest of the loop

                # Handle testing probability
                if person.infectious:
                    test_probs[person.uid] = self['symptomatic'] # They're infectious: high probability of testing
                else:
                    test_probs[person.uid] = 1.0

                # If exposed, check if the person becomes infectious
                if person.exposed:
                    self.results['n_exposed'][t] += 1
                    if not person.infectious and t >= person.date_infectious: # It's the day they become infectious
                        person.infectious = True
                        if verbose>=2:
                            print(f'      Person {person.uid} became infectious!')

                # If infectious, check if anyone gets infected
                if person.infectious:
                    # First, check for recovery
                    if person.date_recovered and t >= person.date_recovered: # It's the day they become infectious
                        person.exposed = False
                        person.infectious = False
                        person.recovered = True
                        self.results['recoveries'][t] += 1
                    else:
                        self.results['n_infectious'][t] += 1 # Count this person as infectious
                        n_contacts = pt(person.contacts) # Draw the number of Poisson contacts for this person
                        contact_inds = cv.choose(max_n=len(self.people), n=n_contacts) # Choose people at random
                        for contact_ind in contact_inds:
                            exposure = bt(self['r_contact']) # Check for exposure per person
                            if exposure:
                                target_person = self.people[contact_ind]
                                if target_person.susceptible: # Skip people who are not susceptible
                                    self.results['infections'][t] += 1
                                    target_person.susceptible = False
                                    target_person.exposed = True
                                    target_person.date_exposed = t
                                    incub_pars = dict(dist='normal_int', par1=self['incub'], par2=self['incub_std'])
                                    dur_pars   = dict(dist='normal_int', par1=self['dur'],   par2=self['dur_std'])
                                    incub_dist = cv.sample(**incub_pars)
                                    dur_dist   = cv.sample(**dur_pars)

                                    target_person.date_infectious = t + incub_dist
                                    target_person.date_recovered = target_person.date_infectious + dur_dist
                                    if verbose>=2:
                                        print(f'        Person {person.uid} infected person {target_person.uid}!')

                # Count people who recovered
                if person.recovered:
                    self.results['n_recovered'][t] += 1

            # Implement testing -- this is outside of the loop over people, but inside the loop over time
            if t<len(daily_tests): # Don't know how long the data is, ensure we don't go past the end
                n_tests = daily_tests.iloc[t] # Number of tests for this day
                if n_tests and not pl.isnan(n_tests): # There are tests this day
                    self.results['tests'][t] = n_tests # Store the number of tests
                    test_probs = pl.array(list(test_probs.values()))
                    test_probs /= test_probs.sum()
                    test_inds = cv.choose_w(probs=test_probs, n=n_tests)
                    uids_to_pop = []
                    for test_ind in test_inds:
                        tested_person = self.people[test_ind]
                        if tested_person.infectious and bt(self['sensitivity']): # Person was tested and is true-positive
                            self.results['diagnoses'][t] += 1
                            tested_person.diagnosed = True
                            if self['evac_positives']:
                                uids_to_pop.append(tested_person.uid)
                            if verbose>=2:
                                        print(f'          Person {person.uid} was diagnosed!')
                    for uid in uids_to_pop: # Remove people from the ship once they're diagnosed
                        self.off_ship[uid] = self.people.pop(uid)

            # Implement quarantine
            if t == self['quarantine']:
                if verbose>=1:
                    print(f'Implementing quarantine on day {t}...')
                for person in self.people.values():
                    if 'quarantine_eff' in self.pars.keys():
                        quarantine_eff = self['quarantine_eff'] # Both
                    else:
                        if person.crew:
                            quarantine_eff = self['quarantine_eff_c'] # Crew
                        else:
                            quarantine_eff = self['quarantine_eff_g'] # Guests
                    person.contacts *= quarantine_eff

            # Implement testing change
            if t == self['testing_change']:
                if verbose>=1:
                    print(f'Implementing testing change on day {t}...')
                self['symptomatic'] *= self['testing_symptoms'] # Reduce the proportion of symptomatic testing

            # Implement evacuations
            if t<len(evacuated):
                n_evacuated = evacuated.iloc[t] # Number of evacuees for this day
                if n_evacuated and not pl.isnan(n_evacuated): # There are evacuees this day # TODO -- refactor with n_tests
                    if verbose>=1:
                        print(f'Implementing evacuation on day {t}')
                    evac_inds = cv.choose(max_n=len(self.people), n=n_evacuated)
                    uids_to_pop = []
                    for evac_ind in evac_inds:
                        evac_person = self.people[evac_ind]
                        if evac_person.infectious and bt(self['sensitivity']):
                            self.results['evac_diagnoses'][t] += 1
                        uids_to_pop.append(evac_person.uid)
                    for uid in uids_to_pop: # Remove people from the ship once they're diagnosed
                        self.off_ship[uid] = self.people.pop(uid)

        # Compute cumulative results
        self.results['cum_exposed']   = pl.cumsum(self.results['infections'])
        self.results['cum_tested']    = pl.cumsum(self.results['tests'])
        self.results['cum_diagnosed'] = pl.cumsum(self.results['diagnoses'])

        # Compute likelihood
        if calc_likelihood:
            self.likelihood()

        # Tidy up
        self.results['ready'] = True
        elapsed = sc.toc(T, output=True)
        if verbose>=1:
            print(f'\nRun finished after {elapsed:0.1f} s.\n')
            summary = self.summary_stats()
            print(f"""Summary:
     {summary['n_susceptible']:5.0f} susceptible
     {summary['n_exposed']:5.0f} exposed
     {summary['n_infectious']:5.0f} infectious
               """)

        if do_plot:
            self.plot(**kwargs)

        return self.results


    def likelihood(self, verbose=None):
        '''
        Compute the log-likelihood of the current simulation based on the number
        of new diagnoses.
        '''
        if verbose is None:
            verbose = self['verbose']

        if not self.results['ready']:
            self.run(calc_likelihood=False, verbose=verbose) # To avoid an infinite loop

        loglike = 0
        for d,datum in enumerate(self.data['new_positives']):
            if not pl.isnan(datum): # Skip days when no tests were performed
                estimate = self.results['diagnoses'][d]
                p = cv.poisson_test(datum, estimate)
                logp = pl.log(p)
                loglike += logp
                if verbose>=2:
                    print(f'  {self.data["date"][d]}, data={datum:3.0f}, model={estimate:3.0f}, log(p)={logp:10.4f}, loglike={loglike:10.4f}')

        self.results['likelihood'] = loglike

        if verbose>=1:
            print(f'Likelihood: {loglike}')

        return loglike



    def plot(self, do_save=None, fig_args=None, plot_args=None, scatter_args=None, axis_args=None, as_days=True, font_size=18, use_grid=True, verbose=None):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Parameters
        ----------
        do_save : bool or str
            Whether or not to save the figure. If a string, save to that filename.

        fig_args : dict
            Dictionary of kwargs to be passed to pl.figure()

        plot_args : dict
            Dictionary of kwargs to be passed to pl.plot()

        as_days : bool
            Whether to plot the x-axis as days or time points

        Returns
        -------
        Figure handle
        '''

        if verbose is None:
            verbose = self['verbose']
        if verbose:
            print('Plotting...')

        if fig_args     is None: fig_args     = {'figsize':(26,16)}
        if plot_args    is None: plot_args    = {'lw':3, 'alpha':0.7}
        if scatter_args is None: scatter_args = {'s':150, 'marker':'s'}
        if axis_args    is None: axis_args    = {'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.25}

        fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)
        pl.rcParams['font.size'] = font_size

        res = self.results # Shorten since heavily used

        # Plot everything
        colors = sc.gridcolors(5)
        to_plot = sc.odict({ # TODO
            'Total counts': sc.odict({'n_susceptible':'Number susceptible',
                                      'n_exposed':'Number exposed',
                                      'n_infectious':'Number infectious',
                                      'cum_diagnosed':'Number diagnosed',
                                    }),
            'Daily counts': sc.odict({'infections':'New infections',
                                      'tests':'Number of tests',
                                      'diagnoses':'New diagnoses',
                                     }),
            })

        data_mapping = {
            'cum_diagnosed': pl.cumsum(self.data['new_positives']),
            'tests':         self.data['new_tests'],
            'diagnoses':     self.data['new_positives'],
            }

        for p,title,keylabels in to_plot.enumitems():
            pl.subplot(2,1,p+1)
            for i,key,label in keylabels.enumitems():
                this_color = colors[i+p]
                y = res[key]
                pl.plot(res['t'], y, label=label, **plot_args, c=this_color)
                if key in data_mapping:
                    pl.scatter(self.data['day'], data_mapping[key], c=[this_color], **scatter_args)
            pl.scatter(pl.nan, pl.nan, c=[(0,0,0)], label='Data', **scatter_args)
            pl.grid(use_grid)
            # cv.fixaxis(self) # Deprecated function
            pl.ylabel('Count')
            pl.xlabel('Days since index case')
            pl.title(title)

        # Ensure the figure actually renders or saves
        if do_save:
            if isinstance(do_save, str):
                filename = do_save # It's a string, assume it's a filename
            else:
                filename = 'covasim_cruise_ship_results.png' # Just give it a default name
            pl.savefig(filename)

        pl.show()

        return fig


    def plot_people(self):
        ''' Use imshow() to show all individuals as rows, with time as columns, one pixel per timestep per person '''
        raise NotImplementedError


#%% Define helper functions, with Numba for performance

@nb.njit((nb.float64,))
def pt(rate):
    ''' A Poisson trial '''
    return np.random.poisson(rate, 1)[0]

@nb.njit((nb.float64,))
def bt(prob):
    ''' Perform a binomial (Bernolli) trial '''
    return np.random.random() < prob