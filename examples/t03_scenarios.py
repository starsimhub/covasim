'''
Example of a simple set of scenarios
'''

import covasim as cv

# Set base parameters -- these will be shared across all scenarios
basepars = {'pop_size':10e3}

# Configure the settings for each scenario
scenarios = {'baseline': {
              'name':'Baseline',
              'pars': {}
              },
            'high_beta': {
              'name':'High beta (0.020)',
              'pars': {
                  'beta': 0.020,
                  }
              },
            'low_beta': {
              'name':'Low beta (0.012)',
              'pars': {
                  'beta': 0.012,
                  }
              },
             }

             
if __name__ == '__main__':
    
    # Run and plot the scenarios
    scens = cv.Scenarios(basepars=basepars, scenarios=scenarios)
    scens.run()
    scens.plot()