'''
Simple script for running Covasim scenarios
'''

import covasim as cv
import csv

def write_result_dict_to_csv(result_dict,cols):
    out = []
    for row in result_dict.keys(): 
       out.append([row] + list(result_dict[row].values()))
    return out 


def create_greenspaces(num_greenspaces=1, daily_access=[100],prob_daily_access=[0.5],max_daily_cap=[100]):
    basic_greenspace = {
      "daily_access": 1000,
      "prob_daily_access": 0.1,
      "max_daily_cap": 1000,
      "daily_visits": [[]]
      }
    
    greenspaces = [basic_greenspace for i in range(num_greenspaces)]

    for i in range(len(daily_access)) : greenspaces[i]['daily_access'] = daily_access[i]
    for i in range(len(prob_daily_access)) : greenspaces[i]['prob_daily_access'] = prob_daily_access[i]
    for i in range(len(max_daily_cap)) : greenspaces[i]['max_daily_cap'] = max_daily_cap[i]

    return greenspaces


# Run options
do_plot = 1
do_show = 1
verbose = 1

# Sim options
basepars = dict(
  pop_size = 2000,
  verbose = verbose,
)

# Scenario metaparameters
metapars = dict(
    n_runs    = 3, # Number of parallel runs; change to 3 for quick, 11 for real
    noise     = 0.1, # Use noise, optionally
    noisepar  = 'beta',
    rand_seed = 1,
    quantiles = {'low':0.1, 'high':0.9},
)


report = ['cum_infections','cum_reinfections','cum_infectious','cum_symptomatic','cum_severe','cum_critical','cum_recoveries','cum_deaths']
runs = []


# Define the actual scenarios
scenarios = {'baseline': {
              'name':'Baseline',
              'pars': {
                  'interventions': None,
                  }
              },} 

for i in range(0,10):
    greenspaces = create_greenspaces(num_greenspaces = 4,prob_daily_access=[i/10 for j in range(4)])
    runs.append(str(i/100)+" spd")
    scenarios[str(i/100)+" spd"] = {
        'name' : str(i/100) +  "spd",
        'pars': {
                  'interventions': cv.introduce_greenspace(num_greenspaces=i,custom_spaces=greenspaces,pop_size=2000,symp_prob_dec=1/100)
                }
    }



# Run the scenarios -- this block is required for parallel processing on Windows
if __name__ == "__main__":

    scens = cv.Scenarios(basepars=basepars, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose)

    results = scens.results
    final_results = {}
    for r in report: 
        final_report = {} 
        single_results = results[r]
        for run in runs: 
            single_report_results = single_results[run]
            final_report[run] = sum(single_report_results['best'])
            print("SINGLE",sum(single_report_results['best']))
        final_results[r] = final_report
    
    
     #summarize the beta changes in a csv file
    with open('beta_changes.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(['-'] + runs)
        for row in write_result_dict_to_csv(final_results,report):
             writer.writerow(row)
    if do_plot:
        fig1 = scens.plot(do_show=do_show)

