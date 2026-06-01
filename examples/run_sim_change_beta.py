'''
Simple script for running Covasim scenarios
'''

import covasim as cv
import csv


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

report = ['cum_infections','cum_reinfections','cum_infectious','cum_symptomatic','cum_severe','cum_critical','cum_recoveries','cum_deaths','cum_tests','cum_diagnoses','cum_known_deaths','cum_quarantined','cum_isolated']
runs = []
# Define the actual scenarios
start_day = '2020-04-04'
scenarios = {} 
for i in range(1,11):
    runs.append(str(i)+"beta")
    scenarios[str(i)+"beta"] = {
        'name' : str(i) + "beta",
        'pars': {
                  'interventions': cv.change_beta(days=start_day, changes=i/100)
                }
    }

def write_result_dict_to_csv(result_dict,cols):
    out = []
    for row in result_dict.keys(): 
       out.append([row] + list(result_dict[row].values()))
    return out 

# Run the scenarios -- this block is required for parallel processing on Windows
if __name__ == "__main__":

    scens = cv.Scenarios(basepars=basepars, metapars=metapars, scenarios=scenarios)
    #scens.run(verbose=verbose)
    scens.run()
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
    
    print(final_results)

    #summarize the beta changes in a csv file
    with open('beta_changes.csv', 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(['-'] + runs)
        for row in write_result_dict_to_csv(final_results,report):
             writer.writerow(row)

     
    print("RESULTS", scens.results)
    if do_plot:
        fig1 = scens.plot(do_show=do_show,style='simple')


