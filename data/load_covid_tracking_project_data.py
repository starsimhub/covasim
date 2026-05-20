'''
This script creates a single file containing all the scraped 
data from the Covid Data Project Data Scraper.
https://covidtracking.com
'''

from cova_epi_scraper import Scraper
import sciris as sc

def covid_tracking_date_to_date(d):
    ''' Date is in format e.g. 20201031 '''
    out = sc.readdate(str(d)) # Should be a supported format
    return out

class CovidTrackingProjectScraper(Scraper):
    def create_date(self):
        self.df['date'] = self.df.date.apply(covid_tracking_date_to_date)


class CovidUSTrackingProjectScraper(CovidTrackingProjectScraper):
    def create_key(self):
        self.df['key'] = 'US'


## Set up parameters 
pars_state = dict()
pars_state['title']         = "Covid Tracking Project Scraper for US states"
pars_state['load_path']     = "https://covidtracking.com/api/v1/states/daily.csv"
pars_state['output_folder'] = "epi_data/covid-tracking"

pars_state['renames'] = dict()
pars_state['renames']['state']                    = "key"
pars_state['renames']['positiveIncrease']         = "new_diagnoses"
pars_state['renames']['negativeIncrease']         = "new_negatives"
pars_state['renames']['totalTestResultsIncrease'] = "new_tests"
pars_state['renames']['hospitalizedIncrease']     = "new_hospitalized"
pars_state['renames']['deathIncrease']            = "new_deaths"
pars_state['renames']['inIcuCumulative']          = "cum_icu"
pars_state['renames']['hospitalizedCumulative']   = "cum_hospitalized"
pars_state['renames']['onVentilatorCumulative']   = "cum_on_ventilator"

pars_state['cumulative_fields'] = dict()
pars_state['cumulative_fields']['cum_icu']           = "new_icu"
pars_state['cumulative_fields']['cum_on_ventilator'] = "num_on_ventilator"


pars_state['fields_to_drop'] = [
    "hash",
    "dateChecked",
    "totalTestResults",
    "posNeg",
    "positive",
    "negative",
    "pending",
    "hospitalizedCurrently",
    "inIcuCurrently",
    "onVentilatorCurrently",
    "recovered",
    "hospitalized",
    "total"
]

# Set up US parameters
pars_us = dict()
pars_us['title']         = "Covid Tracking Project Scraper for US states"
pars_us['load_path']     = "https://covidtracking.com/api/v1/us/daily.csv"
pars_us['output_folder'] = "epi_data/covid-tracking"

pars_us['renames'] = dict()
pars_us['renames']['positiveIncrease']         = "new_diagnoses"
pars_us['renames']['negativeIncrease']         = "new_negatives"
pars_us['renames']['totalTestResultsIncrease'] = "new_tests"
pars_us['renames']['hospitalizedIncrease']     = "new_hospitalized"
pars_us['renames']['deathIncrease']            = "new_deaths"
pars_us['renames']['inIcuCumulative']          = "cum_icu"
pars_us['renames']['hospitalizedCumulative']   = "cum_hospitalized"
pars_us['renames']['onVentilatorCumulative']   = "cum_on_ventilator"

pars_us['cumulative_fields'] = dict()
pars_us['cumulative_fields']['cum_icu']           = "new_icu"
pars_us['cumulative_fields']['cum_on_ventilator'] = "new_on_ventilator"


pars_us['fields_to_drop'] = [
    "hash",
    "dateChecked",
    "totalTestResults",
    "posNeg",
    "positive",
    "negative",
    "pending",
    "hospitalizedCurrently",
    "inIcuCurrently",
    "onVentilatorCurrently",
    "recovered",
    "hospitalized",
    "states",
    "total"
]

# Scrape states
CovidTrackingProjectScraper(pars_state).scrape()
# Scrape US
CovidUSTrackingProjectScraper(pars_us).scrape()


