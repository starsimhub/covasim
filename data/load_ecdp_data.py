from cova_epi_scraper import Scraper
import pandas as pd


class ECDPScraper(Scraper):
    def create_date(self):
        self.df["date"] = pd.to_datetime(self.df[["year", "month", "day"]])


pars = dict()
pars['title'] = 'European Centre for Disease Prevention and Control Covid-19 Data Scraper'
pars['load_path'] = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'

pars['output_folder'] = 'epi_data/ecdp'

pars['renames'] = dict()
pars['renames']['countriesAndTerritories'] = 'key'
pars['renames']['cases']                   = 'new_diagnoses'
pars['renames']['deaths']                  = 'new_deaths'
pars['renames']['popData2018']             = 'population'

pars['fields_to_drop'] = [
    'dateRep',
    'month',
    'year',
    'geoId',
    'countryterritoryCode'
]

ECDPScraper(pars).scrape()

