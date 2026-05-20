import os
import logging
import pandas as pd
import sciris as sc

class Scraper(sc.prettyobj):
    '''
    Standard methods for scrapers
    '''

    def __init__(self, parameters):
        assert parameters.get("load_path", False), "Must provide load_path"
        self.load_path = parameters["load_path"]

        self.output_folder = parameters.get(
            "output_folder", "epi_data")


        self.renames = parameters.get("renames")
        self.fields_to_drop = parameters.get("fields_to_drop")
        self.cumulative_fields = parameters.get("cumulative_fields")

        self.df = None
        self.grouping = None

        self.log = logging.getLogger(__name__)
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        if parameters.get("scape_on_init", False):
            self.scrape()


    def scrape(self):
        self.preload()
        self.load()
        self.transform()
        self.test_quality()
        self.output()

    ## PRELOAD

    def preload(self):
        pass

    ## LOAD DATA

    def load(self):
        # Read ito a dataframe
        self.log.info(f"Loading data from {self.load_path}")
        self.df = pd.read_csv(self.load_path)
        self.log.info(f"Loaded {len(self.df)} records.")
        self.log.info(f"Original columns: {', '.join(self.df.columns)}")

    ## TRANSFORM DATA


    def transform(self):
        self.rename_fields()
        self.create_date()
        self.create_key()
        self.group_data()
        self.create_day()
        self.convert_cumulative_fields()
        self.drop_fields()

    def rename_fields(self):
        if self.renames is not None:
            self.log.info(f"Renaming fields: {self.renames}")
            self.df = self. df.rename(columns=self.renames)

    def create_date(self):
        self.df['date'] = pd.to_datetime(self.df.date)

    def create_key(self):
        pass

    def group_data(self):
        assert 'key' in self.df.columns, 'No column named "key"; do you need to rename?'
        assert 'date' in self.df.columns, 'No column named "date"; do you neeed to define a create_date method?'

        self.df = self.df.sort_values(['key', 'date'])
        self.grouping = self.df.groupby('key')

    def create_day(self):
        print(len(self.df))
        self.df['day'] = (self.grouping['date'].transform(
            lambda x: (x - min(x))).apply(lambda x: x.days))

    def convert_cumulative_fields(self):
        if self.cumulative_fields:
            for cum, num in self.cumulative_fields.items():
                self.convert_cum_to_num(cum, num)

    def convert_cum_to_num(self, cum_field, num_field):
        lag_field = f"lagged_{cum_field}"
        self.df[lag_field] = self.grouping[cum_field].shift(1)
        self.df[num_field] = self.df[cum_field].sub(
            self.df[lag_field]).fillna(self.df[cum_field])
        self.df.drop([lag_field], inplace=True, axis=1)

    def drop_fields(self):
        if self.fields_to_drop:
            self.log.info(f"Dropping fields {', '.join(self.fields_to_drop)}.")
            self.df.drop(self.fields_to_drop, inplace=True, axis=1)

    ## TEST DATA QUALITY

    def test_quality(self):
        self.run_general_data_quality_tests()
        self.run_additional_data_quality_tests()

    def run_general_data_quality_tests(self):
        # date and day present?
        assert 'date' in self.df.columns, f"Data must have a 'date' field. Current columns are {', '.join(self.df.columns)}"
        assert 'day' in self.df.columns, f"Data must have a 'day' field. Current columns are {', '.join(self.df.columns)}"

        # are data in sequence?
        for g in self.grouping['date']:
            number_of_days = (g[1].max() - g[1].min()).days + 1
            records = len(g[1])
            if len(g[1]) != number_of_days:
                self.log.warn(f"Entity {g[0]} does not have as many records ({records}) as days ({number_of_days}). Should be ok, though.")


    def run_additional_data_quality_tests(self):
        pass

    ## OUTPUT DATA

    def output(self):
        self.log.info(f"Final columns: {', '.join(self.df.columns)}")
        self.log.info("First rows of data:")
        self.log.info(self.df.head())
        here = sc.thisdir(__file__)
        data_home = os.path.join(here, self.output_folder)

        for g in self.grouping:
            key_value = g[0]
            filename = f'{sc.sanitizefilename(key_value)}.csv'
            filepath = sc.makefilepath(filename=filename, folder=data_home)
            self.log.info(f'Creating {filepath}')
            mini_df = self.df[self.df.key == key_value]
            mini_df.to_csv(filepath)

        self.log.info(
            f"There are {len(self.grouping)} entities in this dataset.")
        self.log.info(f"Saved {len(self.df)} records.")

