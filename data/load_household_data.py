"""
Load country household size from UN data source.

Returns dictionary with country name as key and average household size as value,
for insertion in covasim/data.
"""

import numpy as np
import pandas as pd
import sciris as sc
import covasim as cv

url = 'https://population.un.org/household/exceldata/population_division_UN_Houseshold_Size_and_Composition_2019.xlsx'
df_raw = pd.read_excel(url, sheet_name='UN HH Size and Composition 2019', skiprows=4)
assert len(df_raw) > 1, "Loaded UN Household size."

# Select and rename columns
target_columns = ['Country or area', 'Reference date (dd/mm/yyyy)', 'Average household size (number of members)']
df = df_raw[target_columns].copy()
df.columns = ['country', 'date', 'size']

# Convert date column to datetime type and replace nodata with NA.
df['date'] = df['date'].apply(lambda d: pd.to_datetime(d, format='%d/%m/%Y'))
df['size'] = df['size'].apply(lambda s: np.nan if isinstance(s, str) and s == '..' else s)
df = df.dropna()

# Take the most recent household size for each country.
df = df.sort_values(by=['country', 'date']).groupby(by=['country']).last()[['size']]
un_country_households_dict = df.to_dict()['size'].copy()

# Add to the dictionary commonly used aliases like USA for United States of America.
us_countries = [k.lower() for k in un_country_households_dict]
country_mappings_dict = cv.data.loaders.get_country_aliases()
for alias, name in country_mappings_dict.items():
    if name.lower() in us_countries:
        un_country_households_dict[alias] = un_country_households_dict[name]

pathname = str(sc.thisdir(cv.__file__, aspath=True) / 'data' / 'household_size_data.py')
print(f'To update, copy the data below into the following file:\n{pathname}\n')
print(un_country_households_dict)