# This script should only be run once to generate country codes
import json
import os
import pandas as pd
import numpy as np
import requests
import pycountry

from definitions import GEN_PATH, ROOT_DIR
from utilities import helper as h

years = {'from': 2010, 'to': 2019}
matches = h.load_matches(years)


def extract_country_name():
    # Get unique names
    unique_tourneys = np.unique(matches.tourney_name.to_numpy())

    # Extract name of location
    locations = np.vectorize(h.filter_tourney_name)(unique_tourneys)

    # Get unique locations
    unique_loc = np.unique(locations)

    # Remove empty strings
    unique_loc = np.array(list(filter(None, unique_loc))).tolist()

    # Generate empty arrays
    countries = []

    # Create new web session
    search_session = requests.Session()

    # Fetch country name from location
    # noinspection PyTypeChecker
    for i in range(len(unique_loc)):
        if i % 50 == 0:
            print(i)

        countries.append(h.fetch_country(unique_loc[i], search_session))

    unique_loc = pd.DataFrame(unique_loc, columns=['location'])
    countries = pd.DataFrame(countries, columns=['country_name'])
    tourney_info = pd.concat([countries, unique_loc], sort=False, axis=1)
    tourney_info.to_csv(os.path.join(GEN_PATH, 'tourneys_raw.csv'))


def manual_country_fix():
    tourney_info = pd.read_csv(os.path.join(GEN_PATH, 'tourneys_raw.csv'), index_col=0)

    with open(os.path.join(ROOT_DIR, 'utilities/location_swaps.json')) as f:
        swaps = json.load(f)

    from_swaps = []
    to_swaps = []

    for swap in swaps:
        for name in swap['from']:
            from_swaps.append(name)
            to_swaps.append(swap['to'])

    tourney_info.country_name.replace(from_swaps, to_swaps, inplace=True)
    tourney_info.sort_values(by=['country_name', 'location'], inplace=True, ascending=True)
    tourney_info.reset_index(drop=True, inplace=True)
    tourney_info.to_csv(os.path.join(GEN_PATH, 'tourneys_fixed.csv'))


def get_country_code():
    tourney_info = pd.read_csv(os.path.join(GEN_PATH, 'tourneys_fixed.csv'))
    unique_countries = np.unique(tourney_info.country_name)
    unique_ccs = []

    for c in list(pycountry.countries):
        print(c.name)

    for country in unique_countries:
        try:
            unique_ccs.append(pycountry.countries.get(name=country).alpha_3)
        except AttributeError:
            unique_ccs.append(country)

    print(unique_ccs)

# 1. Geo-location search for country name (takes time)
# extract_country_name()

# 2 Fix wrongly formatted countries and sort by country
# manual_country_fix()

# 3. Lookup country code
get_country_code()

# 4. Lookup climate
