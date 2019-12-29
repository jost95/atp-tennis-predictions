# This script should only be run ones to generate country codes
import os
import pandas as pd
import numpy as np
import requests

from definitions import GEN_PATH
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
    country_codes = []

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
    country_codes = pd.DataFrame(country_codes, columns=['country_code'])
    tourney_info = pd.concat([country_codes, countries, unique_loc], sort=False, axis=1)
    tourney_info.to_csv(os.path.join(GEN_PATH, 'tourneys.csv'))


def get_country_code():
    tourney_info = pd.read_csv(os.path.join(GEN_PATH, 'tourneys.csv'))

