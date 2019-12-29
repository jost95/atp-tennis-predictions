import pandas as pd
from utilities import helper as h

matches = h.load_matches(2010, 2020)
tourneys = matches.tourney_name
tourneys.to_hdf('../input/generated/tournaments.h5', key='tourneys', mode='w')