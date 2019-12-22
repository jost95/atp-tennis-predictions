# This script calculates weighted win and game difference matrices

import pandas as pd
import numpy as np
import datetime

# Read actual players
players = pd.read_csv('../input/fixed/atp_players_2019.csv')['id'].to_numpy()
no_players = len(players)

# Read all matches
matches = []

for year in range(2010, 2019):
    matches.append(pd.read_csv('../input/raw/atp_matches_futures_' + str(year) + '.csv', parse_dates=['tourney_date']))
    matches.append(
        pd.read_csv('../input/raw/atp_matches_qual_chall_' + str(year) + '.csv', parse_dates=['tourney_date']))
    matches.append(pd.read_csv('../input/raw/atp_matches_' + str(year) + '.csv', parse_dates=['tourney_date']))

# Concatenate data frames
matches = pd.concat(matches, sort=False)

# Create mutual stats matrix
mutual_stats = np.zeros((no_players, no_players))
mutual_stats = pd.DataFrame(mutual_stats, players, players)

# Create conditions matrix
cond_cat = ['level', 'season', 'climate', 'surface']
cond_stats = np.zeros((no_players, len(cond_cat)))
cond_stats = pd.DataFrame(cond_stats, players, cond_cat)
t_weights = {}  # TODO: implement tournement weights

for i, match in matches.iterrows():
    winner_id = match['winner_id']
    loser_id = match['loser_id']
    time_delta = (datetime.date(2019, 1, 1) - match['tourney_date'].date()).days
    time_weight = np.exp(-time_delta / (365 * 3))

    if winner_id in players and loser_id in players:
        mutual_stats[match['winner_id']][match['loser_id']] += time_weight

mutual_stats.to_csv('../input/fixed/mutual_stats_to_20181231.csv')
