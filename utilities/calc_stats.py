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

# Create mutual stats matrices
mutual_matches = np.zeros((no_players, no_players))
mutual_matches = pd.DataFrame(mutual_matches, players, players)

mutual_matches_clay = np.zeros((no_players, no_players))
mutual_matches_clay = pd.DataFrame(mutual_matches_clay, players, players)

mutual_matches_grass = np.zeros((no_players, no_players))
mutual_matches_grass = pd.DataFrame(mutual_matches_grass, players, players)

mutual_matches_hard = np.zeros((no_players, no_players))
mutual_matches_hard = pd.DataFrame(mutual_matches_hard, players, players)

mutual_games = np.zeros((no_players, no_players))
mutual_games = pd.DataFrame(mutual_games, players, players)

# Create general perfomance matrix
# TODO: implement season and climate with lookup table
cond_cat = ['total_won', 'total_lost', 'surface_clay_won', 'surface_clay_lost', 'surface_grass_won',
            'surface_grass_lost', 'surface_hard_won', 'surface_hard_lost', 'surface_carpet_won', 'surface_carpet_lost']
cond_stats = np.zeros((no_players, len(cond_cat)))
cond_stats = pd.DataFrame(cond_stats, players, cond_cat)

# Weights
t_weights = {'S': 0.125, 'C': 0.25, 'A': 0.5, 'M': 0.75, 'D': 0.75, 'G': 1, 'F': 1}
base_weight = 10

for i, match in matches.iterrows():
    winner_id = match['winner_id']
    loser_id = match['loser_id']
    time_delta = (datetime.date(2019, 1, 1) - match['tourney_date'].date()).days
    time_weight = np.exp(-time_delta / (365 * 3))

    # Guess surface as hard if not known
    surface = str(match['surface']).lower()
    surface = 'hard' if surface == 'nan' or surface == 'none' else surface

    # Winner stats
    if winner_id in players:
        cond_stats['total_won'][winner_id] += base_weight * t_weights[match['tourney_level']] * time_weight
        cond_stats['surface_' + surface + '_won'] += base_weight * time_weight

    # Loser stats
    if loser_id in players:
        cond_stats['total_lost'][winner_id] += base_weight * t_weights[match['tourney_level']] * time_weight
        cond_stats['surface_' + surface + '_lost'] += base_weight * time_weight

    # Mutual stats
    if winner_id in players and loser_id in players:
        # Extract win
        mutual_matches[winner_id][loser_id] += base_weight * time_weight

        # Extract win on surface
        if surface == 'clay':
            mutual_matches_clay[winner_id][loser_id] += base_weight * time_weight
        elif surface == 'grass':
            mutual_matches_grass[winner_id][loser_id] += base_weight * time_weight
        else:
            mutual_matches_hard[winner_id][loser_id] += base_weight * time_weight

        # Extract games
        score = str(match['score'])

        # Continue if score is not known
        if score == 'nan':
            continue

        sets = score.split()
        winner_games = 0
        loser_games = 0

        try:
            for s in sets:
                # Remove weird scoring
                games = ''.join(c for c in s if c not in '[]RET').split('-')

                if len(games) == 1:
                    continue

                # This might cause trouble if score not properly formatted
                winner_games += int(games[0][0])
                loser_games += int(games[1][0])
        except ValueError:
            continue

        mutual_games[winner_id][loser_id] += base_weight * time_weight * winner_games
        mutual_games[loser_id][winner_id] += base_weight * time_weight * loser_games

mutual_matches.to_csv('../input/fixed/mutual_matches_to_20181231.csv')
mutual_matches_clay.to_csv('../input/fixed/mutual_matches_clay_to_20181231.csv')
mutual_matches_grass.to_csv('../input/fixed/mutual_matches_grass_to_20181231.csv')
mutual_matches_hard.to_csv('../input/fixed/mutual_matches_hard_to_20181231.csv')
mutual_games.to_csv('../input/fixed/mutual_games_to_20181231.csv')
