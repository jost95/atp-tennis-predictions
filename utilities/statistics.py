# This script calculates weighted win and game difference matrices
import time

import pandas as pd
import numpy as np
import datetime


def extract_player_ids(from_year, to_year):
    # Extract players from correct matches (future ones)
    matches = []

    for year in range(from_year, to_year + 1):
        matches.append(
            pd.read_csv('input/raw/atp_matches_futures_' + str(year) + '.csv', parse_dates=['tourney_date']))
        matches.append(
            pd.read_csv('input/raw/atp_matches_qual_chall_' + str(year) + '.csv', parse_dates=['tourney_date']))
        matches.append(pd.read_csv('input/raw/atp_matches_' + str(year) + '.csv', parse_dates=['tourney_date']))

    matches = pd.concat(matches, sort=False)
    winner_ids = matches.winner_id.to_numpy()
    loser_ids = matches.loser_id.to_numpy()

    # Filter out unique players
    players = np.unique(np.append(winner_ids, loser_ids))

    print('Players loaded, number of players:', len(players))
    return players


def load_matches(from_year, to_year, player_ids=None):
    matches = []

    for year in range(from_year, to_year + 1):
        matches.append(
            pd.read_csv('input/raw/atp_matches_futures_' + str(year) + '.csv', parse_dates=['tourney_date']))
        matches.append(
            pd.read_csv('input/raw/atp_matches_qual_chall_' + str(year) + '.csv', parse_dates=['tourney_date']))
        matches.append(pd.read_csv('input/raw/atp_matches_' + str(year) + '.csv', parse_dates=['tourney_date']))

    # Concatenate data frames
    matches = pd.concat(matches, sort=False)

    if player_ids is not None:
        # Remove not wanted matches
        matches = matches[matches['winner_id'].isin(player_ids) | matches['loser_id'].isin(player_ids)]

    # Drop not relevant columns
    matches = matches.filter(['winner_id', 'loser_id', 'tourney_date', 'tourney_level', 'surface', 'score'])

    print('Matches loaded, number of matches:', len(matches))
    return matches


def get_time_weight(base_year, current_date, sign=1):
    time_delta = (datetime.date(base_year + 1, 1, 1) - current_date.date()).days
    return np.exp(-sign * time_delta / (365 * 3))


def get_surface(surface):
    # Guess surface as hard if not known
    surface = str(surface).lower()
    return 'hard' if surface == 'nan' or surface == 'none' else surface


def get_score(score):
    score = str(score)

    # Continue if score is not known
    if score == 'nan':
        return 0, 0

    sets = score.split()
    winner_games = 0
    loser_games = 0

    for s in sets:
        # Remove weird scoring
        games = ''.join(c for c in s if c not in '[]RET').split('-')

        if len(games) == 1:
            continue

        # This might cause trouble if score not properly formatted
        winner_games += int(games[0][0])
        loser_games += int(games[1][0])

    return winner_games, loser_games


def print_progress(i, no_matches):
    print(i, 'matches (' + str(round(i / no_matches * 100, 2)) + '%) processed')


def generate_match_statistics(t_weights, base_weight, from_stats_year, to_stats_year, from_data_year, to_data_year):
    print('----- GENERATING MATCH STATISTICS -----')
    start_time = time.time()

    # Load players
    player_ids = extract_player_ids(from_data_year, to_data_year)
    no_players = len(player_ids)

    # Load matches to generate statistics
    matches = load_matches(from_stats_year, to_stats_year, player_ids)

    # Create mutual stats matrices
    base_matrix = np.zeros((no_players, no_players), dtype=np.int64)
    mutual_matches_clay = pd.DataFrame(base_matrix, player_ids, player_ids)
    mutual_matches_grass = pd.DataFrame(base_matrix, player_ids, player_ids)
    mutual_matches_hard = pd.DataFrame(base_matrix, player_ids, player_ids)
    mutual_score = pd.DataFrame(base_matrix, player_ids, player_ids)

    # Create general perfomance matrix
    # TODO: implement season and climate with lookup table
    cond_cat = ['total_wins', 'total_losses', 'surface_clay_wins', 'surface_clay_losses', 'surface_grass_wins',
                'surface_grass_losses', 'surface_hard_wins', 'surface_hard_losses', 'surface_carpet_wins',
                'surface_carpet_losses']
    cond_stats = np.zeros((no_players, len(cond_cat)), dtype=np.int64)
    cond_stats = pd.DataFrame(cond_stats, player_ids, cond_cat)

    # Counter for timing purposes
    i = 0
    no_matches = len(matches)

    print('Generating match statistics...')

    # Loop is unavoidable...
    for match in matches.itertuples():
        winner_id = match.winner_id
        loser_id = match.loser_id
        time_weight = get_time_weight(to_stats_year, match.tourney_date)
        surface = get_surface(match.surface)

        # Calculate match weights
        match_d_weight = round(base_weight * time_weight)
        match_dt_weight = round(base_weight * time_weight * t_weights[match.tourney_level])

        # Check flags
        winner_in_ids = False
        loser_in_ids = False

        # Winner stats
        if winner_id in player_ids:
            cond_stats['total_wins'][winner_id] += match_dt_weight
            cond_stats['surface_' + surface + '_wins'][winner_id] += match_d_weight
            winner_in_ids = True

        # Loser stats
        if loser_id in player_ids:
            cond_stats['total_losses'][loser_id] += match_dt_weight
            cond_stats['surface_' + surface + '_losses'][loser_id] += match_d_weight
            loser_in_ids = True

        # Mutual statistics
        if winner_in_ids and loser_in_ids:
            # Extract win on surface
            if surface == 'clay':
                mutual_matches_clay[winner_id][loser_id] += match_d_weight
            elif surface == 'grass':
                mutual_matches_grass[winner_id][loser_id] += match_d_weight
            else:
                mutual_matches_hard[winner_id][loser_id] += match_d_weight

            try:
                winner_games, loser_games = get_score(match.score)
            except ValueError:
                continue

            mutual_score[winner_id][loser_id] += round(base_weight * time_weight * winner_games)
            mutual_score[loser_id][winner_id] += round(base_weight * time_weight * loser_games)

        # Update counter
        i += 1
        if i % 10000 == 0:
            print_progress(i, no_matches)

    print('All', no_matches, 'matches (100%) processed')

    # To avoid running script every training phase
    filename = 'input/generated/match_statistics.h5'
    mutual_matches_clay.to_hdf(filename, key='mm_clay', mode='w')
    mutual_matches_grass.to_hdf(filename, key='mm_grass')
    mutual_matches_hard.to_hdf(filename, key='mm_hard')
    mutual_score.to_hdf(filename, key='ms')
    cond_stats.to_hdf(filename, key='cs')

    print('H5 statistics file saved')

    end_time = time.time()
    time_diff = round(end_time - start_time)
    print('----- MATCH STATISTICS COMPLETED, EXEC TIME:', time_diff, 'SECONDS ----- \n')
