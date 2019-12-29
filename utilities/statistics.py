import time
import datetime
import pandas as pd
import numpy as np
from utilities import helper as h


def generate_match_statistics(t_weights, base_weight, from_stats_year, to_stats_year, from_data_year, to_data_year):
    # Generates match statistics matrices for a certain time period
    print('----- GENERATING MATCH STATISTICS -----')
    start_time = time.time()

    # Load players
    player_ids = h.extract_player_ids(from_data_year, to_data_year)
    no_players = len(player_ids)

    # Load matches to generate statistics
    matches = h.load_matches(from_stats_year, to_stats_year, player_ids)
    no_matches = len(matches)

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

    print('Generating match statistics...')

    # Loop is unavoidable...
    for match in matches.itertuples():
        winner_id = match.winner_id
        loser_id = match.loser_id
        time_weight = h.get_time_weight(to_stats_year, match.tourney_date)
        surface = h.get_surface(match.surface)

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
                winner_games, loser_games = h.get_score(match.score)
            except ValueError:
                continue

            mutual_score[winner_id][loser_id] += round(base_weight * time_weight * winner_games)
            mutual_score[loser_id][winner_id] += round(base_weight * time_weight * loser_games)

        # Update counter
        i += 1
        h.print_progress(i, no_matches)

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
