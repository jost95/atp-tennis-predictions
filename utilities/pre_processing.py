import time
import pandas as pd
import numpy as np
from utilities import helper as h


def process_matches(stats_filepath, proc_match_filepath, t_weights, base_weight, proc_years):
    # Generates a match matrix with certain statistics for each match
    print('----- GENERATING PRE-PROCESSED MATCHES -----')
    start_time = time.time()

    mutual_matches_clay = pd.read_hdf(stats_filepath, key='mm_clay')
    mutual_matches_grass = pd.read_hdf(stats_filepath, key='mm_grass')
    mutual_matches_hard = pd.read_hdf(stats_filepath, key='mm_hard')
    mutual_matches = mutual_matches_clay + mutual_matches_grass + mutual_matches_hard
    mutual_score = pd.read_hdf(stats_filepath, key='ms')
    cond_stats = pd.read_hdf(stats_filepath, key='cs')
    print('Generated statistics loaded')

    # Load rankings
    rankings = h.load_rankings()

    # Load raw_matches and sport by date
    raw_matches = h.load_matches(proc_years)
    raw_matches.sort_values(by=['tourney_date'], inplace=True, ascending=True)

    # TODO: implement home advantage, season and climate, need lookup table
    # TODO: implement very recent performance, last month + tournament
    data_columns = ['date', 'rel_total_wins', 'rel_surface_wins', 'mutual_wins', 'mutual_surface_wins', 'mutual_score',
                    'rank_diff', 'points_grad_diff', 'outcome']
    matches = np.zeros((len(raw_matches), len(data_columns)), dtype=np.int64)
    matches = pd.DataFrame(matches, columns=data_columns)

    i = 0
    no_matches = len(raw_matches)

    print('Pre-processing matches...')

    # Generate training matrix and update statistics matrices
    # Loop unavoidable
    for raw_match in raw_matches.itertuples():
        match = matches.iloc[i].copy()
        winner_id = raw_match.winner_id
        loser_id = raw_match.loser_id
        tourney_date = raw_match.tourney_date
        time_weight = h.get_time_weight(proc_years['from'], tourney_date, sign=-1)
        surface = h.get_surface(raw_match.surface)

        # 0. Set date
        match.date = tourney_date

        # 1. Relative total win raw_matches differences
        rel_total_wins = h.get_relative_total_wins(cond_stats, winner_id, loser_id)
        match.rel_total_wins = round(base_weight * rel_total_wins)

        # 2. Relative surface win differences
        rel_surface_wins = h.get_relative_surface_wins(cond_stats, winner_id, loser_id, surface)
        match.rel_surface_wins = round(base_weight * rel_surface_wins)

        # 3. Mutual wins
        mutual_wins = mutual_matches[winner_id][loser_id] - mutual_matches[loser_id][winner_id]
        match.mutual_wins = mutual_wins

        # 4. Mutual surface wins
        mutual_surface_wins = h.get_mutual_surface_wins(mutual_matches_clay, mutual_matches_grass, mutual_matches_hard,
                                                        surface, winner_id, loser_id)
        match.mutual_surface_wins = mutual_surface_wins

        # 4. Mutual game
        mutual_games = mutual_score[winner_id][loser_id] - mutual_score[loser_id][winner_id]
        match.mutual_games = mutual_games

        # 5. Rank diff
        rank_diff, points_grad_diff = h.get_rankings(rankings, winner_id, loser_id, tourney_date)
        match.rank_diff = rank_diff
        match.points_grad_diff = points_grad_diff

        # 6. Winner is always winner
        match.outcome = 1

        # Create a balanced set with equal outcomes
        if i % 2 == 0:
            match = -match

        # Update entry
        matches.iloc[i] = match

        # Update stats matrices
        match_d_weight = round(base_weight * time_weight)
        match_dt_weight = round(base_weight * time_weight * t_weights[raw_match.tourney_level])

        cond_stats['total_wins'][winner_id] += match_dt_weight
        cond_stats['surface_' + surface + '_wins'][winner_id] += match_d_weight
        cond_stats['total_losses'][loser_id] += match_dt_weight
        cond_stats['surface_' + surface + '_losses'][loser_id] += match_d_weight

        # Update mutual stats
        mutual_matches[winner_id][loser_id] += match_d_weight

        # Extract win on surface
        if surface == 'clay':
            mutual_matches_clay[winner_id][loser_id] += match_d_weight
        elif surface == 'grass':
            mutual_matches_grass[winner_id][loser_id] += match_d_weight
        else:
            mutual_matches_hard[winner_id][loser_id] += match_d_weight

        try:
            winner_games, loser_games = h.get_score(raw_match.score)
        except ValueError:
            continue

        mutual_score[winner_id][loser_id] += round(base_weight * time_weight * winner_games)
        mutual_score[loser_id][winner_id] += round(base_weight * time_weight * loser_games)

        # Update counter
        i += 1
        h.print_progress(i, no_matches)

    print('All', no_matches, 'matches (100%) processed')

    matches.to_hdf(proc_match_filepath, key='matches', mode='w')

    print('Pre-processed H5 matches saved')

    end_time = time.time()
    time_diff = round(end_time - start_time)
    print('----- PRE-PROCESS COMPLETED, EXEC TIME:', time_diff, 'SECONDS ----- \n')
