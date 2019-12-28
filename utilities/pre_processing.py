import datetime
import time

import pandas as pd
import numpy as np
from typing import Dict
from utilities.statistics import get_score, load_matches, print_progress, get_time_weight, get_surface


def load_rankings():
    rankings_10s = pd.read_csv('input/raw/atp_rankings_10s.csv', parse_dates=['ranking_date'])
    rankings_current = pd.read_csv('input/raw/atp_rankings_current.csv', parse_dates=['ranking_date'])
    rankings = pd.concat([rankings_10s, rankings_current], sort=False)

    # Sort by date (oldest ranking first)
    rankings.sort_values(by=['ranking_date'], inplace=True, ascending=True)

    print('Rankings loaded')
    return rankings


def get_relative_surface_wins(cond_stats, winner_id, loser_id, surface):
    surface_wins_winner = cond_stats['surface_' + surface + '_wins'][winner_id]
    surface_losses_winner = cond_stats['surface_' + surface + '_losses'][winner_id]
    surface_played_winner = surface_wins_winner + surface_losses_winner
    surface_wins_loser = cond_stats['surface_' + surface + '_wins'][loser_id]
    surface_losses_loser = cond_stats['surface_' + surface + '_losses'][loser_id]
    surface_played_loser = surface_wins_loser + surface_losses_loser

    if surface_played_winner == 0:
        rel_surface_wins_winner = 0
    else:
        rel_surface_wins_winner = float(surface_wins_winner) / surface_played_winner

    if surface_played_loser == 0:
        rel_surface_wins_loser = 0
    else:
        rel_surface_wins_loser = float(surface_wins_loser) / surface_played_loser

    return rel_surface_wins_winner - rel_surface_wins_loser


def get_relative_total_wins(cond_stats, winner_id, loser_id):
    total_wins_winner = cond_stats['total_wins'][winner_id]
    total_losses_winner = cond_stats['total_losses'][winner_id]
    total_played_winner = total_wins_winner + total_losses_winner
    total_wins_loser = cond_stats['total_wins'][loser_id]
    total_losses_loser = cond_stats['total_losses'][loser_id]
    total_played_loser = total_wins_loser + total_losses_loser

    if total_played_winner == 0:
        rel_total_wins_winner = 0
    else:
        rel_total_wins_winner = float(total_wins_winner) / total_played_winner

    if total_played_loser == 0:
        rel_total_wins_loser = 0
    else:
        rel_total_wins_loser = float(total_wins_loser) / total_played_loser

    return rel_total_wins_winner - rel_total_wins_loser


def get_mutual_surface_wins(mm_clay, mm_grass, mm_hard, surface, winner_id,
                            loser_id):
    if surface == 'clay':
        return mm_clay[winner_id][loser_id] - mm_clay[loser_id][winner_id]
    elif surface == 'grass':
        return mm_grass[winner_id][loser_id] - mm_grass[loser_id][winner_id]
    else:
        return mm_hard[winner_id][loser_id] - mm_hard[loser_id][winner_id]


def get_rankings(rankings, winner_id, loser_id, tourney_date):
    winner_rankings = rankings.loc[rankings['player'] == winner_id]
    loser_rankings = rankings.loc[rankings['player'] == loser_id]
    highest_numbered_ranking = np.max(rankings['rank'])

    # Set date as ranking index, delete possible duplicates due to overlapping lists
    winner_rankings.set_index('ranking_date', inplace=True)
    winner_rankings = winner_rankings.loc[~winner_rankings.index.duplicated(keep='first')]
    loser_rankings.set_index('ranking_date', inplace=True)
    loser_rankings = loser_rankings.loc[~loser_rankings.index.duplicated(keep='first')]

    # It is not certain that all players have a ranking right now
    try:
        winner_current_rank = winner_rankings.iloc[winner_rankings.index.get_loc(tourney_date, method='pad')]
        winner_current_points = winner_current_rank['points']
        winner_current_rank = winner_current_rank['rank']
    except KeyError:
        winner_current_rank = highest_numbered_ranking + 1
        winner_current_points = 0

    try:
        loser_current_rank = loser_rankings.iloc[loser_rankings.index.get_loc(tourney_date, method='pad')]
        loser_current_points = loser_current_rank['points']
        loser_current_rank = loser_current_rank['rank']
    except KeyError:
        loser_current_rank = highest_numbered_ranking + 1
        loser_current_points = 0

    rank_diff = winner_current_rank - loser_current_rank

    last_year_date = tourney_date - pd.DateOffset(years=1)

    # It is not certain that all players had a ranking one year ago
    try:
        winner_old_rank = winner_rankings.iloc[winner_rankings.index.get_loc(last_year_date, method='pad')]
        winner_old_points = winner_old_rank['points']
    except KeyError:
        winner_old_points = 0

    try:
        loser_old_rank = winner_rankings.iloc[winner_rankings.index.get_loc(last_year_date, method='pad')]
        loser_old_points = loser_old_rank['points']
    except KeyError:
        loser_old_points = 0

    winner_points_grad = winner_current_points - winner_old_points
    loser_points_grad = loser_current_points - loser_old_points
    points_grad_diff = winner_points_grad - loser_points_grad

    return rank_diff, points_grad_diff


# noinspection PyTypeChecker
def generate_data(t_weights, base_weight, from_data_year, to_data_year):
    print('----- GENERATING PRE-PROCESSED MATCHES -----')
    start_time = time.time()

    filename = 'input/generated/match_statistics.h5'
    mutual_matches_clay = pd.read_hdf(filename, key='mm_clay')
    mutual_matches_grass = pd.read_hdf(filename, key='mm_grass')
    mutual_matches_hard = pd.read_hdf(filename, key='mm_hard')
    mutual_matches = mutual_matches_clay + mutual_matches_grass + mutual_matches_hard
    mutual_score = pd.read_hdf(filename, key='ms')
    cond_stats = pd.read_hdf(filename, key='cs')
    print('Generated statistics loaded')

    # Load rankings
    rankings = load_rankings()

    # Load raw_matches and sport by date
    raw_matches = load_matches(from_data_year, to_data_year)
    raw_matches.sort_values(by=['tourney_date'], inplace=True, ascending=True)

    # TODO: implement home advantage, season and climate, need lookup table
    # TODO: implement very recent performance, last month + tournament
    data_columns = ['rel_total_wins', 'rel_surface_wins', 'mutual_wins', 'mutual_surface_wins', 'mutual_score',
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
        time_weight = get_time_weight(from_data_year, tourney_date, sign=-1)
        surface = get_surface(raw_match.surface)

        # 1. Relative total win raw_matches differences
        rel_total_wins = get_relative_total_wins(cond_stats, winner_id, loser_id)
        match.rel_total_wins = round(base_weight * rel_total_wins)

        # 2. Relative surface win differences
        rel_surface_wins = get_relative_surface_wins(cond_stats, winner_id, loser_id, surface)
        match.rel_surface_wins = round(base_weight * rel_surface_wins)

        # 3. Mutual wins
        mutual_wins = mutual_matches[winner_id][loser_id] - mutual_matches[loser_id][winner_id]
        match.mutual_wins = mutual_wins

        # 4. Mutual surface wins
        mutual_surface_wins = get_mutual_surface_wins(mutual_matches_clay, mutual_matches_grass, mutual_matches_hard,
                                                      surface, winner_id, loser_id)
        match.mutual_surface_wins = mutual_surface_wins

        # 4. Mutual game
        mutual_games = mutual_score[winner_id][loser_id] - mutual_score[loser_id][winner_id]
        match.mutual_games = mutual_games

        # 5. Rank diff
        rank_diff, points_grad_diff = get_rankings(rankings, winner_id, loser_id, tourney_date)
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
            winner_games, loser_games = get_score(raw_match.score)
        except ValueError:
            continue

        mutual_score[winner_id][loser_id] += round(base_weight * time_weight * winner_games)
        mutual_score[loser_id][winner_id] += round(base_weight * time_weight * loser_games)

        # Update counter
        i += 1
        if i % 10000 == 0:
            print_progress(i, no_matches)

    print('All', no_matches, 'matches (100%) processed')

    matches.to_hdf('input/generated/processed_matches.h5', key='matches', mode='w')

    print('Pre-processed H5 matches saved')

    end_time = time.time()
    time_diff = round(end_time - start_time)
    print('----- PRE-PROCESS COMPLETED, EXEC TIME:', time_diff, 'SECONDS ----- \n')
