import datetime
import pandas as pd
import numpy as np
from typing import Dict
from utilities.statistics import get_score, load_matches, print_progress, get_time_weight, get_surface


def load_rankings():
    print('Loading rankings...')
    rankings_10s = pd.read_csv('input/raw/atp_rankings_10s.csv', parse_dates=['ranking_date'])
    rankings_current = pd.read_csv('input/raw/atp_rankings_current.csv', parse_dates=['ranking_date'])
    rankings = pd.concat([rankings_10s, rankings_current], sort=False)

    # Sort by date (oldest ranking first)
    rankings.sort_values(by=['ranking_date'], inplace=True, ascending=True)
    print('Rankings loaded')
    return rankings


def get_relative_surface_wins(cond_stats, winner_id, loser_id, surface):
    winner_surface_won = cond_stats['surface_' + surface + '_won'][winner_id]
    winner_surface_lost = cond_stats['surface_' + surface + '_lost'][winner_id]
    winner_surface_played = winner_surface_won + winner_surface_lost
    loser_surface_won = cond_stats['surface_' + surface + '_won'][loser_id]
    loser_surface_lost = cond_stats['surface_' + surface + '_lost'][loser_id]
    loser_surface_played = loser_surface_won + loser_surface_lost

    if winner_surface_played == 0:
        winner_rel_surface_won = 0
    else:
        winner_rel_surface_won = float(winner_surface_won) / winner_surface_played

    if loser_surface_played == 0:
        loser_rel_surface_won = 0
    else:
        loser_rel_surface_won = float(loser_surface_won) / loser_surface_played

    return winner_rel_surface_won - loser_rel_surface_won


def get_relative_total_wins(cond_stats, winner_id, loser_id):
    winner_total_won = cond_stats['total_won'][winner_id]
    winner_total_lost = cond_stats['total_lost'][winner_id]
    winner_total_played = winner_total_won + winner_total_lost
    loser_total_won = cond_stats['total_won'][loser_id]
    loser_total_lost = cond_stats['total_lost'][loser_id]
    loser_total_played = loser_total_won + loser_total_lost

    if winner_total_played == 0:
        winner_rel_total_won = 0
    else:
        winner_rel_total_won = float(winner_total_won) / winner_total_played

    if loser_total_played == 0:
        loser_rel_total_won = 0
    else:
        loser_rel_total_won = float(loser_total_won) / loser_total_played

    return winner_rel_total_won - loser_rel_total_won


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
    print('PRE-PROCESSING MATCHES')

    print('Loading statistics...')
    filename = 'input/fixed/match_statistics.h5'
    mutual_matches_clay = pd.read_hdf(filename, key='mm_clay')
    mutual_matches_grass = pd.read_hdf(filename, key='mm_grass')
    mutual_matches_hard = pd.read_hdf(filename, key='mm_hard')
    mutual_matches = mutual_matches_clay + mutual_matches_grass + mutual_matches_hard
    mutual_games = pd.read_hdf(filename, key='mg')
    cond_stats = pd.read_hdf(filename, key='cs')
    print('Statistics loaded')

    # Load rankings
    rankings = load_rankings()

    # Load raw_matches and sport by date
    raw_matches = load_matches(from_data_year, to_data_year)
    raw_matches.sort_values(by=['tourney_date'], inplace=True, ascending=True)

    # TODO: implement home advantage, season and climate, need lookup table
    # TODO: implement very recent performance, last month + tournament
    data_columns = ['rel_total_won', 'rel_surface_won', 'mutual_won', 'mutual_surface_won', 'mutual_game', 'rank_diff',
                    'points_grad_diff', 'outcome']
    matches = np.zeros((len(raw_matches), len(data_columns)), dtype=np.int64)
    matches = pd.DataFrame(matches, columns=data_columns)

    i = 0
    no_matches = len(raw_matches)

    # Generate training matrix and update statistics matrices
    for raw_match in raw_matches.itertuples():
        i += 1

        if i % 1000 == 0:
            print_progress(i, no_matches)

        match = matches.iloc[i].copy()
        winner_id = raw_match.winner_id
        loser_id = raw_match.loser_id
        tourney_date = raw_match.tourney_date
        time_weight = get_time_weight(from_data_year, tourney_date, sign=-1)

        surface = get_surface(raw_match.surface)

        # 1. Relative total won raw_matches differences
        rel_total_wins = get_relative_total_wins(cond_stats, winner_id, loser_id)
        match['rel_total_won'] = round(base_weight * rel_total_wins)

        # 2. Relative surface won differences
        rel_surface_wins = get_relative_surface_wins(cond_stats, winner_id, loser_id, surface)
        match['rel_surface_won'] = round(base_weight * rel_surface_wins)

        # 3. Mutual wins
        mutual_won = mutual_matches[winner_id][loser_id] - mutual_matches[loser_id][winner_id]
        match['mutual_won'] = mutual_won

        # 4. Mutual surface wins
        mutual_surface_wins = get_mutual_surface_wins(mutual_matches_clay, mutual_matches_grass, mutual_matches_hard,
                                                      surface, winner_id, loser_id)
        match['mutual_surface_won'] = mutual_surface_wins

        # 4. Mutual game
        mutual_game = mutual_games[winner_id][loser_id] - mutual_games[loser_id][winner_id]
        match['mutual_game'] = mutual_game

        # 5. Rank diff
        rank_diff, points_grad_diff = get_rankings(rankings, winner_id, loser_id, tourney_date)
        match['rank_diff'] = rank_diff
        match['points_grad_diff'] = points_grad_diff

        # 6. Winner is always winner
        match['outcome'] = 1

        # Create a balanced set with equal outcomes
        if i % 2 == 0:
            match = -match

        # Update entry
        matches.iloc[i] = match

        # Update stats matrices
        match_d_weight = round(base_weight * time_weight)
        match_dt_weight = round(base_weight * time_weight * t_weights[raw_match.tourney_level])

        cond_stats['total_won'][winner_id] += match_dt_weight
        cond_stats['surface_' + surface + '_won'][winner_id] += match_d_weight
        cond_stats['total_lost'][loser_id] += match_dt_weight
        cond_stats['surface_' + surface + '_lost'][loser_id] += match_d_weight

        # Update mutual stats
        mutual_matches[winner_id][loser_id] += match_d_weight

        # Extract win on surface
        if surface == 'clay':
            mutual_matches_clay[winner_id][loser_id] += match_d_weight
        elif surface == 'grass':
            mutual_matches_grass[winner_id][loser_id] += match_d_weight
        else:
            mutual_matches_hard[winner_id][loser_id] += match_d_weight

        winner_games, loser_games = get_score(raw_match.score)

        mutual_games[winner_id][loser_id] += round(base_weight * time_weight * winner_games)
        mutual_games[loser_id][winner_id] += round(base_weight * time_weight * loser_games)

    print(no_matches, 'matches (100%) processed')

    matches.to_hd5('input/fixed/processed_matches.csv')

    print('Pre-processed data saved')
